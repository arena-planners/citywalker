"""CityWalker wrapper for the arena_planners bridge."""

from __future__ import annotations

import collections
import logging
import math
import pathlib

import numpy as np
import torch
from arena_planners.sdk import Step, load_manifest, main_loop

from citywalker_model import modeling_citywalker
from citywalker_model.modeling_citywalker import CityWalkerModel

_DIR = pathlib.Path(__file__).parent
MODEL_DIR = _DIR / "model" / "citywalker"
DINOV2_DIR = _DIR / "model" / "dinov2-base"
_FETCH_HINT = "run `arena feature planners add citywalker` (or `python -m arena_planners fetch citywalker`)"

CONTEXT_SIZE = 5
NUM_WAYPOINTS = 5
GOAL_CLIP_STEPS = 50.0
ARRIVE_PROBABILITY = 0.5
_MIN_STEP_SCALE_M = 1e-2
_TRAIN_CROP = (350, 630)

_log = logging.getLogger("citywalker")


def to_robot_frame(robot: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map-frame points `(N, 2)` as `[forward, left]` meters in the frame of `robot = [x, y, theta]`."""
    rx, ry, rtheta = (float(v) for v in robot[:3])
    delta = np.asarray(points, dtype=np.float64).reshape(-1, 2) - (rx, ry)
    cos, sin = math.cos(rtheta), math.sin(rtheta)
    forward = delta[:, 0] * cos + delta[:, 1] * sin
    left = -delta[:, 0] * sin + delta[:, 1] * cos
    return np.stack([forward, left], axis=1)


def to_model_frame(forward_left: np.ndarray) -> np.ndarray:
    """Model coordinates `[right, forward]` from robot-frame `[forward, left]`."""
    points = np.asarray(forward_left, dtype=np.float64).reshape(-1, 2)
    return np.stack([-points[:, 1], points[:, 0]], axis=1)


def steering_goal(features: dict, goal: dict) -> dict:
    """The global plan's lookahead point when the bridge sends one as `subgoal`, else the episode goal."""
    subgoal = features.get("subgoal")
    if subgoal is None:
        return goal
    x, y, theta = (float(v) for v in np.asarray(subgoal, dtype=np.float64)[:3])
    return {"x": x, "y": y, "theta": theta}


def goal_in_model_frame(robot: np.ndarray, goal: dict, step_scale_m: float) -> np.ndarray:
    """Goal row `[right, forward]` in step units, clipped to the 50-step search window of training."""
    target = to_model_frame(to_robot_frame(robot, np.array([[goal["x"], goal["y"]]])))[0] / step_scale_m
    radius = float(np.hypot(*target))
    if radius > GOAL_CLIP_STEPS:
        target = target * GOAL_CLIP_STEPS / radius
    return target


def coords(poses: list[np.ndarray], goal: dict, step_scale_m: float) -> np.ndarray:
    """Model input `(6, 2)`: the 5 past positions (oldest first, the last one at the origin) and the goal row."""
    robot = poses[-1]
    history = to_model_frame(to_robot_frame(robot, np.asarray(poses)[:, :2])) / step_scale_m
    return np.vstack([history, goal_in_model_frame(robot, goal, step_scale_m)]).astype(np.float32)


def to_chunk(waypoints: np.ndarray, step_scale_m: float) -> list[list[float]]:
    """Waypoints `[x, y]` in meters (x forward, y left) from the model's `(5, 2)` `[right, forward]` step units."""
    points = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2) * step_scale_m
    return np.stack([points[:, 1], -points[:, 0]], axis=1).tolist()


def frames_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """Images `(1, 5, 3, H, W)` in [0, 1] from uint8 `(H, W, 3)` frames, oldest first."""
    stack = np.stack([np.asarray(frame)[..., :3] for frame in frames]).astype(np.float32) / 255.0
    return torch.from_numpy(stack).permute(0, 3, 1, 2).unsqueeze(0)


def arrived(logit: float) -> bool:
    """Upstream's arrival decision, `sigmoid(logit) >= 0.5`."""
    return 1.0 / (1.0 + math.exp(-logit)) >= ARRIVE_PROBABILITY


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_policy(
    model_dir: pathlib.Path,
    dinov2_dir: pathlib.Path,
    device: torch.device,
    crop: tuple[int, int] = _TRAIN_CROP,
    resize: tuple[int, int] = _TRAIN_CROP,
) -> CityWalkerModel:
    """The released checkpoint with the DINOv2 config resolved from `dinov2_dir`, cropping like the training run."""
    if not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"citywalker weights missing at {model_dir}, {_FETCH_HINT}")
    if not (dinov2_dir / "config.json").is_file():
        raise FileNotFoundError(f"dinov2 config missing at {dinov2_dir}, {_FETCH_HINT}")
    modeling_citywalker._DINOV2_HF_REPOS["dinov2_vitb14"] = str(dinov2_dir)
    model = CityWalkerModel.from_pretrained(model_dir)
    model.config.crop = tuple(int(v) for v in crop)
    model.config.resize = tuple(int(v) for v in resize)
    return model.to(device).eval()


class History:
    """The last 5 frames and poses at the planner tick, padded with the oldest entry until full."""

    def __init__(self) -> None:
        self.frames: collections.deque[np.ndarray] = collections.deque(maxlen=CONTEXT_SIZE)
        self.poses: collections.deque[np.ndarray] = collections.deque(maxlen=CONTEXT_SIZE)

    def clear(self) -> None:
        self.frames.clear()
        self.poses.clear()

    def push(self, frame: np.ndarray, pose: np.ndarray) -> None:
        self.frames.append(np.ascontiguousarray(np.asarray(frame)[..., :3], dtype=np.uint8))
        self.poses.append(np.asarray(pose, dtype=np.float64)[:3].copy())

    @staticmethod
    def _padded(items: collections.deque) -> list:
        return [items[0]] * (CONTEXT_SIZE - len(items)) + list(items)

    def context(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return self._padded(self.frames), self._padded(self.poses)


class Policy:
    """CityWalker's DINOv2 token stack, coordinate transformer and the waypoint and arrival heads."""

    def __init__(self, model: CityWalkerModel, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def predict(self, images: torch.Tensor, coordinates: np.ndarray) -> tuple[np.ndarray, float]:
        """Cumulative waypoints `(5, 2)` in `[right, forward]` step units and the arrival logit."""
        rows = torch.from_numpy(np.asarray(coordinates, dtype=np.float32)).unsqueeze(0)
        output = self.model(images.to(self.device), rows.to(self.device))
        return output.waypoints[0].float().cpu().numpy(), float(output.arrive_logits[0, 0])


class CityWalker:
    def __init__(self, policy: Policy, step_scale_m: float) -> None:
        if step_scale_m < _MIN_STEP_SCALE_M:
            raise ValueError(f"citywalker: step_scale_m {step_scale_m} below {_MIN_STEP_SCALE_M}")
        self.policy = policy
        self.step_scale_m = float(step_scale_m)
        self.history = History()
        self._warned: set[str] = set()
        self.reset(None)

    def reset(self, initial_state: dict | None) -> None:
        state = initial_state or {}
        self.goal: dict | None = state.get("goal_pose")
        self.signal = str(state.get("signal", ""))
        self.arrived = False
        self.history.clear()

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _log.warning(msg)

    def _signal(self) -> str:
        return self.signal if self.arrived else ""

    def step(self, features: dict) -> Step:
        image, robot_pose = features.get("camera"), features.get("robot_pose")
        if image is None or robot_pose is None or np.asarray(image).ndim != 3 or np.asarray(robot_pose).size < 3:
            self._warn_once("sensors", "citywalker: camera or robot_pose missing, standing still")
            return Step(signal=self._signal())
        self.history.push(image, robot_pose)
        if self.goal is None:
            return Step()
        frames, poses = self.history.context()
        waypoints, logit = self.policy.predict(
            frames_tensor(frames), coords(poses, steering_goal(features, self.goal), self.step_scale_m)
        )
        if not self.arrived and arrived(logit):
            self.arrived = True
            _log.warning("citywalker: arrival head fired (logit %.2f)", logit)
        return Step(chunk=to_chunk(waypoints, self.step_scale_m), signal=self._signal())


_planner: CityWalker | None = None


def on_init(config: dict) -> None:
    global _planner
    device = select_device(str(config.get("device", "auto")))
    model = load_policy(
        MODEL_DIR,
        DINOV2_DIR,
        device,
        tuple(config.get("crop", _TRAIN_CROP)),
        tuple(config.get("resize", _TRAIN_CROP)),
    )
    _planner = CityWalker(Policy(model, device), float(config.get("step_scale_m", 0.5)))
    _log.warning("citywalker: loaded on %s, step scale %.2f m", device, _planner.step_scale_m)


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    assert _planner is not None
    _planner.reset(initial_state)


def step(features: dict) -> Step:
    assert _planner is not None
    return _planner.step(features)


if __name__ == "__main__":
    logging.basicConfig()
    manifest = load_manifest(_DIR / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset, on_init=on_init)
