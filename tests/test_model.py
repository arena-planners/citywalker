from __future__ import annotations

import math
import pathlib
import time

import numpy as np
import pytest
import torch
from arena_planners import weights
from arena_planners.sdk import Step
from conftest import synthetic_frames

import planner

_DIR = pathlib.Path(planner.__file__).parent
_SCALE = 0.5
_CAMERA_SHAPE = (480, 640, 3)
_PARAMETERS = 208_418_379


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@pytest.fixture(scope="module")
def citywalker() -> planner.CityWalker:
    missing = weights.missing(_DIR)
    if missing:
        pytest.skip(f"weights {missing} missing, run `arena feature planners add citywalker`")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    planner.on_init({"device": "auto", "step_scale_m": _SCALE})
    assert planner._planner is not None
    allocated = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
    print(f"\ncitywalker load {time.perf_counter() - start:.1f} s, {allocated:.2f} GiB allocated")
    return planner._planner


@pytest.fixture(scope="module")
def camera() -> list[np.ndarray]:
    return synthetic_frames(8, _CAMERA_SHAPE)


def _drive(citywalker: planner.CityWalker, camera: list[np.ndarray], goal: dict, ticks: int = 5) -> Step:
    """Walk straight along +x at the step scale for `ticks` ticks and return the last step."""
    citywalker.reset({"goal_pose": goal, "signal": "arrived"})
    result = Step()
    for index in range(ticks):
        features = {"camera": camera[index], "robot_pose": np.array([index * _SCALE, 0.0, 0.0])}
        _sync()
        start = time.perf_counter()
        result = citywalker.step(features)
        _sync()
        print(f"step {1000 * (time.perf_counter() - start):.0f} ms")
    return result


def _chunk(result: Step) -> np.ndarray:
    assert not result.action and not result.command
    chunk = np.asarray(result.chunk)
    assert chunk.shape == (5, 2)
    assert np.isfinite(chunk).all()
    return chunk


def test_checkpoint_matches_the_paper_and_crops_like_the_training_run(citywalker: planner.CityWalker):
    model = citywalker.policy.model
    assert sum(p.numel() for p in model.parameters()) == _PARAMETERS
    assert model.config.crop == (350, 630)
    assert model.config.resize == (350, 630)
    assert model.config.context_size == planner.CONTEXT_SIZE
    assert model.config.len_traj_pred == planner.NUM_WAYPOINTS


def test_goal_ahead_drives_forward(citywalker: planner.CityWalker, camera: list[np.ndarray]):
    goal = {"x": 2.0 + 10.0, "y": 0.0, "theta": 0.0}
    chunk = _chunk(_drive(citywalker, camera, goal))
    np.set_printoptions(precision=3, suppress=True)
    print(f"goal-ahead chunk (m):\n{chunk}")
    assert chunk[-1, 0] > 0.5
    assert abs(math.atan2(chunk[-1, 1], chunk[-1, 0])) < math.radians(30)
    latencies = []
    features = {"camera": camera[5], "robot_pose": np.array([2.5, 0.0, 0.0])}
    for _ in range(5):
        _sync()
        start = time.perf_counter()
        citywalker.step(features)
        _sync()
        latencies.append(1000 * (time.perf_counter() - start))
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
    print(f"steady step {np.mean(latencies):.0f} ms (min {min(latencies):.0f}), peak {peak:.2f} GiB")


@pytest.mark.parametrize(("side", "goal_y"), [("left", 6.0), ("right", -6.0)])
def test_goal_side_decides_the_turn(citywalker: planner.CityWalker, camera: list[np.ndarray], side: str, goal_y: float):
    chunk = _chunk(_drive(citywalker, camera, {"x": 2.0 + 6.0, "y": goal_y, "theta": 0.0}))
    print(f"goal {side} chunk (m):\n{chunk}")
    assert np.sign(chunk[-1, 1]) == np.sign(goal_y)


def test_arrival_fires_near_the_goal_and_not_far_from_it(citywalker: planner.CityWalker, camera: list[np.ndarray]):
    probabilities = {}
    for name, distance_m in (("near", 3.0), ("far", 20.0)):
        result = _drive(citywalker, camera, {"x": 2.0 + distance_m, "y": 0.0, "theta": 0.0})
        frames, poses = citywalker.history.context()
        rows = planner.coords(poses, citywalker.goal, _SCALE)
        _, logit = citywalker.policy.predict(planner.frames_tensor(frames), rows)
        probabilities[name] = 1.0 / (1.0 + math.exp(-logit))
        print(f"{name} goal ({distance_m} m, {distance_m / _SCALE:.0f} steps): arrival {probabilities[name]:.3f}")
        assert (result.signal == "arrived") is (probabilities[name] >= planner.ARRIVE_PROBABILITY)
    assert probabilities["near"] > probabilities["far"]
    assert probabilities["far"] < planner.ARRIVE_PROBABILITY


def test_same_observation_gives_the_same_chunk(citywalker: planner.CityWalker, camera: list[np.ndarray]):
    goal = {"x": 12.0, "y": 0.0, "theta": 0.0}
    first = _chunk(_drive(citywalker, camera, goal))
    second = _chunk(_drive(citywalker, camera, goal))
    np.testing.assert_array_equal(first, second)


def test_no_goal_holds_still(citywalker: planner.CityWalker, camera: list[np.ndarray]):
    citywalker.reset(None)
    assert citywalker.step({"camera": camera[0], "robot_pose": np.zeros(3)}) == Step()
