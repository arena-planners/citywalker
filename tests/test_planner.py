from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import torch
import yaml
from arena_planners import goal, weights
from arena_planners.bridge.chunk import check_chunk, chunk_config_from_manifest

import planner
from citywalker_model.modeling_citywalker import CityWalkerModel

_PLANNER_DIR = pathlib.Path(planner.__file__).parent
_CPU = torch.device("cpu")
_SCALE = 0.5
_GOAL = {"x": 7.0, "y": 0.0, "theta": 0.0}
_HEADS = ("wp_predictor", "arrive_predictor")


def _manifest() -> dict:
    with open(_PLANNER_DIR / "planner.yaml") as fh:
        return yaml.safe_load(fh)


def _walk(count: int, step_m: float = _SCALE) -> list[np.ndarray]:
    return [np.array([index * step_m, 0.0, 0.0]) for index in range(count)]


def _force(model: CityWalkerModel, deltas: list[list[float]], arrive_logit: float) -> None:
    with torch.no_grad():
        model.wp_predictor.weight.zero_()
        model.wp_predictor.bias.copy_(torch.tensor(deltas, dtype=torch.float32).flatten())
        model.arrive_predictor.weight.zero_()
        model.arrive_predictor.bias.fill_(arrive_logit)


@pytest.fixture
def citywalker(tiny_model: CityWalkerModel) -> planner.CityWalker:
    original = {name: {k: v.clone() for k, v in getattr(tiny_model, name).state_dict().items()} for name in _HEADS}
    yield planner.CityWalker(planner.Policy(tiny_model, _CPU), _SCALE)
    for name in _HEADS:
        getattr(tiny_model, name).load_state_dict(original[name])


def _features(frames: list[np.ndarray], index: int = 0) -> dict:
    return {"camera": frames[index % len(frames)], "robot_pose": np.array([index * _SCALE, 0.0, 0.0])}


def test_robot_frame_rotates_by_heading():
    robot = np.array([1.0, 1.0, math.pi / 2])
    points = np.array([[1.0, 3.0], [0.0, 1.0]])
    np.testing.assert_allclose(planner.to_robot_frame(robot, points), [[2, 0], [0, 1]], atol=1e-9)


def test_model_frame_is_right_then_forward():
    np.testing.assert_allclose(planner.to_model_frame(np.array([[1.0, 2.0]])), [[-2.0, 1.0]])


def test_goal_ahead_lies_on_the_forward_axis_in_step_units():
    np.testing.assert_allclose(planner.goal_in_model_frame(np.zeros(3), {"x": 2.0, "y": 0.0}, _SCALE), [0.0, 4.0])


def test_goal_left_has_negative_model_x():
    np.testing.assert_allclose(planner.goal_in_model_frame(np.zeros(3), {"x": 0.0, "y": 2.0}, _SCALE), [-4.0, 0.0])


def test_goal_clips_to_the_training_search_window():
    far = planner.goal_in_model_frame(np.zeros(3), {"x": 60.0, "y": 80.0}, _SCALE)
    np.testing.assert_allclose(far, [-40.0, 30.0], atol=1e-9)
    assert np.hypot(*far) == pytest.approx(planner.GOAL_CLIP_STEPS)


def test_coords_stack_history_oldest_first_with_the_current_pose_at_the_origin():
    stack = planner.coords(_walk(5), _GOAL, _SCALE)
    assert stack.shape == (6, 2)
    assert stack.dtype == np.float32
    np.testing.assert_allclose(stack[:5], [[0, -4], [0, -3], [0, -2], [0, -1], [0, 0]], atol=1e-6)
    np.testing.assert_allclose(stack[5], [0.0, 10.0], atol=1e-6)


def test_coords_follow_the_current_heading():
    poses = [np.array([0.0, -2.0, math.pi / 2]), np.array([0.0, -1.0, math.pi / 2]), np.array([0.0, 0.0, math.pi / 2])]
    stack = planner.coords(poses, {"x": -1.0, "y": 0.0}, 1.0)
    np.testing.assert_allclose(stack, [[0, -2], [0, -1], [0, 0], [-1, 0]], atol=1e-6)


def test_chunk_maps_step_units_to_robot_meters():
    waypoints = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]])
    chunk = planner.to_chunk(waypoints, _SCALE)
    assert len(chunk) == planner.NUM_WAYPOINTS
    np.testing.assert_allclose(chunk[0], [1.0, -0.5])
    np.testing.assert_allclose(chunk[-1], [5.0, -2.5])
    assert check_chunk("waypoints", chunk) == chunk


def test_frames_tensor_drops_alpha_and_scales_to_unit_range(frames: list[np.ndarray]):
    rgba = [np.concatenate([frame, np.full(frame.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1) for frame in frames]
    tensor = planner.frames_tensor(rgba)
    assert tensor.shape == (1, 5, 3, 96, 128)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0
    torch.testing.assert_close(tensor[0, 0, :, 0, 0], torch.from_numpy(frames[0][0, 0]).float() / 255.0)


def test_history_pads_with_the_oldest_entry(frames: list[np.ndarray]):
    history = planner.History()
    history.push(frames[0], np.array([0.0, 0.0, 0.0]))
    history.push(frames[1], np.array([0.5, 0.0, 0.0]))
    padded_frames, padded_poses = history.context()
    assert len(padded_frames) == len(padded_poses) == planner.CONTEXT_SIZE
    assert all(np.array_equal(frame, frames[0]) for frame in padded_frames[:4])
    assert np.array_equal(padded_frames[4], frames[1])
    np.testing.assert_array_equal(np.asarray(padded_poses)[:, 0], [0.0, 0.0, 0.0, 0.0, 0.5])


def test_history_keeps_the_last_five(frames: list[np.ndarray]):
    history = planner.History()
    for index in range(8):
        history.push(frames[index % 5], np.array([float(index), 0.0, 0.0]))
    _, poses = history.context()
    np.testing.assert_array_equal(np.asarray(poses)[:, 0], [3.0, 4.0, 5.0, 6.0, 7.0])


@pytest.mark.parametrize(("logit", "expected"), [(0.0, True), (-0.01, False), (8.0, True), (-8.0, False)])
def test_arrival_decides_at_probability_one_half(logit: float, expected: bool):
    assert planner.arrived(logit) is expected


def test_manifest_is_a_chunked_point_goal_planner_with_an_arrival_signal():
    manifest = _manifest()
    assert manifest["action_type"] == "waypoints"
    assert manifest["rate_hz"] == 1
    assert goal.declared(manifest) == frozenset({"pose"})
    assert manifest["signals"] == ["arrived"]
    config = chunk_config_from_manifest(manifest)
    assert config.execute_s == pytest.approx(1.0 / manifest["rate_hz"])
    assert config.speed_mps == pytest.approx(manifest["config"]["step_scale_m"] * manifest["rate_hz"])
    assert manifest["observations"]["datasources"]["camera"]["params"]["output_size"] == [640, 480]
    assert manifest["config"]["crop"] == manifest["config"]["resize"] == [350, 630]


def test_weights_manifest_pins_the_hub_revisions():
    entries = weights.read(_PLANNER_DIR)
    assert {entry.dest for entry in entries} == {
        "model/citywalker/model.safetensors",
        "model/citywalker/config.json",
        "model/dinov2-base/config.json",
    }
    assert all(entry.revision and len(entry.revision) == 40 for entry in entries)
    assert {entry.repo for entry in entries} == {"ai4ce/citywalker", "facebook/dinov2-base"}


def test_missing_weights_name_the_fetch_command(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match="arena feature planners add citywalker"):
        planner.load_policy(tmp_path / "citywalker", tmp_path / "dinov2-base", _CPU)


def test_step_scale_below_the_training_floor_is_refused(tiny_model: CityWalkerModel):
    with pytest.raises(ValueError, match="step_scale_m"):
        planner.CityWalker(planner.Policy(tiny_model, _CPU), 0.001)


def test_reset_without_goal_buffers_and_stands_still(citywalker: planner.CityWalker, frames: list[np.ndarray]):
    citywalker.reset(None)
    assert citywalker.step(_features(frames)) == planner.Step()
    assert len(citywalker.history.poses) == 1


@pytest.mark.parametrize("missing", ["camera", "robot_pose"])
def test_missing_sensor_stands_still(citywalker: planner.CityWalker, frames: list[np.ndarray], missing: str):
    citywalker.reset({"goal_pose": _GOAL})
    assert citywalker.step({**_features(frames), missing: None}) == planner.Step()
    assert len(citywalker.history.poses) == 0


def test_step_returns_five_waypoints_in_the_robot_frame(citywalker: planner.CityWalker, frames: list[np.ndarray]):
    _force(citywalker.policy.model, [[1.0, 2.0]] * 5, -8.0)
    citywalker.reset({"goal_pose": _GOAL, "signal": "arrived"})
    result = citywalker.step(_features(frames))
    assert result.signal == ""
    assert len(result.chunk) == 5
    np.testing.assert_allclose(result.chunk, [[k, -0.5 * k] for k in range(1, 6)], atol=1e-5)


@pytest.mark.parametrize(("state", "expected"), [({"signal": "arrived"}, "arrived"), ({}, "")])
def test_arrival_latches_and_signals_only_when_asked(
    citywalker: planner.CityWalker, frames: list[np.ndarray], state: dict, expected: str
):
    _force(citywalker.policy.model, [[0.0, 1.0]] * 5, 8.0)
    citywalker.reset({"goal_pose": _GOAL, **state})
    first = citywalker.step(_features(frames))
    assert first.signal == expected
    assert len(first.chunk) == 5
    _force(citywalker.policy.model, [[0.0, 1.0]] * 5, -8.0)
    for index in range(1, 4):
        later = citywalker.step(_features(frames, index))
        assert later.signal == expected
        assert len(later.chunk) == 5
    assert citywalker.arrived


def test_reset_clears_the_latch_and_the_history(citywalker: planner.CityWalker, frames: list[np.ndarray]):
    _force(citywalker.policy.model, [[0.0, 1.0]] * 5, 8.0)
    citywalker.reset({"goal_pose": _GOAL, "signal": "arrived"})
    citywalker.step(_features(frames))
    citywalker.reset({"goal_pose": _GOAL})
    assert not citywalker.arrived
    assert citywalker.signal == ""
    assert len(citywalker.history.frames) == 0
    _force(citywalker.policy.model, [[0.0, 1.0]] * 5, -8.0)
    assert citywalker.step(_features(frames)).signal == ""


def test_steps_are_deterministic(citywalker: planner.CityWalker, frames: list[np.ndarray]):
    runs = []
    for _ in range(2):
        citywalker.reset({"goal_pose": _GOAL})
        runs.append([citywalker.step(_features(frames, index)).chunk for index in range(3)])
    np.testing.assert_array_equal(runs[0], runs[1])


def test_steering_goal_prefers_the_plan_subgoal():
    goal_pose = {"x": 9.0, "y": 0.0, "theta": 0.0}
    assert planner.steering_goal({}, goal_pose) == goal_pose
    assert planner.steering_goal({"subgoal": None}, goal_pose) == goal_pose
    subgoal = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    assert planner.steering_goal({"subgoal": subgoal}, goal_pose) == {"x": 1.0, "y": 2.0, "theta": 0.5}


def test_manifest_steers_by_the_global_plan_subgoal():
    manifest = _manifest()
    assert manifest["depends"]["global_plan"] is True
    with pytest.raises(ValueError):
        goal.withhold_pose(manifest)
    sources = manifest["observations"]["datasources"]
    assert sources["subgoal"] == {"type": "SubgoalGenerator", "params": {"lookahead": 6.0}}
    assert sources["robot_pose_from_tf"]["type"] == "RobotPoseTFGenerator"
    assert sources["goal_pose"] == {"type": "geometry_msgs/PoseStamped", "params": {"topic": "goal_pose"}}
    assert sources["global_plan"] == {"type": "nav_msgs/Path", "params": {"topic": "plan"}}
    assert sources["costmap"] == {"type": "nav_msgs/OccupancyGrid", "params": {"topic": "global_costmap/costmap"}}
    names = list(sources)
    assert names.index("robot_pose_from_tf") < names.index("subgoal")
