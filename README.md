# citywalker

[CityWalker](https://ai4ce.github.io/CityWalker/) (Liu et al., CVPR 2025, arXiv 2411.17820) behind the arena_planners
bridge: a 208M-parameter waypoint policy (frozen DINOv2 ViT-B/14 tokens, a 16-layer transformer and two MLP heads)
trained by imitation on more than 2000 hours of city walking and driving videos from the web. It takes the last five
camera frames, the last five robot positions and the goal as a point in the robot frame, and returns the next five
waypoints with an arrival flag, at 1 Hz, the rate the training samples were drawn at.

```sh
arena feature planners add citywalker
arena launch sim:=gazebo robot:=jackal[camera=vln_navid] robot.planner:=citywalker
```

`robot.planner:=citywalker` selects `robot.mobile:=vla`. The robot needs an `image` sensor in `model_params.yaml`, the
camera frame is resized to 640x480 on the bridge side. The weights take 0.8 GB on disk.

## Inputs

| Datasource | Topic (jackal `camera=vln_navid`) | Model input |
|---|---|---|
| `camera` (`image` sensor) | `vln_camera/image` | last 5 frames at 1 Hz, 640x480 RGB in [0, 1], center-cropped to 350x630 and ImageNet-normalized inside the model |
| `robot_pose` | tf | last 5 positions in the current robot frame, divided by `step_scale_m` |
| goal | the subgoal | the subgoal in the current robot frame, divided by `step_scale_m`, clipped to 50 steps |
| `subgoal` | `plan` from the nav2 planner server, `global_costmap/costmap` | the farthest point up to 6 m along the plan whose straight chord from the robot stays clear of the costmap, used in place of the goal pose, the goal pose itself until a plan arrives |

`depends.global_plan: true` makes the adapter start a nav2 planner server (`robot.mobile.global_planner`, default
`nav2/navfn`) that replans to the goal every second, and the `SubgoalGenerator` datasource turns the plan into
`subgoal`. The model steers at that point, so it only ever sees a goal in line of sight. A goal pose without a route
left the robot stuck behind walls, see Notes.

The model's coordinate axes are `[right, forward]`, the camera axes of the DPVO odometry the training labels come
from. The planner converts from and to the bridge's `[forward, left]` robot frame. Until five ticks have passed since
the reset, the oldest frame and position stand in for the missing ones.

Training frames are 360x640 first-person walking videos, center-cropped to 350x630 like the paper's Table II, and the
paper's robot runs used a webcam on a Unitree Go1. The paper states no field of view or mount height for either. Of the
jackal camera variants, `vln_navid` (640x480 RGB at 90 degrees HFOV, 0.98 m above the floor) comes closest to the
16:9 training frames, so the planner crops the central 350 rows and 630 columns of that frame. `vln_rgbd` publishes
224x224, smaller than the crop, and is not suitable.

The model was trained on urban outdoor walking video and the paper evaluates it on New York City sidewalks, in
scenarios it names turns, crossings, detours, proximity and crowds. It gives no indoor result, so an indoor Arena world
lies outside the evaluated domain. The released checkpoint is the `2000hr` web-video model before the teleoperation
fine-tuning of the paper's Table 1.

## Actions

Each tick the model predicts five cumulative waypoints in step units, one per second of training video. The planner
multiplies them by `step_scale_m` (default 0.5 m, the distance the robot covers in one tick at `chunk.speed_mps`) and
the bridge follows the path with pure pursuit (lookahead 0.5 m, 0.5 m/s, the jackal's linear limit) for one tick,
then asks again. The training data normalizes every trajectory by its mean step, the paper's deployment denormalizes
with the robot's step length, which `step_scale_m` sets.

The arrival head is trained to fire when the target lies 5 to 10 steps ahead (`arrived_threshold` of the training
config), 2.5 to 5 m at the default step scale. When it fires (`sigmoid(logit) >= 0.5`, the upstream validation rule)
the planner latches and sends `arrived` with every later chunk until the next reset, for a goto phase with
`signal: arrived`. Without an expected signal the planner keeps driving and the goal pose decides success.

```yaml
robots:
  - start: [1.0, 1.0, 0.0]
    phases:
      - goto: [8.0, 8.0, 0.0]
        signal: arrived
```

Before a goal arrives (a reset without one) the planner buffers frames and stands still without running the model.

## Config

Set in `planner.yaml > config`, override per launch with `robot.mobile.config.<key>:=<value>`.

| Key | Default | Effect |
|---|---|---|
| `device` | `auto` | Torch device, `auto` picks CUDA when available. |
| `step_scale_m` | `0.5` | Meters per model step unit, the robot's distance per tick. Keep it equal to `chunk.speed_mps / rate_hz`. |
| `crop` | `[350, 630]` | Center crop `[height, width]` of the 640x480 frame, the `citywalk_2000hr.yaml` value. |
| `resize` | `[350, 630]` | Size after the crop, multiples of 14. The Hugging Face config carries `[400, 400]` and `[392, 392]` from the fine-tuning run. |

## Weights and license

`arena feature planners add citywalker` fetches `model/citywalker/model.safetensors` and `config.json` from
[ai4ce/citywalker](https://huggingface.co/ai4ce/citywalker) (revision 1b5d169, the authors' conversion of
`CityWalker_2000hr.ckpt` with the DINOv2 backbone included) and the backbone configuration from
[facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base). Code and weights are Apache-2.0 (upstream
LICENSE, the model card and the DINOv2 model card), which permits redistribution, see LICENSE and ATTRIBUTION.md. Cite
Liu et al. 2025 when using this planner:

```bibtex
@inproceedings{liu2025citywalker,
  title={CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos},
  author={Liu, Xinhao and Li, Jintong and Jiang, Yicheng and Sujay, Niranjan and Yang, Zhicheng and Zhang, Juexiao and Abanes, John and Zhang, Jing and Feng, Chen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={6875--6885},
  year={2025}
}
```

## Notes

- Measured on an RTX PRO 6000 Blackwell with torch 2.14.1+cu130 and transformers 5.18.0: 0.4 s to load, 0.78 GiB
  allocated after load, 1.04 GiB peak, 30 ms per step at 640x480 after a first step of 0.34 s.
- The route decides whether a goal behind walls is reached. On ten start and goal pairs of hospital_1 (the five
  `hospital_complex` stages and five more from shifted seeds, 4 to 24 m apart, 2 to 8 pedestrians, 100 s and one
  episode per pair) the planner reached 6 goals steering by the subgoal and 3 steering by the goal pose.
- On synthetic frames with a straight 0.5 m/s history, a goal 3 m ahead (6 steps) gives arrival probability 0.95 and
  a goal 20 m ahead (40 steps) 0.001, goals 6 m to the left or right bend the path to that side.
- transformers 5 loads the checkpoint: it builds the model on the meta device, so the vendored code takes the
  config-only path for the DINOv2 backbone and `model/dinov2-base/config.json` replaces the hub lookup.
- `tests/test_model.py` needs the weights and runs the documented invariants (goal side, arrival near and far,
  determinism) on synthetic frames, the other tests build a one-layer random backbone and run on CPU.
