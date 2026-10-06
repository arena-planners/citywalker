# Attribution

## Code

Model code adapted from the ai4ce/citywalker Hugging Face repository (revision 1b5d169, 2026-07), the authors' port
of ai4ce/CityWalker (commit ab0ef60, 2025-08-05) to a `transformers.PreTrainedModel`.
Sources: https://huggingface.co/ai4ce/citywalker, https://github.com/ai4ce/CityWalker

`citywalker_model/` holds these files verbatim (the import between them is already relative):

- `configuration_citywalker.py`, `modeling_citywalker.py`

They port `model/citywalker_feat.py` and `model/model_utils.py` (`PolarEmbedding`, `PositionalEncoding`,
`FeatPredictor`) of ai4ce/CityWalker, with the torch.hub DINOv2 backbone replaced by `transformers.Dinov2Model`. The
same two files sit in ai4ce/wanderland-benchmark (commit e718d02) under `src/wanderland_lab/models/citywalker/`.

`planner.py` ports the sample construction of `data/citywalk_feat_dataset.py` for the `citywalk_2000hr.yaml` run:
five past frames and positions at 1 Hz, positions in the frame of the current pose with the model axes `[right,
forward]` (columns 0 and 2 of the DPVO camera-frame positions), the goal appended as the sixth coordinate row, all
positions divided by the step scale, and the five cumulative waypoints multiplied back. The step scale is a config
value instead of the per-video mean step of training, the goal row is clipped to the 50-step `search_window`, and the
arrival decision is the `sigmoid(logit) >= 0.5` of the upstream validation metric. The image crop follows the
`citywalk_2000hr.yaml` training run (350x630 of a 360x640 frame) rather than the 400 to 392 square crop the Hugging Face
config carries from `finetune.yaml`. `planner.py` also points the vendored module's `facebook/dinov2-base` lookup at
`model/dinov2-base/`, so the backbone config loads from the fetched weights instead of the hub.

## Weights

https://huggingface.co/ai4ce/citywalker (revision 1b5d169, model card Apache-2.0): `CityWalker_2000hr.ckpt` of
the GitHub releases converted to safetensors, DINOv2 ViT-B/14 included. `model/dinov2-base/config.json` is the
backbone configuration from https://huggingface.co/facebook/dinov2-base (revision f9e44c8, Apache-2.0). The Apache
License 2.0 permits redistribution (section 4), so an arena-rosnav mirror is allowed but not needed.

See LICENSE for upstream terms (Apache-2.0).

Cite:

```bibtex
@inproceedings{liu2025citywalker,
  title={CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos},
  author={Liu, Xinhao and Li, Jintong and Jiang, Yicheng and Sujay, Niranjan and Yang, Zhicheng and Zhang, Juexiao and Abanes, John and Zhang, Jing and Feng, Chen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={6875--6885},
  year={2025}
}
```
