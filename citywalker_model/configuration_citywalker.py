"""HuggingFace `PretrainedConfig` for the CityWalker waypoint-prediction model.

Mirrors the fields of upstream CityWalker's nested OmegaConf struct
(`config/finetune.yaml`) but in a flat, typed, JSON-serializable form so the
model round-trips through `save_pretrained` / `from_pretrained`.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class CityWalkerConfig(PretrainedConfig):
    model_type = "citywalker"

    def __init__(
        self,
        # Observation encoder (DINOv2 backbone).
        obs_encoder_type: str = "dinov2_vitb14",
        context_size: int = 5,
        crop: tuple[int, int] = (400, 400),
        resize: tuple[int, int] = (392, 392),
        freeze_obs_encoder: bool = True,
        # Coordinate embedding.
        cord_num_freqs: int = 6,
        cord_include_input: bool = True,
        # Image preprocessing inside the model forward pass (upstream behavior).
        do_rgb_normalize: bool = True,
        do_resize: bool = True,
        # Transformer decoder.
        decoder_num_heads: int = 8,
        decoder_num_layers: int = 16,
        decoder_ff_dim_factor: int = 4,
        # Output head.
        len_traj_pred: int = 5,
        **kwargs,
    ):
        self.obs_encoder_type = obs_encoder_type
        self.context_size = int(context_size)
        self.crop = tuple(crop)
        self.resize = tuple(resize)
        self.freeze_obs_encoder = bool(freeze_obs_encoder)
        self.cord_num_freqs = int(cord_num_freqs)
        self.cord_include_input = bool(cord_include_input)
        self.do_rgb_normalize = bool(do_rgb_normalize)
        self.do_resize = bool(do_resize)
        self.decoder_num_heads = int(decoder_num_heads)
        self.decoder_num_layers = int(decoder_num_layers)
        self.decoder_ff_dim_factor = int(decoder_ff_dim_factor)
        self.len_traj_pred = int(len_traj_pred)
        super().__init__(**kwargs)

    @property
    def feature_dim(self) -> int:
        """Feature width of the chosen DINOv2 variant."""
        return {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }[self.obs_encoder_type]
