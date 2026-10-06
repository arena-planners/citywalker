"""CityWalker waypoint-prediction model, ported to a HuggingFace `PreTrainedModel`.

Port of `model/citywalker_feat.py` + supporting modules from
https://github.com/ai4ce/CityWalker, stripped of Lightning/OmegaConf.

Architecture (inference-only):

    images (B,T,3,H,W)  ──► DINOv2 ──► obs tokens  (B,T,D)
    coords (B,T+1,2)    ──► PolarEmbedding + Linear ──► goal token  (B,1,D)
                                                    ──► concat ──► (B,T+2,D)
                            ──► TransformerEncoder (self-attention decoder)
                            ──► MLP head  ──► (waypoints_pred, arrive_pred)

Outputs:
    waypoints_pred  : (B, len_traj_pred, 2)   cumulative XY deltas in body frame
    arrive_pred     : (B, 1)                  logits
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from transformers import Dinov2Config, Dinov2Model, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_citywalker import CityWalkerConfig


def _build_obs_encoder(name: str) -> Dinov2Model:
    """Build the DINOv2 backbone, working under both fresh-init and
    `from_pretrained` (which wraps __init__ in a `with torch.device("meta")`
    context starting in transformers 5.x).

    Inside the meta context, calling ``Dinov2Model.from_pretrained`` raises
    because nested `from_pretrained` calls are an anti-pattern: the outer
    loader is responsible for materializing weights. So when we detect the
    meta context, we just build the empty `Dinov2Model(config)` shell — the
    outer `from_pretrained` will populate the encoder weights from the
    bundled safetensors blob (which contains the encoder's weights via
    Phase 2's full-state-dict save).

    Outside the meta context (direct `CityWalkerModel(cfg)` construction),
    we still pull the real DINOv2 weights from `facebook/dinov2-*` so users
    instantiating from scratch get a useful backbone.
    """
    in_meta = (
        torch.device("meta") == _peek_default_device()
    )
    if in_meta:
        return Dinov2Model(Dinov2Config.from_pretrained(name))
    return Dinov2Model.from_pretrained(name)


def _peek_default_device() -> Optional[torch.device]:
    """Return the device set by the outermost `with torch.device(...)` /
    `torch.set_default_device(...)` context, or None if neither is active."""
    try:
        from transformers.modeling_utils import (
            get_torch_context_manager_or_global_device,
        )

        return get_torch_context_manager_or_global_device()
    except Exception:
        return None


# Map our `obs_encoder_type` strings (matching upstream torch.hub names) to
# the corresponding facebook/dinov2-* HF repo. We mirror only the four LVD142M
# no-register variants — same backbones, same weights, just shipped via HF
# instead of torch.hub. This is what lets us drop torch.hub entirely while
# keeping the legacy CityWalker `obs_encoder_type` strings working.
_DINOV2_HF_REPOS = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov2_vitg14": "facebook/dinov2-giant",
}


@dataclass
class CityWalkerOutput(ModelOutput):
    waypoints: torch.FloatTensor = None
    arrive_logits: torch.FloatTensor = None
    token_features: Optional[torch.FloatTensor] = None
    future_features: Optional[torch.FloatTensor] = None


class PolarEmbedding(nn.Module):
    """Fourier-feature encoding of 2D body-frame coordinates in polar form."""

    def __init__(self, num_freqs: int, include_input: bool):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freq_bands", freq_bands)
        self.out_dim = (2 if include_input else 0) + 4 * num_freqs

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x, y = coords[..., 0], coords[..., 1]
        r = torch.sqrt(x * x + y * y).unsqueeze(-1)
        theta = torch.atan2(y, x).unsqueeze(-1)

        parts = [r, theta] if self.include_input else []
        fb = self.freq_bands.view(1, 1, -1)
        parts.append(torch.sin(theta * fb))
        parts.append(torch.cos(theta * fb))
        parts.append(torch.sin(r * fb))
        parts.append(torch.cos(r * fb))
        return torch.cat(parts, dim=-1)


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (upstream naming preserved for weight-key parity)."""

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_enc[:, : x.size(1), :]


class _FeatPredictor(nn.Module):
    """Transformer self-attention stack over (context_size + 2) tokens."""

    def __init__(self, embed_dim: int, seq_len: int, nhead: int, num_layers: int, ff_dim_factor: int):
        super().__init__()
        self.positional_encoding = _PositionalEncoding(embed_dim, max_seq_len=seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim_factor * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_layer = layer
        self.sa_decoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa_decoder(self.positional_encoding(x))


class CityWalkerModel(PreTrainedModel):
    """HF-compatible CityWalker model. Inference path only; training stays upstream."""

    config_class = CityWalkerConfig
    base_model_prefix = "citywalker"
    supports_gradient_checkpointing = False
    main_input_name = "images"

    def __init__(self, config: CityWalkerConfig):
        super().__init__(config)
        self.config = config

        if config.do_rgb_normalize:
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if config.obs_encoder_type not in _DINOV2_HF_REPOS:
            raise ValueError(
                f"Unsupported obs_encoder_type: {config.obs_encoder_type!r}. "
                f"Expected one of {sorted(_DINOV2_HF_REPOS)}."
            )
        # DINOv2 backbone. See `_build_obs_encoder` — handles the case where
        # we're inside the outer `from_pretrained`'s meta-device context
        # (transformers 5.x) by building an empty shell that the outer
        # loader will fill from our safetensors blob.
        self.obs_encoder = _build_obs_encoder(
            _DINOV2_HF_REPOS[config.obs_encoder_type]
        )
        if config.freeze_obs_encoder:
            for p in self.obs_encoder.parameters():
                p.requires_grad = False
            self.obs_encoder.eval()
        self._feature_dim = config.feature_dim

        self.cord_embedding = PolarEmbedding(
            num_freqs=config.cord_num_freqs,
            include_input=config.cord_include_input,
        )
        cord_enc_dim = self.cord_embedding.out_dim * (config.context_size + 1)
        self.compress_goal_enc = nn.Linear(cord_enc_dim, self._feature_dim)

        self.predictor = _FeatPredictor(
            embed_dim=self._feature_dim,
            seq_len=config.context_size + 1,
            nhead=config.decoder_num_heads,
            num_layers=config.decoder_num_layers,
            ff_dim_factor=config.decoder_ff_dim_factor,
        )
        self.predictor_mlp = nn.Sequential(
            nn.Linear((config.context_size + 1) * self._feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.wp_predictor = nn.Linear(32, config.len_traj_pred * 2)
        self.arrive_predictor = nn.Linear(32, 1)

        self.post_init()

    def _encode_obs(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch through the DINOv2 backbone and return the CLS token.

        Upstream's torch.hub backbone returns ``head(x_norm_clstoken)`` (head
        is Identity for the pretrained variants), giving (B, feature_dim).
        HF's ``Dinov2Model`` returns ``BaseModelOutputWithPooling`` with
        ``last_hidden_state`` of shape (B, num_patches+1, feature_dim); the
        CLS token is at index 0 along the sequence dim. Using ``[:, 0]`` here
        matches upstream byte-for-byte at inference (same weights, same
        layernorm, same tokenization).
        """
        out = self.obs_encoder(pixel_values=x)
        return out.last_hidden_state[:, 0]

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.do_rgb_normalize:
            x = (x - self.mean) / self.std
        if self.config.do_resize:
            x = TF.center_crop(x, list(self.config.crop))
            x = TF.resize(x, list(self.config.resize))
        return x

    def forward(
        self,
        images: torch.Tensor,
        coords: torch.Tensor,
        future_images: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            images:        (B, context_size, 3, H, W) float tensor in [0, 1].
            coords:        (B, context_size + 1, 2) recent body-frame XY positions.
            future_images: optional (B, context_size, 3, H, W) for the
                feature-prediction head (unused at inference).
        """
        B, T, _, H, W = images.shape
        x = self._preprocess(images.view(B * T, 3, H, W))
        obs_enc = self._encode_obs(x).view(B, T, -1)

        future_enc: Optional[torch.Tensor] = None
        if future_images is not None:
            fx = self._preprocess(future_images.view(B * T, 3, H, W))
            future_enc = self._encode_obs(fx).view(B, T, -1)

        cord_enc = self.cord_embedding(coords).view(B, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(B, 1, -1)

        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        features = self.predictor(tokens)
        dec_out = self.predictor_mlp(features.view(B, -1))

        wp = self.wp_predictor(dec_out).view(B, self.config.len_traj_pred, 2)
        wp = torch.cumsum(wp, dim=1)
        arrive = self.arrive_predictor(dec_out).view(B, 1)

        if not return_dict:
            return wp, arrive, features[:, :-1], future_enc
        return CityWalkerOutput(
            waypoints=wp,
            arrive_logits=arrive,
            token_features=features[:, :-1],
            future_features=future_enc,
        )
