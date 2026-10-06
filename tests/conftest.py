from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from citywalker_model import modeling_citywalker
from citywalker_model.configuration_citywalker import CityWalkerConfig
from citywalker_model.modeling_citywalker import CityWalkerModel

FRAME_SHAPE = (96, 128, 3)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> CityWalkerModel:
    """A `CityWalkerModel` with a random one-layer DINOv2-small backbone, built without the hub."""
    torch.manual_seed(0)
    backbone_dir: pathlib.Path = tmp_path_factory.mktemp("dinov2-small")
    backbone = Dinov2Model(Dinov2Config(hidden_size=384, num_hidden_layers=1, num_attention_heads=6))
    backbone.save_pretrained(backbone_dir)
    modeling_citywalker._DINOV2_HF_REPOS["dinov2_vits14"] = str(backbone_dir)
    config = CityWalkerConfig(obs_encoder_type="dinov2_vits14", decoder_num_layers=1, crop=(70, 70), resize=(70, 70))
    return CityWalkerModel(config).eval()


def synthetic_frames(count: int, shape: tuple[int, int, int] = FRAME_SHAPE) -> list[np.ndarray]:
    """Ground-and-sky frames of a camera moving forward, the scene shifting by one row per frame."""
    height, width, _ = shape
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    cols = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    sky = np.array([0.55, 0.7, 0.9], dtype=np.float32)
    ground = np.array([0.45, 0.42, 0.4], dtype=np.float32)
    noise = np.random.default_rng(0).normal(0.0, 0.02, shape).astype(np.float32)
    base = np.where(rows < 0.5, sky * (1.0 - 0.2 * rows), ground * (0.6 + 0.8 * rows) * (0.8 + 0.4 * cols)) + noise
    base = (np.clip(base, 0.0, 1.0) * 255).astype(np.uint8)
    return [np.roll(base, index, axis=0) for index in range(count)]


@pytest.fixture(scope="session")
def frames() -> list[np.ndarray]:
    return synthetic_frames(CityWalkerConfig().context_size)
