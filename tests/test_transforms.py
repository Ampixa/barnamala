# tests/test_transforms.py
import numpy as np
import torch
from torchvision.transforms import v2

from devnet.data import build_eval_transform, build_train_transform, compute_mean_std

FLIP_TYPES = (v2.RandomHorizontalFlip, v2.RandomVerticalFlip)


def _walk(transform):
    yield transform
    for child in getattr(transform, "transforms", []):
        yield from _walk(child)


def test_no_flip_in_any_tier():
    """Mirrored Devanagari characters are invalid; flips must never appear."""
    for tier in ["light", "medium", "heavy"]:
        t = build_train_transform(tier, mean=0.5, std=0.25)
        assert not any(isinstance(s, FLIP_TYPES) for s in _walk(t))


def test_train_transform_output_normalized():
    t = build_train_transform("medium", mean=0.5, std=0.25)
    x = torch.randint(0, 256, (1, 32, 32), dtype=torch.uint8)
    out = t(x)
    assert out.shape == (1, 32, 32)
    assert out.dtype == torch.float32
    assert out.abs().mean() < 5  # roughly standardized, not raw 0-255


def test_eval_transform_is_deterministic():
    t = build_eval_transform(mean=0.5, std=0.25)
    x = torch.randint(0, 256, (1, 32, 32), dtype=torch.uint8)
    assert torch.equal(t(x), t(x))


def test_compute_mean_std_on_train_only():
    images = np.full((10, 32, 32), 128, dtype=np.uint8)
    mean, std = compute_mean_std(images)
    assert abs(mean - 128 / 255) < 1e-6
    assert std >= 0.0
