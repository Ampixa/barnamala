# tests/test_robustness.py
import torch

from devnet.robustness import CORRUPTIONS, corrupt


def test_all_corruptions_preserve_shape_and_range():
    x = torch.rand(4, 1, 32, 32)  # [0,1] floats, pre-normalization
    for kind in CORRUPTIONS:
        out = corrupt(x, kind, severity=3)
        assert out.shape == x.shape
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_severity_zero_is_identity():
    x = torch.rand(2, 1, 32, 32)
    for kind in CORRUPTIONS:
        assert torch.allclose(corrupt(x, kind, severity=0), x, atol=1e-6)


def test_noise_is_deterministic_given_generator():
    x = torch.rand(2, 1, 32, 32)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    assert torch.equal(corrupt(x, "noise", 3, generator=g1),
                       corrupt(x, "noise", 3, generator=g2))
