# tests/test_mallanet_baseline.py
from pathlib import Path

import pytest
import torch

from devnet.models.mallanet_baseline import EnhancedBMCNNwHFCs

WEIGHTS = Path("/tmp/mallanet_repo/models/best_model.pth")


def test_architecture_output_shape():
    model = EnhancedBMCNNwHFCs(num_classes=46)
    out = model(torch.randn(2, 1, 32, 32))
    assert out.shape == (2, 46)


def test_param_count_matches_paper():
    model = EnhancedBMCNNwHFCs(num_classes=46)
    n = sum(p.numel() for p in model.parameters())
    assert abs(n - 17_320_579) / 17_320_579 < 0.02  # paper reports 17,320,579


@pytest.mark.skipif(not WEIGHTS.exists(), reason="MallaNet checkpoint not present")
def test_published_checkpoint_loads():
    model = EnhancedBMCNNwHFCs(num_classes=46)
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])  # raises if mismatched
