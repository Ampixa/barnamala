# tests/test_smoke_real_data.py
"""Integration gate (spec §5): a small model on a data subset must exceed
90% val accuracy in a ~2-minute CPU run BEFORE any rented-GPU run is launched."""
from pathlib import Path

import numpy as np
import pytest

from devnet.config import RunConfig
from devnet.train import Trainer

DATA_ROOT = Path("data/extracted/DevanagariHandwrittenCharacterDataset")


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="DHCD not downloaded")
def test_smoke_train_real_subset(tmp_path):
    from devnet.data import load_split

    images, labels, _ = load_split(DATA_ROOT, "Train")
    keep = np.concatenate(
        [np.where(labels == c)[0][:64] for c in np.unique(labels)])
    cfg = RunConfig(
        widths=(16, 32, 64), depths=(1, 1, 1), epochs=20, batch_size=128,
        lr=3e-3, warmup_epochs=1, aug_tier="light", mix_prob=0.0, ema_decay=0.9,
        num_workers=2, device="cpu", out_dir=str(tmp_path),
        val_fraction=0.15, seed=0,
    )
    trainer = Trainer(cfg, images=images[keep], labels=labels[keep])
    result = trainer.fit()
    assert result["best_val_acc"] > 0.90
