# tests/test_trainer.py
import numpy as np
import torch

from devnet.config import RunConfig
from devnet.train import Trainer


def test_trainer_overfits_tiny_synthetic_set(tmp_path):
    """A tiny model must memorize 4 distinct constant images in a few epochs."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (4, 32, 32), dtype=np.uint8)
    images = np.repeat(base, 16, axis=0)              # 64 images, 4 patterns
    labels = np.repeat(np.arange(4), 16).astype(np.int64)

    cfg = RunConfig(
        widths=(8, 16, 32), depths=(1, 1, 1), epochs=15, batch_size=16,
        lr=3e-3, warmup_epochs=1, mix_prob=0.0, aug_tier="none",
        ema_decay=0.9, num_workers=0, device="cpu",
        out_dir=str(tmp_path), val_fraction=0.25, seed=0,
    )
    trainer = Trainer(cfg, images=images, labels=labels, num_classes=4)
    result = trainer.fit()
    assert result["best_val_acc"] >= 0.99
    assert (tmp_path / "best.pth").exists()
    assert (tmp_path / "log.csv").exists()
