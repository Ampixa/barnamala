# tests/test_distill.py
import numpy as np
import torch

from devnet.distill import DistillTrainer, kd_loss
from devnet.config import RunConfig


def test_kd_loss_zero_when_matching_teacher():
    logits = torch.randn(8, 46)
    targets = torch.randint(0, 46, (8,))
    full = kd_loss(logits, logits, targets, temperature=4.0, alpha=1.0)
    assert full.item() < 1e-6  # pure KD term vanishes when student == teacher


def test_kd_loss_alpha_zero_is_plain_ce():
    s = torch.randn(8, 46)
    t = torch.randn(8, 46)
    y = torch.randint(0, 46, (8,))
    ce = torch.nn.functional.cross_entropy(s, y, label_smoothing=0.1)
    assert torch.allclose(kd_loss(s, t, y, temperature=4.0, alpha=0.0), ce)


def test_distill_trainer_runs_and_uses_teacher(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (4, 32, 32), dtype=np.uint8)
    images = np.repeat(base, 16, axis=0)
    labels = np.repeat(np.arange(4), 16).astype(np.int64)
    teacher_logits = np.eye(4, dtype=np.float32)[labels] * 8  # perfect teacher

    logits_path = tmp_path / "teacher_logits.npz"
    np.savez(logits_path, logits=teacher_logits)

    cfg = RunConfig(
        widths=(8, 16, 32), depths=(1, 1, 1), epochs=10, batch_size=16,
        lr=3e-3, warmup_epochs=1, mix_prob=0.0, aug_tier="none",
        ema_decay=0.9, num_workers=0, device="cpu",
        out_dir=str(tmp_path / "run"), val_fraction=0.25, seed=0,
        teacher_logits=str(logits_path), kd_alpha=0.5,
    )
    trainer = DistillTrainer(cfg, images=images, labels=labels, num_classes=4)
    result = trainer.fit()
    assert result["best_val_acc"] >= 0.99


def test_distill_trainer_rejects_mismatched_teacher_file(tmp_path):
    import pytest

    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, (64, 32, 32), dtype=np.uint8)
    labels = np.repeat(np.arange(4), 16).astype(np.int64)
    bad_logits = np.zeros((10, 4), dtype=np.float32)  # wrong row count

    logits_path = tmp_path / "bad.npz"
    np.savez(logits_path, logits=bad_logits)
    cfg = RunConfig(
        widths=(8, 16, 32), depths=(1, 1, 1), epochs=1, batch_size=16,
        num_workers=0, device="cpu", out_dir=str(tmp_path / "run"),
        val_fraction=0.25, seed=0, teacher_logits=str(logits_path),
    )
    with pytest.raises(ValueError, match="teacher logits shape"):
        DistillTrainer(cfg, images=images, labels=labels, num_classes=4)
