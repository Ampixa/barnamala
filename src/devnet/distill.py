"""Knowledge distillation: KD loss and a Trainer that adds a teacher term.

teacher_logits npz layout: key 'logits', shape [N_train_full, C], row i
corresponding to row i of the FULL training array (before val carve-out) —
produced by scripts/dump_teacher_logits.py using identical data_root and order.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from devnet.train import Trainer


def kd_loss(student_logits, teacher_logits, targets, *, temperature, alpha,
            label_smoothing: float = 0.1):
    """alpha * KL(teacher || student, temperature-softened) + (1-alpha) * CE."""
    kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.log_softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
        log_target=True,
    ) * temperature * temperature
    ce = F.cross_entropy(student_logits, targets, label_smoothing=label_smoothing)
    return alpha * kl + (1 - alpha) * ce


class _IndexedDataset(Dataset):
    """Wraps a dataset so batches carry original indices (for logit lookup)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i


class DistillTrainer(Trainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        full = np.load(cfg.teacher_logits)["logits"]
        n_full = len(self.train_indices) + len(self.val_ds)
        if full.shape != (n_full, self.num_classes):
            raise ValueError(
                f"teacher logits shape {full.shape} does not match "
                f"(full train size, num_classes) = ({n_full}, {self.num_classes}); "
                "stale or mismatched npz?")
        # align teacher logits with the post-split training subset; keep on the
        # training device to avoid a per-batch host-to-device copy
        self.teacher = torch.from_numpy(full[self.train_indices]).float().to(self.device)

    def _loaders(self):
        train, val = super()._loaders()
        indexed = DataLoader(
            _IndexedDataset(self.train_ds), self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True, drop_last=True)
        return indexed, val

    def _loss(self, out, ya, yb, lam, extra=None):
        # mixing is disabled for distillation runs (teacher logits are per-image)
        t = self.teacher[extra.to(self.teacher.device)]
        return kd_loss(out, t, ya, temperature=self.cfg.kd_temperature,
                       alpha=self.cfg.kd_alpha,
                       label_smoothing=self.cfg.label_smoothing)
