"""Training: mixup/cutmix, weight EMA, cosine warmup, Trainer."""
import math

import numpy as np
import torch


def apply_mix(x, y, *, mixup_alpha, cutmix_alpha, mix_prob, rng):
    """Apply mixup OR cutmix (coin flip) with probability mix_prob.

    Returns (x, y_a, y_b, lam); loss = lam*CE(out,y_a) + (1-lam)*CE(out,y_b).
    """
    if rng.random() >= mix_prob:
        return x, y, y, 1.0
    perm = torch.randperm(x.size(0))
    if rng.random() < 0.5:  # mixup
        lam = float(rng.beta(mixup_alpha, mixup_alpha))
        x = lam * x + (1 - lam) * x[perm]
        return x, y, y[perm], lam
    # cutmix
    lam = float(rng.beta(cutmix_alpha, cutmix_alpha))
    H, W = x.shape[-2:]
    rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = int(rng.integers(H)), int(rng.integers(W))
    y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
    x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
    x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, y, y[perm], lam


class EMA:
    """Exponential moving average of model weights (float tensors only)."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
