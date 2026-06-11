"""Training: mixup/cutmix, weight EMA, cosine warmup, Trainer."""
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from devnet.config import RunConfig
from devnet.data import (
    ArrayDataset,
    build_eval_transform,
    build_train_transform,
    compute_mean_std,
    load_split,
    stratified_split,
)
from devnet.models.student import DevNet


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
    x = x.clone()
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
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


class Trainer:
    """Supervised trainer. Subclassed by DistillTrainer (later task) which
    overrides _loaders and _loss; batches may carry a third element (indices)."""

    def __init__(self, cfg: RunConfig, *, images=None, labels=None, num_classes=46):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.num_classes = num_classes
        torch.manual_seed(cfg.seed)

        if images is None:  # load real DHCD train split
            images, labels, _ = load_split(cfg.data_root, "Train")
        self.mean, self.std = compute_mean_std(images)
        tr_idx, va_idx = stratified_split(labels, cfg.val_fraction, seed=cfg.seed)

        self.train_ds = ArrayDataset(
            images[tr_idx], labels[tr_idx],
            build_train_transform(cfg.aug_tier, self.mean, self.std),
        )
        self.val_ds = ArrayDataset(
            images[va_idx], labels[va_idx],
            build_eval_transform(self.mean, self.std),
        )
        self.train_indices = tr_idx  # kept for KD logit alignment (distill task)

        self.model = DevNet(cfg.widths, cfg.depths, num_classes, cfg.dropout).to(self.device)
        self.ema = EMA(self.model, cfg.ema_decay)
        self.rng = np.random.default_rng(cfg.seed)

    def _loaders(self):
        cfg = self.cfg
        train = DataLoader(self.train_ds, cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=True,
                           drop_last=True)
        val = DataLoader(self.val_ds, cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True)
        return train, val

    def _loss(self, out, ya, yb, lam, extra=None):
        ls = self.cfg.label_smoothing
        return lam * F.cross_entropy(out, ya, label_smoothing=ls) + \
            (1 - lam) * F.cross_entropy(out, yb, label_smoothing=ls)

    @torch.no_grad()
    def _validate(self, loader):
        """Validation accuracy of the EMA weights."""
        model = DevNet(self.cfg.widths, self.cfg.depths, self.num_classes,
                       self.cfg.dropout).to(self.device)
        self.ema.copy_to(model)
        model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
        return correct / total

    def fit(self):
        cfg = self.cfg
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        train_loader, val_loader = self._loaders()
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        steps_per_epoch = max(1, len(train_loader))
        sched = cosine_warmup_scheduler(
            opt, cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch)
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)

        (out_dir / "config.json").write_text(
            json.dumps({**cfg.__dict__, "git_sha": git_sha(),
                        "mean": self.mean, "std": self.std}, default=str, indent=2))
        best_val = 0.0
        with open(out_dir / "log.csv", "w", newline="") as f:
            log = csv.writer(f)
            log.writerow(["epoch", "train_loss", "val_acc", "lr"])
            for epoch in range(cfg.epochs):
                self.model.train()
                running = 0.0
                for batch in train_loader:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    extra = batch[2] if len(batch) > 2 else None
                    if extra is not None:  # distillation: per-image teacher logits, no mixing
                        x, ya, yb, lam = x, y, y, 1.0
                    else:
                        x, ya, yb, lam = apply_mix(
                            x, y, mixup_alpha=cfg.mixup_alpha,
                            cutmix_alpha=cfg.cutmix_alpha,
                            mix_prob=cfg.mix_prob, rng=self.rng)
                    with torch.autocast(self.device.type, enabled=use_amp):
                        out = self.model(x)
                        loss = self._loss(out, ya, yb, lam, extra=extra)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    self.ema.update(self.model)
                    running += loss.item() * x.size(0)
                val_acc = self._validate(val_loader)
                log.writerow([epoch + 1, running / len(self.train_ds),
                              val_acc, opt.param_groups[0]["lr"]])
                f.flush()
                if val_acc >= best_val:
                    best_val = val_acc
                    torch.save({"model_state_dict": self.ema.shadow,
                                "config": cfg.__dict__,
                                "mean": self.mean, "std": self.std,
                                "val_acc": val_acc},
                               out_dir / "best.pth")
        return {"best_val_acc": best_val}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = RunConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = RunConfig(**{**cfg.__dict__, "seed": args.seed,
                           "out_dir": f"{cfg.out_dir}_seed{args.seed}"})
    if cfg.teacher_logits:
        from devnet.distill import DistillTrainer  # lazy: avoids circular import
        trainer_cls = DistillTrainer
    else:
        trainer_cls = Trainer
    result = trainer_cls(cfg).fit()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
