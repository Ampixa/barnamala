"""YAML-backed run configuration."""
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunConfig:
    # model
    widths: tuple = (40, 80, 160)
    depths: tuple = (2, 2, 2)
    dropout: float = 0.1
    # optimization
    epochs: int = 300
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    ema_decay: float = 0.999
    # regularization
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5
    aug_tier: str = "medium"
    # data / run
    data_root: str = "data/extracted/DevanagariHandwrittenCharacterDataset"
    val_fraction: float = 0.1
    seed: int = 0
    num_workers: int = 4
    device: str = "auto"  # auto -> cuda if available
    out_dir: str = "results/run"
    # distillation (None = plain supervised run)
    teacher_logits: str | None = None
    kd_temperature: float = 4.0
    kd_alpha: float = 0.7

    @classmethod
    def from_yaml(cls, path) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise TypeError(f"Unknown config keys: {sorted(unknown)}")
        for key in ("widths", "depths"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)
