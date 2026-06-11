# Devanagari SOTA (DevNet) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full training/evaluation pipeline for a statistically-defended SOTA on the 46-class DHCD benchmark: compact SE-ResNet student, teacher ensemble + distillation, script-aware augmentation, and paired statistical comparison against a reproduced MallaNet baseline.

**Architecture:** A small installable Python package (`src/devnet/`) with pure-function statistics, an in-memory DHCD data pipeline with stratified validation carve-out, a width/depth-configurable pre-activation SE-ResNet, a single trainer used for both supervised and distillation runs, and an evaluation module that dumps per-image predictions to disk so every statistic (Wilson CI, McNemar, ECE) is recomputable without retraining.

**Tech Stack:** Python ≥3.10, PyTorch ≥2.3, torchvision ≥0.18 (v2 transforms), numpy, scikit-learn, PyYAML, pytest. Dev/debug on CPU; full runs on a rented single GPU.

**Spec:** `docs/superpowers/specs/2026-06-11-devanagari-sota-design.md`

---

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, pytest config |
| `src/devnet/__init__.py` | Package marker |
| `src/devnet/stats.py` | Wilson CI, exact McNemar, ECE — pure functions |
| `src/devnet/data.py` | DHCD loading, stratified split, script-aware transforms |
| `src/devnet/models/__init__.py` | Model registry |
| `src/devnet/models/student.py` | Pre-act SE-ResNet (student & teacher via width/depth) |
| `src/devnet/models/mallanet_baseline.py` | Ported MallaNet architecture (comparison only) |
| `src/devnet/train.py` | Mixup/CutMix, EMA, cosine-warmup, Trainer |
| `src/devnet/distill.py` | KD loss + distillation training |
| `src/devnet/evaluate.py` | Prediction dumps, metrics, TTA, ensembling, paired tests |
| `src/devnet/robustness.py` | Noise/blur/contrast corruption sweeps |
| `src/devnet/config.py` | YAML-backed run configuration |
| `scripts/download_data.sh` | Fetch + verify DHCD |
| `scripts/reproduce_mallanet.py` | Run MallaNet checkpoint → prediction dump |
| `scripts/run_seeds.sh` | Multi-seed launcher |
| `configs/*.yaml` | Experiment configs |
| `tests/*` | See tasks |

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/devnet/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "devnet"
version = "0.1.0"
description = "Statistically-defended SOTA for handwritten Devanagari character recognition"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.3",
    "torchvision>=0.18",
    "numpy",
    "scikit-learn",
    "pyyaml",
    "pillow",
    "tqdm",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.egg-info/
data/raw/
data/extracted/
results/*/checkpoints/
*.pth
*.npz
```

- [ ] **Step 3: Create package markers**

`src/devnet/__init__.py` and `tests/__init__.py` — both empty files.

- [ ] **Step 4: Create venv and install**

Run: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
Expected: installs torch et al., exits 0. (Use `.venv/bin/python` / `.venv/bin/pytest` for every later step.)

- [ ] **Step 5: Verify pytest runs**

Run: `.venv/bin/pytest`
Expected: `no tests ran` (exit code 5 is fine at this stage)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "chore: project scaffold with devnet package"
```

---

### Task 2: Statistics module (Wilson CI, exact McNemar, ECE)

**Files:**
- Create: `src/devnet/stats.py`
- Test: `tests/test_stats.py`

Reference values below were computed independently during design (spec §2).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stats.py
import numpy as np
import pytest

from devnet.stats import expected_calibration_error, mcnemar_exact, wilson_ci


def test_wilson_ci_matches_reference():
    # 40 errors out of 13800 -> acc 99.710%, CI [99.606, 99.787] (spec §2)
    lo, hi = wilson_ci(correct=13760, n=13800)
    assert lo == pytest.approx(0.99606, abs=2e-5)
    assert hi == pytest.approx(0.99787, abs=2e-5)


def test_wilson_ci_bounds_are_probabilities():
    lo, hi = wilson_ci(correct=0, n=10)
    assert 0.0 <= lo <= hi <= 1.0
    lo, hi = wilson_ci(correct=10, n=10)
    assert 0.0 <= lo <= hi <= 1.0


def test_mcnemar_reference_values():
    # From spec §2 power table
    assert mcnemar_exact(30, 15) == pytest.approx(0.0357, abs=2e-4)
    assert mcnemar_exact(25, 10) == pytest.approx(0.0167, abs=2e-4)
    assert mcnemar_exact(35, 30) == pytest.approx(0.6201, abs=2e-4)


def test_mcnemar_edge_cases():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert mcnemar_exact(10, 3) == mcnemar_exact(3, 10)  # symmetric


def test_ece_perfectly_calibrated_is_zero():
    conf = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    correct = np.array([1, 1, 1, 1, 0])  # 80% accuracy at 80% confidence
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_overconfident():
    conf = np.array([0.99, 0.99, 0.99, 0.99])
    correct = np.array([1, 1, 0, 0])  # 50% accuracy at 99% confidence
    assert expected_calibration_error(conf, correct) == pytest.approx(0.49, abs=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'devnet.stats'`

- [ ] **Step 3: Implement `src/devnet/stats.py`**

```python
"""Pure statistical functions for benchmark evaluation."""
from math import comb, sqrt

import numpy as np


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def mcnemar_exact(n10: int, n01: int) -> float:
    """Two-sided exact McNemar test on discordant pair counts.

    n10: baseline wrong, candidate right.  n01: candidate wrong, baseline right.
    """
    m = n10 + n01
    if m == 0:
        return 1.0
    k = min(n10, n01)
    p = 2 * sum(comb(m, i) for i in range(k + 1)) / 2**m
    return min(p, 1.0)


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """ECE with equal-width confidence bins."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_stats.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/devnet/stats.py tests/test_stats.py
git commit -m "feat: statistics module (Wilson CI, exact McNemar, ECE)"
```

---

### Task 3: Data download script

**Files:**
- Create: `scripts/download_data.sh`

No unit test (network operation); the script self-verifies image counts.

- [ ] **Step 1: Write `scripts/download_data.sh`**

```bash
#!/usr/bin/env bash
# Download the Devanagari Handwritten Character Dataset (DHCD).
# Source: Acharya, Pant & Gyawali 2015 (UCI ML Repository id 389).
set -euo pipefail
cd "$(dirname "$0")/.."

RAW=data/raw
EXTRACTED=data/extracted
URL="https://archive.ics.uci.edu/static/public/389/devanagari+handwritten+character+dataset.zip"

mkdir -p "$RAW" "$EXTRACTED"
if [ ! -f "$RAW/dhcd.zip" ]; then
    curl -L --fail -o "$RAW/dhcd.zip" "$URL"
fi
unzip -qo "$RAW/dhcd.zip" -d "$EXTRACTED"

TRAIN_DIR="$EXTRACTED/DevanagariHandwrittenCharacterDataset/Train"
TEST_DIR="$EXTRACTED/DevanagariHandwrittenCharacterDataset/Test"

n_train=$(find "$TRAIN_DIR" -name '*.png' | wc -l)
n_test=$(find "$TEST_DIR" -name '*.png' | wc -l)
n_classes=$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo "train images: $n_train (expect 78200)"
echo "test images:  $n_test (expect 13800)"
echo "classes:      $n_classes (expect 46)"
[ "$n_train" -eq 78200 ] && [ "$n_test" -eq 13800 ] && [ "$n_classes" -eq 46 ] \
    && echo "OK: DHCD verified" || { echo "FAIL: counts do not match"; exit 1; }
```

- [ ] **Step 2: Make executable and run**

Run: `chmod +x scripts/download_data.sh && ./scripts/download_data.sh`
Expected: `OK: DHCD verified`
(If the UCI URL 404s, fall back to the Kaggle mirror `ashokpant/devanagari-handwritten-character-dataset` and adjust the two `*_DIR` paths to match its layout — but verify the same counts. If counts differ, STOP and surface to the user; do not proceed with a nonstandard split.)

- [ ] **Step 3: Commit**

```bash
git add scripts/download_data.sh
git commit -m "feat: DHCD download script with count verification"
```

---

### Task 4: Data loading and stratified split

**Files:**
- Create: `src/devnet/data.py` (loading + split parts)
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data.py
import numpy as np
import pytest
from PIL import Image

from devnet.data import load_split, stratified_split


@pytest.fixture
def fake_dhcd(tmp_path):
    """3 classes x 8 train images of 32x32, mimicking DHCD layout."""
    root = tmp_path / "DevanagariHandwrittenCharacterDataset"
    rng = np.random.default_rng(0)
    for split in ["Train", "Test"]:
        for cls in ["character_1_ka", "character_2_kha", "digit_0"]:
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(8):
                arr = rng.integers(0, 255, (32, 32), dtype=np.uint8)
                Image.fromarray(arr, mode="L").save(d / f"img_{i}.png")
    return root


def test_load_split_shapes_and_labels(fake_dhcd):
    images, labels, classes = load_split(fake_dhcd, "Train")
    assert images.shape == (24, 32, 32)
    assert images.dtype == np.uint8
    assert sorted(classes) == classes  # deterministic alphabetical order
    assert set(labels) == {0, 1, 2}
    assert (np.bincount(labels) == 8).all()


def test_stratified_split_is_stratified_and_disjoint():
    labels = np.repeat(np.arange(5), 100)
    train_idx, val_idx = stratified_split(labels, val_fraction=0.1, seed=0)
    assert len(set(train_idx) & set(val_idx)) == 0  # no leakage
    assert len(train_idx) + len(val_idx) == 500
    # exactly 10 val per class
    assert (np.bincount(labels[val_idx]) == 10).all()


def test_stratified_split_deterministic_given_seed():
    labels = np.repeat(np.arange(3), 30)
    a = stratified_split(labels, val_fraction=0.2, seed=7)
    b = stratified_split(labels, val_fraction=0.2, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = stratified_split(labels, val_fraction=0.2, seed=8)
    assert not np.array_equal(a[1], c[1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`

- [ ] **Step 3: Implement loading + split in `src/devnet/data.py`**

```python
"""DHCD data pipeline: loading, stratified split, script-aware transforms."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_split(root, split: str):
    """Load a DHCD split directory into memory.

    Returns (images uint8 [N,32,32], labels int64 [N], class_names list[str]).
    Class index = position in the alphabetically sorted folder list.
    """
    split_dir = Path(root) / split
    classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    images, labels = [], []
    for idx, cls in enumerate(classes):
        for f in sorted((split_dir / cls).glob("*.png")):
            images.append(np.array(Image.open(f).convert("L"), dtype=np.uint8))
            labels.append(idx)
    return np.stack(images), np.array(labels, dtype=np.int64), classes


def stratified_split(labels: np.ndarray, val_fraction: float, seed: int):
    """Per-class shuffled split. Returns (train_idx, val_idx), both sorted."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.sort(train_idx), np.sort(val_idx)


class ArrayDataset(Dataset):
    """In-memory dataset over uint8 images; transform applied per item."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = torch.from_numpy(images).unsqueeze(1)  # [N,1,32,32] uint8
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[i]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_data.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/devnet/data.py tests/test_data.py
git commit -m "feat: DHCD loading and stratified validation split"
```

---

### Task 5: Script-aware transforms (no horizontal flips — enforced by test)

**Files:**
- Modify: `src/devnet/data.py` (append)
- Test: `tests/test_transforms.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_transforms.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_train_transform'`

- [ ] **Step 3: Append to `src/devnet/data.py`**

```python
from torchvision.transforms import v2  # noqa: E402  (top of file in practice)

# Augmentation tiers. Script-aware: NO horizontal/vertical flips —
# a mirrored Devanagari character is not a valid character.
_AUG_TIERS = {
    "none": [],
    "light": [
        v2.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.95, 1.05)),
    ],
    "medium": [
        v2.RandomAffine(degrees=12, translate=(0.10, 0.10), scale=(0.90, 1.10), shear=5),
        v2.ElasticTransform(alpha=15.0, sigma=3.0),
    ],
    "heavy": [
        v2.RandomAffine(degrees=15, translate=(0.12, 0.12), scale=(0.85, 1.15), shear=8),
        v2.ElasticTransform(alpha=25.0, sigma=3.0),
    ],
}


def compute_mean_std(images: np.ndarray) -> tuple[float, float]:
    """Mean/std in [0,1] units, computed on the training split only."""
    x = images.astype(np.float64) / 255.0
    return float(x.mean()), float(x.std())


def build_train_transform(tier: str, mean: float, std: float):
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        *_AUG_TIERS[tier],
        v2.Normalize([mean], [std]),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.10)),
    ])


def build_eval_transform(mean: float, std: float):
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([mean], [std]),
    ])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_transforms.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/devnet/data.py tests/test_transforms.py
git commit -m "feat: script-aware augmentation tiers (flips excluded, test-enforced)"
```

---

### Task 6: Student/teacher model (pre-activation SE-ResNet)

**Files:**
- Create: `src/devnet/models/__init__.py`, `src/devnet/models/student.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py
import torch

from devnet.models.student import DevNet


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_default_student_under_param_budget():
    model = DevNet()  # widths (40, 80, 160), depths (2, 2, 2)
    assert n_params(model) <= 1_500_000  # spec C2 budget


def test_output_shape():
    model = DevNet()
    x = torch.randn(4, 1, 32, 32)
    assert model(x).shape == (4, 46)


def test_teacher_configuration_scales_up():
    teacher = DevNet(widths=(96, 192, 384), depths=(3, 3, 3))
    assert n_params(teacher) > 5_000_000
    x = torch.randn(2, 1, 32, 32)
    assert teacher(x).shape == (2, 46)


def test_gradients_flow():
    model = DevNet(widths=(8, 16, 32), depths=(1, 1, 1))
    out = model(torch.randn(2, 1, 32, 32))
    out.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'devnet.models'`

- [ ] **Step 3: Implement**

`src/devnet/models/__init__.py`:

```python
from devnet.models.student import DevNet

__all__ = ["DevNet"]
```

`src/devnet/models/student.py`:

```python
"""Compact pre-activation SE-ResNet for 32x32 single-channel input.

The same class serves as student (default widths) and teacher (wider/deeper).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s[:, :, None, None]


class PreActSEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.se = SqueezeExcite(out_ch)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
            if stride != 1 or in_ch != out_ch
            else None
        )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        sc = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return self.se(out) + sc


class DevNet(nn.Module):
    def __init__(
        self,
        widths: tuple = (40, 80, 160),
        depths: tuple = (2, 2, 2),
        num_classes: int = 46,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Conv2d(1, widths[0], 3, 1, 1, bias=False)
        blocks, in_ch = [], widths[0]
        for stage, (w, d) in enumerate(zip(widths, depths)):
            for j in range(d):
                stride = 2 if (stage > 0 and j == 0) else 1
                blocks.append(PreActSEBlock(in_ch, w, stride))
                in_ch = w
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(in_ch)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = F.relu(self.bn(x)).mean(dim=(2, 3))
        return self.fc(self.drop(x))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: 4 passed. Also print the budget number for the record:
`.venv/bin/python -c "from devnet.models import DevNet; print(sum(p.numel() for p in DevNet().parameters()))"`
Expected: a number ≤ 1,500,000 (~1.1M)

- [ ] **Step 5: Commit**

```bash
git add src/devnet/models tests/test_models.py
git commit -m "feat: pre-activation SE-ResNet student/teacher model"
```

---

### Task 7: Config system

**Files:**
- Create: `src/devnet/config.py`, `configs/student.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
from devnet.config import RunConfig


def test_defaults_match_spec():
    cfg = RunConfig()
    assert cfg.widths == (40, 80, 160)
    assert cfg.label_smoothing == 0.1
    assert cfg.val_fraction == 0.1
    assert cfg.epochs == 300


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("epochs: 5\nwidths: [8, 16, 32]\nseed: 3\n")
    cfg = RunConfig.from_yaml(p)
    assert cfg.epochs == 5
    assert cfg.widths == (8, 16, 32)
    assert cfg.seed == 3
    assert cfg.lr == RunConfig().lr  # unspecified keys keep defaults


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("eposh: 5\n")  # typo must not pass silently
    import pytest
    with pytest.raises(TypeError):
        RunConfig.from_yaml(p)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/devnet/config.py`**

```python
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
```

`configs/student.yaml`:

```yaml
# Compact student, supervised baseline (no distillation)
widths: [40, 80, 160]
depths: [2, 2, 2]
epochs: 300
aug_tier: medium
out_dir: results/student_supervised
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/devnet/config.py configs/student.yaml tests/test_config.py
git commit -m "feat: YAML run configuration with strict key validation"
```

---

### Task 8: Training building blocks (mixup/cutmix, EMA, cosine warmup)

**Files:**
- Create: `src/devnet/train.py` (building blocks only; Trainer in Task 9)
- Test: `tests/test_train_components.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_components.py
import numpy as np
import torch

from devnet.train import EMA, apply_mix, cosine_warmup_scheduler


def test_apply_mix_identity_when_prob_zero():
    x = torch.randn(8, 1, 32, 32)
    y = torch.arange(8)
    rng = np.random.default_rng(0)
    out_x, ya, yb, lam = apply_mix(x.clone(), y, mixup_alpha=0.2,
                                   cutmix_alpha=1.0, mix_prob=0.0, rng=rng)
    assert torch.equal(out_x, x)
    assert torch.equal(ya, y) and torch.equal(yb, y)
    assert lam == 1.0


def test_apply_mix_lambda_in_unit_interval():
    x = torch.randn(8, 1, 32, 32)
    y = torch.arange(8)
    rng = np.random.default_rng(1)
    for _ in range(20):
        _, _, _, lam = apply_mix(x.clone(), y, mixup_alpha=0.2,
                                 cutmix_alpha=1.0, mix_prob=1.0, rng=rng)
        assert 0.0 <= lam <= 1.0


def test_ema_converges_to_constant_weights():
    model = torch.nn.Linear(4, 2)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    for _ in range(60):
        ema.update(model)
    target = torch.nn.Linear(4, 2)
    ema.copy_to(target)
    for p in target.parameters():
        assert torch.allclose(p, torch.ones_like(p), atol=1e-6)


def test_cosine_warmup_shape():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = cosine_warmup_scheduler(opt, warmup_steps=10, total_steps=100)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9]            # warming up
    assert max(lrs) <= 1.0 + 1e-8     # peak at base lr
    assert lrs[-1] < 0.01             # decayed to ~0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_train_components.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement building blocks in `src/devnet/train.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_train_components.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/devnet/train.py tests/test_train_components.py
git commit -m "feat: mixup/cutmix, weight EMA, cosine-warmup scheduler"
```

---

### Task 9: Trainer with synthetic overfit test

**Files:**
- Modify: `src/devnet/train.py` (append Trainer + CLI)
- Test: `tests/test_trainer.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_trainer.py -v`
Expected: FAIL with `ImportError: cannot import name 'Trainer'`

- [ ] **Step 3: Append Trainer to `src/devnet/train.py`**

```python
import csv  # add to imports at top of file
import json
import subprocess
from pathlib import Path

import torch.nn.functional as F
from torch.utils.data import DataLoader

from devnet.config import RunConfig
from devnet.data import (ArrayDataset, build_eval_transform,
                         build_train_transform, compute_mean_std,
                         load_split, stratified_split)
from devnet.models.student import DevNet


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
    """Supervised trainer. If cfg.teacher_logits is set, adds KD loss (distill.py)."""

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
        self.train_indices = tr_idx  # kept for KD logit alignment (distill.py)

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
        loss = lam * F.cross_entropy(out, ya, label_smoothing=ls) + \
               (1 - lam) * F.cross_entropy(out, yb, label_smoothing=ls)
        return loss

    @torch.no_grad()
    def _validate(self, loader, use_ema=True):
        model = DevNet(self.cfg.widths, self.cfg.depths, self.num_classes,
                       self.cfg.dropout).to(self.device)
        if use_ema:
            self.ema.copy_to(model)
        else:
            model.load_state_dict(self.model.state_dict())
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
                    x, ya, yb, lam = apply_mix(
                        x, y, mixup_alpha=cfg.mixup_alpha,
                        cutmix_alpha=cfg.cutmix_alpha,
                        mix_prob=cfg.mix_prob, rng=self.rng)
                    with torch.autocast(self.device.type, enabled=use_amp):
                        out = self.model(x)
                        loss = self._loss(out, ya, yb, lam)
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
    result = Trainer(cfg).fit()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
```

Note: `RunConfig` is frozen, so `cfg.__dict__` reads work; the seed override builds a new instance.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_trainer.py -v`
Expected: 1 passed (≈30-60s on CPU)

- [ ] **Step 5: Run full suite + commit**

Run: `.venv/bin/pytest`
Expected: all tests pass

```bash
git add src/devnet/train.py tests/test_trainer.py
git commit -m "feat: Trainer with EMA checkpointing, CSV logs, CLI entrypoint"
```

---

### Task 10: Evaluation module (prediction dumps, metrics, TTA, ensemble, paired tests)

**Files:**
- Create: `src/devnet/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
import numpy as np
import torch

from devnet.evaluate import (ensemble_predictions, mcnemar_between,
                             metrics_from_predictions, predict,
                             save_predictions, load_predictions, tta_views)


class StubModel(torch.nn.Module):
    """Always predicts class = pixel sum mod 3 (deterministic, input-driven)."""
    def forward(self, x):
        idx = (x.sum(dim=(1, 2, 3)).long().abs() % 3)
        return torch.nn.functional.one_hot(idx, 3).float() * 10


def test_predict_and_roundtrip(tmp_path):
    model = StubModel()
    x = torch.randn(10, 1, 32, 32)
    y = torch.randint(0, 3, (10,))
    logits, preds, labels = predict(model, [(x, y)], device=torch.device("cpu"))
    assert logits.shape == (10, 3)
    p = tmp_path / "preds.npz"
    save_predictions(p, logits, preds, labels)
    l2, p2, y2 = load_predictions(p)
    assert np.array_equal(preds, p2) and np.array_equal(labels, y2)


def test_metrics_from_predictions():
    preds = np.array([0, 1, 2, 2])
    labels = np.array([0, 1, 2, 1])
    m = metrics_from_predictions(preds, labels)
    assert m["accuracy"] == 0.75
    assert 0 < m["macro_f1"] <= 1
    assert m["n_errors"] == 1
    assert "confusion" in m and np.asarray(m["confusion"]).shape == (3, 3)


def test_ensemble_majority_wins(tmp_path):
    labels = np.array([0, 1])
    # model A correct on both; model B wrong on second with low confidence
    la = np.array([[5.0, 0.0], [0.0, 5.0]])
    lb = np.array([[4.0, 0.0], [1.1, 1.0]])
    pa, pb = tmp_path / "a.npz", tmp_path / "b.npz"
    save_predictions(pa, la, la.argmax(1), labels)
    save_predictions(pb, lb, lb.argmax(1), labels)
    preds = ensemble_predictions([pa, pb])
    assert np.array_equal(preds, labels)


def test_mcnemar_between_files(tmp_path):
    labels = np.zeros(100, dtype=np.int64)
    preds_a = np.zeros(100, dtype=np.int64); preds_a[:30] = 1  # 30 errors
    preds_b = np.zeros(100, dtype=np.int64)                    # 0 errors
    la = np.zeros((100, 2)); lb = np.zeros((100, 2))
    pa, pb = tmp_path / "a.npz", tmp_path / "b.npz"
    save_predictions(pa, la, preds_a, labels)
    save_predictions(pb, lb, preds_b, labels)
    result = mcnemar_between(pb, pa)  # candidate b vs baseline a
    assert result["n10"] == 30 and result["n01"] == 0
    assert result["p_value"] < 1e-6


def test_tta_views_count_and_shape():
    x = torch.randn(4, 1, 32, 32)
    views = tta_views(x)
    assert len(views) == 5
    for v in views:
        assert v.shape == x.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/devnet/evaluate.py`**

```python
"""Evaluation: prediction dumps, metrics, TTA, ensembling, paired tests."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torchvision.transforms.v2 import functional as TF

from devnet.stats import mcnemar_exact, wilson_ci


@torch.no_grad()
def predict(model, loader, device, tta: bool = False):
    """Run model over loader. Returns (logits [N,C], preds [N], labels [N]) numpy.

    With tta=True, logits are mean softmax probabilities over the views
    (still usable downstream as 'logits' for argmax/ensembling).
    """
    model.eval()
    all_out, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        if tta:
            probs = torch.stack(
                [F.softmax(model(v), dim=1) for v in tta_views(x)]
            ).mean(0)
            all_out.append(probs.cpu())
        else:
            all_out.append(model(x).cpu())
        all_y.append(torch.as_tensor(y))
    logits = torch.cat(all_out).numpy()
    labels = torch.cat(all_y).numpy()
    return logits, logits.argmax(1), labels


def tta_views(x):
    """5 deterministic views: identity, rotate ±6°, shift ±1px (no flips)."""
    return [
        x,
        TF.rotate(x, 6.0),
        TF.rotate(x, -6.0),
        TF.affine(x, angle=0.0, translate=[1, 0], scale=1.0, shear=[0.0]),
        TF.affine(x, angle=0.0, translate=[0, 1], scale=1.0, shear=[0.0]),
    ]


def save_predictions(path, logits, preds, labels):
    np.savez_compressed(path, logits=logits, preds=preds, labels=labels)


def load_predictions(path):
    d = np.load(path)
    return d["logits"], d["preds"], d["labels"]


def metrics_from_predictions(preds, labels):
    acc = float((preds == labels).mean())
    n = len(labels)
    lo, hi = wilson_ci(int((preds == labels).sum()), n)
    return {
        "accuracy": acc,
        "n": n,
        "n_errors": int((preds != labels).sum()),
        "wilson_ci": (lo, hi),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "per_class_f1": f1_score(labels, preds, average=None).tolist(),
        "confusion": confusion_matrix(labels, preds).tolist(),
    }


def ensemble_predictions(paths):
    """Average softmax probabilities across prediction files -> preds."""
    probs = None
    for p in paths:
        logits, _, _ = load_predictions(p)
        t = torch.from_numpy(logits)
        # logits may already be probabilities (TTA); softmax is idempotent enough
        # for ranking only if applied consistently — so detect:
        sm = t if np.allclose(t.sum(1), 1.0, atol=1e-3) else F.softmax(t, dim=1)
        probs = sm if probs is None else probs + sm
    return probs.argmax(1).numpy()


def mcnemar_between(candidate_path, baseline_path):
    """Paired McNemar test from two prediction dumps over the same test set."""
    _, preds_c, labels_c = load_predictions(candidate_path)
    _, preds_b, labels_b = load_predictions(baseline_path)
    assert np.array_equal(labels_c, labels_b), "prediction files use different test order"
    c_ok = preds_c == labels_c
    b_ok = preds_b == labels_b
    n10 = int((~b_ok & c_ok).sum())  # baseline wrong, candidate right
    n01 = int((b_ok & ~c_ok).sum())
    return {"n10": n10, "n01": n01, "p_value": mcnemar_exact(n10, n01)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_evaluate.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/devnet/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluation with prediction dumps, TTA, ensembling, McNemar"
```

---

### Task 11: Test-set evaluation CLI + real-data smoke gate

**Files:**
- Create: `scripts/evaluate_checkpoint.py`
- Test: `tests/test_smoke_real_data.py` (integration, skipped when data absent)

- [ ] **Step 1: Write `scripts/evaluate_checkpoint.py`**

```python
"""Evaluate a trained checkpoint on the DHCD test split; dump predictions.

Usage: python scripts/evaluate_checkpoint.py results/run/best.pth [--tta]
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from devnet.data import ArrayDataset, build_eval_transform, load_split
from devnet.evaluate import metrics_from_predictions, predict, save_predictions
from devnet.models.student import DevNet
from devnet.stats import expected_calibration_error
from devnet.train import resolve_device
import torch.nn.functional as F
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--data-root",
                    default="data/extracted/DevanagariHandwrittenCharacterDataset")
    args = ap.parse_args()

    device = resolve_device("auto")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DevNet(tuple(cfg["widths"]), tuple(cfg["depths"]),
                   46, cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    images, labels, _ = load_split(args.data_root, "Test")
    ds = ArrayDataset(images, labels,
                      build_eval_transform(ckpt["mean"], ckpt["std"]))
    loader = DataLoader(ds, 512, shuffle=False, num_workers=2)

    logits, preds, labels = predict(model, loader, device, tta=args.tta)
    out = Path(args.checkpoint).parent
    suffix = "_tta" if args.tta else ""
    save_predictions(out / f"test_predictions{suffix}.npz", logits, preds, labels)

    m = metrics_from_predictions(preds, labels)
    probs = logits if args.tta else F.softmax(torch.from_numpy(logits), 1).numpy()
    m["ece"] = expected_calibration_error(probs.max(1), preds == labels)
    del m["per_class_f1"], m["confusion"]  # keep stdout short; npz has everything
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the smoke gate test**

```python
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
        widths=(16, 32, 64), depths=(1, 1, 1), epochs=8, batch_size=128,
        lr=3e-3, warmup_epochs=1, aug_tier="light", mix_prob=0.0,
        num_workers=2, device="cpu", out_dir=str(tmp_path),
        val_fraction=0.15, seed=0,
    )
    trainer = Trainer(cfg, images=images[keep], labels=labels[keep])
    result = trainer.fit()
    assert result["best_val_acc"] > 0.90
```

- [ ] **Step 3: Run the smoke gate**

Run: `.venv/bin/pytest tests/test_smoke_real_data.py -v` (requires Task 3 data; if data missing it must report SKIPPED, then run `./scripts/download_data.sh` and re-run)
Expected: 1 passed in ≤ ~3 minutes. If accuracy < 90%, STOP and debug (systematic-debugging skill) before any GPU spend.

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluate_checkpoint.py tests/test_smoke_real_data.py
git commit -m "feat: checkpoint evaluation CLI and real-data smoke gate"
```

---

### Task 12: MallaNet baseline reproduction

**Files:**
- Create: `src/devnet/models/mallanet_baseline.py`, `scripts/reproduce_mallanet.py`
- Test: `tests/test_mallanet_baseline.py`

The MallaNet repo is cloned at `/tmp/mallanet_repo` (re-clone if missing:
`git clone --depth 1 https://github.com/sahajrajmalla/MallaNet /tmp/mallanet_repo`).

- [ ] **Step 1: Port the architecture**

Copy the classes `ResidualBlock`, `HFCLayer`, `MergingLayer`, `BMCNNBase`,
`EnhancedBMCNNwHFCs` **verbatim** from `/tmp/mallanet_repo/src/main.py`
(lines 43–152) into `src/devnet/models/mallanet_baseline.py`, with this header:

```python
"""MallaNet architecture, ported verbatim from the author's repository
(github.com/sahajrajmalla/MallaNet, src/main.py) for baseline reproduction.

Used ONLY to run the published checkpoint for paired statistical comparison
(spec §7). No architectural inheritance into DevNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Keep the five classes byte-identical apart from removed comments. Do not "fix"
anything in them — fidelity matters more than style.

- [ ] **Step 2: Write the failing test**

```python
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
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/pytest tests/test_mallanet_baseline.py -v`
Expected: 3 passed. If `test_param_count_matches_paper` fails, report the actual
count — that itself is a reproduction finding; loosen only with a comment.

- [ ] **Step 4: Write `scripts/reproduce_mallanet.py`**

Their training data used numeric class folders (`0`–`45`, Kaggle mirror), while
the UCI release uses names (`character_1_ka`...). The checkpoint's output index
order may therefore not match our alphabetical order. The script measures
accuracy with the identity mapping first; if it is low, it recovers the
permutation from the model's own most-confident behavior, then re-reports.

```python
"""Run the published MallaNet checkpoint on the DHCD test set.

Produces results/mallanet_baseline/test_predictions.npz for paired tests.
Handles potential class-index permutation between their numeric training
folders and our alphabetical UCI folder order.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from devnet.data import ArrayDataset, load_split
from devnet.evaluate import metrics_from_predictions, predict, save_predictions
from devnet.models.mallanet_baseline import EnhancedBMCNNwHFCs
from devnet.train import resolve_device

WEIGHTS = Path("/tmp/mallanet_repo/models/best_model.pth")
DATA_ROOT = "data/extracted/DevanagariHandwrittenCharacterDataset"
OUT = Path("results/mallanet_baseline")


def recover_permutation(preds, labels, num_classes=46):
    """Map model output index -> our class index via per-class modal prediction."""
    mapping = np.full(num_classes, -1)
    for c in range(num_classes):
        model_outputs = preds[labels == c]
        mapping[np.bincount(model_outputs, minlength=num_classes).argmax()] = c
    if len(set(mapping)) != num_classes or -1 in mapping:
        raise RuntimeError(f"Recovered mapping is not a bijection: {mapping}")
    return mapping


def main():
    device = resolve_device("auto")
    model = EnhancedBMCNNwHFCs(num_classes=46).to(device)
    ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    images, labels, classes = load_split(DATA_ROOT, "Test")
    # Their exact eval transform: ToTensor + Normalize((0.5,), (0.5,))
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ])
    loader = DataLoader(ArrayDataset(images, labels, transform), 256,
                        shuffle=False, num_workers=2)
    logits, preds, labels = predict(model, loader, device)

    acc_identity = float((preds == labels).mean())
    note = "identity mapping"
    if acc_identity < 0.90:  # class order mismatch — recover permutation
        mapping = recover_permutation(preds, labels)
        preds = mapping[preds]
        logits = logits[:, np.argsort(mapping)]
        note = f"recovered permutation (identity acc was {acc_identity:.4f})"

    OUT.mkdir(parents=True, exist_ok=True)
    save_predictions(OUT / "test_predictions.npz", logits, preds, labels)
    m = metrics_from_predictions(preds, labels)
    m["mapping_note"] = note
    m["published_accuracy"] = 0.9971
    del m["per_class_f1"], m["confusion"]
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the reproduction**

Run: `.venv/bin/python scripts/reproduce_mallanet.py`
Expected: accuracy ≈ 0.9971 (±0.002). **Whatever the number is, record it
faithfully** — a reproduction gap is a finding (spec §2 fallbacks), not a bug
to massage. CPU inference on 13,800 images with a 17M model takes ~10–30 min;
that is acceptable for a one-off (re-run later on GPU if needed).

- [ ] **Step 6: Commit**

```bash
git add src/devnet/models/mallanet_baseline.py scripts/reproduce_mallanet.py \
        tests/test_mallanet_baseline.py results/mallanet_baseline/metrics.json
git commit -m "feat: MallaNet baseline reproduction with permutation recovery"
```

---

### Task 13: Knowledge distillation

**Files:**
- Create: `src/devnet/distill.py`
- Test: `tests/test_distill.py`

Design: distillation reuses `Trainer` unchanged for optimization; `distill.py`
provides (a) the KD loss, (b) a script-level flow that dumps teacher-ensemble
logits for the train set, then trains a student whose loss adds the KD term via
a `DistillTrainer` subclass that aligns per-sample teacher logits by dataset index.

- [ ] **Step 1: Write the failing tests**

```python
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
        num_workers=0, device="cpu", out_dir=str(tmp_path / "run"),
        val_fraction=0.25, seed=0,
        teacher_logits=str(logits_path), kd_alpha=0.5,
    )
    trainer = DistillTrainer(cfg, images=images, labels=labels, num_classes=4)
    result = trainer.fit()
    assert result["best_val_acc"] >= 0.99
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_distill.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/devnet/distill.py`**

```python
"""Knowledge distillation: KD loss and a Trainer that adds a teacher term.

teacher_logits npz layout: key 'logits', shape [N_train_full, C], row i
corresponding to row i of the FULL training array (before val carve-out) —
produced by scripts/dump_teacher_logits.py using identical data_root and order.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from devnet.train import Trainer, apply_mix


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
        # align teacher logits with the post-split training subset
        self.teacher = torch.from_numpy(full[self.train_indices]).float()

    def _loaders(self):
        train, val = super()._loaders()
        indexed = DataLoader(
            _IndexedDataset(self.train_ds), self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True, drop_last=True)
        return indexed, val

    def _loss(self, out, ya, yb, lam, extra=None):
        # mixing is disabled for distillation runs (teacher logits are per-image)
        t = self.teacher[extra].to(out.device)
        return kd_loss(out, t, ya, temperature=self.cfg.kd_temperature,
                       alpha=self.cfg.kd_alpha,
                       label_smoothing=self.cfg.label_smoothing)
```

Then make `Trainer.fit` distillation-aware with a 3-line change in `train.py`:
inside the batch loop, unpack `extra` and pass it through —

```python
# replace:  x, y = batch[0].to(self.device), batch[1].to(self.device)
x, y = batch[0].to(self.device), batch[1].to(self.device)
extra = batch[2] if len(batch) > 2 else None
if extra is not None:  # distillation: no mixing (per-image teacher logits)
    x, ya, yb, lam = x, y, y, 1.0
else:
    x, ya, yb, lam = apply_mix(
        x, y, mixup_alpha=cfg.mixup_alpha,
        cutmix_alpha=cfg.cutmix_alpha, mix_prob=cfg.mix_prob, rng=self.rng)
with torch.autocast(self.device.type, enabled=use_amp):
    out = self.model(x)
    loss = self._loss(out, ya, yb, lam, extra=extra)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_distill.py tests/test_trainer.py -v`
Expected: all pass (trainer test re-run guards the refactor)

- [ ] **Step 5: Write `scripts/dump_teacher_logits.py`**

```python
"""Average TTA softmax of N teacher checkpoints over the FULL train split
(canonical load_split order) -> teacher_logits.npz for distillation."""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from devnet.data import ArrayDataset, build_eval_transform, load_split
from devnet.evaluate import predict
from devnet.models.student import DevNet
from devnet.train import resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", default="results/teacher_logits.npz")
    ap.add_argument("--data-root",
                    default="data/extracted/DevanagariHandwrittenCharacterDataset")
    args = ap.parse_args()

    device = resolve_device("auto")
    images, labels, _ = load_split(args.data_root, "Train")
    acc = None
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = DevNet(tuple(cfg["widths"]), tuple(cfg["depths"]),
                       46, cfg["dropout"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        ds = ArrayDataset(images, labels,
                          build_eval_transform(ckpt["mean"], ckpt["std"]))
        loader = DataLoader(ds, 512, shuffle=False, num_workers=2)
        probs, _, _ = predict(model, loader, device, tta=True)
        acc = probs if acc is None else acc + probs
    mean_probs = acc / len(args.checkpoints)
    # store as log-probs so kd_loss temperature softening behaves like logits
    np.savez_compressed(args.out, logits=np.log(np.clip(mean_probs, 1e-9, 1.0)))
    print(f"wrote {args.out}, shape {mean_probs.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/devnet/distill.py scripts/dump_teacher_logits.py src/devnet/train.py tests/test_distill.py
git commit -m "feat: ensemble knowledge distillation"
```

---

### Task 14: Robustness corruptions

**Files:**
- Create: `src/devnet/robustness.py`
- Test: `tests/test_robustness.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_robustness.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/devnet/robustness.py`**

```python
"""Image corruptions for robustness curves (applied to [0,1] floats,
BEFORE normalization). Severity 0 = identity, 1..5 increasing."""
import torch
from torchvision.transforms.v2 import functional as TF

_NOISE_STD = [0.0, 0.04, 0.08, 0.12, 0.18, 0.25]
_BLUR_SIGMA = [0.0, 0.4, 0.7, 1.0, 1.4, 1.9]
_CONTRAST = [1.0, 0.8, 0.65, 0.5, 0.35, 0.2]

CORRUPTIONS = ("noise", "blur", "contrast")


def corrupt(x: torch.Tensor, kind: str, severity: int,
            generator: torch.Generator | None = None) -> torch.Tensor:
    if severity == 0:
        return x.clone()
    if kind == "noise":
        noise = torch.randn(x.shape, generator=generator) * _NOISE_STD[severity]
        return (x + noise).clamp(0.0, 1.0)
    if kind == "blur":
        return TF.gaussian_blur(x, kernel_size=5,
                                sigma=_BLUR_SIGMA[severity]).clamp(0.0, 1.0)
    if kind == "contrast":
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return (mean + (x - mean) * _CONTRAST[severity]).clamp(0.0, 1.0)
    raise ValueError(f"unknown corruption {kind!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_robustness.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/devnet/robustness.py tests/test_robustness.py
git commit -m "feat: noise/blur/contrast corruption suite"
```

---

### Task 15: Experiment configs, seed runner, README

**Files:**
- Create: `configs/teacher_a.yaml`, `configs/teacher_b.yaml`, `configs/teacher_c.yaml`, `configs/student_distilled.yaml`, `scripts/run_seeds.sh`, `README.md`

- [ ] **Step 1: Write teacher configs**

`configs/teacher_a.yaml` (wide, medium aug):

```yaml
widths: [96, 192, 384]
depths: [3, 3, 3]
epochs: 300
aug_tier: medium
out_dir: results/teacher_a
```

`configs/teacher_b.yaml` (wide, heavy aug):

```yaml
widths: [96, 192, 384]
depths: [3, 3, 3]
epochs: 300
aug_tier: heavy
out_dir: results/teacher_b
```

`configs/teacher_c.yaml` (deeper-narrower, medium aug):

```yaml
widths: [64, 128, 256]
depths: [4, 4, 4]
epochs: 300
aug_tier: medium
out_dir: results/teacher_c
```

`configs/student_distilled.yaml`:

```yaml
widths: [40, 80, 160]
depths: [2, 2, 2]
epochs: 300
aug_tier: light          # distillation runs disable mixing internally
teacher_logits: results/teacher_logits.npz
kd_temperature: 4.0
kd_alpha: 0.7
out_dir: results/student_distilled
```

- [ ] **Step 2: Write `scripts/run_seeds.sh`**

```bash
#!/usr/bin/env bash
# Train one config across 5 seeds: ./scripts/run_seeds.sh configs/student.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="$1"
for seed in 0 1 2 3 4; do
    echo "=== $CONFIG seed $seed ==="
    .venv/bin/python -m devnet.train --config "$CONFIG" --seed "$seed"
done
```

- [ ] **Step 3: Write `README.md`**

```markdown
# DevNet — Statistically-Defended SOTA for Handwritten Devanagari Recognition

Target: beat 99.72% (Mishra et al. 2021) on the 46-class DHCD benchmark with a
McNemar-significant margin, and dominate the parameter-efficiency frontier with
a ~1.1M-parameter distilled student. Spec: `docs/superpowers/specs/`.

## Setup

    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
    ./scripts/download_data.sh
    .venv/bin/pytest                      # all unit tests + real-data smoke gate

## Full experiment program (rented GPU)

    # 1. Teachers: 3 configs x 5 seeds (~15 runs)
    ./scripts/run_seeds.sh configs/teacher_a.yaml
    ./scripts/run_seeds.sh configs/teacher_b.yaml
    ./scripts/run_seeds.sh configs/teacher_c.yaml

    # 2. Evaluate each teacher on test (predictions saved per run)
    for d in results/teacher_*_seed*/; do
        .venv/bin/python scripts/evaluate_checkpoint.py "$d/best.pth" --tta
    done

    # 3. Ensemble + dump train logits for distillation
    .venv/bin/python scripts/dump_teacher_logits.py results/teacher_*_seed*/best.pth

    # 4. Distill the student (5 seeds) + supervised control
    ./scripts/run_seeds.sh configs/student_distilled.yaml
    ./scripts/run_seeds.sh configs/student.yaml

    # 5. Baseline + paired stats
    .venv/bin/python scripts/reproduce_mallanet.py

## Statistical protocol

The test set is evaluated once per final configuration. Model selection uses a
10% stratified validation carve-out. Reported: mean±std over 5 seeds, Wilson
95% CIs, exact McNemar paired tests vs the reproduced MallaNet checkpoint.
```

- [ ] **Step 4: Run full test suite**

Run: `chmod +x scripts/run_seeds.sh && .venv/bin/pytest`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add configs scripts/run_seeds.sh README.md
git commit -m "feat: experiment configs, seed runner, README with run protocol"
```

---

### Task 16: GPU experiment program (operational)

No new code. Executed on the rented GPU box after all previous tasks pass there.

- [ ] **Step 1:** Provision GPU instance (RTX 4090 class), clone repo, `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`, `./scripts/download_data.sh`, `.venv/bin/pytest` — everything must pass before training.
- [ ] **Step 2:** Sanity run: `.venv/bin/python -m devnet.train --config configs/student.yaml --seed 0` with `epochs: 40` override first; confirm val accuracy > 99.3% before committing to full 300-epoch runs.
- [ ] **Step 3:** Run the full program from README §"Full experiment program".
- [ ] **Step 4:** Pull all `results/**/{log.csv,test_predictions*.npz,metrics.json,config.json}` back to this machine (predictions make every statistic recomputable locally; checkpoints optional but keep at least the best student + one teacher set).
- [ ] **Step 5:** Compute the headline table locally: per-model metrics, ensemble accuracy, McNemar vs MallaNet dump, ECE, robustness curves, plus efficiency metrics — parameter counts, FLOPs (`torch.utils.flop_counter.FlopCounterMode` on one forward pass), and CPU/GPU inference latency per image (median of 100 batches) for the student vs the MallaNet baseline. Compare against spec success criteria C1/C2; record outcomes (including misses — spec §2 fallbacks) in `results/SUMMARY.md` and commit.

---

## Verification against spec

| Spec item | Covered by |
|---|---|
| Stats: Wilson/McNemar/ECE (C3) | Task 2 |
| Data protocol: stratified val, test-once (§3) | Tasks 4, 11, 16 |
| Script-aware aug, no flips (§3) | Task 5 (test-enforced) |
| Student ≤1.5M params (C2, §4) | Task 6 (test-enforced) |
| Recipe: AdamW/cosine/EMA/mixup/AMP/5 seeds (§5) | Tasks 8, 9, 15 |
| Smoke gate before GPU spend (§5) | Task 11 |
| Teacher ensemble + TTA (C1, §6) | Tasks 10, 15, 16 |
| Distillation (§6) | Task 13 |
| MallaNet reproduction, paired McNemar (§7) | Tasks 10, 12 |
| Robustness curves (C4, §7) | Task 14, 16 |
| Calibration ECE (§7) | Tasks 2, 11 |
| Per-run config+SHA+CSV logging, prediction dumps (§8) | Tasks 9, 10 |
| Error-ceiling audit (C4) | Task 16 step 5 (analysis over saved predictions) |
```
