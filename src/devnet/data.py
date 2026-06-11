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
