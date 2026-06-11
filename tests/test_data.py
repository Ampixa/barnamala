# tests/test_data.py
import numpy as np
import pytest
import torch
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


def test_array_dataset_item_and_batching():
    from torch.utils.data import DataLoader

    from devnet.data import ArrayDataset

    images = np.zeros((6, 32, 32), dtype=np.uint8)
    labels = np.arange(6, dtype=np.int64)
    ds = ArrayDataset(images, labels, transform=None)
    assert len(ds) == 6
    x, y = ds[2]
    assert x.shape == (1, 32, 32) and x.dtype == torch.uint8
    assert y.item() == 2
    xb, yb = next(iter(DataLoader(ds, batch_size=3)))
    assert xb.shape == (3, 1, 32, 32) and yb.shape == (3,)

    doubled = ArrayDataset(images, labels, transform=lambda t: t.float() * 2)
    x2, _ = doubled[0]
    assert x2.dtype == torch.float32
