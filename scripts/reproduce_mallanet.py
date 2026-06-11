"""Run the published MallaNet checkpoint on the DHCD test set.

Produces results/mallanet_baseline/test_predictions.npz for paired tests.
Handles potential class-index permutation between their numeric training
folders and our alphabetical UCI folder order.
"""
import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/tmp/mallanet_repo/models/best_model.pth")
    ap.add_argument("--data-root",
                    default="data/extracted/DevanagariHandwrittenCharacterDataset")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(
            f"MallaNet checkpoint not found: {args.weights}\n"
            "Clone it first: git clone --depth 1 "
            "https://github.com/sahajrajmalla/MallaNet /tmp/mallanet_repo")

    device = resolve_device("auto")
    model = EnhancedBMCNNwHFCs(num_classes=46).to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    images, labels, classes = load_split(args.data_root, "Test")
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
