"""Evaluate a trained checkpoint on the DHCD test split; dump predictions.

Usage: python scripts/evaluate_checkpoint.py results/run/best.pth [--tta]
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from devnet.data import ArrayDataset, build_eval_transform, load_split
from devnet.evaluate import metrics_from_predictions, predict, save_predictions
from devnet.models.student import DevNet
from devnet.stats import expected_calibration_error
from devnet.train import resolve_device


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

    scores, preds, labels = predict(model, loader, device, tta=args.tta)
    out = Path(args.checkpoint).parent
    suffix = "_tta" if args.tta else ""
    save_predictions(out / f"test_predictions{suffix}.npz", scores, preds, labels)

    m = metrics_from_predictions(preds, labels)
    probs = scores if args.tta else F.softmax(torch.from_numpy(scores), 1).numpy()
    m["ece"] = expected_calibration_error(probs.max(1), preds == labels)
    del m["per_class_f1"], m["confusion"]  # keep stdout short; npz has everything
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
