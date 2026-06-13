"""Evaluate a soft-vote ensemble of checkpoints on the DHCD test split and run
McNemar vs the MallaNet baseline. The run program never scores the ensemble as
a predictor (it only uses teachers to make distillation targets); this fills
that gap. Use --tta to compare against the TTA-degraded variant.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from devnet.data import ArrayDataset, build_eval_transform, load_split
from devnet.evaluate import predict, mcnemar_exact, load_predictions
from devnet.models.student import DevNet
from devnet.train import resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--baseline", default="results/mallanet_baseline/test_predictions.npz")
    ap.add_argument("--data-root",
                    default="data/extracted/DevanagariHandwrittenCharacterDataset")
    args = ap.parse_args()

    device = resolve_device("auto")
    images, labels, _ = load_split(args.data_root, "Test")
    acc, per_model = None, []
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        nc = ckpt["model_state_dict"]["fc.weight"].shape[0]
        model = DevNet(tuple(cfg["widths"]), tuple(cfg["depths"]), nc, cfg["dropout"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        loader = DataLoader(ArrayDataset(images, labels, build_eval_transform(ckpt["mean"], ckpt["std"])),
                            512, shuffle=False, num_workers=2)
        scores, _, _ = predict(model, loader, device, tta=args.tta)
        probs = scores / scores.sum(1, keepdims=True) if args.tta else _softmax(scores)
        per_model.append(int((probs.argmax(1) != labels).sum()))
        acc = probs if acc is None else acc + probs

    preds = (acc / len(args.checkpoints)).argmax(1)
    errs = int((preds != labels).sum())
    n = len(labels)
    _, mb_preds, _ = load_predictions(args.baseline)
    mb_ok, c_ok = mb_preds == labels, preds == labels
    n10 = int((~mb_ok & c_ok).sum())
    n01 = int((mb_ok & ~c_ok).sum())
    p = mcnemar_exact(n10, n01)
    tag = "TTA" if args.tta else "clean"
    print(f"ensemble ({len(args.checkpoints)} models, {tag}): {errs} errors  "
          f"acc {(1-errs/n)*100:.4f}%   per-model errors {sorted(per_model)}")
    print(f"McNemar vs baseline: n10={n10} n01={n01} p={p:.4f}  "
          f"{'SIGNIFICANT' if p < 0.05 else 'not significant'} at 0.05")


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


if __name__ == "__main__":
    main()
