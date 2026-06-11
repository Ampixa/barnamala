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
