"""Corruption robustness: compact student vs MallaNet on DHCD test (experiment #3).

For each corruption (noise/blur/contrast) x severity 0..5 we corrupt the DHCD
test images once in [0,1] space and feed the *same* corrupted batch to both
models, each with its own normalization (student: DHCD mean/std; MallaNet:
0.5/0.5 + its recovered class permutation). Severity 0 = clean.

Reports per-curve accuracy, mean-corruption accuracy (mCA, severities 1..5),
and relative accuracy retained vs clean. Produces Figure 4 + Table 4 (robustness).

Usage: .venv/bin/python scripts/robustness_eval.py
       [--student results/student_distilled_seed0/best.pth]
       [--mallanet /tmp/mallanet_repo/models/best_model.pth]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from devnet.data import load_split
from devnet.models.mallanet_baseline import EnhancedBMCNNwHFCs
from devnet.models.student import DevNet
from devnet.robustness import CORRUPTIONS, corrupt
from devnet.train import resolve_device

DHCD_ROOT = "data/extracted/DevanagariHandwrittenCharacterDataset"
SEVERITIES = [0, 1, 2, 3, 4, 5]


def recover_permutation(preds, labels, num_classes=46):
    """Map MallaNet output index -> our class index via per-class modal prediction."""
    mapping = np.full(num_classes, -1)
    for c in range(num_classes):
        mapping[np.bincount(preds[labels == c], minlength=num_classes).argmax()] = c
    if len(set(mapping)) != num_classes or -1 in mapping:
        raise RuntimeError(f"Recovered mapping is not a bijection: {mapping}")
    return mapping


def load_student(path, device):
    c = torch.load(path, map_location=device, weights_only=False)
    m = DevNet(tuple(c["config"]["widths"]), tuple(c["config"]["depths"]),
               46, c["config"]["dropout"]).to(device).eval()
    m.load_state_dict(c["model_state_dict"])
    return m, float(c["mean"]), float(c["std"])


@torch.no_grad()
def predict_corrupted(models_norm, x01, kind, severity, device, gen):
    """x01: [N,1,32,32] in [0,1]. Returns {name: preds[N]} for the corrupted set."""
    preds = {name: [] for name in models_norm}
    for i in range(0, len(x01), 256):
        b = x01[i:i + 256].to(device)
        b = corrupt(b, kind, severity, generator=gen)
        for name, (model, mean, std) in models_norm.items():
            out = model((b - mean) / std)
            preds[name].append(out.argmax(1).cpu())
    return {name: torch.cat(p).numpy() for name, p in preds.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="results/student_distilled_seed0/best.pth")
    ap.add_argument("--mallanet", default="/tmp/mallanet_repo/models/best_model.pth")
    ap.add_argument("--out", default="results/robustness.json")
    args = ap.parse_args()
    device = resolve_device("auto")

    images, labels, _ = load_split(DHCD_ROOT, "Test")
    x01 = torch.from_numpy(images).float().unsqueeze(1) / 255.0  # [N,1,32,32]

    student, s_mean, s_std = load_student(args.student, device)
    mallanet = EnhancedBMCNNwHFCs(num_classes=46).to(device).eval()
    mallanet.load_state_dict(
        torch.load(args.mallanet, map_location=device, weights_only=False)["model_state_dict"])

    # Recover MallaNet's class permutation from a clean pass.
    gen = torch.Generator(device=device)
    clean = predict_corrupted({"mallanet": (mallanet, 0.5, 0.5)},
                              x01, "noise", 0, device, gen)["mallanet"]
    mapping = recover_permutation(clean, labels)  # model idx -> our class idx
    inv = np.argsort(mapping)

    models_norm = {"student": (student, s_mean, s_std),
                   "mallanet": (mallanet, 0.5, 0.5)}

    curves = {name: {} for name in models_norm}
    for kind in CORRUPTIONS:
        for sev in SEVERITIES:
            gen.manual_seed(1000 * (CORRUPTIONS.index(kind) + 1) + sev)
            preds = predict_corrupted(models_norm, x01, kind, sev, device, gen)
            for name in models_norm:
                p = mapping[preds[name]] if name == "mallanet" else preds[name]
                acc = float((p == labels).mean())
                curves[name].setdefault(kind, []).append(round(acc, 4))

    def summarize(name):
        clean_acc = curves[name][CORRUPTIONS[0]][0]
        per = {}
        mca_all = []
        for kind in CORRUPTIONS:
            corr = curves[name][kind][1:]  # severities 1..5
            per[kind] = {"clean": curves[name][kind][0],
                         "by_severity": curves[name][kind],
                         "mCA": round(float(np.mean(corr)), 4)}
            mca_all.extend(corr)
        return {"clean": clean_acc,
                "mCA_overall": round(float(np.mean(mca_all)), 4),
                "retained_vs_clean": round(float(np.mean(mca_all)) / clean_acc, 4),
                "per_corruption": per}

    result = {
        "student_checkpoint": args.student,
        "severity_levels": {"noise_std": [0.0, 0.04, 0.08, 0.12, 0.18, 0.25],
                            "blur_sigma": [0.0, 0.4, 0.7, 1.0, 1.4, 1.9],
                            "contrast_factor": [1.0, 0.8, 0.65, 0.5, 0.35, 0.2]},
        "n_test": int(len(labels)),
        "student": summarize("student"),
        "mallanet": summarize("mallanet"),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
