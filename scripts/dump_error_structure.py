"""Persist the error-overlap structure behind the saturation analysis (F3 + Sec 5).

Loads the 5 distilled-student prediction dumps + the MallaNet baseline dump,
computes how many test images are misclassified by exactly k models, the
all-model floor, mean pairwise error-set Jaccard, and per-class / confusable-pair
error counts. Read source of truth for results/error_structure.json.
"""
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from devnet.data import load_split
from devnet.evaluate import load_predictions

DATA = "data/extracted/DevanagariHandwrittenCharacterDataset"
_, _, CLASSES = load_split(DATA, "Test")


def main():
    # All 15 student models (distilled, clean-distilled, supervised) + MallaNet baseline
    # SUMMARY.md floor is "11 images missed by ALL 15 students AND MallaNet" = 16 models
    paths = sorted(glob.glob("results/student_*/test_predictions.npz"))
    paths += ["results/mallanet_baseline/test_predictions.npz"]
    wrongs, labels = [], None
    for p in paths:
        _, preds, lab = load_predictions(p)
        labels = lab if labels is None else labels
        wrongs.append(preds != lab)
    W = np.stack(wrongs)                       # [M, N] bool, M models
    per_image = W.sum(0)                       # how many models miss each image
    shared_by_k = [int((per_image == k).sum()) for k in range(len(paths) + 1)]
    floor = int((per_image == len(paths)).sum())

    jac = []
    for a, b in combinations(range(len(paths)), 2):
        inter = int((W[a] & W[b]).sum())
        union = int((W[a] | W[b]).sum())
        if union:
            jac.append(inter / union)

    # per-class + confusable pairs over the 5 distilled students only
    stu = sorted(glob.glob("results/student_distilled_seed*/test_predictions.npz"))
    per_class = np.zeros(len(CLASSES), dtype=int)
    pair = {}
    for p in stu:
        _, preds, lab = load_predictions(p)
        for t, q in zip(lab[preds != lab], preds[preds != lab]):
            per_class[t] += 1
            pair[(int(t), int(q))] = pair.get((int(t), int(q)), 0) + 1
    top = sorted(pair.items(), key=lambda kv: -kv[1])[:8]

    out = {
        "models": [Path(p).parent.name for p in paths],
        "n_models": len(paths),
        "shared_by_k": shared_by_k,
        "floor_all_models": floor,
        "mean_pairwise_jaccard": round(float(np.mean(jac)), 4),
        "per_class_errors_distilled": {CLASSES[i]: int(per_class[i])
                                       for i in range(len(CLASSES)) if per_class[i]},
        "top_confusions": [[CLASSES[t], CLASSES[q], c] for (t, q), c in top],
    }
    Path("results/error_structure.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
