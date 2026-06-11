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
    """Run model over loader. Returns (scores [N,C], preds [N], labels [N]) numpy.

    Without TTA, scores are raw logits. With tta=True, scores are mean softmax
    probabilities over the views. Downstream code must not assume which —
    use ensemble_predictions / argmax, which handle both.
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
    scores = torch.cat(all_out).numpy()
    labels = torch.cat(all_y).numpy()
    return scores, scores.argmax(1), labels


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


def _to_probs(scores: np.ndarray) -> torch.Tensor:
    """Logits -> softmax; already-probabilities pass through unchanged."""
    t = torch.from_numpy(np.asarray(scores, dtype=np.float64))
    if torch.allclose(t.sum(1), torch.ones(t.shape[0], dtype=torch.float64), atol=1e-3) \
            and t.min() >= 0:
        return t
    return F.softmax(t, dim=1)


def ensemble_predictions(paths):
    """Average per-model probabilities across prediction files -> preds."""
    probs = None
    for p in paths:
        scores, _, _ = load_predictions(p)
        sm = _to_probs(scores)
        probs = sm if probs is None else probs + sm
    return probs.argmax(1).numpy()


def mcnemar_between(candidate_path, baseline_path):
    """Paired McNemar test from two prediction dumps over the same test set."""
    _, preds_c, labels_c = load_predictions(candidate_path)
    _, preds_b, labels_b = load_predictions(baseline_path)
    if not np.array_equal(labels_c, labels_b):
        raise ValueError("prediction files use different test-set order")
    c_ok = preds_c == labels_c
    b_ok = preds_b == labels_b
    n10 = int((~b_ok & c_ok).sum())  # baseline wrong, candidate right
    n01 = int((b_ok & ~c_ok).sum())
    return {"n10": n10, "n01": n01, "p_value": mcnemar_exact(n10, n01)}
