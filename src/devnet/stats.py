"""Pure statistical functions for benchmark evaluation."""
from math import comb, sqrt

import numpy as np


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


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
