# tests/test_stats.py
import numpy as np
import pytest

from devnet.stats import expected_calibration_error, mcnemar_exact, wilson_ci


def test_wilson_ci_matches_reference():
    # 40 errors out of 13800 -> acc 99.710%, CI [99.606, 99.787] (spec §2)
    lo, hi = wilson_ci(correct=13760, n=13800)
    assert lo == pytest.approx(0.99606, abs=2e-5)
    assert hi == pytest.approx(0.99787, abs=2e-5)


def test_wilson_ci_bounds_are_probabilities():
    lo, hi = wilson_ci(correct=0, n=10)
    assert 0.0 <= lo <= hi <= 1.0
    lo, hi = wilson_ci(correct=10, n=10)
    assert 0.0 <= lo <= hi <= 1.0


def test_mcnemar_reference_values():
    # From spec §2 power table
    assert mcnemar_exact(30, 15) == pytest.approx(0.0357, abs=2e-4)
    assert mcnemar_exact(25, 10) == pytest.approx(0.0167, abs=2e-4)
    assert mcnemar_exact(35, 30) == pytest.approx(0.6201, abs=2e-4)


def test_mcnemar_edge_cases():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert mcnemar_exact(10, 3) == mcnemar_exact(3, 10)  # symmetric


def test_ece_perfectly_calibrated_is_zero():
    conf = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    correct = np.array([1, 1, 1, 1, 0])  # 80% accuracy at 80% confidence
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_overconfident():
    conf = np.array([0.99, 0.99, 0.99, 0.99])
    correct = np.array([1, 1, 0, 0])  # 50% accuracy at 99% confidence
    assert expected_calibration_error(conf, correct) == pytest.approx(0.49, abs=1e-9)
