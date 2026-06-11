# tests/test_evaluate.py
import numpy as np
import torch

from devnet.evaluate import (ensemble_predictions, mcnemar_between,
                             metrics_from_predictions, predict,
                             save_predictions, load_predictions, tta_views)


class StubModel(torch.nn.Module):
    """Always predicts class = pixel sum mod 3 (deterministic, input-driven)."""
    def forward(self, x):
        idx = (x.sum(dim=(1, 2, 3)).long().abs() % 3)
        return torch.nn.functional.one_hot(idx, 3).float() * 10


def test_predict_and_roundtrip(tmp_path):
    model = StubModel()
    x = torch.randn(10, 1, 32, 32)
    y = torch.randint(0, 3, (10,))
    logits, preds, labels = predict(model, [(x, y)], device=torch.device("cpu"))
    assert logits.shape == (10, 3)
    p = tmp_path / "preds.npz"
    save_predictions(p, logits, preds, labels)
    l2, p2, y2 = load_predictions(p)
    assert np.array_equal(preds, p2) and np.array_equal(labels, y2)


def test_metrics_from_predictions():
    preds = np.array([0, 1, 2, 2])
    labels = np.array([0, 1, 2, 1])
    m = metrics_from_predictions(preds, labels)
    assert m["accuracy"] == 0.75
    assert 0 < m["macro_f1"] <= 1
    assert m["n_errors"] == 1
    assert "confusion" in m and np.asarray(m["confusion"]).shape == (3, 3)


def test_ensemble_majority_wins(tmp_path):
    labels = np.array([0, 1])
    # model A correct on both; model B wrong on second with low confidence
    la = np.array([[5.0, 0.0], [0.0, 5.0]])
    lb = np.array([[4.0, 0.0], [1.1, 1.0]])
    pa, pb = tmp_path / "a.npz", tmp_path / "b.npz"
    save_predictions(pa, la, la.argmax(1), labels)
    save_predictions(pb, lb, lb.argmax(1), labels)
    preds = ensemble_predictions([pa, pb])
    assert np.array_equal(preds, labels)


def test_mcnemar_between_files(tmp_path):
    labels = np.zeros(100, dtype=np.int64)
    preds_a = np.zeros(100, dtype=np.int64); preds_a[:30] = 1  # 30 errors
    preds_b = np.zeros(100, dtype=np.int64)                    # 0 errors
    la = np.zeros((100, 2)); lb = np.zeros((100, 2))
    pa, pb = tmp_path / "a.npz", tmp_path / "b.npz"
    save_predictions(pa, la, preds_a, labels)
    save_predictions(pb, lb, preds_b, labels)
    result = mcnemar_between(pb, pa)  # candidate b vs baseline a
    assert result["n10"] == 30 and result["n01"] == 0
    assert result["p_value"] < 1e-6


def test_tta_views_count_and_shape():
    x = torch.randn(4, 1, 32, 32)
    views = tta_views(x)
    assert len(views) == 5
    for v in views:
        assert v.shape == x.shape
