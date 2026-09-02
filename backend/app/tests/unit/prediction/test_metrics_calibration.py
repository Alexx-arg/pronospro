"""Tests for calibration metrics (Sprint 5.3)."""

from __future__ import annotations

import numpy as np
import pytest
from app.prediction.metrics.calibration import (
    DEFAULT_CONFIDENCE_BUCKETS,
    confidence_bucket_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)


def _perfect_proba(n: int = 9) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.tile(np.array([0, 1, 2]), (n // 3 + 1))[:n]
    y_proba = np.zeros((n, 3))
    y_proba[np.arange(n), y_true] = 0.95
    # distribute remainder
    for i in range(n):
        others = [c for c in range(3) if c != int(y_true[i])]
        y_proba[i, others[0]] = 0.025
        y_proba[i, others[1]] = 0.025
    return y_true, y_proba


def test_ece_perfect_low() -> None:
    y_true, y_proba = _perfect_proba(30)
    ece = expected_calibration_error(y_true, y_proba, n_bins=10)
    # 95% confidence, 100% accuracy → gap ~0.05
    assert ece == pytest.approx(0.05, abs=0.02)


def test_ece_uniform_high() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_proba = np.array([[1 / 3, 1 / 3, 1 / 3]] * 6)
    ece = expected_calibration_error(y_true, y_proba, n_bins=10)
    # confidence ~0.333, accuracy ~0.333 (argmax always 0, 1/3 correct)
    assert 0 <= ece <= 0.5


def test_mce_ge_ece() -> None:
    y_true, y_proba = _perfect_proba(20)
    ece = expected_calibration_error(y_true, y_proba)
    mce = maximum_calibration_error(y_true, y_proba)
    assert mce >= ece


def test_mce_perfect_small() -> None:
    y_true, y_proba = _perfect_proba(20)
    mce = maximum_calibration_error(y_true, y_proba, n_bins=5)
    assert mce < 0.2


def test_reliability_bins_count_sum() -> None:
    y_true, y_proba = _perfect_proba(10)
    bins = reliability_bins(y_true, y_proba, n_bins=5)
    assert len(bins) == 5
    assert sum(b["count"] for b in bins) == 10
    # empty bins have weight 0
    for b in bins:
        if b["count"] == 0:
            assert b["weight"] == 0


def test_reliability_bins_all_same_confidence() -> None:
    y_true = np.array([0, 0, 0])
    y_proba = np.array([[0.7, 0.2, 0.1]] * 3)
    bins = reliability_bins(y_true, y_proba, n_bins=10)
    # only one bin populated
    populated = [b for b in bins if b["count"] > 0]
    assert len(populated) == 1
    assert populated[0]["count"] == 3


def test_reliability_bins_n_zero_error() -> None:
    with pytest.raises(ValueError):
        reliability_bins([0], [[0.33, 0.33, 0.34]], n_bins=0)
    with pytest.raises(ValueError):
        expected_calibration_error([0], [[0.33, 0.33, 0.34]], n_bins=1)


def test_confidence_buckets_default() -> None:
    assert DEFAULT_CONFIDENCE_BUCKETS == [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_confidence_buckets_sum_n() -> None:
    y_true = np.array([0, 1, 2, 0])
    y_proba = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.4, 0.3, 0.3], [0.45, 0.45, 0.1]])
    buckets = confidence_bucket_metrics(y_true, y_proba)
    # Only predictions with conf >=0.4 counted
    conf = y_proba.max(axis=1)
    expected_n = int((conf >= 0.4).sum())
    assert sum(b["n"] for b in buckets) == expected_n


def test_confidence_buckets_below_04_excluded() -> None:
    # All confidences 1/3 <0.4 → bucket total 0
    y_true = np.array([0, 1, 2])
    y_proba = np.array([[1 / 3, 1 / 3, 1 / 3]] * 3)
    buckets = confidence_bucket_metrics(y_true, y_proba)
    assert sum(b["n"] for b in buckets) == 0
    for b in buckets:
        assert b["n"] == 0


def test_confidence_buckets_boundary() -> None:
    y_true = np.array([0, 0])
    y_proba = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    # conf=0.5 → should fall in [0.5,0.6) bucket (index 1)
    buckets = confidence_bucket_metrics(y_true, y_proba)
    # buckets: [0.4-0.5), [0.5-0.6) ...
    assert buckets[0]["n"] == 0
    assert buckets[1]["n"] == 2


def test_confidence_buckets_invalid_edges() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        confidence_bucket_metrics([0], [[0.5, 0.3, 0.2]], bucket_edges=[0.5, 0.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        confidence_bucket_metrics([0], [[0.5, 0.3, 0.2]], bucket_edges=[-0.1, 0.5])
    with pytest.raises(ValueError, match=">=2"):
        confidence_bucket_metrics([0], [[0.5, 0.3, 0.2]], bucket_edges=[0.5])


def test_confidence_buckets_n_one() -> None:
    y_true = np.array([2])
    y_proba = np.array([[0.1, 0.2, 0.7]])
    buckets = confidence_bucket_metrics(y_true, y_proba)
    assert sum(b["n"] for b in buckets) == 1


def test_calibration_rejects_nan_proba() -> None:
    with pytest.raises(ValueError, match="NaN"):
        expected_calibration_error([0], [[float("nan"), 0.5, 0.5]])


def test_calibration_rejects_proba_not_sum_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        expected_calibration_error([0], [[0.5, 0.5, 0.5]])
