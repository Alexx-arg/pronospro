"""Tests for FoldReport (Sprint 5.3)."""

from __future__ import annotations

import json

import pytest
from app.prediction.metrics.report import FoldReport


def _report(**overrides) -> FoldReport:
    base = {
        "model_name": "poisson",
        "model_version": "v001",
        "dataset_version": "v001",
        "fold_id": 0,
        "n_predictions": 3,
        "accuracy": 0.66,
        "log_loss": 0.5,
        "brier_home": 0.2,
        "brier_draw": 0.1,
        "brier_away": 0.2,
        "brier_multiclass": 0.5,
        "ece": 0.05,
        "mce": 0.1,
        "n_bins": 10,
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "confidence_buckets": [],
        "reliability_bins": [],
    }
    base.update(overrides)
    return FoldReport(**base)  # type: ignore[arg-type]


def test_report_serialises_to_json() -> None:
    r = _report()
    js = r.to_json()
    data = json.loads(js)
    assert data["model_name"] == "poisson"
    assert data["n_predictions"] == 3


def test_report_from_dict_roundtrip() -> None:
    r = _report(accuracy=0.9)
    r2 = FoldReport.from_dict(r.to_dict())
    assert r2.accuracy == 0.9
    assert r2.confusion_matrix == r.confusion_matrix


def test_report_confusion_sum_must_equal_n() -> None:
    with pytest.raises(ValueError, match="sum"):
        _report(n_predictions=5, confusion_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_report_rejects_n_zero() -> None:
    with pytest.raises(ValueError, match="n_predictions"):
        _report(n_predictions=0)


def test_report_rejects_negative_fold() -> None:
    with pytest.raises(ValueError, match="fold_id"):
        _report(fold_id=-1)


def test_report_rejects_bad_matrix_shape() -> None:
    with pytest.raises(ValueError, match="3x3"):
        _report(confusion_matrix=[[1, 0], [0, 1]])


def test_report_immutable() -> None:
    r = _report()
    with pytest.raises(AttributeError):
        r.accuracy = 1.0  # type: ignore[misc]


def test_report_goal_metrics_optional() -> None:
    r = _report(mae_home_goals=0.5, poisson_loglik=-2.1)
    d = r.to_dict()
    assert d["mae_home_goals"] == 0.5
    assert d["poisson_loglik"] == -2.1


def test_report_calibration_bins_alias() -> None:
    r = _report()
    assert r.calibration_bins == r.n_bins
