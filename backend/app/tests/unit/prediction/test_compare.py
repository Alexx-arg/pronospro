"""Tests for backtesting/compare.py Sprint 5.10."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from app.prediction.backtesting.compare import COLUMNS, compare_runs
from app.prediction.storage import layout as layout_mod


def _make_run(base_path: Path, run_id: str, dataset_version: str, iterator_params: dict, model_name: str, folds: list[dict]) -> None:
    # Write config and summary via layout
    config = {
        "dataset_version": dataset_version,
        "iterator_params": iterator_params,
        "model_name": model_name,
        "model_version": "v001",
        "seed": 42,
        "run_id": run_id,
    }
    summary = {
        "run_id": run_id,
        "model_name": model_name,
        "model_version": "v001",
        "n_folds": len(folds),
        "folds": folds,
    }
    cfg_path = layout_mod.run_config_path(base_path, run_id=run_id)
    sum_path = layout_mod.run_summary_path(base_path, run_id=run_id)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    sum_path.write_text(json.dumps(summary), encoding="utf-8")


def _fold(accuracy: float, log_loss: float, ece: float, n: int = 10) -> dict:
    return {
        "accuracy": accuracy,
        "log_loss": log_loss,
        "ece": ece,
        "mce": 0.1,
        "brier_home": 0.2,
        "brier_draw": 0.1,
        "brier_away": 0.2,
        "brier_multiclass": 0.5,
        "n_predictions": n,
        "fold_id": 0,
        "model_name": "elo_baseline",
        "model_version": "v001",
        "dataset_version": "v001",
        "n_bins": 10,
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "confidence_buckets": [],
        "reliability_bins": [],
    }


def test_compare_aggregates_correctly(tmp_path: Path) -> None:
    base = tmp_path
    params = {"min_train_size": 100, "test_size": 10, "gap_days": 0, "val_ratio": 0.15, "mode": "expanding"}
    _make_run(base, "run_a", "v001", params, "elo_baseline", [_fold(0.6, 0.5, 0.05, n=10), _fold(0.8, 0.4, 0.07, n=20)])
    _make_run(base, "run_b", "v001", params, "poisson", [_fold(0.7, 0.6, 0.06, n=15)])
    out = compare_runs(base_path=base, dataset_version="v001", iterator_params=params, date_tag="20260513")
    assert out.is_file()
    assert out.name == "comparison_20260513.csv"
    # Read CSV
    with out.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        rows = list(reader)
    assert len(rows) == 2
    # Check weighted for run_a: total 30, weighted log_loss = (10*0.5+20*0.4)/30=0.433...
    row_a = next(r for r in rows if r["model_name"] == "elo_baseline")
    assert float(row_a["weighted_log_loss"]) == pytest.approx((10 * 0.5 + 20 * 0.4) / 30)
    assert int(row_a["total_n_predictions"]) == 30
    assert float(row_a["mean_accuracy"]) == pytest.approx(0.7)


def test_compare_fails_on_different_dataset_version(tmp_path: Path) -> None:
    base = tmp_path
    params = {"min_train_size": 100, "test_size": 10}
    _make_run(base, "run1", "v001", params, "elo_baseline", [_fold(0.6, 0.5, 0.05)])
    _make_run(base, "run2", "v002", params, "poisson", [_fold(0.6, 0.5, 0.05)])
    with pytest.raises(ValueError, match="no runs found"):
        compare_runs(base_path=base, dataset_version="v999", iterator_params=params, date_tag="20260514")
    # When using run_ids with mismatch, should fail
    with pytest.raises(ValueError, match="dataset_version mismatch"):
        compare_runs(base_path=base, run_ids=["run1", "run2"], date_tag="20260515")


def test_compare_fails_on_different_iterator_params(tmp_path: Path) -> None:
    base = tmp_path
    params_a = {"min_train_size": 100, "test_size": 10}
    params_b = {"min_train_size": 200, "test_size": 10}
    _make_run(base, "run_a", "v001", params_a, "elo_baseline", [_fold(0.6, 0.5, 0.05)])
    _make_run(base, "run_b", "v001", params_b, "poisson", [_fold(0.6, 0.5, 0.05)])
    with pytest.raises(ValueError, match="iterator_params mismatch"):
        compare_runs(base_path=base, run_ids=["run_a", "run_b"], date_tag="20260516")


def test_compare_empty_folds(tmp_path: Path) -> None:
    base = tmp_path
    params = {"min_train_size": 100}
    _make_run(base, "run_empty", "v001", params, "elo_baseline", [])
    out = compare_runs(base_path=base, dataset_version="v001", iterator_params=params, date_tag="20260517")
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert int(rows[0]["n_folds"]) == 0
    assert int(rows[0]["total_n_predictions"]) == 0
    # Metrics NaN
    assert rows[0]["mean_accuracy"] == "nan" or rows[0]["mean_accuracy"] == "" or float(rows[0]["mean_accuracy"]) != float(rows[0]["mean_accuracy"])  # NaN check


def test_compare_deterministic(tmp_path: Path) -> None:
    base = tmp_path
    params = {"min_train_size": 100}
    _make_run(base, "run1", "v001", params, "elo_baseline", [_fold(0.6, 0.5, 0.05)])
    out1 = compare_runs(base_path=base, dataset_version="v001", iterator_params=params, date_tag="same")
    out2 = compare_runs(base_path=base, dataset_version="v001", iterator_params=params, date_tag="same")
    assert out1.read_text() == out2.read_text()


def test_compare_csv_not_parquet(tmp_path: Path) -> None:
    base = tmp_path
    params = {"min_train_size": 100}
    _make_run(base, "run1", "v001", params, "elo_baseline", [_fold(0.6, 0.5, 0.05)])
    out = compare_runs(base_path=base, dataset_version="v001", iterator_params=params, date_tag="20260518")
    assert out.suffix == ".csv"
    # Ensure no parquet file created
    assert not any(p.suffix == ".parquet" for p in base.rglob("*"))


def test_compare_no_train_leakage(tmp_path: Path) -> None:
    # Ensure compare only reads summary/config, not raw data
    base = tmp_path
    params = {"min_train_size": 100}
    _make_run(base, "run1", "v001", params, "elo_baseline", [_fold(0.6, 0.5, 0.05)])
    # No dataset needed; compare should succeed without v001 files
    out = compare_runs(base_path=base, dataset_version="v001", iterator_params=params, date_tag="20260519")
    assert out.is_file()
