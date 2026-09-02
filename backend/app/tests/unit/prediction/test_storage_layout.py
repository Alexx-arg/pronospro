"""Sprint 5.0 — storage layout produces deterministic, traversal-safe
paths.

Tests are pure (no disk touch).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.prediction.contracts import ModelName
from app.prediction.storage import layout


def test_dataset_dir_layout() -> None:
    p = layout.dataset_dir(Path("/tmp/ws"), dataset_version="v001")
    assert p == Path("/tmp/ws/data/datasets/v001")


def test_model_dir_layout() -> None:
    p = layout.model_dir(Path("ws"), model_name=ModelName.GRADIENT_BOOSTING, model_version="v001")
    assert p == Path("ws/data/models/gradient_boosting/v001")


def test_artifact_paths_are_children_of_model_dir() -> None:
    root = Path("ws")
    bin_p = layout.artifact_bin_path(root, model_name=ModelName.POISSON, model_version="v009")
    man_p = layout.artifact_manifest_path(root, model_name=ModelName.POISSON, model_version="v009")
    cal_p = layout.calibrator_path(root, model_name=ModelName.POISSON, model_version="v009")
    expected_parent = Path("ws/data/models/poisson/v009")
    assert bin_p == expected_parent / "artifact.bin"
    assert man_p == expected_parent / "manifest.json"
    assert cal_p == expected_parent / "calibrator.json"


def test_run_paths() -> None:
    root = Path("ws")
    run_id = "abc123"
    assert layout.run_dir(root, run_id=run_id) == Path("ws/data/models/runs/abc123")
    assert layout.run_config_path(root, run_id=run_id).name == "config.json"
    assert layout.run_summary_path(root, run_id=run_id).name == "summary.json"


def test_fold_paths_under_run_dir() -> None:
    root = Path("ws")
    p = layout.fold_predictions_path(root, run_id="r1", fold_id="fold_001")
    assert p == Path("ws/data/models/runs/r1/folds/fold_001/predictions.csv")


def test_prediction_record_path_layout() -> None:
    root = Path("ws")
    p = layout.prediction_record_path(
        root, model_name=ModelName.ELO_BASELINE, model_version="v001", fixture_id=42
    )
    assert p == Path("ws/data/predictions/elo_baseline/v001/42.json")


def test_comparison_csv_path_uses_date_tag() -> None:
    p = layout.comparison_csv_path(Path("ws"), date_tag="2026-08-15")
    assert p == Path("ws/data/models/runs/comparison_2026-08-15.csv")


def test_layout_is_deterministic() -> None:
    """Identical inputs → identical outputs across calls."""
    root = Path("ws")
    a = layout.model_dir(root, model_name=ModelName.POISSON, model_version="v001")
    b = layout.model_dir(root, model_name=ModelName.POISSON, model_version="v001")
    assert a == b


def test_safe_segment_rejects_traversal_attempts() -> None:
    # We exercise the seg-extraction indirectly through layout.model_dir.
    with pytest.raises(ValueError):
        layout.model_dir(Path("ws"), model_name=ModelName.POISSON, model_version="..")
    with pytest.raises(ValueError):
        layout.model_dir(Path("ws"), model_name=ModelName.POISSON, model_version="a/b")


def test_safe_segment_rejects_empty() -> None:
    with pytest.raises(ValueError):
        layout.run_dir(Path("ws"), run_id="")
