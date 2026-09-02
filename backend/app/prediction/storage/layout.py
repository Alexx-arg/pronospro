"""Canonical deterministic filesystem paths for the prediction engine.

The layout is **relative** by design: callers should anchor it against
a configurable base (the repo root or a per-test ``tmp_path``), which
the Sprint 5.8 stores will receive at construction time. This keeps
artifacts relocatable — a property that the Phase 4 dataset contract
also relies on.

Layout (see ``docs/PHASE_5.md`` §2):

::

    <root>/data/datasets/v001/                 # Phase 4 (already exists)
    <root>/data/models/<model_name>/<model_version>/
        artifact.bin
        manifest.json
        calibrator.json
    <root>/data/models/runs/<run_id>/
        config.json
        folds/fold_001/
            train_meta.json
            test_meta.json
            predictions.csv
            metrics.json
        summary.json
    <root>/data/predictions/<model_name>/<model_version>/<fixture_id>.json

No I/O happens in this module — it only produces paths. The existence /
creation / locking of files is the responsibility of the Sprint 5.8
stores (``ModelArtifactStore``, ``RunsStore``, ``PredictionStore``).
"""

from __future__ import annotations

from pathlib import Path

from app.prediction.contracts import ModelName


def _safe_segment(value: str) -> str:
    """Whitelist sanitisation: model_name / model_version are
    canonical in our domain (e.g. ``elo_baseline``, ``v001``) but we
    still refuse any path-traversal-ish characters defensively.
    """
    if not value:
        raise ValueError("path segment must be non-empty")
    if "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"unsafe path segment: {value!r}")
    return value


def dataset_dir(root: Path, *, dataset_version: str) -> Path:
    """Path to the Phase 4 dataset directory
    ``<root>/data/datasets/<dataset_version>/``.

    The directory's existence is NOT checked here — the Phase 4 loader
    already raises ``DatasetLoadError`` when missing.
    """
    return root / "data" / "datasets" / _safe_segment(dataset_version)


def model_dir(
    root: Path, *, model_name: ModelName, model_version: str
) -> Path:
    """``<root>/data/models/<model_name>/<model_version>/``."""
    return (
        root
        / "data"
        / "models"
        / _safe_segment(str(model_name))
        / _safe_segment(model_version)
    )


def artifact_bin_path(
    root: Path, *, model_name: ModelName, model_version: str
) -> Path:
    """``<model_dir>/artifact.bin`` — the payload (weights/parameters)."""
    return model_dir(root, model_name=model_name, model_version=model_version) / "artifact.bin"


def artifact_manifest_path(
    root: Path, *, model_name: ModelName, model_version: str
) -> Path:
    """``<model_dir>/manifest.json`` — the :class:`ModelArtifact`
    serialised to JSON.
    """
    return model_dir(root, model_name=model_name, model_version=model_version) / "manifest.json"


def calibrator_path(
    root: Path, *, model_name: ModelName, model_version: str
) -> Path:
    """``<model_dir>/calibrator.json`` — fitted calibrator params.

    Always present under a trained-and-calibrated artifact even when
    the chosen method is ``identity`` (the JSON records the choice
    + validation-time ECE for auditability).
    """
    return model_dir(root, model_name=model_name, model_version=model_version) / "calibrator.json"


def run_dir(root: Path, *, run_id: str) -> Path:
    """``<root>/data/models/runs/<run_id>/`` — every backtest run dir."""
    return root / "data" / "models" / "runs" / _safe_segment(run_id)


def run_config_path(root: Path, *, run_id: str) -> Path:
    """``<run_dir>/config.json`` — frozen copy of the iterator + model
    params at run start (so re-deriving ``run_id`` outside this machine
    produces an identical summary path).
    """
    return run_dir(root, run_id=run_id) / "config.json"


def run_summary_path(root: Path, *, run_id: str) -> Path:
    """``<run_dir>/summary.json`` — aggregated metrics across all folds
    for this run_id.
    """
    return run_dir(root, run_id=run_id) / "summary.json"


def fold_dir(root: Path, *, run_id: str, fold_id: str) -> Path:
    """``<run_dir>/folds/<fold_id>/`` — fold-specific outputs."""
    return run_dir(root, run_id=run_id) / "folds" / _safe_segment(fold_id)


def fold_train_meta_path(
    root: Path, *, run_id: str, fold_id: str
) -> Path:
    return fold_dir(root, run_id=run_id, fold_id=fold_id) / "train_meta.json"


def fold_test_meta_path(root: Path, *, run_id: str, fold_id: str) -> Path:
    return fold_dir(root, run_id=run_id, fold_id=fold_id) / "test_meta.json"


def fold_predictions_path(
    root: Path, *, run_id: str, fold_id: str
) -> Path:
    """``<fold_dir>/predictions.csv`` — one row per predicted fixture
    for that fold. CSV (not Parquet) per the no-pyarrow convention of
    Phase 4.
    """
    return fold_dir(root, run_id=run_id, fold_id=fold_id) / "predictions.csv"


def fold_metrics_path(root: Path, *, run_id: str, fold_id: str) -> Path:
    return fold_dir(root, run_id=run_id, fold_id=fold_id) / "metrics.json"


def prediction_record_path(
    root: Path,
    *,
    model_name: ModelName,
    model_version: str,
    fixture_id: int,
) -> Path:
    """``<root>/data/predictions/<model_name>/<model_version>/<fixture_id>.json``.

    The triple ``(model_name, model_version, fixture_id)`` is unique
    per prediction contract (see
    :class:`app.prediction.artifacts.PredictionRecord`). Sprint 5.8's
    ``PredictionStore`` rejects a duplicate write.
    """
    return (
        root
        / "data"
        / "predictions"
        / _safe_segment(str(model_name))
        / _safe_segment(model_version)
        / f"{int(fixture_id)}.json"
    )


def comparison_csv_path(root: Path, *, date_tag: str) -> Path:
    """``<root>/data/models/runs/comparison_<date_tag>.csv`` — summary
    of multiple runs side-by-side (one row per ``(model_name,
    model_version)`` on the same ``(dataset_version, iterator_params)``).
    """
    return root / "data" / "models" / "runs" / f"comparison_{_safe_segment(date_tag)}.csv"


__all__ = [
    "artifact_bin_path",
    "artifact_manifest_path",
    "calibrator_path",
    "comparison_csv_path",
    "dataset_dir",
    "fold_dir",
    "fold_metrics_path",
    "fold_predictions_path",
    "fold_test_meta_path",
    "fold_train_meta_path",
    "model_dir",
    "prediction_record_path",
    "run_config_path",
    "run_dir",
    "run_summary_path",
]
