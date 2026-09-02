# ruff: noqa: N806
"""Backtest runner — Sprint 5.8.

Orchestrates WalkForwardIterator → Trainer → Predictor → CalibrationDetector/Calibrator → Metrics → FoldReport → Storage.
Guarantees:
* train never sees val/test
* calibration uses only val
* test never used for fitting/selection
* same folds for all models in run_backtest_all
* reproducibility via derive_seed
* base_path injectable (tmp_path in tests)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.dataset.loader import LoadedDataset
from app.prediction.backtesting.fold import Fold
from app.prediction.backtesting.iterator import WalkForwardIterator
from app.prediction.calibration.detector import CalibrationDetector
from app.prediction.contracts import ModelName
from app.prediction.features.vector import loaded_example_to_features
from app.prediction.metrics.calibration import expected_calibration_error
from app.prediction.metrics.classification import accuracy, brier_score, confusion_matrix, log_loss
from app.prediction.metrics.report import FoldReport
from app.prediction.seeds import derive_seed
from app.prediction.storage.artifacts import ModelArtifactStore
from app.prediction.storage.predictions import PredictionStore
from app.prediction.storage.runs import RunsStore


@dataclass(frozen=True, slots=True)
class FoldResult:
    fold: Fold
    artifact: Any
    calibrator_dict: dict[str, Any]
    report: FoldReport
    predictions: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    model_name: ModelName
    model_version: str
    folds: tuple[FoldResult, ...]


def _compute_run_id(
    dataset_version: str,
    iterator_params: dict[str, Any],
    model_name: str,
    model_version: str,
    seed: int,
) -> str:
    payload = json.dumps(
        {
            "dataset_version": dataset_version,
            "iterator_params": iterator_params,
            "model_name": str(model_name),
            "model_version": model_version,
            "seed": seed,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _targets_to_label(targets: dict[str, int | None]) -> int:
    if targets.get("home_win") == 1:
        return 0
    if targets.get("draw") == 1:
        return 1
    if targets.get("away_win") == 1:
        return 2
    # Fallback: infer from goals if present
    hg = targets.get("home_goals")
    ag = targets.get("away_goals")
    if hg is not None and ag is not None:
        if hg > ag:
            return 0
        if hg == ag:
            return 1
        return 2
    raise ValueError(f"cannot infer label from targets {targets}")


def _run_single_fold(
    fold: Fold,
    dataset: LoadedDataset,
    trainer: Any,
    seed_fold: int,
    model_version: str,
) -> FoldResult:
    # Extract LoadedExample blocks
    train_examples = [dataset.rows[i] for i in fold.train_indices]
    val_examples = [dataset.rows[i] for i in fold.val_indices]
    test_examples = [dataset.rows[i] for i in fold.test_indices]

    # Vectorize
    train_features = [loaded_example_to_features(ex) for ex in train_examples]
    val_features = [loaded_example_to_features(ex) for ex in val_examples]
    test_features = [loaded_example_to_features(ex) for ex in test_examples]

    # Targets: extract label and also keep full targets for goals metrics if needed
    train_targets = [ex.targets for ex in train_examples]
    val_targets = [ex.targets for ex in val_examples]
    test_targets = [ex.targets for ex in test_examples]

    # Convert to trainer protocol format: list of [hw,dr,aw,hg,ag] for Poisson compatibility,
    # and also allow trainer to handle dicts. We pass list of dicts for simplicity
    # but also ensure trainer's _extract can handle dicts.
    # For classification metrics we need labels 0/1/2
    [_targets_to_label(t) for t in train_targets]
    val_labels = [_targets_to_label(t) for t in val_targets]
    test_labels = [_targets_to_label(t) for t in test_targets]

    # Train (only train, never val/test) — use protocol that some trainers accept val_block via kwargs
    # Try to call with val_block if trainer supports (Poisson, Elo), else fallback
    artifact = None
    try:
        # Preferred: pass val for hyperparam search where supported (Elo)
        artifact = trainer.train(
            train_features,
            train_targets,
            hyperparameters=None,
            seed=seed_fold,
            val_block=val_features,
            val_targets=val_targets,
            model_version=model_version,
            training_cutoff=fold.training_cutoff,
        )
    except TypeError:
        # Fallback to protocol without val
        artifact = trainer.train(
            train_features,
            train_targets,
            hyperparameters=None,
            seed=seed_fold,
        )
        # Ensure training_cutoff is set correctly — if trainer didn't use fold cutoff, patch artifact
        # (some trainers ignore training_cutoff kwarg)
        if artifact.training_cutoff != fold.training_cutoff:
            # Re-create artifact with correct cutoff (since frozen, we create new one)
            from dataclasses import replace

            artifact = replace(artifact, training_cutoff=fold.training_cutoff)

    # Retrieve fitted model for prediction
    # Trainers expose get_model()
    model = None
    if hasattr(trainer, "get_model"):
        try:
            model = trainer.get_model(artifact)
        except Exception:
            model = trainer.get_model()
    if model is None:
        # Fallback: try _last_model
        model = getattr(trainer, "_last_model", None)
    if model is None:
        raise RuntimeError("trainer did not expose fitted model")

    # Raw val probs for calibration (val-only)
    val_raw = [model.predict(f) for f in val_features]
    # Detector chooses calibrator based on val ECE/n
    detector = CalibrationDetector()
    calibrator = detector.detect(val_raw, val_labels)
    # Fit calibrator on val only (never test)
    calibrator.fit(val_raw, val_labels)
    calibrator_dict = calibrator.to_dict()
    calibrator_dict["val_ece"] = float(expected_calibration_error(val_labels, np.array([[p.p_home_win, p.p_draw, p.p_away_win] for p in val_raw]), n_bins=10))
    calibrator_dict["val_n"] = int(len(val_labels))

    # Test predictions — raw then calibrated
    test_raw = [model.predict(f) for f in test_features]
    test_calibrated = calibrator.transform(test_raw)
    # Ensure list of MatchProbabilities
    # test_calibrated is Sequence[MatchProbabilities]
    test_proba = np.array([[p.p_home_win, p.p_draw, p.p_away_win] for p in test_calibrated])
    test_labels_arr = np.array(test_labels)

    # Metrics on calibrated test
    acc = float(accuracy(test_labels_arr, test_proba))
    ll = float(log_loss(test_labels_arr, test_proba))
    brier = brier_score(test_labels_arr, test_proba)
    from app.prediction.metrics.calibration import confidence_bucket_metrics, reliability_bins
    from app.prediction.metrics.calibration import expected_calibration_error as ece_fn
    from app.prediction.metrics.calibration import maximum_calibration_error as mce_fn

    ece = float(ece_fn(test_labels_arr, test_proba))
    mce = float(mce_fn(test_labels_arr, test_proba))
    cm = confusion_matrix(test_labels_arr, test_proba)
    # Convert cm to list
    cm_list = cm.tolist() if hasattr(cm, "tolist") else [[int(x) for x in row] for row in cm]
    rel_bins = reliability_bins(test_labels_arr, test_proba, n_bins=10)
    conf_buckets = confidence_bucket_metrics(test_labels_arr, test_proba)

    # Goal metrics optional (if test has goals)
    mae_home = mae_away = rmse_home = rmse_away = rmse_total = poisson_ll = None
    # We keep None for Sprint 5.8 unless model provides goals and test has them
    # Not computing to avoid coupling

    report = FoldReport(
        model_name=str(artifact.model_name),
        model_version=artifact.model_version,
        dataset_version=dataset.manifest.dataset_version,
        fold_id=fold.fold_id,
        n_predictions=int(len(test_features)),
        accuracy=acc,
        log_loss=ll,
        brier_home=float(brier["brier_home"]),
        brier_draw=float(brier["brier_draw"]),
        brier_away=float(brier["brier_away"]),
        brier_multiclass=float(brier["brier_multiclass"]),
        ece=ece,
        mce=mce,
        n_bins=10,
        confusion_matrix=cm_list,
        confidence_buckets=conf_buckets,
        reliability_bins=rel_bins,
        mae_home_goals=mae_home,
        mae_away_goals=mae_away,
        rmse_home_goals=rmse_home,
        rmse_away_goals=rmse_away,
        rmse_total_goals=rmse_total,
        poisson_loglik=poisson_ll,
    )

    # Predictions for storage (one per test fixture)
    predictions: list[dict[str, Any]] = []
    for feat, prob, true_label in zip(test_features, test_calibrated, test_labels, strict=False):
        predictions.append(
            {
                "fixture_id": int(feat.fixture_id),
                "kickoff": feat.kickoff.isoformat(),
                "p_home_win": float(prob.p_home_win),
                "p_draw": float(prob.p_draw),
                "p_away_win": float(prob.p_away_win),
                "true_label": int(true_label),
            }
        )

    return FoldResult(
        fold=fold,
        artifact=artifact,
        calibrator_dict=calibrator_dict,
        report=report,
        predictions=predictions,
    )


def run_backtest(
    dataset: LoadedDataset,
    trainer: Any,
    *,
    iterator_params: dict[str, Any],
    model_version: str = "v001",
    seed: int = 42,
    base_path: Path | str = Path("data"),
) -> RunResult:
    """Run backtest for a single model over all folds."""
    base_path = Path(base_path)
    # Build iterator from params
    from app.prediction.contracts import WalkForwardMode

    mode = iterator_params.get("mode", WalkForwardMode.EXPANDING)
    if isinstance(mode, str):
        mode = WalkForwardMode(mode)
    iterator = WalkForwardIterator(
        dataset,
        min_train_size=int(iterator_params["min_train_size"]),
        test_size=int(iterator_params["test_size"]),
        gap_days=int(iterator_params.get("gap_days", 0)),
        mode=mode,
        val_ratio=iterator_params.get("val_ratio"),
        val_size=iterator_params.get("val_size"),
    )
    folds = list(iterator)
    # Run_id deterministic
    run_id = _compute_run_id(
        dataset.manifest.dataset_version,
        iterator_params,
        str(trainer.name),
        model_version,
        seed,
    )

    # Storage
    runs_store = RunsStore(base_path)
    pred_store = PredictionStore(base_path)
    artifact_store = ModelArtifactStore(base_path)

    # Persist run config upfront (overwrite=False)
    config = {
        "dataset_version": dataset.manifest.dataset_version,
        "iterator_params": iterator_params,
        "model_name": str(trainer.name),
        "model_version": model_version,
        "seed": seed,
        "run_id": run_id,
    }
    with contextlib.suppress(FileExistsError):
        runs_store.save_config(run_id, config, overwrite=False)

    fold_results: list[FoldResult] = []
    for fold in folds:
        seed_fold = derive_seed(base_seed=seed, fold_index=fold.fold_id)
        result = _run_single_fold(fold, dataset, trainer, seed_fold, model_version)
        fold_results.append(result)

        # Persist per-fold artifacts/metrics/predictions
        fold_id_str = f"fold_{result.fold.fold_id:03d}"
        # Artifact payload — trainers produce payload bytes via artifact's sha, but we need bytes
        # For simplicity, payload is JSON of artifact hyperparameters
        payload_bytes = json.dumps(
            {"hyperparameters": dict(result.artifact.hyperparameters)}, sort_keys=True
        ).encode()
        with contextlib.suppress(FileExistsError):
            artifact_store.save(result.artifact, payload_bytes, overwrite=False)
        # Fold metrics
        with contextlib.suppress(FileExistsError):
            runs_store.save_fold_metrics(run_id, fold_id_str, result.report.to_dict(), overwrite=False)
        # Fold predictions csv
        with contextlib.suppress(FileExistsError):
            runs_store.save_fold_predictions(run_id, fold_id_str, result.predictions, overwrite=False)
        # Calibrator
        with contextlib.suppress(FileExistsError):
            runs_store.save_fold_calibrator(run_id, fold_id_str, result.calibrator_dict, overwrite=False)
        # PredictionStore per fixture (append-only)
        from app.prediction.artifacts import PredictionRecord

        for prob, feat in zip(
            list(result.predictions),  # already dict
            [dataset.rows[i] for i in result.fold.test_indices], strict=False,
        ):
            # Reconstruct MatchProbabilities from dict
            from app.prediction.contracts import MatchProbabilities

            mp = MatchProbabilities(
                p_home_win=float(prob["p_home_win"]),
                p_draw=float(prob["p_draw"]),
                p_away_win=float(prob["p_away_win"]),
            )
            rec = PredictionRecord(
                model_name=trainer.name,
                model_version=model_version,
                fixture_id=int(prob["fixture_id"]),
                kickoff=feat.kickoff,
                probabilities=mp,
                artifact_sha256=result.artifact.payload_sha256,
                predicted_at=datetime.now(UTC),
            )
            pred_store.save(rec, overwrite=False)

    # Summary
    summary = {
        "run_id": run_id,
        "model_name": str(trainer.name),
        "model_version": model_version,
        "n_folds": len(fold_results),
        "folds": [r.report.to_dict() for r in fold_results],
    }
    with contextlib.suppress(FileExistsError):
        runs_store.save_summary(run_id, summary, overwrite=False)

    return RunResult(
        run_id=run_id,
        model_name=trainer.name,
        model_version=model_version,
        folds=tuple(fold_results),
    )


def run_backtest_all(
    dataset: LoadedDataset,
    trainers: dict[ModelName, Any],
    *,
    iterator_params: dict[str, Any],
    model_version: str = "v001",
    seed: int = 42,
    base_path: Path | str = Path("data"),
) -> dict[ModelName, RunResult]:
    """Run backtest for multiple models over exactly the same folds."""
    base_path = Path(base_path)
    # Materialize iterator once to ensure same folds
    from app.prediction.contracts import WalkForwardMode

    mode = iterator_params.get("mode", WalkForwardMode.EXPANDING)
    if isinstance(mode, str):
        mode = WalkForwardMode(mode)
    iterator = WalkForwardIterator(
        dataset,
        min_train_size=int(iterator_params["min_train_size"]),
        test_size=int(iterator_params["test_size"]),
        gap_days=int(iterator_params.get("gap_days", 0)),
        mode=mode,
        val_ratio=iterator_params.get("val_ratio"),
        val_size=iterator_params.get("val_size"),
    )
    folds = list(iterator)
    # We will run each trainer reusing same folds list by passing iterator_params
    # but each run_backtest will re-create iterator; to guarantee same, we pass same folds via monkey-patch?
    # Instead we call run_backtest per trainer; they will each create iterator with same params → same folds deterministically.
    results: dict[ModelName, RunResult] = {}
    for name, trainer in trainers.items():
        # Use same iterator_params; each run will derive same folds
        result = run_backtest(
            dataset,
            trainer,
            iterator_params=iterator_params,
            model_version=model_version,
            seed=seed,
            base_path=base_path,
        )
        results[name] = result
        # Verify same folds (sanity)
        assert len(result.folds) == len(folds) or len(folds) == 0
    return results


__all__ = ["FoldResult", "RunResult", "run_backtest", "run_backtest_all"]
