"""Integration tests for BacktestRunner Sprint 5.8."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.dataset.loader import LoadedDataset, LoadedExample
from app.dataset.manifest import DatasetManifest
from app.features.example import FEATURE_NAMES
from app.prediction.backtesting.runner import run_backtest, run_backtest_all
from app.prediction.contracts import ModelName
from app.prediction.training.elo_trainer import EloBaselineTrainer
from app.prediction.training.gb_trainer import GradientBoostingTrainer
from app.prediction.training.poisson_trainer import PoissonTrainer


def _make_dataset(n: int = 20) -> LoadedDataset:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[LoadedExample] = []
    for i in range(n):
        kickoff = t0.replace(day=1)  # keep simple
        # Use timedelta to ensure increasing
        from datetime import timedelta

        kickoff = t0 + timedelta(days=i)
        feats = {name: float(i % 3) for name in FEATURE_NAMES}
        # Ensure Elo features are realistic
        feats["home_elo_pre_match"] = 1500.0 + (i % 5) * 10
        feats["away_elo_pre_match"] = 1500.0
        feats["elo_difference"] = feats["home_elo_pre_match"] - feats["away_elo_pre_match"]
        # Targets: cycle home/draw/away
        label = i % 3
        if label == 0:
            targets = {"home_win": 1, "draw": 0, "away_win": 0, "home_goals": 2, "away_goals": 0}
        elif label == 1:
            targets = {"home_win": 0, "draw": 1, "away_win": 0, "home_goals": 1, "away_goals": 1}
        else:
            targets = {"home_win": 0, "draw": 0, "away_win": 1, "home_goals": 0, "away_goals": 2}
        rows.append(
            LoadedExample(
                fixture_id=i + 1,
                kickoff=kickoff,
                competition_id=39,
                season_id=1,
                home_team_id=10,
                away_team_id=20,
                features=feats,
                targets=targets,
            )
        )
    manifest = DatasetManifest(
        dataset_version="v001",
        generated_at=t0,
        feature_definition_version="fd_v1",
        data_cutoff=t0,
        source_schema_version="schema_v1",
        row_count=len(rows),
        feature_names=tuple(FEATURE_NAMES),
        target_names=("home_win", "draw", "away_win", "home_goals", "away_goals"),
        start_date=rows[0].kickoff,
        end_date=rows[-1].kickoff,
        competitions=(39,),
        seasons=(1,),
        csv_sha256="x" * 64,
        extras={},
    )
    return LoadedDataset(manifest=manifest, rows=rows)


def _iterator_params() -> dict:
    return {
        "min_train_size": 6,
        "val_ratio": 0.3,
        "test_size": 2,
        "gap_days": 0,
        "mode": "expanding",
    }


def test_runner_no_leakage(tmp_path: Path) -> None:
    dataset = _make_dataset(20)
    trainer = EloBaselineTrainer()
    result = run_backtest(dataset, trainer, iterator_params=_iterator_params(), seed=42, base_path=tmp_path)
    for fr in result.folds:
        train_ids = {dataset.rows[i].fixture_id for i in fr.fold.train_indices}
        val_ids = {dataset.rows[i].fixture_id for i in fr.fold.val_indices}
        test_ids = {dataset.rows[i].fixture_id for i in fr.fold.test_indices}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)


def test_runner_calibration_uses_only_val(tmp_path: Path) -> None:
    dataset = _make_dataset(20)
    trainer = EloBaselineTrainer()
    result = run_backtest(dataset, trainer, iterator_params=_iterator_params(), seed=1, base_path=tmp_path)
    for fr in result.folds:
        # Calibrator dict must have val_ece/val_n (from val only)
        assert "val_ece" in fr.calibrator_dict
        assert "val_n" in fr.calibrator_dict
        assert fr.calibrator_dict["val_n"] == len(fr.fold.val_indices)


def test_runner_metrics_persisted(tmp_path: Path) -> None:
    dataset = _make_dataset(20)
    trainer = PoissonTrainer()
    result = run_backtest(dataset, trainer, iterator_params=_iterator_params(), seed=2, base_path=tmp_path)
    # Check files exist
    for fr in result.folds:
        fold_id = f"fold_{fr.fold.fold_id:03d}"
        metrics_path = tmp_path / "data" / "models" / "runs" / result.run_id / "folds" / fold_id / "metrics.json"
        assert metrics_path.is_file()
        assert fr.report.n_predictions == len(fr.fold.test_indices)


def test_runner_artifact_immutable(tmp_path: Path) -> None:
    dataset = _make_dataset(20)
    trainer = EloBaselineTrainer()
    result = run_backtest(dataset, trainer, iterator_params=_iterator_params(), seed=3, base_path=tmp_path)
    # Second run with same params should not overwrite (overwrite=False)
    result2 = run_backtest(dataset, trainer, iterator_params=_iterator_params(), seed=3, base_path=tmp_path)
    assert result.run_id == result2.run_id
    # Artifacts still exist
    assert result.folds[0].artifact.payload_sha256 == result2.folds[0].artifact.payload_sha256


def test_runner_predictions_immutable(tmp_path: Path) -> None:
    dataset = _make_dataset(20)
    trainer = GradientBoostingTrainer()
    result = run_backtest(dataset, trainer, iterator_params=_iterator_params(), seed=4, base_path=tmp_path)
    # Try to save duplicate prediction — should be warning but not overwrite
    from app.prediction.storage.predictions import PredictionStore

    store = PredictionStore(tmp_path)
    # First prediction already saved; second save with same fixture should not overwrite
    first_fold = result.folds[0]
    first_pred = first_fold.predictions[0]
    # Already saved via runner, now try again
    from app.prediction.artifacts import PredictionRecord
    from app.prediction.contracts import MatchProbabilities

    rec = PredictionRecord(
        model_name=ModelName.GRADIENT_BOOSTING,
        model_version="v001",
        fixture_id=int(first_pred["fixture_id"]),
        kickoff=dataset.rows[first_fold.fold.test_indices[0]].kickoff,
        probabilities=MatchProbabilities(p_home_win=0.5, p_draw=0.3, p_away_win=0.2),
        artifact_sha256="xxx",
        predicted_at=datetime.now(UTC),
    )
    path1 = store.save(rec, overwrite=False)
    path2 = store.save(rec, overwrite=False)
    assert path1 == path2


def test_runner_empty_folds(tmp_path: Path) -> None:
    dataset = _make_dataset(5)
    trainer = EloBaselineTrainer()
    params = {"min_train_size": 10, "val_ratio": 0.2, "test_size": 2, "gap_days": 0, "mode": "expanding"}
    result = run_backtest(dataset, trainer, iterator_params=params, seed=5, base_path=tmp_path)
    assert len(result.folds) == 0
    # Summary should still exist
    summary_path = tmp_path / "data" / "models" / "runs" / result.run_id / "summary.json"
    assert summary_path.is_file()


def test_runner_reproducibility(tmp_path: Path) -> None:
    dataset = _make_dataset(20)
    trainer = PoissonTrainer()
    params = _iterator_params()
    r1 = run_backtest(dataset, trainer, iterator_params=params, seed=42, base_path=tmp_path / "a")
    r2 = run_backtest(dataset, trainer, iterator_params=params, seed=42, base_path=tmp_path / "b")
    assert r1.run_id == r2.run_id
    assert len(r1.folds) == len(r2.folds)
    for f1, f2 in zip(r1.folds, r2.folds, strict=False):
        assert f1.report.accuracy == f2.report.accuracy


def test_runner_three_models_same_folds(tmp_path: Path) -> None:
    dataset = _make_dataset(20)
    trainers = {
        ModelName.ELO_BASELINE: EloBaselineTrainer(),
        ModelName.POISSON: PoissonTrainer(),
        ModelName.GRADIENT_BOOSTING: GradientBoostingTrainer(),
    }
    results = run_backtest_all(dataset, trainers, iterator_params=_iterator_params(), seed=7, base_path=tmp_path)
    folds_lens = [len(r.folds) for r in results.values()]
    assert len(set(folds_lens)) == 1
    # Check same fold indices across models
    first = list(results.values())[0].folds[0].fold.train_indices
    for r in results.values():
        assert r.folds[0].fold.train_indices == first
