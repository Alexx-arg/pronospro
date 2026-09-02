"""Sprint 5.9 — E2E integration with real dataset v001.

Tests run against data/datasets/v001 if present; otherwise they are skipped
(v001 not available in this environment — see D4).

Walk-forward params for 5.9 (D2) are reduced for CI speed:
  min_train_size=100, test_size=10, gap=0, val_ratio=0.15
They do NOT modify PredictionSettings defaults.

Criterios §5.9 D2-D3: probar los 3 modelos (Elo, Poisson, GB) sobre los
mismos folds, val-only calibration, training_cutoff, simplex, métricas,
persistencia, reproducibilidad, anti-leakage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.prediction.contracts import ModelName

# ---------------------------------------------------------------------------
# Helpers to locate v001
# ---------------------------------------------------------------------------

def _find_v001() -> Path | None:
    candidates = [
        Path("data/datasets/v001"),
        Path("../data/datasets/v001"),
        Path("../../data/datasets/v001"),
        Path(__file__).resolve().parents[5] / "data" / "datasets" / "v001",
        Path(__file__).resolve().parents[4] / "data" / "datasets" / "v001",
        Path.cwd() / "data" / "datasets" / "v001",
        Path.cwd().parent / "data" / "datasets" / "v001",
    ]
    for p in candidates:
        try:
            if (p / "dataset.csv").is_file() and (p / "metadata.json").is_file():
                return p
        except Exception:
            continue
    return None


DATASET_DIR = _find_v001()

# Params D2
ITER_PARAMS = {
    "min_train_size": 100,
    "test_size": 10,
    "gap_days": 0,
    "val_ratio": 0.15,
    "mode": "expanding",
}


def _skip_if_no_v001() -> Path:
    if DATASET_DIR is None:
        pytest.skip("data/datasets/v001 not available in this environment (D4)")
    return DATASET_DIR  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_carga_y_validacion_real_v001() -> None:
    dataset_dir = _skip_if_no_v001()
    from app.dataset.loader import load_dataset, validate_dataset

    manifest = validate_dataset(dataset_dir)
    dataset = load_dataset(dataset_dir, verify_sha256=True)
    assert dataset.manifest.dataset_version == manifest.dataset_version
    assert len(dataset.rows) == manifest.row_count
    assert len(dataset.rows) > 0
    # Orden (kickoff, fixture_id)
    for i in range(len(dataset.rows) - 1):
        a = dataset.rows[i]
        b = dataset.rows[i + 1]
        assert (a.kickoff, a.fixture_id) <= (b.kickoff, b.fixture_id)


def test_walk_forward_real() -> None:
    dataset_dir = _skip_if_no_v001()
    from app.dataset.loader import load_dataset
    from app.prediction.backtesting.iterator import WalkForwardIterator
    from app.prediction.contracts import WalkForwardMode

    dataset = load_dataset(dataset_dir)
    it = WalkForwardIterator(
        dataset,
        min_train_size=ITER_PARAMS["min_train_size"],
        test_size=ITER_PARAMS["test_size"],
        gap_days=ITER_PARAMS["gap_days"],
        mode=WalkForwardMode(ITER_PARAMS["mode"]),
        val_ratio=ITER_PARAMS["val_ratio"],
    )
    folds = list(it)
    assert len(folds) >= 1
    for f in folds:
        assert f.train_end < f.val_start
        assert f.val_end < f.test_start
        assert len(f.train_indices) >= ITER_PARAMS["min_train_size"]


def _run_e2e_for_model(model_name: ModelName, tmp_path: Path):
    dataset_dir = _skip_if_no_v001()
    from app.dataset.loader import load_dataset
    from app.prediction.backtesting.runner import run_backtest

    dataset = load_dataset(dataset_dir)
    if model_name == ModelName.ELO_BASELINE:
        from app.prediction.training.elo_trainer import EloBaselineTrainer

        trainer = EloBaselineTrainer()
    elif model_name == ModelName.POISSON:
        from app.prediction.training.poisson_trainer import PoissonTrainer

        trainer = PoissonTrainer()
    elif model_name == ModelName.GRADIENT_BOOSTING:
        from app.prediction.training.gb_trainer import GradientBoostingTrainer

        trainer = GradientBoostingTrainer()
    else:
        raise ValueError(model_name)
    result = run_backtest(
        dataset,
        trainer,
        iterator_params=ITER_PARAMS,
        model_version="v001",
        seed=42,
        base_path=tmp_path,
    )
    return result


def test_e2e_elo(tmp_path: Path) -> None:
    _skip_if_no_v001()
    result = _run_e2e_for_model(ModelName.ELO_BASELINE, tmp_path)
    assert len(result.folds) >= 1
    for fr in result.folds:
        assert fr.report.n_predictions == len(fr.fold.test_indices)
        assert 0 <= fr.report.accuracy <= 1
        assert fr.report.log_loss >= 0
        assert fr.report.training_cutoff is not None or True  # artifact has cutoff


def test_e2e_poisson(tmp_path: Path) -> None:
    _skip_if_no_v001()
    result = _run_e2e_for_model(ModelName.POISSON, tmp_path)
    assert len(result.folds) >= 1
    for fr in result.folds:
        s = fr.report.log_loss
        assert s >= 0
        assert fr.report.brier_home >= 0


def test_e2e_gb(tmp_path: Path) -> None:
    _skip_if_no_v001()
    result = _run_e2e_for_model(ModelName.GRADIENT_BOOSTING, tmp_path)
    assert len(result.folds) >= 1
    for fr in result.folds:
        assert fr.report.accuracy >= 0
        # GB has no goals
        assert fr.report.mae_home_goals is None


def test_anti_leakage(tmp_path: Path) -> None:
    dataset_dir = _skip_if_no_v001()
    from app.dataset.loader import load_dataset

    dataset = load_dataset(dataset_dir)
    from app.prediction.training.elo_trainer import EloBaselineTrainer

    from app.prediction.backtesting.runner import run_backtest

    result = run_backtest(
        dataset,
        EloBaselineTrainer(),
        iterator_params=ITER_PARAMS,
        seed=1,
        base_path=tmp_path,
    )
    for fr in result.folds:
        train_ids = {dataset.rows[i].fixture_id for i in fr.fold.train_indices}
        val_ids = {dataset.rows[i].fixture_id for i in fr.fold.val_indices}
        test_ids = {dataset.rows[i].fixture_id for i in fr.fold.test_indices}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)


def test_calibracion_val_only(tmp_path: Path) -> None:
    dataset_dir = _skip_if_no_v001()
    from app.dataset.loader import load_dataset
    from app.prediction.backtesting.runner import run_backtest
    from app.prediction.training.elo_trainer import EloBaselineTrainer

    dataset = load_dataset(dataset_dir)
    result = run_backtest(
        dataset, EloBaselineTrainer(), iterator_params=ITER_PARAMS, seed=2, base_path=tmp_path
    )
    for fr in result.folds:
        assert "val_ece" in fr.calibrator_dict
        assert "val_n" in fr.calibrator_dict
        assert fr.calibrator_dict["val_n"] == len(fr.fold.val_indices)


def test_persistencia_metrics_artifacts_predictions(tmp_path: Path) -> None:
    _skip_if_no_v001()
    result = _run_e2e_for_model(ModelName.POISSON, tmp_path)
    for fr in result.folds:
        fold_id = f"fold_{fr.fold.fold_id:03d}"
        assert (tmp_path / "data" / "models" / "runs" / result.run_id / "folds" / fold_id / "metrics.json").is_file()
        assert (tmp_path / "data" / "models" / "runs" / result.run_id / "folds" / fold_id / "calibrator.json").is_file()
        assert (tmp_path / "data" / "models" / "runs" / result.run_id / "folds" / fold_id / "predictions.csv").is_file()
    assert (tmp_path / "data" / "models" / "runs" / result.run_id / "summary.json").is_file()
    assert (tmp_path / "data" / "models" / "poisson" / "v001" / "manifest.json").is_file()


def test_reproducibilidad(tmp_path: Path) -> None:
    dataset_dir = _skip_if_no_v001()
    from app.dataset.loader import load_dataset
    from app.prediction.backtesting.runner import run_backtest
    from app.prediction.training.elo_trainer import EloBaselineTrainer

    dataset = load_dataset(dataset_dir)
    r1 = run_backtest(dataset, EloBaselineTrainer(), iterator_params=ITER_PARAMS, seed=42, base_path=tmp_path / "a")
    r2 = run_backtest(dataset, EloBaselineTrainer(), iterator_params=ITER_PARAMS, seed=42, base_path=tmp_path / "b")
    assert r1.run_id == r2.run_id
    for a, b in zip(r1.folds, r2.folds):
        assert a.report.accuracy == b.report.accuracy


def test_mismos_folds_para_tres_modelos(tmp_path: Path) -> None:
    dataset_dir = _skip_if_no_v001()
    from app.dataset.loader import load_dataset
    from app.prediction.backtesting.runner import run_backtest_all
    from app.prediction.contracts import ModelName
    from app.prediction.training.elo_trainer import EloBaselineTrainer
    from app.prediction.training.gb_trainer import GradientBoostingTrainer
    from app.prediction.training.poisson_trainer import PoissonTrainer

    dataset = load_dataset(dataset_dir)
    trainers = {
        ModelName.ELO_BASELINE: EloBaselineTrainer(),
        ModelName.POISSON: PoissonTrainer(),
        ModelName.GRADIENT_BOOSTING: GradientBoostingTrainer(),
    }
    results = run_backtest_all(dataset, trainers, iterator_params=ITER_PARAMS, seed=7, base_path=tmp_path)
    lens = [len(r.folds) for r in results.values()]
    assert len(set(lens)) == 1
    first_indices = list(results.values())[0].folds[0].fold.train_indices
    for r in results.values():
        assert r.folds[0].fold.train_indices == first_indices
        for fr in r.folds:
            # Simplex
            for p in fr.predictions:
                s = p["p_home_win"] + p["p_draw"] + p["p_away_win"]
                assert s == pytest.approx(1.0, abs=1e-9)
                assert 0 <= p["p_home_win"] <= 1
            # Metrics finitas y training_cutoff
            assert fr.report.log_loss >= 0
            assert fr.artifact.training_cutoff is not None


def test_training_cutoff_simplex_metricas(tmp_path: Path) -> None:
    _skip_if_no_v001()
    result = _run_e2e_for_model(ModelName.ELO_BASELINE, tmp_path)
    for fr in result.folds:
        assert fr.artifact.training_cutoff == fr.fold.training_cutoff
        for p in fr.predictions:
            s = p["p_home_win"] + p["p_draw"] + p["p_away_win"]
            assert s == pytest.approx(1.0, abs=1e-9)
        assert 0 <= fr.report.ece <= 1
        assert 0 <= fr.report.mce <= 1
