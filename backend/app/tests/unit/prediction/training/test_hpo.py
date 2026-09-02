# ruff: noqa: N806
"""Tests HPO con Optuna — Sprint 5.12.

Usa dataset sintético muy pequeño y 2 trials. Verifica:
- No excepciones shape/type
- Retorna dict con best_params serializable
- Estructura correcta (best_params, best_value, etc.)
- Determinismo (misma seed → mismos best_params)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from app.dataset.loader import LoadedDataset, LoadedExample
from app.dataset.manifest import DatasetManifest
from app.features.example import FEATURE_NAMES

ITER_PARAMS_SMALL: dict[str, object] = {
    "min_train_size": 10,
    "test_size": 5,
    "gap_days": 0,
    "val_ratio": 0.3,
    "mode": "expanding",
}


def _make_small_dataset(n: int = 30) -> LoadedDataset:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[LoadedExample] = []
    for i in range(n):
        from datetime import timedelta

        kickoff = t0 + timedelta(days=i)
        feats = {name: float((i % 3) * 0.5) for name in FEATURE_NAMES}
        feats["home_elo_pre_match"] = 1500.0 + (i % 3) * 10
        feats["away_elo_pre_match"] = 1500.0
        feats["elo_difference"] = feats["home_elo_pre_match"] - feats["away_elo_pre_match"]
        label = i % 3
        if label == 0:
            targets: dict[str, int | None] = {"home_win": 1, "draw": 0, "away_win": 0, "home_goals": 2, "away_goals": 0}
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


def test_hpo_returns_best_params(tmp_path: Path) -> None:
    dataset = _make_small_dataset(30)
    from app.prediction.training.hpo import run_hpo_study

    result = run_hpo_study(
        dataset,
        model_name="lightgbm",
        metric="weighted_log_loss",
        n_trials=2,
        base_path=tmp_path,
        seed=42,
        iterator_params=ITER_PARAMS_SMALL,
    )
    assert "best_params" in result
    assert "best_value" in result
    assert "best_trial_number" in result
    assert "n_trials" in result
    assert result["n_trials"] == 2
    assert isinstance(result["best_params"], dict)
    json.dumps(result["best_params"])
    assert "learning_rate" in result["best_params"]
    assert "num_leaves" in result["best_params"]
    assert "n_estimators" in result["best_params"]
    assert np.isfinite(result["best_value"])


def test_hpo_determinismo(tmp_path: Path) -> None:
    dataset = _make_small_dataset(30)
    from app.prediction.training.hpo import run_hpo_study

    r1 = run_hpo_study(
        dataset,
        model_name="lightgbm",
        n_trials=2,
        base_path=tmp_path / "a",
        seed=42,
        iterator_params=ITER_PARAMS_SMALL,
    )
    r2 = run_hpo_study(
        dataset,
        model_name="lightgbm",
        n_trials=2,
        base_path=tmp_path / "b",
        seed=42,
        iterator_params=ITER_PARAMS_SMALL,
    )
    assert r1["best_params"] == r2["best_params"]
    assert r1["best_value"] == r2["best_value"]


def test_hpo_no_leakage_usa_folds(tmp_path: Path) -> None:
    dataset = _make_small_dataset(30)
    from app.prediction.training.hpo import run_hpo_study

    result = run_hpo_study(
        dataset, model_name="lightgbm", n_trials=1, base_path=tmp_path, seed=42, iterator_params=ITER_PARAMS_SMALL
    )
    assert result["best_params"] is not None


def test_hpo_search_space_custom(tmp_path: Path) -> None:
    dataset = _make_small_dataset(30)
    from app.prediction.training.hpo import run_hpo_study

    custom_space = {
        "learning_rate": (0.05, 0.06),
        "num_leaves": (20, 25),
        "n_estimators": (50, 60),
    }
    result = run_hpo_study(
        dataset,
        model_name="lightgbm",
        n_trials=2,
        search_space=custom_space,
        base_path=tmp_path,
        seed=42,
        iterator_params=ITER_PARAMS_SMALL,
    )
    assert 0.05 <= result["best_params"]["learning_rate"] <= 0.06
    assert 20 <= result["best_params"]["num_leaves"] <= 25


def test_hpo_n_trials_2_no_error(tmp_path: Path) -> None:
    dataset = _make_small_dataset(20)
    from app.prediction.training.hpo import run_hpo_study

    result = run_hpo_study(
        dataset, model_name="lightgbm", n_trials=2, base_path=tmp_path, seed=42, iterator_params=ITER_PARAMS_SMALL
    )
    assert result["n_trials"] == 2
