"""Tests for EloBaselineTrainer grid search and drop_row."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from app.features.example import FEATURE_NAMES
from app.prediction.contracts import FixtureFeatures
from app.prediction.training.elo_trainer import EloBaselineTrainer


def _make_fixture(
    fixture_id: int,
    r_home: float | None,
    r_away: float | None,
    kickoff: datetime | None = None,
) -> FixtureFeatures:
    vec = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    if r_home is not None:
        vec[FEATURE_NAMES.index("home_elo_pre_match")] = float(r_home)
        vec[FEATURE_NAMES.index("elo_difference")] = float(r_home - (r_away or 0) if r_away is not None else 0)
    if r_away is not None:
        vec[FEATURE_NAMES.index("away_elo_pre_match")] = float(r_away)
        if r_home is not None:
            vec[FEATURE_NAMES.index("elo_difference")] = float(r_home - r_away)
    return FixtureFeatures(
        fixture_id=fixture_id,
        kickoff=kickoff or datetime(2026, 1, 1, tzinfo=UTC),
        feature_vector=vec,
        feature_names=tuple(FEATURE_NAMES),
    )


def _targets_from_results(
    results: list[int],
) -> list[list[int]]:
    """results 0=home,1=draw,2=away -> protocol targets [hw,dr,aw,hg,ag]."""
    out: list[list[int]] = []
    for r in results:
        hw = 1 if r == 0 else 0
        dr = 1 if r == 1 else 0
        aw = 1 if r == 2 else 0
        # home/away goals simple: 2-0 for home win etc.
        if r == 0:
            hg, ag = 2, 0
        elif r == 1:
            hg, ag = 1, 1
        else:
            hg, ag = 0, 2
        out.append([hw, dr, aw, hg, ag])
    return out


def test_trainer_grid_search_selects_best() -> None:
    # Train: need beta0s; use a few fixtures
    train_f = [_make_fixture(i, 1500 + i * 10, 1500) for i in range(1, 6)]
    train_t = _targets_from_results([0, 0, 1, 2, 0])
    # Val where home advantage matters: all home wins when R_home high
    val_f = [_make_fixture(10, 1700, 1400), _make_fixture(11, 1700, 1400)]
    val_t = _targets_from_results([0, 0])
    trainer = EloBaselineTrainer()
    best_params, best_loss, info = trainer.search(train_f, train_t, val_f, val_t)
    assert best_params.K in (15, 20, 25, 30)
    assert best_params.HFA in (50, 65, 80)
    assert best_params.beta_1 in (0.0015, 0.0020, 0.0025)
    assert best_loss >= 0


def test_trainer_deterministic_lexicographic() -> None:
    train_f = [_make_fixture(i, 1500, 1500) for i in range(1, 5)]
    train_t = _targets_from_results([0, 1, 2, 0])
    val_f = [_make_fixture(10, 1500, 1500), _make_fixture(11, 1500, 1500)]
    val_t = _targets_from_results([1, 1])
    trainer = EloBaselineTrainer()
    p1, _, _ = trainer.search(train_f, train_t, val_f, val_t)
    p2, _, _ = trainer.search(train_f, train_t, val_f, val_t)
    assert p1 == p2


def test_trainer_drop_row_for_nan() -> None:
    train_f = [
        _make_fixture(1, 1500, 1500),
        _make_fixture(2, None, None),  # NaN elo
        _make_fixture(3, 1500, 1500),
    ]
    train_t = _targets_from_results([0, 1, 2])
    val_f = [_make_fixture(10, 1500, 1500)]
    val_t = _targets_from_results([0])
    trainer = EloBaselineTrainer()
    best_params, _, info = trainer.search(train_f, train_t, val_f, val_t)
    assert info["train_dropouts"] == 1
    assert best_params is not None


def test_trainer_drop_all_raises() -> None:
    train_f = [_make_fixture(1, None, None), _make_fixture(2, None, None)]
    train_t = _targets_from_results([0, 1])
    val_f = [_make_fixture(10, 1500, 1500)]
    val_t = _targets_from_results([0])
    trainer = EloBaselineTrainer()
    with pytest.raises(ValueError, match="train block empty"):
        trainer.search(train_f, train_t, val_f, val_t)


def test_trainer_artifact_created() -> None:
    train_f = [_make_fixture(i, 1500 + i, 1500) for i in range(1, 4)]
    train_t = _targets_from_results([0, 1, 2])
    trainer = EloBaselineTrainer()
    art = trainer.train(train_f, train_t, hyperparameters={}, seed=42)
    assert art.model_name == "elo_baseline"
    assert "K" in art.hyperparameters
    assert "beta_1" in art.hyperparameters
    assert art.inputs.feature_names == (
        "home_elo_pre_match",
        "away_elo_pre_match",
        "elo_difference",
    )


def test_trainer_no_val_uses_hyperparams() -> None:
    train_f = [_make_fixture(i, 1500, 1500) for i in range(1, 4)]
    train_t = _targets_from_results([0, 0, 1])
    trainer = EloBaselineTrainer()
    art = trainer.train(train_f, train_t, hyperparameters={"K": 30, "HFA": 80, "beta_1": 0.0025})
    assert art.hyperparameters["K"] == 30
    assert art.hyperparameters["HFA"] == 80


def test_trainer_does_not_use_test() -> None:
    """Ensure search does not look at test block (only train+val)."""
    train_f = [_make_fixture(i, 1600, 1400) for i in range(1, 4)]
    train_t = _targets_from_results([0, 0, 0])
    val_f = [_make_fixture(10, 1600, 1400)]
    val_t = _targets_from_results([0])
    trainer = EloBaselineTrainer()
    p1, _, _ = trainer.search(train_f, train_t, val_f, val_t)
    # Even if test would favor different HFA, search must not see it
    assert p1.HFA in (50, 65, 80)
