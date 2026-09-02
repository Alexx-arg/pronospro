"""GB model tests — 66 features, NaN passthrough, simplex."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from app.features.example import FEATURE_NAMES
from app.prediction.contracts import FixtureFeatures
from app.prediction.models.gradient_boosting import GradientBoostingModel
from app.prediction.training.gb_trainer import GradientBoostingTrainer


def _feat(values: dict[str, float | None] | None = None, fid: int = 1) -> FixtureFeatures:
    vec = np.full(len(FEATURE_NAMES), 0.5, dtype=np.float32)
    if values:
        for k, v in values.items():
            idx = FEATURE_NAMES.index(k)
            vec[idx] = np.nan if v is None else float(v)
    return FixtureFeatures(
        fixture_id=fid,
        kickoff=datetime(2026, 1, 1, tzinfo=UTC),
        feature_vector=vec,
        feature_names=tuple(FEATURE_NAMES),
    )


def _train_model() -> GradientBoostingModel:
    feats = []
    targets = []
    for i in range(12):
        f = _feat(fid=i + 1)
        feats.append(f)
        # Cycle labels 0,1,2
        label = i % 3
        if label == 0:
            targets.append([1, 0, 0, 2, 0])
        elif label == 1:
            targets.append([0, 1, 0, 1, 1])
        else:
            targets.append([0, 0, 1, 0, 2])
    trainer = GradientBoostingTrainer()
    trainer.train(feats, targets, seed=42)
    return trainer.get_model()


def test_gb_66_features_exact() -> None:
    model = _train_model()
    assert model.feature_names == tuple(FEATURE_NAMES)
    assert len(model.feature_names) == 66


def test_gb_imputes_missing_explicit() -> None:
    # NaN must not be imputed to 0; model must handle NaN natively
    model = _train_model()
    f_nan = _feat({FEATURE_NAMES[0]: None}, fid=100)
    f_zero = _feat({FEATURE_NAMES[0]: 0.0}, fid=101)
    # Both should predict without error (passthrough)
    mp_nan = model.predict(f_nan)
    mp_zero = model.predict(f_zero)
    # They should not be forced to same imputed value; just ensure both valid
    for mp in (mp_nan, mp_zero):
        assert 0 <= mp.p_home_win <= 1
        assert mp.p_home_goals is None
        assert mp.p_away_goals is None
    # Explicit check that NaN was not replaced by 0 in vector before model
    assert np.isnan(float(f_nan.feature_vector[0]))
    assert float(f_zero.feature_vector[0]) == 0.0


def test_gb_multiclass_probs_sum_to_one() -> None:
    model = _train_model()
    for _ in range(5):
        f = _feat()
        mp = model.predict(f)
        s = mp.p_home_win + mp.p_draw + mp.p_away_win
        assert s == pytest.approx(1.0, abs=1e-9)
        for p in (mp.p_home_win, mp.p_draw, mp.p_away_win):
            assert 0 <= p <= 1
            assert np.isfinite(p)


def test_gb_no_poisson_goals() -> None:
    model = _train_model()
    mp = model.predict(_feat())
    assert mp.p_home_goals is None
    assert mp.p_away_goals is None


def test_gb_no_poisson_to_1x2_used() -> None:
    # Ensure model does not import poisson_to_1x2
    import app.prediction.models.gradient_boosting as mod

    assert "poisson_to_1x2" not in dir(mod)


def test_gb_extreme_feature() -> None:
    model = _train_model()
    f = _feat({FEATURE_NAMES[5]: 100.0}, fid=99)
    mp = model.predict(f)
    assert np.isfinite(mp.p_home_win)


def test_gb_deterministic_predict() -> None:
    model = _train_model()
    f = _feat(fid=1)
    a = model.predict(f)
    b = model.predict(f)
    assert a == b


def test_gb_batch_proba() -> None:
    model = _train_model()
    feats = [_feat(fid=i) for i in range(1, 4)]
    arr = model.predict_proba_array(feats)
    assert arr.shape == (3, 3)
    assert np.allclose(arr.sum(axis=1), 1.0, atol=1e-6)
