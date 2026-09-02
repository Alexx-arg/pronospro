"""Tests for Elo baseline + shared poisson conversor (Sprint 5.5)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from app.features.example import FEATURE_NAMES
from app.prediction.contracts import FixtureFeatures
from app.prediction.models._poisson import poisson_to_1x2
from app.prediction.models.elo_baseline import EloBaselineModel, EloBaselineParams


def _elo_features(
    r_home: float,
    r_away: float,
    fixture_id: int = 1,
) -> FixtureFeatures:
    """Create FixtureFeatures with only Elo features set, rest NaN."""
    vec = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    idx_h = FEATURE_NAMES.index("home_elo_pre_match")
    idx_a = FEATURE_NAMES.index("away_elo_pre_match")
    idx_d = FEATURE_NAMES.index("elo_difference")
    vec[idx_h] = float(r_home)
    vec[idx_a] = float(r_away)
    vec[idx_d] = float(r_home - r_away)
    from datetime import UTC, datetime

    return FixtureFeatures(
        fixture_id=fixture_id,
        kickoff=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        feature_vector=vec,
        feature_names=tuple(FEATURE_NAMES),
    )


# ------------------------------------------------------------------
# Poisson shared
# ------------------------------------------------------------------

def test_poisson_to_1x2_sums_to_one() -> None:
    for lh in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for la in [0.1, 0.5, 1.0, 2.0, 5.0]:
            ph, pd, pa, _, _ = poisson_to_1x2(lh, la, max_goals=10)
            assert ph + pd + pa == pytest.approx(1.0, abs=1e-9)
            assert 0 <= ph <= 1 and 0 <= pd <= 1 and 0 <= pa <= 1


def test_poisson_to_1x2_equal_lambdas_symmetry() -> None:
    ph, pd, pa, _, _ = poisson_to_1x2(1.5, 1.5)
    assert ph == pytest.approx(pa, abs=1e-9)
    assert pd > 0


def test_poisson_to_1x2_extreme_lambda() -> None:
    ph, pd, pa, _, _ = poisson_to_1x2(30, 0.1)
    assert ph > 0.9
    assert pa < 0.05
    assert np.isfinite(ph) and np.isfinite(pd) and np.isfinite(pa)


def test_poisson_to_1x2_invalid_raises() -> None:
    with pytest.raises(ValueError):
        poisson_to_1x2(-1, 1)
    with pytest.raises(ValueError):
        poisson_to_1x2(float("nan"), 1)


def test_poisson_to_1x2_overflow_capped() -> None:
    # log lambda >30 should be capped, not raise overflow
    ph, pd, pa, _, _ = poisson_to_1x2(math.exp(31), 1.0)
    assert np.isfinite(ph)


# ------------------------------------------------------------------
# Elo baseline
# ------------------------------------------------------------------

def test_elo_simplex() -> None:
    params = EloBaselineParams(K=20, HFA=65, beta_0_home=0.2, beta_0_away=0.1, beta_1=0.002)
    model = EloBaselineModel(params)
    for dh in [-200, 0, 200]:
        f = _elo_features(1500 + dh, 1500)
        mp = model.predict(f)
        assert 0 <= mp.p_home_win <= 1
        assert 0 <= mp.p_draw <= 1
        assert 0 <= mp.p_away_win <= 1
        assert mp.p_home_win + mp.p_draw + mp.p_away_win == pytest.approx(1.0, abs=1e-9)
        assert mp.p_home_goals is not None and mp.p_away_goals is not None


def test_elo_symmetry_no_hfa() -> None:
    params = EloBaselineParams(K=20, HFA=0, beta_0_home=0.2, beta_0_away=0.2, beta_1=0.002)
    model = EloBaselineModel(params)
    f = _elo_features(1500, 1500)
    mp = model.predict(f)
    assert mp.p_home_win == pytest.approx(mp.p_away_win, abs=1e-9)
    assert mp.p_draw > 0


def test_elo_hfa_advantage() -> None:
    params_no = EloBaselineParams(K=20, HFA=0, beta_0_home=0.2, beta_0_away=0.2, beta_1=0.002)
    params_hfa = EloBaselineParams(K=20, HFA=65, beta_0_home=0.2, beta_0_away=0.2, beta_1=0.002)
    f = _elo_features(1500, 1500)
    mp0 = EloBaselineModel(params_no).predict(f)
    mph = EloBaselineModel(params_hfa).predict(f)
    assert mph.p_home_win > mp0.p_home_win


def test_elo_extreme_rating() -> None:
    params = EloBaselineParams(K=20, HFA=65, beta_0_home=0.2, beta_0_away=0.1, beta_1=0.002)
    model = EloBaselineModel(params)
    f = _elo_features(2000, 1200)
    mp = model.predict(f)
    assert mp.p_home_win > 0.75
    assert mp.p_away_win < 0.15
    assert mp.p_home_win > mp.p_away_win


def test_elo_consumes_only_elo_features() -> None:
    """Changing non-Elo features must not affect Elo prediction."""
    params = EloBaselineParams(K=20, HFA=65, beta_0_home=0.2, beta_0_away=0.1, beta_1=0.002)
    model = EloBaselineModel(params)
    f1 = _elo_features(1600, 1500)
    # Mutate a non-Elo feature
    vec2 = np.array(f1.feature_vector, copy=True)
    # Change first feature (home_wins_last_3) which is not used
    vec2[0] = 999.0
    from app.prediction.contracts import FixtureFeatures

    f2 = FixtureFeatures(
        fixture_id=f1.fixture_id,
        kickoff=f1.kickoff,
        feature_vector=vec2,
        feature_names=f1.feature_names,
    )
    mp1 = model.predict(f1)
    mp2 = model.predict(f2)
    assert mp1.p_home_win == pytest.approx(mp2.p_home_win)
    assert mp1.p_draw == pytest.approx(mp2.p_draw)


def test_elo_nan_raises() -> None:
    params = EloBaselineParams(K=20, HFA=65, beta_0_home=0.2, beta_0_away=0.1, beta_1=0.002)
    model = EloBaselineModel(params)
    vec = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    # leave elo as nan (default)
    from datetime import UTC, datetime

    from app.prediction.contracts import FixtureFeatures

    f = FixtureFeatures(
        fixture_id=1,
        kickoff=datetime(2026, 1, 1, tzinfo=UTC),
        feature_vector=vec,
        feature_names=tuple(FEATURE_NAMES),
    )
    with pytest.raises(ValueError, match="NaN"):
        model.predict(f)


def test_elo_deterministic() -> None:
    params = EloBaselineParams(K=20, HFA=65, beta_0_home=0.2, beta_0_away=0.1, beta_1=0.002)
    model = EloBaselineModel(params)
    f = _elo_features(1550, 1450)
    a = model.predict(f)
    b = model.predict(f)
    assert a == b
