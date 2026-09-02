"""Tests for goal metrics (Sprint 5.3)."""

from __future__ import annotations

import math

import pytest
from app.prediction.metrics.goals import (
    mae_home_goals,
    poisson_loglik,
    rmse_home_goals,
    rmse_total_goals,
)


def test_mae_home_perfect() -> None:
    assert mae_home_goals([2, 1], [2, 1]) == pytest.approx(0.0)


def test_mae_home_simple() -> None:
    assert mae_home_goals([2, 0], [1, 1]) == pytest.approx(1.0)


def test_rmse_home_perfect() -> None:
    assert rmse_home_goals([1, 2], [1, 2]) == pytest.approx(0.0)


def test_rmse_total() -> None:
    # true total 3, pred total 2 → rmse 1
    assert rmse_total_goals([1], [2], [1], [1]) == pytest.approx(1.0)


def test_goals_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mae_home_goals([1], [-1])
    with pytest.raises(ValueError):
        mae_home_goals([-1], [1])


def test_goals_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        mae_home_goals([1], [float("nan")])


def test_goals_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mae_home_goals([], [])


def test_goals_mismatch_length() -> None:
    with pytest.raises(ValueError, match="length"):
        mae_home_goals([1, 2], [1])


def test_poisson_rate_form_perfect() -> None:
    # When lambda equals observed k, likelihood relatively high
    ll = poisson_loglik([1], [2], [1.0], [2.0])
    # Manual: Poisson(1;1)=e^-1*1/1!=0.3679, Poisson(2;2)=e^-2*4/2=0.2706, log sum ≈ -2.3
    assert ll == pytest.approx(math.log(0.3679 * 0.2706), rel=0.02)


def test_poisson_rate_form_zero_lambda_rejects() -> None:
    with pytest.raises(ValueError, match="> 0"):
        poisson_loglik([1], [0], [0.0], [1.0])


def test_poisson_dict_form() -> None:
    p_home = [{1: 0.5, 0: 0.3, 2: 0.2}]
    p_away = [{0: 0.6, 1: 0.4}]
    ll = poisson_loglik([1], [0], p_home, p_away)
    assert ll == pytest.approx(math.log(0.5 * 0.6), rel=1e-5)


def test_poisson_dict_missing_k_uses_eps() -> None:
    p_home = [{0: 1.0}]
    p_away = [{0: 1.0}]
    # true k=5 not in dict → eps → log(eps)
    ll = poisson_loglik([5], [5], p_home, p_away)
    assert ll < -20


def test_poisson_mixed_form_raises() -> None:
    with pytest.raises(ValueError, match="same form"):
        poisson_loglik([1], [1], [{0: 0.5}], [1.0])


def test_poisson_rejects_negative_prob_in_dict() -> None:
    with pytest.raises(ValueError, match="negative"):
        poisson_loglik([0], [0], [{0: -0.1}], [{0: 0.5}])


def test_rmse_total_mismatch_shape() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        rmse_total_goals([1, 2], [1, 2], [1], [1, 2])
