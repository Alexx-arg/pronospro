"""Dirichlet calibration tests."""

from __future__ import annotations

import numpy as np
import pytest
from app.prediction.calibration.dirichlet import DirichletCalibrator
from app.prediction.contracts import MatchProbabilities


def _mp(arr: np.ndarray) -> list[MatchProbabilities]:
    return [MatchProbabilities(float(r[0]), float(r[1]), float(r[2])) for r in arr]


def _arr(probas: list[MatchProbabilities]) -> np.ndarray:
    return np.array([[p.p_home_win, p.p_draw, p.p_away_win] for p in probas])


def test_dirichlet_output_simplex() -> None:
    raw = np.array([[0.5, 0.3, 0.2]] * 10)
    targets = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    cal = DirichletCalibrator()
    cal.fit(_mp(raw), targets)
    out = _arr(cal.transform(_mp(raw)))
    assert np.all(out >= 0) and np.all(out <= 1)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.isfinite(out))


def test_dirichlet_uniform_input() -> None:
    raw = np.array([[1 / 3, 1 / 3, 1 / 3]] * 6)
    targets = np.array([0, 1, 2, 0, 1, 2])
    cal = DirichletCalibrator()
    cal.fit(_mp(raw), targets)
    out = _arr(cal.transform(_mp(raw)))
    assert np.all(np.isfinite(out))


def test_dirichlet_extreme_probs() -> None:
    raw = np.array([[0.99, 0.005, 0.005], [0.005, 0.99, 0.005]])
    raw = raw / raw.sum(axis=1, keepdims=True)
    cal = DirichletCalibrator()
    cal.fit(_mp(raw), np.array([0, 1]))
    out = _arr(cal.transform(_mp(raw)))
    assert np.all(np.isfinite(out))


def test_dirichlet_reproducible() -> None:
    raw = np.array([[0.5, 0.3, 0.2]] * 8)
    targets = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    c1 = DirichletCalibrator()
    c2 = DirichletCalibrator()
    c1.fit(_mp(raw), targets)
    c2.fit(_mp(raw), targets)
    assert np.allclose(c1.W, c2.W)
    assert np.allclose(c1.b, c2.b)


def test_dirichlet_no_mutation() -> None:
    raw = np.array([[0.6, 0.2, 0.2]])
    copy = np.array(raw, copy=True)
    cal = DirichletCalibrator()
    cal.fit(_mp(raw), np.array([0]))
    _ = cal.transform(_mp(raw))
    assert np.array_equal(raw, copy)


def test_dirichlet_regularization() -> None:
    # With high lambda, W should stay close to I
    raw = np.array([[0.5, 0.3, 0.2]] * 6)
    targets = np.array([0, 1, 2, 0, 1, 2])
    cal = DirichletCalibrator(lambda_l2=10.0)
    cal.fit(_mp(raw), targets)
    assert np.allclose(cal.W, np.eye(3), atol=0.5)


def test_dirichlet_serialization() -> None:
    raw = np.array([[0.5, 0.3, 0.2]] * 4)
    cal = DirichletCalibrator()
    cal.fit(_mp(raw), np.array([0, 1, 2, 0]))
    d = cal.to_dict()
    assert d["kind"] == "dirichlet"
    cal2 = DirichletCalibrator.from_dict(d)
    assert np.allclose(cal2.W, cal.W)
    assert np.allclose(cal2.b, cal.b)


def test_dirichlet_n_zero_raises() -> None:
    cal = DirichletCalibrator()
    with pytest.raises(ValueError):
        cal.fit([], [])
