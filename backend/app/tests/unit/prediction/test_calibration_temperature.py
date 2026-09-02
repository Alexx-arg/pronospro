"""Temperature scaling tests."""

from __future__ import annotations

import numpy as np
import pytest
from app.prediction.calibration.temperature import TemperatureCalibrator
from app.prediction.contracts import MatchProbabilities


def _mp(arr: np.ndarray) -> list[MatchProbabilities]:
    return [MatchProbabilities(float(r[0]), float(r[1]), float(r[2])) for r in arr]


def _arr(probas: list[MatchProbabilities]) -> np.ndarray:
    return np.array([[p.p_home_win, p.p_draw, p.p_away_win] for p in probas])


def test_temperature_T1_is_identity() -> None:
    cal = TemperatureCalibrator()
    raw = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]])
    cal.temperature = 1.0
    out = cal.transform(_mp(raw))
    assert np.allclose(_arr(out), raw, atol=1e-9)


def test_temperature_gt1_smooths() -> None:
    cal = TemperatureCalibrator()
    raw = np.array([[0.9, 0.05, 0.05]])
    cal.temperature = 2.0
    out = _arr(cal.transform(_mp(raw)))
    # Smoothed → max prob decreases
    assert out[0, 0] < 0.9
    assert out[0, 1] > 0.05


def test_temperature_lt1_sharpens() -> None:
    cal = TemperatureCalibrator()
    raw = np.array([[0.5, 0.3, 0.2]])
    cal.temperature = 0.5
    out = _arr(cal.transform(_mp(raw)))
    assert out[0, 0] > 0.5


def test_temperature_output_simplex() -> None:
    cal = TemperatureCalibrator()
    raw = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    cal.fit(_mp(raw), np.array([0, 2]))
    out = _arr(cal.transform(_mp(raw)))
    assert np.all(out >= 0) and np.all(out <= 1)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.isfinite(out))


def test_temperature_no_mutation() -> None:
    cal = TemperatureCalibrator()
    raw = np.array([[0.6, 0.2, 0.2]])
    copy = np.array(raw, copy=True)
    cal.fit(_mp(raw), np.array([0]))
    _ = cal.transform(_mp(raw))
    assert np.array_equal(raw, copy)


def test_temperature_bounds() -> None:
    raw = np.array([[0.9, 0.05, 0.05]] * 10)
    targets = np.array([0] * 10)
    cal = TemperatureCalibrator()
    cal.fit(_mp(raw), targets)
    assert 0.1 <= cal.temperature <= 10.0


def test_temperature_reproducible() -> None:
    raw = np.array([[0.6, 0.2, 0.2]] * 20)
    targets = np.array([0, 1, 2] * 6 + [0, 1])
    c1 = TemperatureCalibrator()
    c2 = TemperatureCalibrator()
    c1.fit(_mp(raw), targets)
    c2.fit(_mp(raw), targets)
    assert c1.temperature == pytest.approx(c2.temperature)


def test_temperature_handles_small_prob() -> None:
    raw = np.array([[1e-12, 0.5, 0.5 - 1e-12]])
    raw = raw / raw.sum(axis=1, keepdims=True)
    cal = TemperatureCalibrator()
    cal.fit(_mp(np.array([[0.33, 0.33, 0.34]] * 5)), np.array([0, 1, 2, 0, 1]))
    out = _arr(cal.transform(_mp(raw)))
    assert np.all(np.isfinite(out))


def test_temperature_n_zero_raises() -> None:
    cal = TemperatureCalibrator()
    with pytest.raises(ValueError):
        cal.fit([], [])


def test_temperature_serialization() -> None:
    cal = TemperatureCalibrator()
    raw = np.array([[0.5, 0.3, 0.2]] * 5)
    cal.fit(_mp(raw), np.array([0, 1, 2, 0, 1]))
    d = cal.to_dict()
    assert d["kind"] == "temperature"
    cal2 = TemperatureCalibrator.from_dict(d)
    assert cal2.temperature == pytest.approx(cal.temperature)
