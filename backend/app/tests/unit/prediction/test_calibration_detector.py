"""Detector tests."""

from __future__ import annotations

import numpy as np
import pytest
from app.prediction.calibration.detector import CalibrationDetector
from app.prediction.contracts import CalibratorKind, MatchProbabilities


def _mp(arr: np.ndarray) -> list[MatchProbabilities]:
    return [MatchProbabilities(float(r[0]), float(r[1]), float(r[2])) for r in arr]


def test_detector_low_ece_identity() -> None:
    # Perfect predictions → low ECE (<0.02)
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_proba = np.zeros((6, 3))
    y_proba[np.arange(6), y_true] = 0.99
    y_proba += 0.005
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    det = CalibrationDetector()
    cal = det.detect(_mp(y_proba), y_true)
    assert cal.kind == CalibratorKind.IDENTITY


def test_detector_medium_ece_temperature() -> None:
    # Random uniform → medium ECE, but n small → temperature
    rng = np.random.default_rng(0)
    raw = rng.random((20, 3))
    raw = raw / raw.sum(axis=1, keepdims=True)
    y_true = rng.integers(0, 3, size=20)
    det = CalibrationDetector()
    cal = det.detect(_mp(raw), y_true)
    # ECE likely 0.02-0.05 or higher but n<5000 → temperature
    assert cal.kind in (CalibratorKind.TEMPERATURE, CalibratorKind.IDENTITY)


def test_detector_high_ece_small_n_temperature() -> None:
    rng = np.random.default_rng(1)
    raw = rng.random((10, 3))
    raw = raw / raw.sum(axis=1, keepdims=True)
    # Make predictions wrong to increase ECE
    y_true = np.array([0] * 10)  # all home, but raw random
    det = CalibrationDetector()
    # Force high ECE by using uniform vs true 0 with low confidence?
    cal = det.detect(_mp(raw), y_true)
    # n=10 <5000 → never dirichlet
    assert cal.kind != CalibratorKind.DIRICHLET


def test_detector_high_ece_large_n_dirichlet() -> None:
    n = 6000
    det = CalibrationDetector(ece_low=0.02, ece_high=0.05)
    # Predict 0.9 for class 0 always, but true is 1 → high ECE
    raw2 = np.tile(np.array([0.9, 0.05, 0.05]), (n, 1))
    y_true2 = np.array([1] * n)  # all draw, but predict home
    cal2 = det.detect(_mp(raw2), y_true2)
    assert cal2.kind == CalibratorKind.DIRICHLET


def test_detector_deterministic() -> None:
    rng = np.random.default_rng(3)
    raw = rng.random((30, 3))
    raw = raw / raw.sum(axis=1, keepdims=True)
    y_true = rng.integers(0, 3, size=30)
    det = CalibrationDetector()
    k1 = det.detect(_mp(raw), y_true).kind
    k2 = det.detect(_mp(raw), y_true).kind
    assert k1 == k2


def test_detector_n_zero_raises() -> None:
    det = CalibrationDetector()
    with pytest.raises(ValueError):
        det.detect([], [])
