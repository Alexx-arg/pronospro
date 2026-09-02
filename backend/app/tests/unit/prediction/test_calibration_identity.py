"""Identity calibrator tests."""

from __future__ import annotations

import numpy as np
import pytest
from app.prediction.calibration.identity import IdentityCalibrator
from app.prediction.contracts import CalibratorKind, MatchProbabilities


def _mp_array(arr: np.ndarray) -> list[MatchProbabilities]:
    return [
        MatchProbabilities(float(r[0]), float(r[1]), float(r[2])) for r in arr
    ]


def test_identity_preserves_simplex() -> None:
    cal = IdentityCalibrator()
    raw = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
    targets = np.array([0, 1])
    cal.fit(_mp_array(raw), targets)
    out = cal.transform(_mp_array(raw))
    out_arr = np.array([[p.p_home_win, p.p_draw, p.p_away_win] for p in out])
    assert np.allclose(out_arr, raw)


def test_identity_shape_preserved() -> None:
    cal = IdentityCalibrator()
    raw = np.array([[0.33, 0.33, 0.34]] * 5)
    cal.fit(_mp_array(raw), np.array([0, 1, 2, 0, 1]))
    out = cal.transform(_mp_array(raw))
    assert len(out) == 5


def test_identity_no_mutation() -> None:
    cal = IdentityCalibrator()
    raw = np.array([[0.6, 0.2, 0.2]])
    raw_copy = np.array(raw, copy=True)
    cal.fit(_mp_array(raw), np.array([0]))
    _ = cal.transform(_mp_array(raw))
    assert np.array_equal(raw, raw_copy)


def test_identity_kind() -> None:
    assert IdentityCalibrator().kind == CalibratorKind.IDENTITY


def test_identity_serialization() -> None:
    cal = IdentityCalibrator()
    d = cal.to_dict()
    assert d["kind"] == "identity"
    cal2 = IdentityCalibrator.from_dict(d)
    assert cal2.kind == CalibratorKind.IDENTITY


def test_identity_rejects_invalid() -> None:
    cal = IdentityCalibrator()
    with pytest.raises(ValueError):
        cal.fit(_mp_array(np.array([[0.5, 0.5, 0.5]])), np.array([0]))
