"""Sprint 5.0 — ``PredictionSettings`` validates bounds + parses
the CSV-string ``confidence_buckets`` like the existing ``Settings``
pattern (parity with ``app.config.Settings``).
"""

from __future__ import annotations

import pytest
from app.prediction.config import PredictionSettings
from app.prediction.contracts import NanPolicy, WalkForwardMode
from pydantic import ValidationError


def test_default_settings_construct_successfully() -> None:
    s = PredictionSettings()  # type: ignore[call-arg]
    assert s.dataset_version == "v001"
    assert s.walk_forward_min_train_size == 200
    assert s.walk_forward_val_ratio == 0.15
    assert s.walk_forward_test_size == 50
    assert s.walk_forward_gap_days == 1
    assert s.walk_forward_mode == WalkForwardMode.EXPANDING
    assert s.calibration_bins == 10
    assert s.random_seed == 42
    assert s.max_goals == 10
    assert s.nan_policy == NanPolicy.DROP_ROW


def test_confidence_bucket_list_parses_csv() -> None:
    s = PredictionSettings()  # type: ignore[call-arg]
    assert s.confidence_bucket_list() == [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_min_train_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        PredictionSettings(walk_forward_min_train_size=0)  # type: ignore[call-arg]


def test_val_ratio_must_be_strictly_in_unit_interval() -> None:
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValidationError):
            PredictionSettings(walk_forward_val_ratio=bad)  # type: ignore[call-arg]


def test_test_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        PredictionSettings(walk_forward_test_size=0)  # type: ignore[call-arg]


def test_gap_days_must_be_non_negative() -> None:
    # gap == 0 is allowed (no buffer); gap < 0 is not.
    s = PredictionSettings(walk_forward_gap_days=0)  # type: ignore[call-arg]
    assert s.walk_forward_gap_days == 0
    with pytest.raises(ValidationError):
        PredictionSettings(walk_forward_gap_days=-1)  # type: ignore[call-arg]


def test_calibration_bins_bounds() -> None:
    s = PredictionSettings(calibration_bins=2)  # type: ignore[call-arg]
    assert s.calibration_bins == 2
    with pytest.raises(ValidationError):
        PredictionSettings(calibration_bins=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        PredictionSettings(calibration_bins=101)  # type: ignore[call-arg]


def test_max_goals_bounds() -> None:
    s = PredictionSettings(max_goals=15)  # type: ignore[call-arg]
    assert s.max_goals == 15
    with pytest.raises(ValidationError):
        PredictionSettings(max_goals=0)  # type: ignore[call-arg]


def test_confidence_buckets_must_be_strictly_increasing() -> None:
    with pytest.raises(ValidationError):
        PredictionSettings(confidence_buckets="0.4,0.4,1.0")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        PredictionSettings(confidence_buckets="0.5,0.4")  # type: ignore[call-arg]


def test_confidence_buckets_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        PredictionSettings(confidence_buckets="0.4,1.5")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        PredictionSettings(confidence_buckets="-0.1,0.4")  # type: ignore[call-arg]


def test_confidence_buckets_must_have_at_least_two_values() -> None:
    with pytest.raises(ValidationError):
        PredictionSettings(confidence_buckets="0.5")  # type: ignore[call-arg]


def test_env_prefix_prediction_is_read_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PredictionSettings`` uses ``env_prefix="PREDICTION_"`` — aka
    variables prefixed ``PREDICTION_`` populate fields. Test mirrors
    the contract encoded in ``.env.example``.
    """
    monkeypatch.setenv("PREDICTION_DATASET_VERSION", "v042")
    monkeypatch.setenv("PREDICTION_WALK_FORWARD_MIN_TRAIN_SIZE", "333")
    monkeypatch.setenv("PREDICTION_RANDOM_SEED", "7")
    monkeypatch.setenv("PREDICTION_MAX_GOALS", "8")
    s = PredictionSettings()  # type: ignore[call-arg]
    assert s.dataset_version == "v042"
    assert s.walk_forward_min_train_size == 333
    assert s.random_seed == 7
    assert s.max_goals == 8


def test_nan_policy_enum_round_trip() -> None:
    s = PredictionSettings(  # type: ignore[call-arg]
        nan_policy=NanPolicy.IMPUTE_TRAIN_MEAN
    )
    assert s.nan_policy == NanPolicy.IMPUTE_TRAIN_MEAN
    assert s.nan_policy == "impute_train_mean"


def test_walk_forward_mode_enum_round_trip() -> None:
    s = PredictionSettings(  # type: ignore[call-arg]
        walk_forward_mode=WalkForwardMode.SLIDING
    )
    assert s.walk_forward_mode == WalkForwardMode.SLIDING
    assert s.walk_forward_mode == "sliding"
