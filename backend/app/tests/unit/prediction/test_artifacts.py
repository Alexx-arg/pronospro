"""Sprint 5.0 — ``ModelArtifact`` and ``PredictionRecord`` are frozen
and serialise to plain JSON-friendly dicts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest
from app.prediction import artifacts
from app.prediction.contracts import ModelName


def _minimal_artifact() -> artifacts.ModelArtifact:
    return artifacts.ModelArtifact(
        model_name=ModelName.GRADIENT_BOOSTING,
        model_version="v001",
        training_data_version="v001",
        feature_definition_version="fd_v1",
        inputs=artifacts.ModelArtifactInputs(
            feature_names=("home_elo_pre_match", "away_elo_pre_match"),
            feature_definition_version="fd_v1",
            head_features=None,
        ),
        hyperparameters=artifacts.make_hyperparameters(
            n_estimators=500, learning_rate=0.05
        ),
        training_cutoff=datetime(2024, 12, 31, tzinfo=UTC),
        created_at=datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
        metrics=artifacts.frozen_metrics(log_loss=1.012),
        fitted_seed=42,
        payload_ref="artifact.bin",
        payload_sha256="0" * 64,
    )


def test_model_artifact_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    a = _minimal_artifact()
    with pytest.raises(FrozenInstanceError):
        a.model_version = "v002"  # type: ignore[misc]


def test_model_artifact_hyperparameters_are_read_only_mapping() -> None:
    a = _minimal_artifact()
    assert isinstance(a.hyperparameters, MappingProxyType)
    with pytest.raises(TypeError):
        a.hyperparameters["n_estimators"] = 999  # type: ignore[index]


def test_model_artifact_metrics_are_read_only_mapping() -> None:
    a = _minimal_artifact()
    assert isinstance(a.metrics, MappingProxyType)
    with pytest.raises(TypeError):
        a.metrics["log_loss"] = 5.0  # type: ignore[index]


def test_model_artifact_to_dict_is_json_serialisable() -> None:
    # If to_dict contains a non JSON-encodable value, json.dumps raises.
    import json

    a = _minimal_artifact()
    s = json.dumps(a.to_dict())  # should not raise
    d = json.loads(s)
    assert d["model_name"] == "gradient_boosting"
    assert d["model_version"] == "v001"
    assert d["inputs"]["feature_names"] == [
        "home_elo_pre_match",
        "away_elo_pre_match",
    ]
    assert d["hyperparameters"] == {"n_estimators": 500, "learning_rate": 0.05}


def test_model_artifact_to_dict_iso_datetimes() -> None:
    a = _minimal_artifact()
    d = a.to_dict()
    # ISO8601 with ``+00:00`` tz suffix (tz-aware UTC).
    assert d["training_cutoff"] == "2024-12-31T00:00:00+00:00"
    assert d["created_at"] == "2026-08-15T12:00:00+00:00"


def test_model_artifact_inputs_with_head_features_round_trips() -> None:
    """Poisson uses ``head_features`` not ``None`` — round-trip the
    serialised shape so the format is documented."""
    import json

    a = artifacts.ModelArtifact(
        model_name=ModelName.POISSON,
        model_version="v001",
        training_data_version="v001",
        feature_definition_version="fd_v1",
        inputs=artifacts.ModelArtifactInputs(
            # truncated for brevity — full partition verified in Sprint 5.6
            feature_names=("home_elo_pre_match", "away_elo_pre_match"),
            feature_definition_version="fd_v1",
            head_features=(
                ("home_elo_pre_match", "elo_difference"),
                ("away_elo_pre_match", "elo_difference"),
            ),
        ),
        hyperparameters=artifacts.make_hyperparameters(regularization_l2=0.01),
        training_cutoff=datetime(2024, 12, 31, tzinfo=UTC),
        created_at=datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
        metrics=artifacts.frozen_metrics(),
        fitted_seed=7,
        payload_ref="artifact.bin",
        payload_sha256="0" * 64,
    )
    d = json.loads(json.dumps(a.to_dict()))
    assert d["inputs"]["head_features"] == [
        ["home_elo_pre_match", "elo_difference"],
        ["away_elo_pre_match", "elo_difference"],
    ]


def test_prediction_record_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    from app.prediction.contracts import MatchProbabilities

    rec = artifacts.PredictionRecord(
        model_name=ModelName.ELO_BASELINE,
        model_version="v001",
        fixture_id=42,
        kickoff=datetime(2026, 8, 15, 15, 0, tzinfo=UTC),
        probabilities=MatchProbabilities(0.5, 0.3, 0.2),
        artifact_sha256="0" * 64,
        predicted_at=datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
    )
    with pytest.raises(FrozenInstanceError):
        rec.fixture_id = 999  # type: ignore[misc]


def test_make_hyperparameters_rejects_post_construction_mutation() -> None:
    h = artifacts.make_hyperparameters(a=1, b=2)
    with pytest.raises(TypeError):
        h["a"] = 99  # type: ignore[index]
    assert h["a"] == 1
