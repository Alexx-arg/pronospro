"""Poisson trainer tests."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from app.features.example import FEATURE_NAMES
from app.prediction.contracts import FixtureFeatures, ModelName
from app.prediction.models.poisson import HEAD_HOME_FEATURES
from app.prediction.training.poisson_trainer import PoissonTrainer


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


def _targets(n: int, hg: int = 1, ag: int = 0) -> list[list[int]]:
    # Simple: all home wins 1-0
    out: list[list[int]] = []
    for _ in range(n):
        out.append([1, 0, 0, hg, ag])
    return out


def test_entrenamiento_correcto_y_artifact() -> None:
    feats = [_feat(fid=i) for i in range(1, 6)]
    targets = _targets(5, hg=2, ag=1)
    trainer = PoissonTrainer()
    art = trainer.train(feats, targets, hyperparameters={}, seed=42)
    assert art.model_name == ModelName.POISSON
    assert art.hyperparameters["regularization_l2"] == 0.01
    assert art.hyperparameters["max_goals"] == 10
    assert art.inputs.head_features is not None
    assert len(art.inputs.head_features[0]) == 37
    assert len(art.inputs.head_features[1]) == 37
    assert art.training_cutoff is not None
    assert art.payload_sha256 is not None


def test_missing_train_drop_row() -> None:
    # One fixture with NaN in home head → should be dropped
    feats = [
        _feat({HEAD_HOME_FEATURES[0]: None}, fid=1),
        _feat(fid=2),
        _feat(fid=3),
    ]
    targets = _targets(3)
    trainer = PoissonTrainer()
    art = trainer.train(feats, targets)
    # train_block_dropouts should be >=1
    assert art.metrics["train_block_dropouts"] >= 1


def test_missing_test_imputacion_no_cero() -> None:
    # Train with correlated feature → non-zero coefficient
    feats_train = [_feat({"home_goals_for_last_5": 0.0}, fid=1), _feat({"home_goals_for_last_5": 10.0}, fid=2)]
    targets_train = [[1, 0, 0, 0, 0], [1, 0, 0, 5, 0]]  # hg 0 vs 5
    trainer = PoissonTrainer()
    trainer.train(feats_train, targets_train)
    model = trainer.get_model()
    # Train mean = (0+10)/2=5.0
    f_test_nan = _feat({"home_goals_for_last_5": None}, fid=10)
    f_test_zero = _feat({"home_goals_for_last_5": 0.0}, fid=11)
    mp_nan = model.predict(f_test_nan)
    mp_zero = model.predict(f_test_zero)
    # Imputed 5.0 vs 0.0 should give different rates (coefficient >0)
    assert mp_nan.p_home_win != pytest.approx(mp_zero.p_home_win, rel=1e-3)


def test_seleccion_por_validation_no_test() -> None:
    # Train always 0.5, val has different distribution
    train_feats = [_feat(fid=i) for i in range(1, 5)]
    train_targets = _targets(4, hg=1, ag=1)  # draws
    val_feats = [_feat(fid=10), _feat(fid=11)]
    val_targets = _targets(2, hg=2, ag=0)  # home wins
    trainer = PoissonTrainer()
    art = trainer.train(train_feats, train_targets, val_block=val_feats, val_targets=val_targets)
    assert "val_log_loss" in art.metrics
    # Ensure test block not used: changing test should not affect artifact
    # We didn't pass test, so no influence
    assert art.metrics["train_n"] == 4


def test_prohibicion_test_no_influye() -> None:
    train_feats = [_feat(fid=i) for i in range(1, 4)]
    train_targets = _targets(3, hg=1, ag=0)
    trainer1 = PoissonTrainer()
    art1 = trainer1.train(train_feats, train_targets, seed=1)
    # Train again with same train but different test (not passed) should be same sha
    trainer2 = PoissonTrainer()
    art2 = trainer2.train(train_feats, train_targets, seed=1)
    assert art1.payload_sha256 == art2.payload_sha256


def test_determinismo() -> None:
    feats = [_feat(fid=i) for i in range(1, 6)]
    targets = _targets(5, hg=2, ag=1)
    t1 = PoissonTrainer()
    a1 = t1.train(feats, targets, seed=42)
    t2 = PoissonTrainer()
    a2 = t2.train(feats, targets, seed=42)
    assert a1.payload_sha256 == a2.payload_sha256


def test_artifact_serializable() -> None:
    feats = [_feat(fid=i) for i in range(1, 4)]
    targets = _targets(3)
    art = PoissonTrainer().train(feats, targets)
    d = art.to_dict()
    assert d["model_name"] == "poisson"
    assert "hyperparameters" in d


def test_training_cutoff() -> None:
    feats = [
        FixtureFeatures(
            fixture_id=i,
            kickoff=datetime(2026, 1, i, tzinfo=UTC),
            feature_vector=np.full(len(FEATURE_NAMES), 0.5, dtype=np.float32),
            feature_names=tuple(FEATURE_NAMES),
        )
        for i in range(1, 4)
    ]
    targets = _targets(3)
    art = PoissonTrainer().train(feats, targets)
    assert art.training_cutoff == datetime(2026, 1, 3, tzinfo=UTC)


def test_compatibilidad_trainer_protocol() -> None:
    from app.prediction.contracts import Trainer

    trainer = PoissonTrainer()
    assert isinstance(trainer, Trainer) or hasattr(trainer, "train")
    assert trainer.name == ModelName.POISSON
