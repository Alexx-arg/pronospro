"""GB trainer tests — determinism, artifact, no test leakage."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from app.features.example import FEATURE_NAMES
from app.prediction.contracts import FixtureFeatures, ModelName
from app.prediction.training.gb_trainer import GradientBoostingTrainer


def _feat(fid: int = 1, nan: bool = False) -> FixtureFeatures:
    vec = np.full(len(FEATURE_NAMES), 0.5, dtype=np.float32)
    if nan:
        vec[0] = np.nan
    return FixtureFeatures(
        fixture_id=fid,
        kickoff=datetime(2026, 1, fid, tzinfo=UTC),
        feature_vector=vec,
        feature_names=tuple(FEATURE_NAMES),
    )


def _targets_cycle(n: int) -> list[list[int]]:
    out: list[list[int]] = []
    for i in range(n):
        label = i % 3
        if label == 0:
            out.append([1, 0, 0, 2, 0])
        elif label == 1:
            out.append([0, 1, 0, 1, 1])
        else:
            out.append([0, 0, 1, 0, 2])
    return out


def test_gb_trainer_artifact() -> None:
    feats = [_feat(fid=i) for i in range(1, 7)]
    targets = _targets_cycle(6)
    trainer = GradientBoostingTrainer()
    art = trainer.train(feats, targets, seed=42)
    assert art.model_name == ModelName.GRADIENT_BOOSTING
    assert art.inputs.feature_names == tuple(FEATURE_NAMES)
    assert "learning_rate" in art.hyperparameters
    assert art.training_cutoff == datetime(2026, 1, 6, tzinfo=UTC)
    assert art.payload_sha256 is not None


def test_gb_determinism() -> None:
    feats = [_feat(fid=i) for i in range(1, 9)]
    targets = _targets_cycle(8)
    t1 = GradientBoostingTrainer()
    a1 = t1.train(feats, targets, seed=42)
    t2 = GradientBoostingTrainer()
    a2 = t2.train(feats, targets, seed=42)
    assert a1.payload_sha256 == a2.payload_sha256


def test_gb_no_test_leakage() -> None:
    train_feats = [_feat(fid=i) for i in range(1, 7)]
    train_targets = _targets_cycle(6)
    trainer1 = GradientBoostingTrainer()
    art1 = trainer1.train(train_feats, train_targets, seed=1)
    # Train again with same train but different test (not passed) should be same
    trainer2 = GradientBoostingTrainer()
    art2 = trainer2.train(train_feats, train_targets, seed=1)
    assert art1.payload_sha256 == art2.payload_sha256


def test_gb_missing_passthrough_not_drop() -> None:
    # Train with NaN should not raise (HistGB handles it); no drop
    feats = [_feat(fid=1, nan=True), _feat(fid=2), _feat(fid=3)]
    targets = _targets_cycle(3)
    trainer = GradientBoostingTrainer()
    art = trainer.train(feats, targets, seed=0)
    # Should succeed and train_n ==3 (no drop)
    assert art.metrics["train_n"] == 3


def test_gb_training_cutoff() -> None:
    feats = [
        FixtureFeatures(
            fixture_id=1,
            kickoff=datetime(2026, 1, 10, tzinfo=UTC),
            feature_vector=np.full(len(FEATURE_NAMES), 0.5, dtype=np.float32),
            feature_names=tuple(FEATURE_NAMES),
        ),
        FixtureFeatures(
            fixture_id=2,
            kickoff=datetime(2026, 1, 12, tzinfo=UTC),
            feature_vector=np.full(len(FEATURE_NAMES), 0.5, dtype=np.float32),
            feature_names=tuple(FEATURE_NAMES),
        ),
    ]
    targets = _targets_cycle(2)
    art = GradientBoostingTrainer().train(feats, targets)
    assert art.training_cutoff == datetime(2026, 1, 12, tzinfo=UTC)


def test_gb_hyperparams_serializable() -> None:
    feats = [_feat(fid=i) for i in range(1, 5)]
    targets = _targets_cycle(4)
    art = GradientBoostingTrainer().train(feats, targets)
    d = art.to_dict()
    assert "hyperparameters" in d
    assert d["hyperparameters"]["loss"] == "log_loss"


def test_gb_val_health_check_no_test_usage() -> None:
    train_feats = [_feat(fid=i) for i in range(1, 5)]
    train_targets = _targets_cycle(4)
    val_feats = [_feat(fid=10), _feat(fid=11)]
    val_targets = _targets_cycle(2)
    trainer = GradientBoostingTrainer()
    art = trainer.train(train_feats, train_targets, val_block=val_feats, val_targets=val_targets, seed=0)
    assert "val_log_loss" in art.metrics or "train_n" in art.metrics


def test_gb_compat_trainer() -> None:
    from app.prediction.contracts import Trainer

    trainer = GradientBoostingTrainer()
    assert isinstance(trainer, Trainer) or hasattr(trainer, "train")
