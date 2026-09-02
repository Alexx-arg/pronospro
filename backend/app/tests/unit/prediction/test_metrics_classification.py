"""Tests for classification metrics (Sprint 5.3)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from app.prediction.metrics.classification import (
    accuracy,
    brier_score,
    confusion_matrix,
    log_loss,
    validate_multiclass_probabilities,
)


def _proba(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 3))
    return raw / raw.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ validation
def test_validate_rejects_nan() -> None:
    proba = np.array([[0.5, 0.3, float("nan")]])
    with pytest.raises(ValueError, match="NaN"):
        validate_multiclass_probabilities(proba)


def test_validate_rejects_inf() -> None:
    proba = np.array([[0.5, 0.3, float("inf")]])
    with pytest.raises(ValueError, match="inf"):
        validate_multiclass_probabilities(proba)


def test_validate_rejects_negative() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_multiclass_probabilities([[-0.1, 0.6, 0.5]])


def test_validate_rejects_gt_one() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_multiclass_probabilities([[1.2, -0.1, -0.1]])


def test_validate_rejects_sum_not_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        validate_multiclass_probabilities([[0.5, 0.5, 0.5]])


def test_validate_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="3 columns"):
        validate_multiclass_probabilities([[0.5, 0.5]])


def test_validate_rejects_n_zero() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        validate_multiclass_probabilities(np.empty((0, 3)))


# ---------------------------------------------------------- accuracy
def test_accuracy_perfect() -> None:
    y_true = np.array([0, 1, 2])
    y_proba = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    assert accuracy(y_true, y_proba) == pytest.approx(1.0)


def test_accuracy_uniform() -> None:
    y_true = np.array([0, 1, 2, 0])
    y_proba = np.array([[1 / 3, 1 / 3, 1 / 3]] * 4)
    # argmax of ties picks 0 → 2 correct (those with true 0)
    assert accuracy(y_true, y_proba) == pytest.approx(0.5)


def test_accuracy_n_zero_error() -> None:
    with pytest.raises(ValueError):
        accuracy([], np.empty((0, 3)))


def test_accuracy_targets_invalid() -> None:
    with pytest.raises(ValueError, match=r"\{0,1,2\}"):
        accuracy([3], [[0.33, 0.33, 0.34]])


# ---------------------------------------------------------- log_loss
def test_log_loss_perfect_near_zero() -> None:
    y_true = np.array([0])
    y_proba = np.array([[0.999, 0.0005, 0.0005]])
    ll = log_loss(y_true, y_proba)
    assert ll == pytest.approx(-math.log(0.999), rel=1e-5)
    assert ll < 0.01


def test_log_loss_uniform() -> None:
    y_true = np.array([0, 1, 2])
    y_proba = np.array([[1 / 3, 1 / 3, 1 / 3]] * 3)
    ll = log_loss(y_true, y_proba)
    assert ll == pytest.approx(-math.log(1 / 3), rel=1e-5)


def test_log_loss_tiny_probability() -> None:
    y_true = np.array([0])
    y_proba = np.array([[1e-6, 0.5, 0.5 - 1e-6]])
    # Should clip to eps=1e-15, not raise
    ll = log_loss(y_true, y_proba)
    assert ll == pytest.approx(-math.log(1e-6), rel=1e-6)


def test_log_loss_near_zero_true_class_gives_large() -> None:
    y_true = np.array([0])
    y_proba = np.array([[1e-12, 0.5, 0.5 - 1e-12]])
    ll = log_loss(y_true, y_proba)
    assert ll > 20  # -log(1e-12) ≈ 27


def test_log_loss_invalid_proba_raises() -> None:
    with pytest.raises(ValueError):
        log_loss([0], [[0.5, 0.5, 0.5]])


# ---------------------------------------------------------- brier
def test_brier_perfect() -> None:
    y_true = np.array([0])
    y_proba = np.array([[1.0, 0.0, 0.0]])
    b = brier_score(y_true, y_proba)
    assert b["brier_home"] == pytest.approx(0.0)
    assert b["brier_draw"] == pytest.approx(0.0)
    assert b["brier_away"] == pytest.approx(0.0)
    assert b["brier_multiclass"] == pytest.approx(0.0)


def test_brier_uniform() -> None:
    y_true = np.array([0])
    y_proba = np.array([[1 / 3, 1 / 3, 1 / 3]])
    b = brier_score(y_true, y_proba)
    # (1/3-1)^2=4/9 for home, (1/3)^2=1/9 for draw/away
    assert b["brier_home"] == pytest.approx(4 / 9)
    assert b["brier_draw"] == pytest.approx(1 / 9)
    assert b["brier_away"] == pytest.approx(1 / 9)
    assert b["brier_multiclass"] == pytest.approx(6 / 9)


def test_brier_n_classes_missing() -> None:
    # Only home wins in y_true, but brier for other classes still defined
    y_true = np.array([0, 0])
    y_proba = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    b = brier_score(y_true, y_proba)
    assert "brier_home" in b


# ---------------------------------------------------------- confusion
def test_confusion_diagonal_perfect() -> None:
    y_true = np.array([0, 1, 2])
    y_proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    cm = confusion_matrix(y_true, y_proba)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[1, 1] == 1 and cm[2, 2] == 1
    assert cm.sum() == 3


def test_confusion_missing_classes() -> None:
    y_true = np.array([0, 0])
    y_proba = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
    cm = confusion_matrix(y_true, y_proba)
    assert cm[1].sum() == 0
    assert cm[2].sum() == 0
    assert cm.sum() == 2


def test_confusion_all_same_pred() -> None:
    y_true = np.array([0, 1, 2])
    y_proba = np.array([[0.8, 0.1, 0.1]] * 3)
    cm = confusion_matrix(y_true, y_proba)
    assert cm[:, 0].sum() == 3
