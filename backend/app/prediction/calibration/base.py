"""Shared helpers for calibration (Sprint 5.4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.prediction.contracts import MatchProbabilities
from app.prediction.metrics.classification import validate_multiclass_probabilities

CLIP_EPS: float = 1e-12
PROBA_SUM_TOL: float = 1e-6


def _probas_to_array(probas: Any) -> Any:
    """Convert ``Sequence[MatchProbabilities]`` or ndarray to ``(n,3)`` array."""
    if isinstance(probas, np.ndarray):
        return np.asarray(probas, dtype=np.float64)
    # Assume sequence of MatchProbabilities
    try:
        arr = np.array(
            [[p.p_home_win, p.p_draw, p.p_away_win] for p in probas],
            dtype=np.float64,
        )
        return arr
    except Exception as exc:
        raise ValueError(f"cannot convert probas to array: {exc}") from exc


def _array_to_probas(arr: Any) -> list[MatchProbabilities]:
    out: list[MatchProbabilities] = []
    for row in arr:
        out.append(
            MatchProbabilities(
                p_home_win=float(row[0]),
                p_draw=float(row[1]),
                p_away_win=float(row[2]),
            )
        )
    return out


def _validate_targets(targets: Any, n: int) -> Any:
    arr = np.asarray(targets, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"targets must be 1-D, got shape {arr.shape}")
    if arr.shape[0] != n:
        raise ValueError(f"targets length {arr.shape[0]} != n {n}")
    if arr.shape[0] == 0:
        raise ValueError("targets must have at least one element")
    if (arr < 0).any() or (arr >= 3).any():
        raise ValueError(f"targets must be in {{0,1,2}}, got {np.unique(arr)}")
    return arr


def _validate_probas_and_targets(
    probas: Any,
    targets: Any,
) -> tuple[Any, Any]:
    arr = _probas_to_array(probas)
    # Use central validator for probabilities (finite, [0,1], sum≈1)
    arr = validate_multiclass_probabilities(arr)
    n = arr.shape[0]
    t = _validate_targets(targets, n)
    return arr, t


def _softmax(logits: Any) -> Any:
    """Numerically stable softmax over last axis."""
    # logits shape (n,3) or (3,)
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(axis=-1, keepdims=True)
    return e / s


def _clip_for_log(proba: Any, eps: float = CLIP_EPS) -> Any:
    return np.clip(proba, eps, 1.0)


__all__ = [
    "CLIP_EPS",
    "PROBA_SUM_TOL",
    "_array_to_probas",
    "_clip_for_log",
    "_probas_to_array",
    "_softmax",
    "_validate_probas_and_targets",
    "_validate_targets",
]
