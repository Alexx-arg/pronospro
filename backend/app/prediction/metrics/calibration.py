"""Calibration metrics: ECE, MCE, reliability bins, confidence buckets (Sprint 5.3).

Definitions follow ``docs/PHASE_5.md`` §8.3 / Sprint 5.3 spec:
* ECE = Σ w_b * |acc_b - conf_b|  where w_b = n_b / n_total.
* MCE = max_b |acc_b - conf_b| over non-empty bins.
* Confidence = max(p) per sample, predicted class = argmax(p).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.prediction.metrics.classification import (
    validate_multiclass_probabilities,
)

NUM_CLASSES: int = 3


def _validate_targets_for_calibration(y_true: Any, n: int) -> Any:
    arr = np.asarray(y_true, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {arr.shape}")
    if arr.shape[0] != n:
        raise ValueError(f"y_true length {arr.shape[0]} != n {n}")
    if arr.shape[0] == 0:
        raise ValueError("y_true must have at least one element")
    if (arr < 0).any() or (arr >= NUM_CLASSES).any():
        raise ValueError(f"y_true must be in {{0,1,2}}, got {np.unique(arr)}")
    return arr


def _prepare(y_true: Any, y_proba: Any) -> tuple[Any, Any, Any, Any]:
    proba = validate_multiclass_probabilities(y_proba)
    n = proba.shape[0]
    true = _validate_targets_for_calibration(y_true, n)
    conf = proba.max(axis=1)
    pred = np.argmax(proba, axis=1)
    correct = (pred == true).astype(np.float64)
    return true, proba, conf, correct


# ---------------------------------------------------------------------------
# ECE / MCE / Reliability bins
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: Any,
    y_proba: Any,
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Bins are uniform over ``[0, 1]`` with ``n_bins`` equal-width intervals.
    Empty bins contribute 0.

    Args:
        y_true: 1-D int array.
        y_proba: 2-D float array ``(n, 3)``.
        n_bins: Number of bins (``>=2``).

    Returns:
        ECE in ``[0, 1]``.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >=2, got {n_bins}")
    _, _, conf, correct = _prepare(y_true, y_proba)
    n = conf.shape[0]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]  # type: ignore[index]
        # Last bin inclusive of 1.0
        if i == n_bins - 1:  # noqa: SIM108
            mask = (conf >= low) & (conf <= high)
        else:  # noqa: SIM108
            mask = (conf >= low) & (conf < high)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        ece += (cnt / n) * abs(acc_bin - conf_bin)
    return float(ece)


def maximum_calibration_error(
    y_true: Any,
    y_proba: Any,
    *,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error (worst bin).

    Same binning as :func:`expected_calibration_error`; empty bins ignored.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >=2, got {n_bins}")
    _, _, conf, correct = _prepare(y_true, y_proba)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]  # type: ignore[index]
        if i == n_bins - 1:  # noqa: SIM108
            mask = (conf >= low) & (conf <= high)
        else:  # noqa: SIM108
            mask = (conf >= low) & (conf < high)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        err = abs(acc_bin - conf_bin)
        if err > mce:
            mce = err
    return float(mce)


def reliability_bins(
    y_true: Any,
    y_proba: Any,
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Reliability diagram data.

    Returns a list of dicts, one per bin, with keys:
    ``bin_id, bin_lower, bin_upper, count, accuracy, mean_confidence,
    abs_gap, weight``. Empty bins have ``count==0, accuracy==0,
    mean_confidence==0`` and ``weight==0``.

    The list length is always ``n_bins`` and is JSON-serialisable.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >=2, got {n_bins}")
    _, _, conf, correct = _prepare(y_true, y_proba)
    n = conf.shape[0]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, Any]] = []
    for i in range(n_bins):
        low, high = float(bins[i]), float(bins[i + 1])  # type: ignore[index]
        if i == n_bins - 1:  # noqa: SIM108
            mask = (conf >= low) & (conf <= high)
        else:  # noqa: SIM108
            mask = (conf >= low) & (conf < high)
        cnt = int(mask.sum())
        if cnt == 0:
            out.append(
                {
                    "bin_id": i,
                    "bin_lower": low,
                    "bin_upper": high,
                    "count": 0,
                    "accuracy": 0.0,
                    "mean_confidence": 0.0,
                    "abs_gap": 0.0,
                    "weight": 0.0,
                }
            )
        else:
            acc = float(correct[mask].mean())
            mconf = float(conf[mask].mean())
            out.append(
                {
                    "bin_id": i,
                    "bin_lower": low,
                    "bin_upper": high,
                    "count": cnt,
                    "accuracy": acc,
                    "mean_confidence": mconf,
                    "abs_gap": abs(acc - mconf),
                    "weight": cnt / n,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Confidence buckets (configurable)
# ---------------------------------------------------------------------------

DEFAULT_CONFIDENCE_BUCKETS: list[float] = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def confidence_bucket_metrics(
    y_true: Any,
    y_proba: Any,
    *,
    bucket_edges: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Per-bucket accuracy / confidence / log_loss.

    Args:
        y_true: 1-D int array.
        y_proba: 2-D float array ``(n, 3)``.
        bucket_edges: Sorted increasing list of upper bounds.
            Default ``[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`` meaning
            intervals ``[0.4,0.5)`` ... ``[0.9,1.0]``.
            The lower bound of the first bucket is ``bucket_edges[0]``.

    Returns:
        List of dicts with keys ``lower, upper, n, accuracy,
        mean_confidence, log_loss``. One dict per bucket (``len==len(edges)-1``).
        Predictions with ``confidence < bucket_edges[0]`` are **not**
        assigned to any bucket; they are excluded from the per-bucket
        report (but counted in overall accuracy/log_loss elsewhere).
        ``sum(n for b in result)`` equals the number of predictions
        with ``confidence >= bucket_edges[0]`` and ``confidence <= 1.0``.
        This is documented explicitly so callers do not silently lose
        observations.

    Raises:
        ValueError: if ``bucket_edges`` not strictly increasing or
            not in ``[0, 1]`` or ``len < 2``.
    """
    _, proba, conf, correct = _prepare(y_true, y_proba)
    true = _validate_targets_for_calibration(y_true, proba.shape[0])
    edges = list(bucket_edges) if bucket_edges is not None else list(DEFAULT_CONFIDENCE_BUCKETS)

    if len(edges) < 2:
        raise ValueError(f"bucket_edges must have >=2 values, got {edges}")
    if any(e < 0.0 or e > 1.0 for e in edges):
        raise ValueError(f"bucket_edges must be in [0, 1], got {edges}")
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        if not (a < b):
            raise ValueError(f"bucket_edges must be strictly increasing, got {edges}")

    # For log_loss per bucket we reuse EPS clipping (same as classification).
    eps = 1e-15
    p_true = proba[np.arange(true.shape[0]), true]
    p_true = np.clip(p_true, eps, 1.0)

    buckets: list[dict[str, Any]] = []
    for i in range(len(edges) - 1):
        low, high = float(edges[i]), float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf >= low) & (conf < high)
        cnt = int(mask.sum())
        if cnt == 0:
            buckets.append(
                {
                    "lower": low,
                    "upper": high,
                    "n": 0,
                    "accuracy": 0.0,
                    "mean_confidence": 0.0,
                    "log_loss": 0.0,
                }
            )
        else:
            acc = float(correct[mask].mean())
            mconf = float(conf[mask].mean())
            ll = float(-np.log(p_true[mask]).mean())
            buckets.append(
                {
                    "lower": low,
                    "upper": high,
                    "n": cnt,
                    "accuracy": acc,
                    "mean_confidence": mconf,
                    "log_loss": ll,
                }
            )
    return buckets


__all__ = [
    "DEFAULT_CONFIDENCE_BUCKETS",
    "confidence_bucket_metrics",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_bins",
]
