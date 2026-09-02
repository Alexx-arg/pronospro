"""Classification metrics for 1X2 (Sprint 5.3).

Conventions (see ``docs/PHASE_5.md`` §8):
* targets: 0 = home win, 1 = draw, 2 = away win.
* probabilities: ``y_proba`` shape ``(n, 3)`` column order [home, draw, away].
* All functions raise ``ValueError`` on invalid probabilities — never
  silently clip or renormalise.

The central validator :func:`validate_multiclass_probabilities` is
reused by :mod:`app.prediction.metrics.calibration` and can be imported
directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Absolute tolerance for ``sum(p) == 1``. Chosen to be tighter than
#: typical float32 conversion (1e-6) but looser than 1e-9 used for
#: simplex invariant in ``MatchProbabilities`` tests.
PROBA_SUM_TOL: float = 1e-6

#: Small epsilon for clipping inside ``log_loss`` — not used to fix
#: invalid inputs, only to avoid ``log(0)`` when the input *is* valid.
LOG_LOSS_EPS: float = 1e-15

NUM_CLASSES: int = 3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_multiclass_probabilities(
    y_proba: Any,
    *,
    atol: float = PROBA_SUM_TOL,
) -> Any:
    """Validate multiclass probabilities and return a ``(n, 3)`` float64 array.

    Checks:
    * 2-D array with shape ``(n, 3)``.
    * ``n > 0``.
    * Finite values (no NaN, no inf).
    * ``0 <= p <= 1`` element-wise.
    * Row sums within ``atol`` of 1.0.

    Args:
        y_proba: Array-like of shape ``(n, 3)``.
        atol: Tolerance for ``sum(p) == 1``.

    Returns:
        ``y_proba`` as ``np.ndarray`` with ``dtype float64``.

    Raises:
        ValueError: on any violated invariant.
    """
    arr = np.asarray(y_proba, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"y_proba must be 2-D, got shape {arr.shape}")
    if arr.shape[1] != NUM_CLASSES:
        raise ValueError(f"y_proba must have 3 columns, got {arr.shape[1]}")
    if arr.shape[0] == 0:
        raise ValueError("y_proba must have at least one row (n > 0)")
    if not np.isfinite(arr).all():
        raise ValueError("y_proba contains NaN or inf")
    if (arr < 0.0).any() or (arr > 1.0).any():
        raise ValueError("y_proba values must be in [0, 1]")
    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        # Provide first offending row for debuggability.
        bad = int(np.argmax(np.abs(row_sums - 1.0)))
        raise ValueError(
            f"y_proba rows must sum to 1 (atol={atol}); "
            f"row {bad} sum={row_sums[bad]:.8f}"
        )
    return arr


def _validate_targets(y_true: Any, n: int) -> Any:
    arr = np.asarray(y_true, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {arr.shape}")
    if arr.shape[0] != n:
        raise ValueError(f"y_true length {arr.shape[0]} != y_proba rows {n}")
    if arr.shape[0] == 0:
        raise ValueError("y_true must have at least one element (n > 0)")
    if not np.isfinite(arr).all():
        raise ValueError("y_true contains NaN or inf")
    if (arr < 0).any() or (arr >= NUM_CLASSES).any():
        raise ValueError(f"y_true values must be in {{0,1,2}}, got {np.unique(arr)}")
    return arr


def _validate_pair(y_true: Any, y_proba: Any) -> tuple[Any, Any]:
    proba = validate_multiclass_probabilities(y_proba)
    n = proba.shape[0]
    true = _validate_targets(y_true, n)
    return true, proba


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(y_true: Any, y_proba: Any) -> float:
    """Multiclass accuracy: ``mean(argmax(p) == y_true)``.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    true, proba = _validate_pair(y_true, y_proba)
    preds = np.argmax(proba, axis=1)
    return float((preds == true).mean())


def log_loss(
    y_true: Any,
    y_proba: Any,
    *,
    eps: float = LOG_LOSS_EPS,
) -> float:
    """Multiclass cross-entropy: ``-mean(log(p_true))``.

    Clips ``p_true`` to ``[eps, 1]`` *only* to avoid ``log(0)`` for
    valid tiny probabilities; invalid rows are already rejected by the
    validator (so clipping never hides a ``p<0`` or ``sum!=1`` bug).

    Args:
        y_true: 1-D array of ints in ``{0,1,2}``.
        y_proba: 2-D array ``(n, 3)``.
        eps: Floor for ``p_true`` before ``log``.

    Returns:
        Mean negative log likelihood.
    """
    true, proba = _validate_pair(y_true, y_proba)
    # Gather p_true per row.
    p_true = proba[np.arange(true.shape[0]), true]
    # Clip only for numerical stability — validator already ensured p>=0
    p_true = np.clip(p_true, eps, 1.0)
    return float(-np.log(p_true).mean())


def brier_score(
    y_true: Any,
    y_proba: Any,
) -> dict[str, float]:
    """Brier scores per class and multiclass aggregate.

    Per class: ``mean((p_k - y_k_onehot)^2)``.
    Multiclass: ``mean(sum_k (p_k - y_k_onehot)^2)``  ==  sum of per-class.

    Returns:
        Dict with keys ``brier_home``, ``brier_draw``, ``brier_away``,
        ``brier_multiclass``.
    """
    true, proba = _validate_pair(y_true, y_proba)
    n = true.shape[0]
    # One-hot encode y_true shape (n, 3)
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(n), true] = 1.0
    sq_err = (proba - one_hot) ** 2
    b_home = float(sq_err[:, 0].mean())
    b_draw = float(sq_err[:, 1].mean())
    b_away = float(sq_err[:, 2].mean())
    b_multi = float(sq_err.sum(axis=1).mean())
    return {
        "brier_home": b_home,
        "brier_draw": b_draw,
        "brier_away": b_away,
        "brier_multiclass": b_multi,
    }


def confusion_matrix(y_true: Any, y_proba: Any) -> Any:
    """3×3 confusion matrix.

    Rows = true class, columns = predicted class (``argmax``).
    ``sum(matrix) == n_predictions`` always.

    Returns:
        ``np.ndarray`` shape ``(3, 3)`` dtype ``int64``.
    """
    true, proba = _validate_pair(y_true, y_proba)
    preds = np.argmax(proba, axis=1)
    mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(true, preds, strict=False):  # type: ignore[var-annotated]
        mat[int(t), int(p)] += 1
    return mat


__all__ = [
    "LOG_LOSS_EPS",
    "NUM_CLASSES",
    "PROBA_SUM_TOL",
    "accuracy",
    "brier_score",
    "confusion_matrix",
    "log_loss",
    "validate_multiclass_probabilities",
]
