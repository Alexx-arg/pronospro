"""Goal-based metrics (Sprint 5.3).

Metrics for Poisson-style goal predictions. All functions are pure
numpy and raise ``ValueError`` on invalid inputs — never silently
ignore ``NaN`` or ``inf``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_goals(y_true: Any, n: int | None = None) -> Any:
    arr = np.asarray(y_true, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"goals array must be 1-D, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("goals array must have at least one element")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"goals length {arr.shape[0]} != expected {n}")
    if not np.isfinite(arr).all():
        raise ValueError("goals array contains NaN or inf")
    if (arr < 0).any():
        raise ValueError("goals must be non-negative")
    # Goals should be integer-valued, but accept float that is int-like.
    if not np.allclose(arr, np.round(arr)):
        raise ValueError(f"goals must be integer-valued, got {arr[:5]}")
    return arr


def mae_home_goals(y_true_home: Any, y_pred_home: Any) -> float:
    """Mean Absolute Error for home goals."""
    true = _validate_goals(y_true_home)
    pred = _validate_goals(y_pred_home, n=true.shape[0])
    # Also validate pred finite (allow float expectations)
    if not np.isfinite(pred).all():
        raise ValueError("y_pred_home contains NaN or inf")
    return float(np.abs(true - pred).mean())


def mae_away_goals(y_true_away: Any, y_pred_away: Any) -> float:
    """Mean Absolute Error for away goals."""
    true = _validate_goals(y_true_away)
    pred = _validate_goals(y_pred_away, n=true.shape[0])
    if not np.isfinite(pred).all():
        raise ValueError("y_pred_away contains NaN or inf")
    return float(np.abs(true - pred).mean())


def rmse_home_goals(y_true_home: Any, y_pred_home: Any) -> float:
    """Root Mean Squared Error for home goals."""
    true = _validate_goals(y_true_home)
    pred = np.asarray(y_pred_home, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"y_pred_home shape {pred.shape} != {true.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("y_pred_home contains NaN or inf")
    if (pred < 0).any():
        raise ValueError("y_pred_home must be non-negative")
    return float(np.sqrt(((true - pred) ** 2).mean()))


def rmse_away_goals(y_true_away: Any, y_pred_away: Any) -> float:
    """Root Mean Squared Error for away goals."""
    true = _validate_goals(y_true_away)
    pred = np.asarray(y_pred_away, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"y_pred_away shape {pred.shape} != {true.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("y_pred_away contains NaN or inf")
    if (pred < 0).any():
        raise ValueError("y_pred_away must be non-negative")
    return float(np.sqrt(((true - pred) ** 2).mean()))


def rmse_total_goals(
    y_true_home: Any,
    y_true_away: Any,
    y_pred_home: Any,
    y_pred_away: Any,
) -> float:
    """RMSE on total goals ``(home + away)``.

    Args:
        y_true_home, y_true_away: 1-D int arrays.
        y_pred_home, y_pred_away: 1-D float arrays (expected goals).

    Returns:
        RMSE of ``(y_true_home + y_true_away)`` vs ``(y_pred_home + y_pred_away)``.
    """
    th = _validate_goals(y_true_home)
    ta = _validate_goals(y_true_away, n=th.shape[0])
    ph = np.asarray(y_pred_home, dtype=np.float64)
    pa = np.asarray(y_pred_away, dtype=np.float64)
    if ph.shape != th.shape or pa.shape != th.shape:
        raise ValueError("predicted goals shape mismatch")
    if not np.isfinite(ph).all() or not np.isfinite(pa).all():
        raise ValueError("predicted goals contain NaN or inf")
    if (ph < 0).any() or (pa < 0).any():
        raise ValueError("predicted goals must be non-negative")
    true_total = th + ta
    pred_total = ph + pa
    return float(np.sqrt(((true_total - pred_total) ** 2).mean()))


def poisson_loglik(
    y_true_home: Any,
    y_true_away: Any,
    y_pred_home_goals: Any,
    y_pred_away_goals: Any,
    *,
    eps: float = 1e-15,
) -> float:
    """Mean Poisson log-likelihood for observed goals.

    For each sample ``i``:

    ``loglik_i = log Poisson(k_home_i; λ_pred_home_i) + log Poisson(k_away_i; λ_pred_away_i)``

    where ``λ`` are the predicted Poisson rates (expected goals).
    The function also accepts per-goal probability dicts (``p_home_goals``)
    from :class:`MatchProbabilities` — if a dict is passed, the
    probability of the observed ``k`` is looked up directly.

    This implementation accepts **two forms** for the prediction args:

    * **Rate form**: 1-D float array of ``λ`` values (``>0``).
    * **Dict form**: sequence of dicts ``{k: prob}`` as produced by the
      Poisson → 1X2 conversor. In that case ``actual`` goals select the
      dict entry; missing ``k`` is treated as ``eps``.

    Args:
        y_true_home, y_true_away: 1-D int arrays of actual goals.
        y_pred_home_goals, y_pred_away_goals: either 1-D float arrays of
            ``λ`` or sequences of ``dict[int, float]``.
        eps: Floor for probability before ``log`` (avoids ``-inf``).

    Returns:
        Mean log-likelihood (higher is better, 0 is unattainable optimum
        for integer goals with Poisson).
    """
    th = _validate_goals(y_true_home)
    ta = _validate_goals(y_true_away, n=th.shape[0])
    n = th.shape[0]

    # Detect form: sequence of dicts vs array of lambdas.
    # We inspect the first element — if it's a dict, treat as dict form.
    def _is_dict_sequence(x: Any) -> bool:
        if isinstance(x, dict):
            return True
        try:
            first = x[0]
            return isinstance(first, dict)
        except Exception:
            return False

    is_dict_home = _is_dict_sequence(y_pred_home_goals)
    is_dict_away = _is_dict_sequence(y_pred_away_goals)

    if is_dict_home or is_dict_away:
        # Dict form — y_pred_* is sequence[dict[int, float]]
        if not (is_dict_home and is_dict_away):
            raise ValueError("both home and away predictions must be same form (both dict or both rate)")
        # Normalize to list of dicts
        ph_list = list(y_pred_home_goals)
        pa_list = list(y_pred_away_goals)
        if len(ph_list) != n or len(pa_list) != n:
            raise ValueError(f"predicted dict sequences length mismatch with n={n}")
        logliks: list[float] = []
        for i in range(n):
            k_h = int(th[i])
            k_a = int(ta[i])
            d_h = ph_list[i]
            d_a = pa_list[i]
            if not isinstance(d_h, dict) or not isinstance(d_a, dict):
                raise ValueError(f"expected dict at index {i}")
            p_h = float(d_h.get(k_h, eps))
            p_a = float(d_a.get(k_a, eps))
            if not np.isfinite(p_h) or not np.isfinite(p_a):
                raise ValueError(f"dict probability at index {i} is non-finite")
            if p_h < 0 or p_a < 0:
                raise ValueError(f"dict probability at index {i} is negative")
            p_h = max(p_h, eps)
            p_a = max(p_a, eps)
            logliks.append(float(np.log(p_h) + np.log(p_a)))
        return float(np.mean(logliks))

    # Rate form — y_pred_* are 1-D arrays of lambdas
    ph = np.asarray(y_pred_home_goals, dtype=np.float64)
    pa = np.asarray(y_pred_away_goals, dtype=np.float64)
    if ph.shape != th.shape or pa.shape != th.shape:
        raise ValueError(f"predicted lambda shape mismatch: {ph.shape}, {pa.shape} vs {th.shape}")
    if not np.isfinite(ph).all() or not np.isfinite(pa).all():
        raise ValueError("lambda contains NaN or inf")
    if (ph <= 0).any() or (pa <= 0).any():
        raise ValueError("lambda must be > 0")
    # Poisson log pmf: -λ + k*log(λ) - gammaln(k+1)
    # Use math.lgamma for per-element gammaln
    import math as _math

    ll_home = -ph + th * np.log(np.maximum(ph, eps))
    ll_away = -pa + ta * np.log(np.maximum(pa, eps))
    # Subtract log(k!)
    for i in range(n):
        ll_home[i] -= _math.lgamma(float(th[i]) + 1.0)
        ll_away[i] -= _math.lgamma(float(ta[i]) + 1.0)
    total = ll_home + ll_away
    return float(total.mean())


__all__ = [
    "mae_away_goals",
    "mae_home_goals",
    "poisson_loglik",
    "rmse_away_goals",
    "rmse_home_goals",
    "rmse_total_goals",
]
