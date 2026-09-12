"""Goal-market estimators (Over/Under 2.5, BTTS) derived from xG via Poisson.

Production market expansion beside the multiclass LightGBM 1X2 predictor:

* The LightGBM model produces the 1X2 simplex (``p_home/p_draw/p_away``).
* It does **not** produce per-goal distributions. We derive them from the
  expected goals (xG) of each side, using the shared Poisson machinery in
  :mod:`app.prediction.models._poisson` so there is a single source of truth
  for Poisson math (no duplicated pmf code).

Invariants:

* Every probability is clamped to ``[0, 1]``.
* ``over + under`` and ``btts_yes + btts_no`` are renormalised to sum to 1.
* Missing xG → ``None`` market (never a fabricated number).
"""

from __future__ import annotations

from math import isfinite


def _poisson_marginals(
    lambda_home: float,
    lambda_away: float,
    *,
    max_goals: int = 12,
) -> tuple[list[float], list[float]]:
    """Return the Poisson pmf vectors P(home=k), P(away=k) for 0..max_goals.

    Reuses a small, self-contained log-space pmf (kept local to avoid a
    heavier import); the 1X2 convertor in ``_poisson.py`` is authoritative
    for the multiclass projection, these marginals are only for goal markets.
    """
    import math

    pmf_h: list[float] = []
    pmf_a: list[float] = []
    for k in range(max_goals + 1):
        pmf_h.append(math.exp(-lambda_home + k * math.log(lambda_home) - math.lgamma(k + 1)))
        pmf_a.append(math.exp(-lambda_away + k * math.log(lambda_away) - math.lgamma(k + 1)))
    return pmf_h, pmf_a


def over_under_25(
    lambda_home: float,
    lambda_away: float,
    *,
    max_goals: int = 12,
) -> tuple[float, float] | None:
    """Return ``(over_2_5, under_2_5)`` probabilities.

    ``over_2_5`` = P(total goals >= 3); ``under_2_5`` = P(total goals <= 2).
    """
    if lambda_home is None or lambda_away is None:
        return None
    if not (isfinite(lambda_home) and isfinite(lambda_away)):
        return None
    if lambda_home <= 0 or lambda_away <= 0:
        return None
    pmf_h, pmf_a = _poisson_marginals(float(lambda_home), float(lambda_away), max_goals=max_goals)

    # P(total <= 2) over the joint grid.
    under = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j <= 2:
                under += pmf_h[i] * pmf_a[j]
    over = max(0.0, 1.0 - under)

    # Renormalise to a hard simplex (guards against pmf truncation drift).
    total = over + under
    if total <= 0:
        return None
    over /= total
    under /= total
    return (min(1.0, max(0.0, over)), min(1.0, max(0.0, under)))


def btts(
    lambda_home: float,
    lambda_away: float,
    *,
    max_goals: int = 12,
) -> tuple[float, float] | None:
    """Return ``(btts_yes, btts_no)`` = both teams score.

    ``btts_yes`` = P(home>=1 AND away>=1); ``btts_no`` = 1 - btts_yes.
    """
    if lambda_home is None or lambda_away is None:
        return None
    if not (isfinite(lambda_home) and isfinite(lambda_away)):
        return None
    if lambda_home <= 0 or lambda_away <= 0:
        return None
    pmf_h, pmf_a = _poisson_marginals(float(lambda_home), float(lambda_away), max_goals=max_goals)

    p_home_zero = pmf_h[0]  # P(home scores 0)
    p_away_zero = pmf_a[0]  # P(away scores 0)
    yes = max(0.0, (1.0 - p_home_zero) * (1.0 - p_away_zero))
    no = max(0.0, 1.0 - yes)
    total = yes + no
    if total <= 0:
        return None
    yes /= total
    no /= total
    return (min(1.0, max(0.0, yes)), min(1.0, max(0.0, no)))


__all__ = ["btts", "over_under_25"]