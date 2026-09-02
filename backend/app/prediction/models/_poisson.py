"""Shared Poisson → 1X2 conversor (Sprint 5.5).

Unique implementation for Elo baseline (Sprint 5.5) and Poisson (Sprint 5.6).
See ``docs/PHASE_5.md`` §11.1 / §12.3-12.5.

Formula:
    P(i, j) = Poisson(i; λ_home) · Poisson(j; λ_away)  for 0 ≤ i,j ≤ M
    P(home) = Σ_{i>j} P(i,j)
    P(draw) = Σ_k P(k,k)
    P(away) = 1 - P(home) - P(draw)  (cierre por resta)

Stability:
* ``log λ > 30`` → cap to 30 (prevents exp overflow).
* Poisson pmf in log-space: ``-λ + k·log λ - gammaln(k+1)``.
* Total probability mass check is done in tests, not here.
"""

from __future__ import annotations

import math

import numpy as np


def _poisson_log_pmf(k: int, lam: float) -> float:
    """Log PMF in stable log-space."""
    # Caller ensures lam > 0 and capped.
    return -lam + k * math.log(lam) - math.lgamma(k + 1)


def poisson_to_1x2(
    lambda_home: float,
    lambda_away: float,
    *,
    max_goals: int = 10,
) -> tuple[float, float, float, dict[int, float], dict[int, float]]:
    """Convert two Poisson rates to 1X2 and per-goal distributions.

    Args:
        lambda_home: Expected home goals (>0).
        lambda_away: Expected away goals (>0).
        max_goals: Cap for goal matrix (inclusive). Default 10.

    Returns:
        Tuple ``(p_home, p_draw, p_away, p_home_goals, p_away_goals)`` where
        ``p_home_goals[k]`` is Poisson(k; λ_home) for 0..max_goals,
        similarly for away. The 1X2 probs sum to 1 within 1e-9.

    Raises:
        ValueError: if lambdas not finite/positive or max_goals <1.
    """
    if not (np.isfinite(lambda_home) and np.isfinite(lambda_away)):
        raise ValueError(f"lambdas must be finite, got {lambda_home}, {lambda_away}")
    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError(f"lambdas must be >0, got {lambda_home}, {lambda_away}")
    if max_goals < 1 or max_goals > 30:
        raise ValueError(f"max_goals must be in [1,30], got {max_goals}")

    # Cap log λ to avoid overflow in exp(log λ) — per spec 12.5
    if math.log(lambda_home) > 30:
        lambda_home = math.exp(30)
    if math.log(lambda_away) > 30:
        lambda_away = math.exp(30)
    lam_h = float(lambda_home)
    lam_a = float(lambda_away)

    # Build pmf vectors for 0..max_goals in log-space then exp (return dicts)
    pmf_home: list[float] = []
    pmf_away: list[float] = []
    for k in range(max_goals + 1):
        lh = -lam_h + k * math.log(lam_h) - math.lgamma(k + 1)
        la = -lam_a + k * math.log(lam_a) - math.lgamma(k + 1)
        pmf_home.append(math.exp(lh))
        pmf_away.append(math.exp(la))

    p_home_goals = {k: pmf_home[k] for k in range(max_goals + 1)}
    p_away_goals = {k: pmf_away[k] for k in range(max_goals + 1)}

    # Effective max for 1X2 calculation — expand when lambdas large to capture tail
    eff_max = max_goals
    max_lam = max(lam_h, lam_a)
    if max_lam > max_goals:
        # Cover ~5 sigma beyond mean (Poisson sigma = sqrt(lam))
        eff_needed = int(math.ceil(max_lam + 5 * math.sqrt(max_lam)))
        eff_max = max(max_goals, min(eff_needed, 60))
        # Build extended pmfs for effective range (reuse log-space)
        pmf_home_eff: list[float] = []
        pmf_away_eff: list[float] = []
        for k in range(eff_max + 1):
            lh = -lam_h + k * math.log(lam_h) - math.lgamma(k + 1)
            la = -lam_a + k * math.log(lam_a) - math.lgamma(k + 1)
            pmf_home_eff.append(math.exp(lh))
            pmf_away_eff.append(math.exp(la))
    else:
        pmf_home_eff = pmf_home
        pmf_away_eff = pmf_away

    # Compute joint matrix symmetrically for 1X2
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for i in range(eff_max + 1):
        for j in range(eff_max + 1):
            p = pmf_home_eff[i] * pmf_away_eff[j]
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
    # Clamp tiny negatives due to floating error
    if p_away < 0 and p_away > -1e-12:
        p_away = 0.0
    if p_home < 0 and p_home > -1e-12:
        p_home = 0.0
    if p_draw < 0 and p_draw > -1e-12:
        p_draw = 0.0

    # Final renormalisation if needed to ensure sum ==1 within tolerance
    total = p_home + p_draw + p_away
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        if total > 1e-12:
            p_home /= total
            p_draw /= total
            p_away /= total
        else:
            p_away = 1.0 - p_home - p_draw

    # Validate simplex
    for v in (p_home, p_draw, p_away):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"poisson_to_1x2 produced out-of-range prob {v}")
    if not math.isclose(p_home + p_draw + p_away, 1.0, abs_tol=1e-9):
        raise ValueError("poisson_to_1x2 does not sum to 1")

    return p_home, p_draw, p_away, p_home_goals, p_away_goals


__all__ = ["poisson_to_1x2"]
