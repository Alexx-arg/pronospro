"""Elo baseline predictor (Sprint 5.5).

Consumes ONLY the three CSV Elo features (Phase 4) and maps them to
1X2 via Poisson rates and the shared ``poisson_to_1x2`` conversor.

Formula (§11.1):
    D = (R_home + HFA) - R_away
    log λ_home = β0_home + β1·D
    log λ_away = β0_away - β1·D
    (λ_home, λ_away) → poisson_to_1x2 → MatchProbabilities

No DB, no recomputation of Elo ratings (K stored but not used).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.features.example import (
    AWAY_ELO_PRE_MATCH,
    HOME_ELO_PRE_MATCH,
)
from app.prediction.contracts import FixtureFeatures, MatchProbabilities
from app.prediction.models._poisson import poisson_to_1x2


@dataclass(frozen=True, slots=True)
class EloBaselineParams:
    """Hyperparameters for the Elo baseline.

    Attributes:
        K: Elo update factor (stored, not used in prediction — see spec).
        HFA: Home field advantage added to R_home before diff.
        beta_0_home: Intercept for log λ_home.
        beta_0_away: Intercept for log λ_away.
        beta_1: Sensitivity of log λ to Elo difference.
        max_goals: Cap for Poisson matrix.
    """

    K: int
    HFA: float
    beta_0_home: float
    beta_0_away: float
    beta_1: float
    max_goals: int = 10


class EloBaselineModel:
    """Stateless predictor — all state lives in :class:`EloBaselineParams`."""

    def __init__(self, params: EloBaselineParams) -> None:
        self.params: EloBaselineParams = params
        # Pre-resolve feature indices for fast lookup — but we look up
        # by name at predict time to stay robust to feature order changes
        # validated via FixtureFeatures.feature_names.

    def _extract_elos(self, features: FixtureFeatures) -> tuple[float, float]:
        """Extract R_home, R_away from FixtureFeatures.

        Raises:
            ValueError: if any required feature is NaN/missing or not found.
        """
        names = features.feature_names
        vec = features.feature_vector
        try:
            idx_home = names.index(HOME_ELO_PRE_MATCH)
            idx_away = names.index(AWAY_ELO_PRE_MATCH)
        except ValueError as exc:
            raise ValueError(f"Elo baseline requires {HOME_ELO_PRE_MATCH} and {AWAY_ELO_PRE_MATCH}") from exc

        r_home = float(vec[idx_home])  # type: ignore[index]
        r_away = float(vec[idx_away])  # type: ignore[index]
        if not (np.isfinite(r_home) and np.isfinite(r_away)):
            raise ValueError(f"Elo features contain NaN/inf: R_home={r_home}, R_away={r_away}")
        return r_home, r_away

    def predict(self, features: FixtureFeatures) -> MatchProbabilities:
        """Predict 1X2 for one fixture."""
        r_home, r_away = self._extract_elos(features)
        p = self.params
        elo_diff = (r_home + p.HFA) - r_away
        log_lam_home = p.beta_0_home + p.beta_1 * elo_diff
        log_lam_away = p.beta_0_away - p.beta_1 * elo_diff
        # Cap log λ per spec to prevent overflow
        if log_lam_home > 30:
            log_lam_home = 30.0
        if log_lam_away > 30:
            log_lam_away = 30.0
        # Also floor to avoid exp(-inf) for very negative; but keep as is
        lam_home = math.exp(log_lam_home)
        lam_away = math.exp(log_lam_away)
        # poisson_to_1x2 validates lambdas
        ph, pd, pa, d_home, d_away = poisson_to_1x2(
            lam_home, lam_away, max_goals=p.max_goals
        )
        return MatchProbabilities(
            p_home_win=ph,
            p_draw=pd,
            p_away_win=pa,
            p_home_goals=d_home,
            p_away_goals=d_away,
        )

    def predict_proba_array(
        self,
        features_list: list[FixtureFeatures],
    ) -> Any:
        """Batch helper returning (n,3) array."""
        n = len(features_list)
        arr = np.zeros((n, 3), dtype=np.float64)
        for i, f in enumerate(features_list):
            mp = self.predict(f)
            arr[i, 0] = mp.p_home_win
            arr[i, 1] = mp.p_draw
            arr[i, 2] = mp.p_away_win
        return arr


__all__ = ["EloBaselineModel", "EloBaselineParams"]
