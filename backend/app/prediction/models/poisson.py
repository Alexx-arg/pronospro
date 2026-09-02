"""Poisson model — two independent Poisson heads → 1X2 (Sprint 5.6).

Uses the shared ``poisson_to_1x2`` conversor from ``_poisson.py``.
Follows ``docs/PHASE_5.md`` §12.2 partition with Option A (8 shared).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import PoissonRegressor  # type: ignore[import-untyped]

from app.features.example import FEATURE_NAMES
from app.prediction.contracts import FixtureFeatures, MatchProbabilities
from app.prediction.models._poisson import poisson_to_1x2

# ---------------------------------------------------------------------------
# Feature partition — Option A (37 / 37, overlap 8, total 66)
# ---------------------------------------------------------------------------

HEAD_HOME_FEATURES: tuple[str, ...] = (
    # Forma local (11)
    "home_wins_last_3",
    "home_draws_last_3",
    "home_losses_last_3",
    "home_wins_last_5",
    "home_draws_last_5",
    "home_losses_last_5",
    "home_wins_last_10",
    "home_draws_last_10",
    "home_losses_last_10",
    "home_points_last_5",
    "home_points_last_10",
    # Goles local (8)
    "home_goals_for_last_5",
    "home_goals_against_last_5",
    "home_goals_for_last_10",
    "home_goals_against_last_10",
    "home_goals_for_mean_last_5",
    "home_goals_against_mean_last_5",
    "home_goals_for_mean_last_10",
    "home_goals_against_mean_last_10",
    # Home/away split (3)
    "home_home_points_last_5",
    "home_home_goals_for_last_5",
    "home_home_goals_against_last_5",
    # Standings local (3)
    "home_table_position",
    "home_points_table",
    "home_goal_difference_table",
    # Rest (1)
    "home_rest_days",
    # Elo (2) — elo_difference is shared
    "home_elo_pre_match",
    "elo_difference",
    # xG local (2)
    "home_xg_season_asof",
    "home_xga_season_asof",
    # H2H (5) — shared
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "h2h_total_goals_mean",
    "h2h_sample_size",
    # Tabla diffs (2) — shared per Option A
    "table_points_difference",
    "table_goal_difference_difference",
)

HEAD_AWAY_FEATURES: tuple[str, ...] = (
    # Forma visitante (11)
    "away_wins_last_3",
    "away_draws_last_3",
    "away_losses_last_3",
    "away_wins_last_5",
    "away_draws_last_5",
    "away_losses_last_5",
    "away_wins_last_10",
    "away_draws_last_10",
    "away_losses_last_10",
    "away_points_last_5",
    "away_points_last_10",
    # Goles visitante (8)
    "away_goals_for_last_5",
    "away_goals_against_last_5",
    "away_goals_for_last_10",
    "away_goals_against_last_10",
    "away_goals_for_mean_last_5",
    "away_goals_against_mean_last_5",
    "away_goals_for_mean_last_10",
    "away_goals_against_mean_last_10",
    # Away split (3)
    "away_away_points_last_5",
    "away_away_goals_for_last_5",
    "away_away_goals_against_last_5",
    # Standings visitante (3)
    "away_table_position",
    "away_points_table",
    "away_goal_difference_table",
    # Rest (1)
    "away_rest_days",
    # Elo (2) — elo_difference shared
    "away_elo_pre_match",
    "elo_difference",
    # xG visitante (2)
    "away_xg_season_asof",
    "away_xga_season_asof",
    # H2H (5) — shared
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "h2h_total_goals_mean",
    "h2h_sample_size",
    # Tabla diffs (2) — shared
    "table_points_difference",
    "table_goal_difference_difference",
)

# The 8 shared features (must be exactly those allowed)
SHARED_FEATURES: tuple[str, ...] = (
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "h2h_total_goals_mean",
    "h2h_sample_size",
    "elo_difference",
    "table_points_difference",
    "table_goal_difference_difference",
)


def _validate_partition() -> None:
    """Run at import time — fail fast if spec drifts."""
    # All must be in FEATURE_NAMES
    feat_set = set(FEATURE_NAMES)
    for name in HEAD_HOME_FEATURES:
        if name not in feat_set:
            raise ValueError(f"HEAD_HOME contains unknown feature {name!r}")
    for name in HEAD_AWAY_FEATURES:
        if name not in feat_set:
            raise ValueError(f"HEAD_AWAY contains unknown feature {name!r}")
    # No targets
    from app.features.example import TARGET_NAMES

    for name in HEAD_HOME_FEATURES + HEAD_AWAY_FEATURES:
        if name in set(TARGET_NAMES):
            raise ValueError(f"feature partition contains target {name!r}")
    # Disjoint except shared (8)
    set_home = set(HEAD_HOME_FEATURES)
    set_away = set(HEAD_AWAY_FEATURES)
    overlap = set_home & set_away
    if overlap != set(SHARED_FEATURES):
        raise ValueError(
            f"overlap must be exactly SHARED_FEATURES {SHARED_FEATURES}, got {sorted(overlap)}"
        )
    # Coverage 37+37-8=66
    union = set_home | set_away
    if len(union) != len(FEATURE_NAMES) or union != feat_set:
        raise ValueError(
            f"partition must cover exactly FEATURE_NAMES (66): "
            f"union len {len(union)} vs FEATURE_NAMES {len(feat_set)}; "
            f"missing {sorted(feat_set - union)} extra {sorted(union - feat_set)}"
        )
    if len(HEAD_HOME_FEATURES) != 37 or len(HEAD_AWAY_FEATURES) != 37:
        raise ValueError("each head must have 37 features")


_validate_partition()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PoissonParams:
    """Fitted Poisson rates are inside sklearn regressors; this holds meta."""

    max_goals: int
    regularization_l2: float
    # The regressors themselves are not stored here (they are payload)
    # but we keep feature lists for validation
    train_means_home: tuple[float, ...]
    train_means_away: tuple[float, ...]


class PoissonModel:
    """Two-head Poisson predictor."""

    def __init__(
        self,
        reg_home: PoissonRegressor,
        reg_away: PoissonRegressor,
        train_means_home: dict[str, float],
        train_means_away: dict[str, float],
        *,
        max_goals: int = 10,
    ) -> None:
        self.reg_home: PoissonRegressor = reg_home
        self.reg_away: PoissonRegressor = reg_away
        self.train_means_home: dict[str, float] = dict(train_means_home)
        self.train_means_away: dict[str, float] = dict(train_means_away)
        self.max_goals: int = int(max_goals)
        # Cache index maps for fast lookup
        self._idx_home: dict[str, int] = {}
        self._idx_away: dict[str, int] = {}

    def _vector_for_head(
        self,
        features: FixtureFeatures,
        head: str,
    ) -> Any:
        """Extract feature vector for a head, imputing NaN with train mean."""
        if head == "home":
            wanted = HEAD_HOME_FEATURES
            means = self.train_means_home
        else:
            wanted = HEAD_AWAY_FEATURES
            means = self.train_means_away

        # Build index map lazily per features.feature_names (usually FEATURE_NAMES)
        # We look up by name each time to stay robust to order changes.
        vals: list[float] = []
        for name in wanted:
            try:
                idx = features.feature_names.index(name)
            except ValueError as exc:
                raise ValueError(f"PoissonModel requires feature {name!r}") from exc
            v = float(features.feature_vector[idx])  # type: ignore[index]
            if not np.isfinite(v):  # NaN or inf → impute
                v = float(means.get(name, 0.0))
            vals.append(v)
        arr = np.asarray(vals, dtype=np.float64).reshape(1, -1)  # type: ignore[attr-defined]
        return arr

    def _predict_lambda(
        self,
        features: FixtureFeatures,
        head: str,
    ) -> float:
        vec = self._vector_for_head(features, head)
        reg = self.reg_home if head == "home" else self.reg_away
        lam = float(reg.predict(vec)[0])
        # PoissonRegressor guarantees >=0, but ensure >0 for poisson_to_1x2
        if not np.isfinite(lam) or lam <= 0:
            lam = 0.1  # minimal positive
        # Cap log λ per spec (poisson_to_1x2 also caps)
        if lam > 1e6:
            lam = 1e6
        return float(lam)

    def predict(self, features: FixtureFeatures) -> MatchProbabilities:
        """Predict 1X2 via Poisson rates → poisson_to_1x2."""
        from app.prediction.contracts import MatchProbabilities

        lam_home = self._predict_lambda(features, "home")
        lam_away = self._predict_lambda(features, "away")
        p_home, p_draw, p_away, d_home, d_away = poisson_to_1x2(
            lam_home, lam_away, max_goals=self.max_goals
        )
        return MatchProbabilities(
            p_home_win=p_home,
            p_draw=p_draw,
            p_away_win=p_away,
            p_home_goals=d_home,
            p_away_goals=d_away,
        )

    def predict_proba_array(self, features_list: list[FixtureFeatures]) -> Any:
        """Batch helper for metrics."""
        n = len(features_list)
        arr = np.zeros((n, 3), dtype=np.float64)
        for i, f in enumerate(features_list):
            mp = self.predict(f)
            arr[i, 0] = mp.p_home_win
            arr[i, 1] = mp.p_draw
            arr[i, 2] = mp.p_away_win
        return arr


__all__ = [
    "HEAD_AWAY_FEATURES",
    "HEAD_HOME_FEATURES",
    "PoissonModel",
    "PoissonParams",
    "SHARED_FEATURES",
]
