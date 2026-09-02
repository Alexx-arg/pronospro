# ruff: noqa: N806
"""Gradient Boosting model — HistGradientBoostingClassifier (Sprint 5.7).

Wraps sklearn.ensemble.HistGradientBoostingClassifier for 1X2 multiclass.
- Uses all 66 FEATURE_NAMES (no partition).
- NaN passthrough nativo (no imputación).
- p_home_goals / p_away_goals = None (no goles en v1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]

from app.features.example import FEATURE_NAMES
from app.prediction.contracts import FixtureFeatures, MatchProbabilities


class GradientBoostingModel:
    """Predictor wrapper around HistGradientBoostingClassifier."""

    def __init__(
        self,
        clf: HistGradientBoostingClassifier,
        *,
        feature_names: tuple[str, ...] = tuple(FEATURE_NAMES),
    ) -> None:
        self.clf: HistGradientBoostingClassifier = clf
        self.feature_names: tuple[str, ...] = feature_names
        # Build index map for fast lookup (feature_names order → vector)
        self._name_to_idx: dict[str, int] = {n: i for i, n in enumerate(feature_names)}

    def _vector_for_predict(self, features: FixtureFeatures) -> Any:
        """Extract 66-dim vector in model order, preserving NaN."""
        # Validate feature_names matches model
        if features.feature_names != self.feature_names and set(
            features.feature_names
        ) != set(self.feature_names):
            raise ValueError(
                f"GradientBoostingModel expects {self.feature_names[:2]}..., got {features.feature_names[:2]}"
            )
        vals: list[float] = []
        for name in self.feature_names:
            try:
                idx = features.feature_names.index(name)
            except ValueError as exc:
                raise ValueError(f"missing feature {name!r}") from exc
            v = float(features.feature_vector[idx])  # type: ignore[index]
            # Keep NaN as is (native HistGradientBoosting handling)
            vals.append(v)
        arr = np.asarray(vals, dtype=np.float64).reshape(1, -1)  # type: ignore[attr-defined]
        return arr

    def predict(self, features: FixtureFeatures) -> MatchProbabilities:
        vec = self._vector_for_predict(features)
        proba = self.clf.predict_proba(vec)[0]
        # proba shape (3,) in order of clf.classes_ which should be [0,1,2]
        # Ensure mapping by classes_
        classes = list(self.clf.classes_)
        # Build dict class->prob
        mapping = {int(c): float(p) for c, p in zip(classes, proba, strict=False)}
        p_home = mapping.get(0, 0.0)
        p_draw = mapping.get(1, 0.0)
        p_away = mapping.get(2, 0.0)
        # Handle missing class (if training had only 2 classes)
        # Fill uniform for missing? But should not happen with proper train.
        # Normalize to sum 1 if needed
        total = p_home + p_draw + p_away
        if not np.isclose(total, 1.0, atol=1e-6) and total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total
        # Clamp
        p_home = float(np.clip(p_home, 0.0, 1.0))
        p_draw = float(np.clip(p_draw, 0.0, 1.0))
        p_away = float(np.clip(p_away, 0.0, 1.0))
        s = p_home + p_draw + p_away
        if not np.isclose(s, 1.0, atol=1e-6):
            # Renormalize
            p_home /= s
            p_draw /= s
            p_away /= s
        return MatchProbabilities(
            p_home_win=p_home,
            p_draw=p_draw,
            p_away_win=p_away,
            p_home_goals=None,
            p_away_goals=None,
        )

    def predict_proba_array(self, features_list: list[FixtureFeatures]) -> Any:
        n = len(features_list)
        arr = np.zeros((n, 3), dtype=np.float64)
        for i, f in enumerate(features_list):
            mp = self.predict(f)
            arr[i, 0] = mp.p_home_win
            arr[i, 1] = mp.p_draw
            arr[i, 2] = mp.p_away_win
        return arr


__all__ = ["GradientBoostingModel"]
