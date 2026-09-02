"""Predictor implementations per model.

Sprint 5.0 reserve: package skeleton only.
* ``elo_baseline.py`` (Sprint 5.5) — "Poisson-Elo" baseline.
* ``poisson.py`` (Sprint 5.6) — regressor per head + shared
  ``poisson_to_1x2`` conversor.
* ``gradient_boosting.py`` (Sprint 5.7) — multiclass 1X2 classifier.
* ``lightgbm.py`` (Sprint 5.11) — LightGBM multiclass, NaN passthrough.
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = ["MODEL_REGISTRY", "get_model_class"]


def get_model_class(model_name: str) -> Any:
    """Factory de modelos por string de configuración.

    Soporta ``elo_baseline``, ``poisson``, ``gradient_boosting`` y
    ``lightgbm`` (alias Sprint 5.11). Lazy import para evitar ciclos.
    """
    name = model_name.strip().lower()
    if name == "elo_baseline":
        from app.prediction.models.elo_baseline import EloBaselineModel

        return EloBaselineModel
    if name == "poisson":
        from app.prediction.models.poisson import PoissonModel

        return PoissonModel
    if name in ("gradient_boosting", "gb", "histgb"):
        from app.prediction.models.gradient_boosting import GradientBoostingModel

        return GradientBoostingModel
    if name == "lightgbm":
        from app.prediction.models.lightgbm import LightGBMModel

        return LightGBMModel
    raise ValueError(f"model_name desconocido: {model_name!r}")


MODEL_REGISTRY: dict[str, Any] = {
    "elo_baseline": "app.prediction.models.elo_baseline:EloBaselineModel",
    "poisson": "app.prediction.models.poisson:PoissonModel",
    "gradient_boosting": "app.prediction.models.gradient_boosting:GradientBoostingModel",
    "lightgbm": "app.prediction.models.lightgbm:LightGBMModel",
}
