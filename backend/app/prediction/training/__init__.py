"""Trainers: produce immutable ``ModelArtifact`` instances.

Sprint 5.0 reserve: package skeleton only
(see ``docs/PHASE_5.md`` §16 — Sprints 5.5 / 5.6 / 5.7).
Sprint 5.11 añade LightGBM sin romper 5.0–5.7.
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = ["TRAINER_REGISTRY", "get_trainer"]


def get_trainer(model_name: str, params: dict[str, Any] | None = None) -> Any:
    """Factory de trainers por string de configuración.

    Reconoce ``elo_baseline``, ``poisson``, ``gradient_boosting`` y
    ``lightgbm`` (Sprint 5.11). Lazy import.
    """
    name = model_name.strip().lower()
    if name == "elo_baseline":
        from app.prediction.training.elo_trainer import EloBaselineTrainer

        return EloBaselineTrainer(**(params or {}))
    if name == "poisson":
        from app.prediction.training.poisson_trainer import PoissonTrainer

        return PoissonTrainer(**(params or {}))
    if name in ("gradient_boosting", "gb", "histgb"):
        from app.prediction.training.gb_trainer import GradientBoostingTrainer

        return GradientBoostingTrainer(**(params or {}))
    if name == "lightgbm":
        from app.prediction.models.lightgbm import LightGBMTrainer

        return LightGBMTrainer(params)
    raise ValueError(f"model_name desconocido: {model_name!r}")


TRAINER_REGISTRY: dict[str, Any] = {
    "elo_baseline": "app.prediction.training.elo_trainer:EloBaselineTrainer",
    "poisson": "app.prediction.training.poisson_trainer:PoissonTrainer",
    "gradient_boosting": "app.prediction.training.gb_trainer:GradientBoostingTrainer",
    "lightgbm": "app.prediction.models.lightgbm:LightGBMTrainer",
}
