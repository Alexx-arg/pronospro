"""Pydantic schemas for inference API — Sprint 6.1."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MatchPredictionRequest(BaseModel):
    """Entrada mínima para predicción.

    Contiene los 6 campos de identidad + vector de 66 features.
    El vector puede contener ``None`` para missing (se convierte a
    ``np.nan`` antes de pasar al modelo). Si el front no construye
    las 66, puede enviar ``features`` dict como alternativa — pero
    para v1 exigimos el vector completo para mantener tipado estricto
    y evitar DB en la API (D8.1).
    """

    fixture_id: int = Field(..., ge=1, description="ID del fixture")
    kickoff: datetime = Field(..., description="Fecha/hora kickoff UTC")
    home_team_id: int = Field(..., ge=1)
    away_team_id: int = Field(..., ge=1)
    competition_id: int = Field(default=39, ge=1)
    season_id: int = Field(default=1, ge=1)
    feature_vector: list[float | None] = Field(
        ..., min_length=66, max_length=66, description="66 features en orden FEATURE_NAMES, None=missing"
    )
    feature_names: list[str] = Field(
        ..., min_length=66, max_length=66, description="66 nombres en orden canónico"
    )

    model_config = {"extra": "forbid"}


class MatchPredictionResponse(BaseModel):
    """Salida estructurada de probabilidades."""

    fixture_id: int
    prob_home: float = Field(..., ge=0, le=1)
    prob_draw: float = Field(..., ge=0, le=1)
    prob_away: float = Field(..., ge=0, le=1)
    model_version: str = Field(..., description="Versión del modelo cargado")
    predicted_at: datetime

    model_config = {"extra": "forbid"}


__all__ = ["MatchPredictionRequest", "MatchPredictionResponse"]
