"""Pydantic schemas for the fixtures API (PronosPro v2)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TeamInfo(BaseModel):
    id: int
    external_id: int
    name: str
    short_name: str | None = None
    code: str | None = None
    country: str | None = None
    logo: str | None = None
    venue: str | None = None


class CompetitionInfo(BaseModel):
    id: int
    external_id: int
    name: str
    logo: str | None = None


class SeasonInfo(BaseModel):
    id: int
    external_id: int
    year: int


class MatchMetrics(BaseModel):
    """Quantitative match metrics from TeamStatistics (as-of kickoff)."""

    home_form: str | None = None
    away_form: str | None = None
    home_xg: float | None = None
    home_xga: float | None = None
    away_xg: float | None = None
    away_xga: float | None = None
    home_corners_avg: float | None = None
    away_corners_avg: float | None = None
    home_yellow_cards_avg: float | None = None
    away_yellow_cards_avg: float | None = None
    home_red_cards_avg: float | None = None
    away_red_cards_avg: float | None = None
    home_possession_avg: float | None = None
    away_possession_avg: float | None = None


class PredictionInfo(BaseModel):
    id: int
    model_version: str
    home_probability: float = Field(..., ge=0, le=1)
    draw_probability: float = Field(..., ge=0, le=1)
    away_probability: float = Field(..., ge=0, le=1)
    expected_home_goals: float | None = None
    expected_away_goals: float | None = None
    confidence: float | None = None
    created_at: datetime


class W1X2Market(BaseModel):
    """Mercado 1X2 — probabilidades del modelo LightGBM."""

    home: float = Field(..., ge=0, le=1)
    draw: float = Field(..., ge=0, le=1)
    away: float = Field(..., ge=0, le=1)
    model_version: str | None = None


class OverUnder25Market(BaseModel):
    """Mercado goles: Over/Under 2.5."""

    over_2_5: float = Field(..., ge=0, le=1)
    under_2_5: float = Field(..., ge=0, le=1)


class BttsMarket(BaseModel):
    """Mercado goles: BTTS (ambos anotan)."""

    yes: float = Field(..., ge=0, le=1)
    no: float = Field(..., ge=0, le=1)


class GoalsMarket(BaseModel):
    """Bloque de mercados de goles."""

    over_under_2_5: OverUnder25Market | None = None
    btts: BttsMarket | None = None


class Markets(BaseModel):
    """Todos los mercados disponibles para un fixture."""

    w1x2: W1X2Market | None = None
    goals: GoalsMarket | None = None
    metrics: MatchMetrics | None = None


class FixtureResponse(BaseModel):
    id: int
    external_id: int
    competition: CompetitionInfo
    season: SeasonInfo
    home_team: TeamInfo
    away_team: TeamInfo
    kickoff_time: datetime
    status: str
    status_short: str | None = None
    venue: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    markets: Markets | None = None


class PaginatedFixtures(BaseModel):
    items: list[FixtureResponse]
    limit: int
    offset: int
    total: int


__all__ = [
    "BttsMarket",
    "CompetitionInfo",
    "FixtureResponse",
    "GoalsMarket",
    "Markets",
    "MatchMetrics",
    "OverUnder25Market",
    "PaginatedFixtures",
    "PredictionInfo",
    "SeasonInfo",
    "TeamInfo",
    "W1X2Market",
]