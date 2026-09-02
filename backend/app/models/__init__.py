"""SQLAlchemy ORM models, 1:1 mapping with ``docs/SCHEMA.md``.

Import order matters because relationships between models are resolved at
import time. Tables that depend on others are loaded after their parents.
The ``Base.metadata`` object ends up populated with every table, which
Alembic introspects in ``alembic/env.py``.
"""

from __future__ import annotations

from app.models.competition import Competition
from app.models.season import Season
from app.models.team import Team
from app.models.player import Player, PlayerTeamSeason
from app.models.fixture import Fixture
from app.models.team_statistics import TeamStatistics
from app.models.player_statistics import PlayerStatistics
from app.models.injury import Injury
from app.models.lineup import Lineup, LineupPlayer
from app.models.model_version import ModelVersion
from app.models.prediction import (
    Prediction,
    PredictionExplanationPayload,
)
from app.models.prediction_explanation import PredictionExplanation
from app.models.prediction_outcome import PredictionOutcome

__all__ = [
    "Competition",
    "Fixture",
    "Injury",
    "Lineup",
    "LineupPlayer",
    "ModelVersion",
    "Player",
    "PlayerStatistics",
    "PlayerTeamSeason",
    "Prediction",
    "PredictionExplanation",
    "PredictionExplanationPayload",
    "PredictionOutcome",
    "Season",
    "Team",
    "TeamStatistics",
]
