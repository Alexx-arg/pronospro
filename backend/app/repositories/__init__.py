"""Async repositories built on top of :class:`AsyncSession`.

Repositories in Phase 2 cover the basic CRUD primitives required to run the
integration test suite and to back the upcoming services. They deliberately
**never commit**: the caller (FastAPI dependency ``get_session`` or the
``session_scope`` context manager) owns the transaction. This keeps commit
semantics explicit and makes testing easier.

The :class:`PredictionRepository` intentionally exposes ONLY ``insert`` and
read methods — no ``update`` / ``delete`` — honoring the immutability
contract (SCHEMA.md §11). Even if someone calls
``session.delete(prediction)``, the database trigger
``trg_no_delete_predictions`` will reject it with an exception.
"""

from __future__ import annotations

from app.repositories.base import BaseRepository
from app.repositories.competition import CompetitionRepository
from app.repositories.fixture import FixtureRepository
from app.repositories.injury import InjuryRepository
from app.repositories.lineup import LineupRepository
from app.repositories.model_version import ModelVersionRepository
from app.repositories.player import PlayerRepository
from app.repositories.player_statistics import PlayerStatisticsRepository
from app.repositories.prediction import PredictionRepository
from app.repositories.prediction_outcome import PredictionOutcomeRepository
from app.repositories.season import SeasonRepository
from app.repositories.team import TeamRepository
from app.repositories.team_statistics import TeamStatisticsRepository

__all__ = [
    "BaseRepository",
    "CompetitionRepository",
    "FixtureRepository",
    "InjuryRepository",
    "LineupRepository",
    "ModelVersionRepository",
    "PlayerRepository",
    "PlayerStatisticsRepository",
    "PredictionOutcomeRepository",
    "PredictionRepository",
    "SeasonRepository",
    "TeamRepository",
    "TeamStatisticsRepository",
]
