"""Sync sub-package: provider → repository orchestration.

Public surface (used by :mod:`app.tasks.sync_*` and tests):

* :func:`sync_competitions_and_seasons`
* :func:`sync_upcoming_fixtures`
* :func:`sync_finished_fixtures`
* :func:`sync_teams`
* :func:`sync_team_statistics`
* :func:`sync_player_statistics`
* :func:`sync_injuries`
* :func:`sync_lineups`
* :class:`SyncMetrics`
"""

from __future__ import annotations

from app.services.sync.fixtures import (
    sync_finished_fixtures,
    sync_upcoming_fixtures,
)
from app.services.sync.injuries import sync_injuries
from app.services.sync.lineups import sync_lineups
from app.services.sync.metadata import (
    current_season_for,
    sync_competitions_and_seasons,
)
from app.services.sync.metrics import SyncMetrics
from app.services.sync.player_statistics import sync_player_statistics
from app.services.sync.team_statistics import sync_team_statistics
from app.services.sync.teams import sync_teams

__all__ = [
    "SyncMetrics",
    "current_season_for",
    "sync_competitions_and_seasons",
    "sync_finished_fixtures",
    "sync_injuries",
    "sync_lineups",
    "sync_player_statistics",
    "sync_team_statistics",
    "sync_teams",
    "sync_upcoming_fixtures",
]
