"""Frozen in-memory rows for feature computation.

Loading rows straight from SQLAlchemy ORM objects into the feature
calculator would risk (a) accidentally mutating state and (b) implicitly
lazy-loading relationships that re-hit the database. To keep the feature
layer pure and leak-proof, the :mod:`app.features.asof` loader **always
projects the ORM rows onto these small immutable dataclasses**.

Two reasons:

1. **No DB round-trips inside feature math.** Once projected, the
   rolling / h2h / standings / elo code only sees plain immutable
   values. There is no way an aggregation can trigger a SELECT.
2. **No accidental leakage.** The :class:`~app.models.fixture.Fixture`
   ORM exposes ``home_goals``/``away_goals`` (NULL until finished) and
   ``status``. The :class:`FixtureRow` intentionally only carries the
   columns the feature math **needs** and exposes whether the fixture is
   finished — never a back-reference.

Equality and ``__hash__`` are not implemented on purpose. The rows are
data carriers; we index them only by ``fixture_id`` externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class FixtureRow:
    """As-of view of a normalised ``fixtures`` row.

    A row is only created by :mod:`app.features.asof` for fixtures that
    are **already finished** (``home_goals`` and ``away_goals`` non-NULL,
    ``status='finished'``) — feature math never touches unfinished rows
    because they cannot contribute form / goals / standings.
    """

    fixture_id: int
    competition_id: int
    season_id: int
    home_team_id: int
    away_team_id: int
    kickoff_time: datetime
    home_goals: int
    away_goals: int
    matchday: int | None
    venue: str | None

    @property
    def kickoff_date(self) -> date:
        """Date part of the kickoff (used to compare against snapshots)."""
        return self.kickoff_time.date()

    def goals_for(self, team_id: int) -> int:
        """Goals scored by ``team_id`` in this fixture."""
        if team_id == self.home_team_id:
            return self.home_goals
        if team_id == self.away_team_id:
            return self.away_goals
        raise ValueError(f"team {team_id} not in fixture {self.fixture_id}")

    def goals_against(self, team_id: int) -> int:
        """Goals conceded by ``team_id`` in this fixture."""
        if team_id == self.home_team_id:
            return self.away_goals
        if team_id == self.away_team_id:
            return self.home_goals
        raise ValueError(f"team {team_id} not in fixture {self.fixture_id}")

    def is_home(self, team_id: int) -> bool:
        """``True`` iff ``team_id`` is the home team of this fixture."""
        return team_id == self.home_team_id

    def outcome_for(self, team_id: int) -> str:
        """``'W'`` / ``'D'`` / ``'L'`` for ``team_id``'s perspective."""
        gf = self.goals_for(team_id)
        ga = self.goals_against(team_id)
        if gf > ga:
            return "W"
        if gf == ga:
            return "D"
        return "L"

    def points_for(self, team_id: int) -> int:
        """3 / 1 / 0 from ``team_id``'s perspective."""
        return {"W": 3, "D": 1, "L": 0}[self.outcome_for(team_id)]


@dataclass(frozen=True, slots=True)
class TeamStatsRow:
    """As-of view of a ``team_statistics`` row.

    Only xG / xGA are needed by Phase 4 features for now
    (``home_xg_for_last_5`` etc.). Other columns from the table
    (possession, fouls, ...) are NOT projected yet: feature code only sees
    fields it actually uses, which keeps the missing-data semantics
    explicit.
    """

    team_id: int
    competition_id: int
    season_id: int
    as_of_date: date
    xg: float | None
    xga: float | None


__all__ = ["FixtureRow", "TeamStatsRow"]
