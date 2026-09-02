"""Internal normalised DTOs returned by data providers.

These DTOs are the **only** shape the rest of the backend expects from the
data layer. They are **provider-agnostic**: changing API-Football for
another source only requires translating the new provider's payload into
these models inside ``app/providers/<new_provider>.py``.

Conventions
------------
* Every DTO is a Pydantic v2 ``BaseModel``. Validation runs in the adapter
  (the provider's adapter is responsible for building the DTO; if a field
  is missing from the upstream payload and is required, an
  :class:`InvalidProviderResponse` is raised and the offending item is
  skipped — the sync service counts it as ``failed``).
* Identifiers upstream are kept under ``external_id`` (the provider's
  internal integer id). Internal DB ids are NEVER present in these DTOs:
  mapping ``external_id -> internal id`` is the repository's job.
* All datetimes are timezone-aware UTC. When the provider gives a naive
  datetime plus a tz string, the adapter normalises to UTC and AL SO keeps
  the original tz name under ``kickoff_timezone`` so DB consumers can
  reconstruct the local time if needed.
* Optional fields are ``... | None`` with default ``None``; the docstring
  of the field explains when ``None`` is expected. We never invent data:
  a missing optional stays ``None``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------
CompetitionType: TypeAlias = Literal["league", "cup", "playoff", "super_cup"]
PlayerPosition: TypeAlias = Literal["GK", "DF", "MF", "FW"]
FixtureStatus: TypeAlias = Literal[
    "scheduled",
    "in_play",
    "finished",
    "postponed",
    "cancelled",
    "suspended",
]
InjuryStatus: TypeAlias = Literal["active", "doubtful", "recovered", "suspended"]
MatchResult: TypeAlias = Literal["home", "draw", "away"]


# ---------------------------------------------------------------------------
# Competition / Season
# ---------------------------------------------------------------------------
class CompetitionDTO(BaseModel):
    """Football competition (league, cup, ...)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    external_id: int
    name: str
    type: CompetitionType = "league"
    country: str | None = None
    logo: str | None = None


class SeasonDTO(BaseModel):
    """Season within a competition."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    competition_external_id: int
    external_id: int
    year: int
    start_date: date | None = None
    end_date: date | None = None
    is_current: bool = False


# ---------------------------------------------------------------------------
# Team / Player
# ---------------------------------------------------------------------------
class TeamDTO(BaseModel):
    """A football team."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    external_id: int
    name: str
    short_name: str | None = None
    code: str | None = None
    country: str | None = None
    logo: str | None = None
    venue: str | None = None
    founded: int | None = None


class PlayerDTO(BaseModel):
    """A football player.

    Some providers expose partial info (e.g. lineups only carry id/name/photo).
    All optional fields default to ``None`` so partial players are still valid
    DTOs; the repository layer is responsible for merging new attributes onto
    already-known players (idempotent upsert).
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    external_id: int
    name: str
    photo: str | None = None
    nationality: str | None = None
    birth_date: date | None = None
    height_cm: int | None = None
    weight_kg: int | None = None
    position: PlayerPosition | None = None


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
class FixtureDTO(BaseModel):
    """A single fixture, either upcoming or finished.

    ``kickoff_time`` is ALWAYS timezone-aware UTC. ``kickoff_timezone``
    carries the source tz string for display purposes (e.g. ``"Europe/London"``);
    when the provider does not supply one, it is ``None``.

    Statuses are normalised to the same vocabulary used by ``fixtures.status``
    in PostgreSQL (SCHEMA.md §5). Provider-specific short codes (e.g. ``"1H"``,
    ``"FT"``) are kept under ``status_short`` for debugging only.

    Goals are present only when the match has reached a stage that yields
    goals (in_play/finished). Absent goals must be represented as ``None``,
    never ``0`` — otherwise a 0-0 draw would be indistinguishable from
    "score not yet known".
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    external_id: int
    competition_external_id: int
    season_external_id: int
    season_year: int
    home_team_external_id: int
    away_team_external_id: int
    kickoff_time: datetime
    kickoff_timezone: str | None = None
    matchday: int | None = None
    round: str | None = None
    venue: str | None = None
    status: FixtureStatus = "scheduled"
    status_short: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def _goals_paired(self) -> "FixtureDTO":
        """Mirror the SCHEMA.md §5 CHECK constraint on goals pairing."""
        if (self.home_goals is None) != (self.away_goals is None):
            raise ValueError(
                "home_goals/away_goals must be both set or both None "
                f"(got home={self.home_goals}, away={self.away_goals})"
            )
        return self

    @property
    def is_finished(self) -> bool:
        """True iff the fixture has a final score."""
        return self.status == "finished" and self.home_goals is not None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class TeamStatisticsDTO(BaseModel):
    """Aggregated team stats for one (team, competition, season, as_of_date).

    ``as_of_date`` is the date at which these stats were valid (the provider
    endpoint is typically called once and reflects the cumulative season
    stats up to that day). Optional fields default to ``None`` because some
    providers don't expose xG/xGA for every competition.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    team_external_id: int
    competition_external_id: int
    season_external_id: int
    as_of_date: date

    fixtures_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    clean_sheets: int = 0
    failed_to_score: int = 0
    form: str | None = None
    shots_total: int | None = None
    shots_on_target: int | None = None
    shots_inside_box: int | None = None
    shots_outside_box: int | None = None
    fouls: int | None = None
    corners: int | None = None
    offsides: int | None = None
    possession_avg: float | None = None
    yellow_cards: int | None = None
    red_cards: int | None = None
    passes_total: int | None = None
    passes_accuracy: float | None = None
    xg: float | None = None
    xga: float | None = None


class PlayerStatisticsDTO(BaseModel):
    """Aggregated player stats for one (player, team, competition, season)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    player_external_id: int
    team_external_id: int
    competition_external_id: int
    season_external_id: int
    appearances: int = 0
    starts: int = 0
    minutes_played: int = 0
    goals: int = 0
    assists: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    rating: float | None = None


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------
class InjuryDTO(BaseModel):
    """Player injury / suspension."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    external_id: int
    player_external_id: int
    team_external_id: int
    competition_external_id: int | None = None
    fixture_external_id: int | None = None
    type: str | None = None
    reason: str | None = None
    status: InjuryStatus = "active"
    start_date: date
    end_date: date | None = None
    updated_external_at: datetime | None = None


# ---------------------------------------------------------------------------
# Lineup
# ---------------------------------------------------------------------------
class LineupPlayerDTO(BaseModel):
    """A player slot in a lineup."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    player_external_id: int
    position: PlayerPosition | None = None
    position_x: int | None = None
    position_y: int | None = None
    shirt_number: int | None = None
    is_starter: bool = True


class LineupDTO(BaseModel):
    """A team's lineup for a fixture."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    fixture_external_id: int
    team_external_id: int
    is_home: bool
    formation: str | None = None
    coach: str | None = None
    updated_external_at: datetime | None = None
    players: list[LineupPlayerDTO] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Minimal lookup DTO used by the sync services to enumerate leagues+seasons
# ---------------------------------------------------------------------------
class LeagueSeasonDTO(BaseModel):
    """A ``(league_external_id, season_external_id, year)`` triple.

    Used by sync services to iterate over the configured leagues and their
    current season without pulling other entities.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    league_external_id: int
    season_external_id: int
    year: int


__all__ = [
    "CompetitionDTO",
    "CompetitionType",
    "FixtureDTO",
    "FixtureStatus",
    "InjuryDTO",
    "InjuryStatus",
    "LeagueSeasonDTO",
    "LineupDTO",
    "LineupPlayerDTO",
    "MatchResult",
    "PlayerDTO",
    "PlayerPosition",
    "PlayerStatisticsDTO",
    "SeasonDTO",
    "TeamDTO",
    "TeamStatisticsDTO",
]
