"""``DataProvider`` Protocol — the pluggable integration contract.

Per ``docs/ARCHITECTURE.md`` §2 the rest of the backend depends ONLY on
this Protocol and the DTOs in :mod:`app.providers.dto`. Concrete adapters
(:mod:`app.providers.api_football`) implement the contract; the registry
(:mod:`app.providers.registry`) selects one via the ``DATA_PROVIDER`` env
var.

Methods return DTOs (or lists thereof); never raw HTTP payloads. Failures
must be typed :class:`app.providers.exceptions.ProviderError` subclasses.

Crystal contract discipline:
* **No** method here ever returns the provider's identifier under
  something other than ``external_id``.
* **No** method here ever leaks credentials: the DTOs do not have any
  key field.
* **No** method here ever writes to the database. Persistence is the
  repository layer's responsibility.

The Protocol is intentionally async to play well with httpx/asyncio and
with the existing async SQLAlchemy session.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, runtime_checkable

from app.providers.dto import (
    CompetitionDTO,
    FixtureDTO,
    InjuryDTO,
    LeagueSeasonDTO,
    LineupDTO,
    PlayerDTO,
    PlayerStatisticsDTO,
    SeasonDTO,
    TeamDTO,
    TeamStatisticsDTO,
)


@runtime_checkable
class DataProvider(Protocol):
    """Pluggable data-provider contract.

    All methods are ``async``. Implementations must be safe to call
    concurrently from multiple sync services (the HTTP client is
    responsible for rate limiting + retries).
    """

    name: str

    # ----- League / season metadata --------------------------------------
    async def fetch_leagues(
        self,
        *,
        league_ids: list[int],
    ) -> list[CompetitionDTO]:
        """Fetch competition metadata for the given league ids."""
        ...

    async def fetch_seasons(
        self,
        *,
        league_id: int,
    ) -> list[SeasonDTO]:
        """Fetch the seasons known for a competition."""
        ...

    # ----- Fixtures --------------------------------------------------------
    async def fetch_upcoming_fixtures(
        self,
        *,
        league_id: int,
        season_year: int,
        date_from: date,
        date_to: date,
    ) -> list[FixtureDTO]:
        """Fetch fixtures whose kickoff falls inside ``[date_from, date_to]``.

        Statuses returned are typically ``scheduled``, ``in_play`` and may
        include ``finished`` matches that happened within the window.
        """
        ...

    async def fetch_finished_fixtures(
        self,
        *,
        league_id: int,
        season_year: int,
        date_from: date,
        date_to: date,
    ) -> list[FixtureDTO]:
        """Fetch fixtures that have already finished within the window.

        The sync service uses this to back-fill scores. The DTO carries
        the final goals and ``finished_at``.
        """
        ...

    # ----- Teams -----------------------------------------------------------
    async def fetch_teams(
        self,
        *,
        league_id: int,
        season_year: int,
    ) -> list[TeamDTO]:
        """Fetch teams participating in a competition+season."""
        ...

    async def fetch_team_statistics(
        self,
        *,
        team_id: int,
        league_id: int,
        season_year: int,
        as_of_date: date,
    ) -> TeamStatisticsDTO | None:
        """Fetch aggregated team stats for one season.

        Returns ``None`` if the provider doesn't have stats for this
        combination (e.g. the team is not in the league that season).
        """
        ...

    # ----- Players ---------------------------------------------------------
    async def fetch_players(
        self,
        *,
        team_id: int,
        season_year: int,
    ) -> list[PlayerDTO]:
        """Fetch the squad/roster of a team for a season."""
        ...

    async def fetch_player_statistics(
        self,
        *,
        player_id: int,
        league_id: int,
        season_year: int,
    ) -> PlayerStatisticsDTO | None:
        """Fetch aggregated player stats.

        Returns ``None`` if the provider has no stats for this combination.
        """
        ...

    # ----- Injuries & lineups ---------------------------------------------
    async def fetch_injuries(
        self,
        *,
        league_id: int,
        season_year: int,
    ) -> list[InjuryDTO]:
        """Fetch current and historical injuries of a competition's season."""
        ...

    async def fetch_lineup(
        self,
        *,
        fixture_id: int,
    ) -> list[LineupDTO]:
        """Fetch the lineups (home + away) of a fixture.

        Returns a list of :class:`LineupDTO` of length 2 (home + away)
        when both are available. Empty list when the lineup is not yet
        published (typical before kick-off).
        """
        ...

    async def close(self) -> None:
        """Release HTTP resources owned by this provider instance."""
        ...


__all__ = ["DataProvider", "LeagueSeasonDTO"]
