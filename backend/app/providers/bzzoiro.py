"""Bzzoiro adapter — Sprint 7.x (reemplazo total de API-Football).

Este es el ÚNICO módulo que conoce el wire format de Bzzoiro Sports Data
(https://sports.bzzoiro.com/api/v2/). Consume httpx directamente con
`Authorization: Token` y retorna DTOs provider-agnostic.

Endpoints Bzzoiro (v2) usados:
  GET /api/v2/leagues/?limit=&offset=
  GET /api/v2/teams/?limit=&offset=&league=&season=
  GET /api/v2/events/?date_from=&date_to=&league=&status=&limit=&offset=
  GET /api/v2/events/{id}/  (detalle, no usado para lista)

Paginación: limit/offset, default 50, max 200. Fechas ISO date_from/date_to.
Auth: Header `Authorization: Token YOUR_API_KEY`.

Mapeo de status Bzzoiro → FixtureStatus:
  upcoming/scheduled/ns/tbd → scheduled
  live/1h/2h/in_play → in_play
  finished/ft/aet/pen → finished
  postponed/pst → postponed
  cancelled/canc/abd → cancelled
  suspended/susp/int → suspended
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Final

import httpx

from app.core.logging import get_logger
from app.providers.dto import (
    CompetitionDTO,
    FixtureDTO,
    FixtureStatus,
    InjuryDTO,
    LineupDTO,
    PlayerDTO,
    PlayerStatisticsDTO,
    SeasonDTO,
    TeamDTO,
    TeamStatisticsDTO,
)
from app.providers.exceptions import InvalidProviderResponse, ProviderError

_LOG = get_logger()

_BZZOIRO_STATUS_MAP: Final[dict[str, FixtureStatus]] = {
    "scheduled": "scheduled",
    "upcoming": "scheduled",
    "ns": "scheduled",
    "tbd": "scheduled",
    "notstarted": "scheduled",
    "not_started": "scheduled",
    "live": "in_play",
    "in_play": "in_play",
    "inplay": "in_play",
    "1h": "in_play",
    "2h": "in_play",
    "ht": "in_play",
    "et": "in_play",
    "p": "in_play",
    "finished": "finished",
    "ft": "finished",
    "aet": "finished",
    "pen": "finished",
    "postponed": "postponed",
    "pst": "postponed",
    "cancelled": "cancelled",
    "canc": "cancelled",
    "abd": "cancelled",
    "suspended": "suspended",
    "susp": "suspended",
    "int": "suspended",
}


def _map_status(raw: str | None) -> FixtureStatus:
    if not raw:
        return "scheduled"
    return _BZZOIRO_STATUS_MAP.get(raw.lower(), "scheduled")


class BzzoiroProvider:
    """DataProvider backed by Bzzoiro Sports Data v2."""

    name: Final[str] = "bzzoiro"

    def __init__(
        self,
        *,
        base_url: str = "https://sports.bzzoiro.com/api/v2",
        api_key: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("BzzoiroProvider requires a non-empty API key")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Token {self._api_key}", "Accept": "application/json"},
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        client = self._ensure_client()
        # Bzzoiro expects date_from/date_to as YYYY-MM-DD, league as id
        resp = await client.get(endpoint, params=params or {})
        if resp.status_code in (401, 403):
            from app.providers.exceptions import ProviderAuthError

            raise ProviderAuthError(f"Bzzoiro auth failed {resp.status_code}", details={"endpoint": endpoint})
        if resp.status_code == 429:
            from app.providers.exceptions import ProviderRateLimitError

            raise ProviderRateLimitError(f"Bzzoiro rate limited on {endpoint}", details={"endpoint": endpoint})
        if resp.status_code >= 500:
            from app.providers.exceptions import ProviderUnavailable

            raise ProviderUnavailable(f"Bzzoiro {resp.status_code} on {endpoint}", details={"endpoint": endpoint})
        if resp.status_code >= 400:
            from app.providers.exceptions import ProviderUnavailable

            raise ProviderUnavailable(f"Bzzoiro {resp.status_code} on {endpoint}", details={"endpoint": endpoint})
        try:
            return resp.json()
        except Exception as exc:
            raise InvalidProviderResponse(f"Non-JSON from {endpoint}", details={"endpoint": endpoint}) from exc

    # ----- League / season -----
    async def fetch_leagues(self, *, league_ids: list[int]) -> list[CompetitionDTO]:
        # Bzzoiro /leagues returns all; filter by ids
        data = await self._get("/leagues/", params={"limit": 200})
        results = data.get("results", []) if isinstance(data, dict) else []
        out: list[CompetitionDTO] = []
        wanted = set(league_ids)
        for entry in results:
            try:
                lid = int(entry["id"])
                if lid not in wanted:
                    continue
                out.append(
                    CompetitionDTO(
                        external_id=lid,
                        name=str(entry.get("name") or ""),
                        type="league",
                        country=entry.get("country"),
                        logo=None,
                    )
                )
            except Exception as exc:
                _LOG.warning("skip bzzoiro league entry: {}", str(exc)[:256])
                continue
        return out

    async def fetch_seasons(self, *, league_id: int) -> list[SeasonDTO]:
        # Bzzoiro leagues contain current_season; we synthesize one season per league
        data = await self._get("/leagues/", params={"limit": 200})
        results = data.get("results", []) if isinstance(data, dict) else []
        for entry in results:
            if int(entry.get("id", -1)) == league_id:
                cur = entry.get("current_season") or {}
                year = int(cur.get("year") or 2024)
                return [
                    SeasonDTO(
                        competition_external_id=league_id,
                        external_id=int(cur.get("id", year)),
                        year=year,
                        start_date=None,
                        end_date=None,
                        is_current=bool(cur.get("is_current", True)),
                    )
                ]
        return []

    # ----- Fixtures -----
    async def fetch_upcoming_fixtures(
        self, *, league_id: int, season_year: int, date_from: date, date_to: date
    ) -> list[FixtureDTO]:
        return await self._fetch_events(league_id, season_year, date_from, date_to, status_filter={"scheduled", "upcoming"})

    async def fetch_finished_fixtures(
        self, *, league_id: int, season_year: int, date_from: date, date_to: date
    ) -> list[FixtureDTO]:
        return await self._fetch_events(league_id, season_year, date_from, date_to, status_filter={"finished"})

    async def _fetch_events(
        self,
        league_id: int,
        season_year: int,
        date_from: date,
        date_to: date,
        status_filter: set[str] | None = None,
    ) -> list[FixtureDTO]:
        out: list[FixtureDTO] = []
        offset = 0
        limit = 100
        while True:
            params: dict[str, Any] = {
                "league": league_id,
                "date_from": date_from.isoformat(),
                "date_to": date_to.isoformat(),
                "limit": limit,
                "offset": offset,
            }
            data = await self._get("/events/", params=params)
            results = data.get("results", []) if isinstance(data, dict) else []
            if not results:
                break
            for entry in results:
                try:
                    dto = self._parse_event(entry)
                    if status_filter and dto.status not in status_filter:
                        continue
                    # Season filter: Bzzoiro season_id may not match our season_year, but we still check if needed
                    # For now, accept all in date range
                    out.append(dto)
                except Exception as exc:
                    _LOG.warning("skip bzzoiro event: {}", str(exc)[:256])
                    continue
            if len(results) < limit:
                break
            offset += limit
            if offset > 1000:  # safety cap
                break
        return out

    def _parse_event(self, entry: dict[str, Any]) -> FixtureDTO:
        # Bzzoiro event shape from live test
        ext_id = int(entry["id"])
        league_id = int(entry.get("league_id") or 0)
        season_id = int(entry.get("season_id") or 0)
        # season_year: try to infer from event_date year
        event_date_str = entry.get("event_date") or entry.get("kickoff_time") or entry.get("date")
        kickoff = self._parse_datetime(event_date_str)
        if kickoff is None:
            raise InvalidProviderResponse("bzzoiro event missing event_date")
        status_raw = str(entry.get("status") or "scheduled")
        status = _map_status(status_raw)
        home_goals = entry.get("home_score")
        away_goals = entry.get("away_score")
        # Bzzoiro may have null for upcoming
        if home_goals is not None:
            home_goals = int(home_goals)
        if away_goals is not None:
            away_goals = int(away_goals)
        if home_goals is None and away_goals is None:
            home_goals = None
            away_goals = None
        elif (home_goals is None) != (away_goals is None):
            home_goals = None
            away_goals = None
        season_year = kickoff.year
        # Try to get season_year from entry if present
        if "season_year" in entry:
            try:
                season_year = int(entry["season_year"])
            except Exception:
                pass
        return FixtureDTO(
            external_id=ext_id,
            competition_external_id=league_id,
            season_external_id=season_id,
            season_year=season_year,
            home_team_external_id=int(entry.get("home_team_id") or 0),
            away_team_external_id=int(entry.get("away_team_id") or 0),
            kickoff_time=kickoff,
            kickoff_timezone=None,
            matchday=entry.get("round_number"),
            round=entry.get("round_name"),
            venue=None,
            status=status,
            status_short=status_raw,
            home_goals=home_goals,
            away_goals=away_goals,
            finished_at=kickoff if status == "finished" else None,
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            iso = value.replace("Z", "+00:00") if value.endswith("Z") else value
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    # ----- Teams -----
    async def fetch_teams(self, *, league_id: int, season_year: int) -> list[TeamDTO]:
        # Bzzoiro /teams with league filter? Try with league param
        try:
            data = await self._get("/teams/", params={"league": league_id, "limit": 200})
        except Exception:
            data = await self._get("/teams/", params={"limit": 200})
        results = data.get("results", []) if isinstance(data, dict) else []
        out: list[TeamDTO] = []
        for entry in results:
            try:
                out.append(
                    TeamDTO(
                        external_id=int(entry["id"]),
                        name=str(entry.get("name") or ""),
                        short_name=entry.get("short_name"),
                        code=entry.get("code"),
                        country=entry.get("country"),
                        logo=None,
                        venue=None,
                        founded=None,
                    )
                )
            except Exception as exc:
                _LOG.warning("skip bzzoiro team: {}", str(exc)[:256])
                continue
        return out

    async def fetch_team_statistics(
        self, *, team_id: int, league_id: int, season_year: int, as_of_date: date
    ) -> TeamStatisticsDTO | None:
        # Bzzoiro does not have a direct team statistics endpoint like API-Football.
        # Return None to let the feature builder treat xG as missing (NaN).
        return None

    # ----- Players / injuries / lineups — not needed for MVP, return empty
    async def fetch_players(self, *, team_id: int, season_year: int) -> list[PlayerDTO]:
        return []

    async def fetch_player_statistics(self, *, player_id: int, league_id: int, season_year: int) -> PlayerStatisticsDTO | None:
        return None

    async def fetch_injuries(self, *, league_id: int, season_year: int) -> list[InjuryDTO]:
        return []

    async def fetch_lineup(self, *, fixture_id: int) -> list[LineupDTO]:
        return []


__all__ = ["BzzoiroProvider"]
