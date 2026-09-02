"""API-Football adapter.

This is the ONLY module in the codebase that knows API-Football's wire
format. It consumes :class:`ProviderHttpClient` for transport and
returns provider-agnostic DTOs from :mod:`app.providers.dto`.

Endpoints used
----------------
All endpoints return ``{get, parameters, errors, results, paging, response}``
where:

* ``response`` is the actual payload (list or single object).
* ``errors`` is a list (empty on success).
* ``results`` is the item count.
* ``paging`` carries ``{total, current, ...}`` when pagination applies.

The adapter normalises:

* Endpoint paths:
    - ``/leagues?id=...``                  - competitions and seasons
    - ``/fixtures?league=...&season=YYYY&from=...&to=...``
    - ``/fixtures?id=FIXTURE_ID``          - for upcoming/finished conflicts,
                                            we use the same /fixtures endpoint
                                            with different filters
    - ``/fixtures/lineups?fixture=FIXTURE_ID``
    - ``/teams?id=TEAM_ID`` / ``/teams?league=...&season=YYYY``
    - ``/teams/statistics?league=...&season=YYYY&team=TEAM_ID``
    - ``/players?team=TEAM_ID&season=YYYY``
    - ``/players?id=PLAYER_ID&season=YYYY&league=LEAGUE_ID``
    - ``/injuries?league=...&season=YYYY``

* Timezone normalisation: API-Football accepts a ``timezone`` query
  parameter; we ALWAYS pass ``timezone=Europe/London`` is intentionally
  NOT used — we ask for UTC by passing ``timezone=UTC``. The returned
  ``fixture.date`` strings are already UTC ISO8601. We also keep the
  league-provided tz string under :attr:`FixtureDTO.kickoff_timezone`
  when the upstream payload exposes one (it does not for v3, so we
  leave it ``None`` and document this clearly).

Failure isolation:
    Each parser raises a typed :class:`InvalidProviderResponse` on
    malformed entries. Aggregate fetch methods (``fetch_*``) catch
    per-item failures and skip the offending entry, returning the
    successfully parsed list. The sync service reports these skips
    via the metrics counters (``received`` vs ``failed``).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Final

from app.core.logging import get_logger
from app.providers.dto import (
    CompetitionDTO,
    CompetitionType,
    FixtureDTO,
    FixtureStatus,
    InjuryDTO,
    InjuryStatus,
    LineupDTO,
    LineupPlayerDTO,
    PlayerDTO,
    PlayerPosition,
    PlayerStatisticsDTO,
    SeasonDTO,
    TeamDTO,
    TeamStatisticsDTO,
)
from app.providers.exceptions import (
    InvalidProviderResponse,
    ProviderError,
)
from app.providers.http_client import ProviderHttpClient

_LOG = get_logger()


# Mapping of API-Football short fixture statuses to our normalised vocabulary.
# Source: https://www.api-football.com/documentation-v3#tag/Fixtures
# (long-short pairs documented there).
_FIXTURE_STATUS_MAP: Final[dict[str, FixtureStatus]] = {
    "TBD": "scheduled",
    "NS": "scheduled",
    "1H": "in_play",
    "2H": "in_play",
    "ET": "in_play",
    "BT": "in_play",
    "P": "in_play",
    "SUSP": "suspended",
    "INT": "suspended",
    "PST": "postponed",
    "CANC": "cancelled",
    "ABD": "cancelled",
    "AWD": "finished",   # technical loss; treated as finalized
    "WO": "finished",
    "FT": "finished",
    "AET": "finished",
    "PEN": "finished",
}

_POSITION_MAP: Final[dict[str, PlayerPosition]] = {
    "GK": "GK",
    "DF": "DF",
    "MF": "MF",
    "FW": "FW",
    "Goalkeeper": "GK",
    "Defender": "DF",
    "Midfielder": "MF",
    "Attacker": "FW",
}

_INJURY_STATUS_MAP: Final[dict[str, InjuryStatus]] = {
    "active": "active",
    "doubtful": "doubtful",
    "recovered": "recovered",
    "suspended": "suspended",
}

# The external League equivalents of the fixtures' "type" field on the
# /leagues endpoint. API-Football returns such values as lowercase words.
_COMPETITION_TYPE_MAP: Final[dict[str, CompetitionType]] = {
    "league": "league",
    "cup": "cup",
    "playoff": "playoff",
    "super_cup": "super_cup",
}


class APIFootballProvider:
    """Concrete :class:`DataProvider` backed by API-Football v3.

    Lifecycle: construct via :func:`app.providers.registry.get_provider`
    (which injects :class:`Settings`). The constructor never opens the
    HTTP client; that happens lazily on first request.
    """

    name: Final[str] = "api_football"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        rate_per_minute: int = 30,
        max_connections: int = 20,
        max_keepalive_connections: int = 10,
        user_agent: str = "football-prediction-app/0.1",
    ) -> None:
        self._http = ProviderHttpClient(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            rate_per_minute=rate_per_minute,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            user_agent=user_agent,
        )
        # Cache competitions/seasons per (league_id) for the current sync
        # run; not a long-lived cache (the sync services re-call over time).
        self._seasons_cache: dict[int, list[SeasonDTO]] = {}

    async def close(self) -> None:
        """Release resources owned by this provider."""
        await self._http.aclose()

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    async def _get(
        self,
        endpoint: str,
        *,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """GET ``endpoint`` and return the parsed ``response`` body."""
        # Always request UTC so the kickoff is canonical.
        params.setdefault("timezone", "UTC")
        body = await self._http.get_json(endpoint, params=params)
        if not isinstance(body, dict):
            raise InvalidProviderResponse(
                f"Expected object wrapper from {endpoint}, got {type(body).__name__}",
                details={"endpoint": endpoint},
            )
        errors = body.get("errors")
        # ``errors`` may be an object {field: [msg]} or a list of strings.
        if errors:
            raise InvalidProviderResponse(
                f"API-Football returned errors on {endpoint}: "
                f"{self._summarise_errors(errors)}",
                details={"endpoint": endpoint, "errors": self._summarise_errors(errors)},
            )
        return body

    @staticmethod
    def _summarise_errors(errors: Any) -> str:
        """Convert the upstream ``errors`` payload into a printable string."""
        if isinstance(errors, list):
            return "; ".join(str(e) for e in errors)[:512]
        if isinstance(errors, dict):
            parts = []
            for key, value in errors.items():
                parts.append(f"{key}: {value}")
            return "; ".join(parts)[:512]
        return str(errors)[:512]

    @staticmethod
    def _response_list(body: dict[str, Any]) -> list[Any]:
        """Extract ``body['response']`` as a list (defaulting to empty)."""
        response = body.get("response")
        if isinstance(response, list):
            return response
        # Some endpoints legitimately return a single object (e.g.
        # /teams/statistics). For fetch_* methods that delegate to this,
        # the caller knows to treat it differently.
        if response is None:
            return []
        return [response]

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        """Parse API-Football's ISO-8601 UTC timestamps (with or without Z)."""
        if not value:
            return None
        try:
            # API-Football returns e.g. "2026-08-15T19:00:00+00:00".
            # ``datetime.fromisoformat`` accepts everything but trailing "Z"
            # in Python 3.11+ it accepts "Z" too, but be defensive.
            iso = value.replace("Z", "+00:00") if value.endswith("Z") else value
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError as exc:
            raise InvalidProviderResponse(
                f"Invalid timestamp '{value[:64]}'",
                details={"value": value[:128]},
            ) from exc

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        """Parse YYYY-MM-DD."""
        if not value:
            return None
        try:
            return date.fromisoformat(value[:10])
        except ValueError as exc:
            raise InvalidProviderResponse(
                f"Invalid date '{value[:64]}'",
                details={"value": value[:128]},
            ) from exc

    @staticmethod
    def _safe_int(value: Any, field_name: str = "value") -> int:
        """Coerce ``value`` to ``int`` with a typed error."""
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise InvalidProviderResponse(
                f"Expected int for {field_name!r}, got {value!r}",
                details={"field_name": field_name, "value": repr(value)[:64]},
            ) from exc

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Coerce ``value`` to ``float``; return ``None`` on missing/invalid."""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ----------------------------------------------------------------------
    # leagues & seasons
    # ----------------------------------------------------------------------
    async def fetch_leagues(
        self, *, league_ids: list[int]
    ) -> list[CompetitionDTO]:
        """Fetch competitions for the given league ids."""
        out: list[CompetitionDTO] = []
        for league_id in league_ids:
            body = await self._get(
                "/leagues", params={"id": league_id, "current": "true"}
            )
            for entry in self._response_list(body):
                try:
                    league = entry["league"]
                    out.append(
                        CompetitionDTO(
                            external_id=self._safe_int(league["id"], "league.id"),
                            name=str(league.get("name") or ""),
                            type=_COMPETITION_TYPE_MAP.get(
                                str(league.get("type") or "league").lower(),
                                "league",
                            ),
                            country=(entry.get("country") or {}).get("name"),
                            logo=league.get("logo"),
                        )
                    )
                except (KeyError, InvalidProviderResponse) as exc:
                    _LOG.warning("skipping league entry: {}", str(exc)[:256])
                    continue
        return out

    async def fetch_seasons(self, *, league_id: int) -> list[SeasonDTO]:
        """Fetch seasons for a competition."""
        if league_id in self._seasons_cache:
            return self._seasons_cache[league_id]
        body = await self._get("/leagues", params={"id": league_id})
        out: list[SeasonDTO] = []
        for entry in self._response_list(body):
            try:
                league = entry["league"]
                for season_obj in entry.get("seasons", []):
                    start = self._parse_date(season_obj.get("start"))
                    end = self._parse_date(season_obj.get("end"))
                    out.append(
                        SeasonDTO(
                            competition_external_id=self._safe_int(
                                league["id"], "league.id"
                            ),
                            external_id=self._safe_int(
                                season_obj["year"], "seasons[].year"
                            ),
                            year=self._safe_int(season_obj["year"], "seasons[].year"),
                            start_date=start,
                            end_date=end,
                            is_current=bool(season_obj.get("current")),
                        )
                    )
            except (KeyError, InvalidProviderResponse) as exc:
                _LOG.warning("skipping season entry: {}", str(exc)[:256])
                continue
        self._seasons_cache[league_id] = out
        return out

    # ----------------------------------------------------------------------
    # fixtures
    # ----------------------------------------------------------------------
    async def fetch_upcoming_fixtures(
        self,
        *,
        league_id: int,
        season_year: int,
        date_from: date,
        date_to: date,
    ) -> list[FixtureDTO]:
        """Fetch fixtures (any status) whose kickoff falls in ``[date_from,
        date_to]``."""
        params: dict[str, Any] = {
            "league": league_id,
            "season": season_year,
            "from": date_from.isoformat(),
            "to": date_to.isoformat(),
        }
        body = await self._get("/fixtures", params=params)
        return self._parse_fixtures(body)

    async def fetch_finished_fixtures(
        self,
        *,
        league_id: int,
        season_year: int,
        date_from: date,
        date_to: date,
    ) -> list[FixtureDTO]:
        """Finished fixtures only."""
        params: dict[str, Any] = {
            "league": league_id,
            "season": season_year,
            "from": date_from.isoformat(),
            "to": date_to.isoformat(),
            "status": "FT-AET-PEN",  # API-Football: comma-separated filter
        }
        body = await self._get("/fixtures", params=params)
        return self._parse_fixtures(body)

    def _parse_fixtures(self, body: dict[str, Any]) -> list[FixtureDTO]:
        """Parse the ``response[]`` list of ``/fixtures`` payloads."""
        out: list[FixtureDTO] = []
        for entry in self._response_list(body):
            try:
                fx_obj = entry["fixture"]
                league = entry["league"]
                teams = entry["teams"]
                home = teams["home"]
                away = teams["away"]
                goals = entry.get("goals") or {}
                kickoff = self._parse_timestamp(fx_obj.get("date"))
                if kickoff is None:
                    raise InvalidProviderResponse("fixture missing kickoff date")
                status_short = (fx_obj.get("status") or {}).get("short")
                status = _FIXTURE_STATUS_MAP.get(status_short or "NS", "scheduled")
                home_goals = goals.get("home")
                away_goals = goals.get("away")
                # Only carry goals when non-null (matches SCHEMA.md §5
                # paired-goals CHECK constraint).
                if home_goals is not None and away_goals is not None:
                    home_goals_int: int | None = int(home_goals)
                    away_goals_int: int | None = int(away_goals)
                elif home_goals is None and away_goals is None:
                    home_goals_int = None
                    away_goals_int = None
                else:
                    # Provider inconsistency: only one side has a value.
                    _LOG.warning(
                        "fixture {} has mismatched goals: home={}, away={}",
                        fx_obj.get("id"), home_goals, away_goals,
                    )
                    home_goals_int = None
                    away_goals_int = None
                season_year_val = int(league.get("season") or 0)
                if season_year_val == 0:
                    raise InvalidProviderResponse(
                        f"fixture {fx_obj.get('id')}: missing league.season"
                    )
                # API-Football's "matchday"/"round" live under
                # ``league.round`` ("Regular Season - 1") and ``fixture.round``
                # is not present in v3. Extract the matchday number from the
                # round string when it ends with one.
                round_str = league.get("round") or None
                matchday = self._extract_matchday(round_str)
                out.append(
                    FixtureDTO(
                        external_id=self._safe_int(fx_obj["id"], "fixture.id"),
                        competition_external_id=self._safe_int(
                            league["id"], "league.id"
                        ),
                        season_external_id=season_year_val,
                        season_year=season_year_val,
                        home_team_external_id=self._safe_int(
                            home["id"], "teams.home.id"
                        ),
                        away_team_external_id=self._safe_int(
                            away["id"], "teams.away.id"
                        ),
                        kickoff_time=kickoff,
                        kickoff_timezone=None,
                        matchday=matchday,
                        round=round_str,
                        venue=((fx_obj.get("venue") or {}).get("name")),
                        status=status,
                        status_short=status_short,
                        home_goals=home_goals_int,
                        away_goals=away_goals_int,
                        finished_at=(kickoff if status == "finished" else None),
                    )
                )
            except (KeyError, InvalidProviderResponse, ValueError, TypeError) as exc:
                _LOG.warning("skipping fixture entry: {}", str(exc)[:256])
                continue
        return out

    @staticmethod
    def _extract_matchday(round_str: str | None) -> int | None:
        """Extract the matchday number from a round string.

        API-Football writes ``"Regular Season - 1"`` or ``"Final"`` etc.
        We return the trailing integer when present, ``None`` otherwise.
        """
        if not round_str:
            return None
        parts = round_str.split("-")
        if len(parts) < 2:
            return None
        tail = parts[-1].strip()
        try:
            return int(tail)
        except ValueError:
            return None

    # ----------------------------------------------------------------------
    # teams
    # ----------------------------------------------------------------------
    async def fetch_teams(
        self, *, league_id: int, season_year: int
    ) -> list[TeamDTO]:
        """Fetch teams for a competition+season."""
        body = await self._get(
            "/teams",
            params={"league": league_id, "season": season_year},
        )
        out: list[TeamDTO] = []
        for entry in self._response_list(body):
            try:
                team = entry["team"]
                venue = entry.get("venue") or {}
                out.append(
                    TeamDTO(
                        external_id=self._safe_int(team["id"], "team.id"),
                        name=str(team.get("name") or ""),
                        short_name=team.get("code"),
                        code=team.get("code"),
                        country=team.get("country"),
                        logo=team.get("logo"),
                        venue=venue.get("name"),
                        founded=self._coerce_year(team.get("founded")),
                    )
                )
            except (KeyError, InvalidProviderResponse) as exc:
                _LOG.warning("skipping team entry: {}", str(exc)[:256])
                continue
        return out

    async def fetch_team_statistics(
        self,
        *,
        team_id: int,
        league_id: int,
        season_year: int,
        as_of_date: date,
    ) -> TeamStatisticsDTO | None:
        """Fetch aggregated team statistics. ``as_of_date`` is informational:
        API-Football endpoints return the cumulative season stats (no
        per-day snapshot exists); our caller stamps the snapshot date as
        ``as_of_date`` so the DB captures the date of the sync.
        """
        body = await self._get(
            "/teams/statistics",
            params={
                "league": league_id,
                "season": season_year,
                "team": team_id,
            },
        )
        # /teams/statistics returns the payload directly under "response".
        response = body.get("response")
        if not isinstance(response, dict):
            return None
        try:
            return self._parse_team_statistics(
                response,
                team_external_id=team_id,
                league_external_id=league_id,
                season_external_id=season_year,
                as_of_date=as_of_date,
            )
        except (KeyError, InvalidProviderResponse) as exc:
            _LOG.warning("skipping team_statistics entry: {}", str(exc)[:256])
            return None

    def _parse_team_statistics(
        self,
        data: dict[str, Any],
        *,
        team_external_id: int,
        league_external_id: int,
        season_external_id: int,
        as_of_date: date,
    ) -> TeamStatisticsDTO:
        """Translate the ``/teams/statistics`` payload into a DTO."""
        fixtures = data.get("fixtures") or {}
        played = (fixtures.get("played") or {}).get("total") or 0
        wins = ((fixtures.get("wins") or {}).get("total")) or 0
        draws = ((fixtures.get("draws") or {}).get("total")) or 0
        losses = ((fixtures.get("loses") or {}).get("total")) or 0
        goals = data.get("goals") or {}
        goals_for = ((goals.get("for") or {}).get("total")) or 0
        goals_against = ((goals.get("against") or {}).get("total")) or 0
        clean = data.get("clean_sheet") or {}
        failed = data.get("failed_to_score") or {}
        big = data.get("biggest") or {}

        return TeamStatisticsDTO(
            team_external_id=team_external_id,
            competition_external_id=league_external_id,
            season_external_id=season_external_id,
            as_of_date=as_of_date,
            fixtures_played=int(played),
            wins=int(wins),
            draws=int(draws),
            losses=int(losses),
            goals_for=int(goals_for) if goals_for is not None else 0,
            goals_against=int(goals_against) if goals_against is not None else 0,
            clean_sheets=int(self._first_int_value(clean)) if clean else 0,
            failed_to_score=int(self._first_int_value(failed)) if failed else 0,
            form=(data.get("form") or None),
            shots_total=self._first_int_value(data.get("shots") or {}),
            possession_avg=None,  # not in /teams/statistics
            yellow_cards=None,
            red_cards=None,
            xg=self._safe_float(data.get("xg")),
            xga=self._safe_float(data.get("xga")),
        )

    @staticmethod
    def _first_int_value(data: dict[str, Any] | Any) -> int | None:
        """Best-effort: return the first integer value found in a dict/list."""
        if isinstance(data, dict):
            # Patterns like {"total": 5, "percentage": "21%"}
            if (total := data.get("total")) is not None:
                try:
                    return int(total)
                except (TypeError, ValueError):
                    pass
            for value in data.values():
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)
        if isinstance(data, list):
            for value in data:
                if isinstance(value, int):
                    return value
        return None

    @staticmethod
    def _coerce_year(value: Any) -> int | None:
        """Coerce ``founded`` field to int year (``"1886"`` → ``1886``)."""
        if value is None or value == "":
            return None
        try:
            return int(str(value)[:4])
        except ValueError:
            return None

    # ----------------------------------------------------------------------
    # players
    # ----------------------------------------------------------------------
    async def fetch_players(
        self, *, team_id: int, season_year: int
    ) -> list[PlayerDTO]:
        """Fetch the squad of a team for a season (paginated).

        The v3 ``/players`` endpoint paginates 20 players per page. We loop
        through all pages up to a hard limit to avoid runaway cost.
        """
        MAX_PAGES = 30
        out: list[PlayerDTO] = []
        page = 1
        while page <= MAX_PAGES:
            try:
                body = await self._get(
                    "/players",
                    params={"team": team_id, "season": season_year, "page": page},
                )
            except ProviderError as exc:
                _LOG.warning(
                    "fetch_players: stopping pagination at page={} reason={}",
                    page, str(exc)[:256],
                )
                break

            items = self._response_list(body)
            if not items:
                break

            for item in items:
                try:
                    out.append(
                        PlayerDTO(
                            external_id=self._safe_int(item["id"], "player.id"),
                            name=str(item.get("name") or ""),
                            photo=item.get("photo"),
                            nationality=item.get("nationality"),
                            birth_date=self._parse_date(
                                ((item.get("birth") or {}).get("date"))
                            ),
                            height_cm=self._coerce_height(item.get("height")),
                            weight_kg=self._coerce_weight(item.get("weight")),
                            position=_POSITION_MAP.get(
                                str((item.get("position") or "MF")),
                                None,
                            ),
                        )
                    )
                except (KeyError, InvalidProviderResponse) as exc:
                    _LOG.warning("skipping player entry: {}", str(exc)[:256])
                    continue

            paging_total = ((body.get("paging") or {}).get("total", 1)) or 1
            if page >= paging_total:
                break
            page += 1
        return out

    @staticmethod
    def _coerce_height(value: Any) -> int | None:
        """Coerce ``"180 cm"`` → ``180``."""
        if not isinstance(value, str):
            return None
        parts = value.split()
        if not parts:
            return None
        try:
            return int(parts[0])
        except ValueError:
            return None

    @staticmethod
    def _coerce_weight(value: Any) -> int | None:
        """Coerce ``"75 kg"`` → ``75``."""
        if not isinstance(value, str):
            return None
        parts = value.split()
        if not parts:
            return None
        try:
            return int(parts[0])
        except ValueError:
            return None

    async def fetch_player_statistics(
        self,
        *,
        player_id: int,
        league_id: int,
        season_year: int,
    ) -> PlayerStatisticsDTO | None:
        """Fetch aggregated player stats for a league+season.

        The match in the player payload is found by ``league.id``.
        """
        # /players?id=PLAYER_ID&season=YEAR  (no league filter)
        body = await self._get(
            "/players",
            params={"id": player_id, "season": season_year},
        )
        items = self._response_list(body)
        if not items:
            return None
        first = items[0]
        for entry in first.get("statistics") or []:
            league_obj = entry.get("league") or {}
            league_id_raw = league_obj.get("id")
            try:
                if league_id_raw is not None and int(league_id_raw) != league_id:
                    continue
            except (TypeError, ValueError):
                continue
            games = entry.get("games") or {}
            goals = entry.get("goals") or {}
            cards = entry.get("cards") or {}
            try:
                return PlayerStatisticsDTO(
                    player_external_id=player_id,
                    team_external_id=self._safe_int(
                        (entry.get("team") or {}).get("id")
                        or 0,
                        "team.id",
                    ),
                    competition_external_id=league_id,
                    season_external_id=season_year,
                    appearances=self._safe_int(
                        games.get("appearences") or games.get("appearances") or 0,
                        "games.appearences",
                    ),
                    starts=self._safe_int(
                        games.get("lineups") or 0, "games.lineups",
                    ),
                    minutes_played=self._safe_int(
                        games.get("minutes") or 0, "games.minutes",
                    ),
                    goals=self._safe_int(goals.get("total") or 0, "goals.total"),
                    assists=self._safe_int(
                        (entry.get("assists") or 0), "assists.total",
                    ),
                    yellow_cards=self._safe_int(
                        cards.get("yellow") or 0, "cards.yellow",
                    ),
                    red_cards=self._safe_int(
                        cards.get("red") or 0, "cards.red",
                    ),
                    rating=self._safe_float((games.get("rating"))),
                )
            except (KeyError, InvalidProviderResponse) as exc:
                _LOG.warning("skipping player_stats entry: {}", str(exc)[:256])
                continue
        return None

    # ----------------------------------------------------------------------
    # injuries
    # ----------------------------------------------------------------------
    async def fetch_injuries(
        self, *, league_id: int, season_year: int
    ) -> list[InjuryDTO]:
        """Fetch current and historical injuries of a league+season."""
        body = await self._get(
            "/injuries",
            params={"league": league_id, "season": season_year},
        )
        out: list[InjuryDTO] = []
        for item in self._response_list(body):
            try:
                player = item["player"]
                team = item["team"]
                league = item.get("league") or {}
                fixture_obj = item.get("fixture") or {}
                status_raw = str(item.get("status") or "active").lower()
                # The /injuries endpoint reports `updated_at` per row.
                updated = self._parse_timestamp(item.get("updated_at"))
                out.append(
                    InjuryDTO(
                        external_id=self._safe_int(item["id"], "injury.id"),
                        player_external_id=self._safe_int(
                            player["id"], "player.id"
                        ),
                        team_external_id=self._safe_int(team["id"], "team.id"),
                        competition_external_id=(
                            int(league["id"]) if "id" in league else None
                        ),
                        fixture_external_id=(
                            int(fixture_obj["id"]) if "id" in fixture_obj else None
                        ),
                        type=item.get("type"),
                        reason=item.get("reason"),
                        status=_INJURY_STATUS_MAP.get(status_raw, "active"),
                        start_date=self._parse_date(item.get("date"))
                        or date(1970, 1, 1),
                        end_date=None,
                        updated_external_at=updated,
                    )
                )
            except (KeyError, InvalidProviderResponse, ValueError, TypeError) as exc:
                _LOG.warning("skipping injury entry: {}", str(exc)[:256])
                continue
        return out

    # ----------------------------------------------------------------------
    # lineups
    # ----------------------------------------------------------------------
    async def fetch_lineup(self, *, fixture_id: int) -> list[LineupDTO]:
        """Fetch the home and away lineups of a fixture."""
        body = await self._get(
            "/fixtures/lineups", params={"fixture": fixture_id}
        )
        out: list[LineupDTO] = []
        for item in self._response_list(body):
            try:
                team = item["team"]
                team_id = self._safe_int(team["id"], "team.id")
                coach = (item.get("coach") or {}).get("name")
                formation = item.get("formation")
                # /fixtures/lineups doesn't return the home/away label
                # directly; we infer it from team.colors isn't reliable, so
                # we leave the caller (sync service) to look it up against
                # the fixture. We carry ``is_home`` as a guess from the
                # team ID coherence vs the fixture, defaulting to False.
                start_xi = (item.get("startXI") or {}).get("players") or []
                subs = (item.get("substitutes") or {}).get("players") or []
                players: list[LineupPlayerDTO] = []
                for slot in start_xi:
                    players.append(self._parse_lineup_player(slot, starter=True))
                for slot in subs:
                    players.append(self._parse_lineup_player(slot, starter=False))
                # We don't know the side here; the sync service looks up the
                # fixture and decides ``is_home`` from the known home/away
                # team IDs.
                out.append(
                    LineupDTO(
                        fixture_external_id=fixture_id,
                        team_external_id=team_id,
                        is_home=False,  # filled by sync service
                        formation=formation,
                        coach=coach,
                        updated_external_at=self._parse_timestamp(item.get("updated_at")),
                        players=players,
                    )
                )
            except (KeyError, InvalidProviderResponse, ValueError, TypeError) as exc:
                _LOG.warning("skipping lineup entry: {}", str(exc)[:256])
                continue
        return out

    def _parse_lineup_player(
        self, slot: dict[str, Any], *, starter: bool
    ) -> LineupPlayerDTO:
        """Parse one lineup player slot."""
        player = slot.get("player") or {}
        pos = slot.get("pos") or slot.get("position") or "MF"
        pos_x = slot.get("posX")
        pos_y = slot.get("posY")
        number = slot.get("number")
        return LineupPlayerDTO(
            player_external_id=self._safe_int(
                player.get("id") or 0, "lineup.player.id"
            ),
            position=_POSITION_MAP.get(str(pos), None),
            position_x=(self._safe_int(pos_x, field_name="posX") if pos_x is not None else None),
            position_y=(self._safe_int(pos_y, field_name="posY") if pos_y is not None else None),
            shirt_number=(self._safe_int(number, field_name="number") if number is not None else None),
            is_starter=starter,
        )
