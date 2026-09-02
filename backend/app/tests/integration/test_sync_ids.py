"""Integration tests for the sync services.

These tests exercise the full vertical slice:

    DataProvider (fake) → sync service → repository → PostgreSQL

The fake provider returns DTOs whose ``external_id`` values are picked to
NOT collide with the internal ``id``s assigned by PostgreSQL's BIGSERIAL
sequence. This lets the tests assert that:

* The fixture is persisted keyed by its ``external_id`` (one row per
  ``external_id``, regardless of how many times sync runs).
* The repository received INTERNAL ids (1, 2, 3 — assigned by Postgres)
  for the FK columns, not the external_ids (large numbers chosen by the
  fake provider).
* Re-running the same sync N times produces ONE row each (idempotency).

Requirements:
* A PostgreSQL instance pre-migrated with `alembic upgrade head` — see
  ``app/tests/conftest.py``. The session fixture rolls back changes.
* The tests do NOT hit the real API-Football. They construct a fake
  provider implementing :class:`DataProvider`.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select

from app.models import (
    Competition,
    Fixture,
    Player,
    PlayerStatistics,
    Season,
    Team,
    TeamStatistics,
)
from app.providers.base import DataProvider
from app.providers.dto import (
    CompetitionDTO,
    FixtureDTO,
    PlayerDTO,
    SeasonDTO,
    TeamDTO,
    TeamStatisticsDTO,
)
from app.services.sync import (
    sync_competitions_and_seasons,
    sync_finished_fixtures,
    sync_player_statistics,
    sync_team_statistics,
    sync_teams,
    sync_upcoming_fixtures,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# External ids chosen so they're numerically distinct from internal ids:
# all start with 9_***_*** (so they CANNOT match any small BIGSERIAL id).
# ---------------------------------------------------------------------------
_EXT_COMP = 9_000_001
_EXT_SEASON = 2024
_EXT_TEAM_HOME = 9_000_010
_EXT_TEAM_AWAY = 9_000_011
_EXT_PLAYER = 9_000_020
_EXT_FIXTURE = 9_000_100


# ---------------------------------------------------------------------------
# Fake provider implementing the Protocol
# ---------------------------------------------------------------------------
class _FakeProvider:
    """In-memory provider that yields fully-populated DTOs."""

    name = "fake"

    def __init__(self) -> None:
        self.upcoming_call_count = 0
        self.finished_call_count = 0

    async def fetch_leagues(self, *, league_ids: list[int]) -> list[CompetitionDTO]:
        return [
            CompetitionDTO(
                external_id=_EXT_COMP, name="Premier League", type="league",
                country="England", logo=None,
            )
        ]

    async def fetch_seasons(self, *, league_id: int) -> list[SeasonDTO]:
        return [
            SeasonDTO(
                competition_external_id=_EXT_COMP,
                external_id=_EXT_SEASON,
                year=_EXT_SEASON,
                start_date=date(2024, 8, 1),
                end_date=date(2025, 6, 30),
                is_current=True,
            )
        ]

    async def fetch_upcoming_fixtures(
        self, *, league_id: int, season_year: int,
        date_from: date, date_to: date,
    ) -> list[FixtureDTO]:
        self.upcoming_call_count += 1
        return [
            FixtureDTO(
                external_id=_EXT_FIXTURE, competition_external_id=_EXT_COMP,
                season_external_id=_EXT_SEASON, season_year=_EXT_SEASON,
                home_team_external_id=_EXT_TEAM_HOME,
                away_team_external_id=_EXT_TEAM_AWAY,
                kickoff_time=datetime(2026, 8, 20, 19, 0, tzinfo=timezone.utc),
                kickoff_timezone=None,
                matchday=1, round="Regular Season - 1",
                venue="Test Venue",
                status="scheduled", status_short="NS",
                home_goals=None, away_goals=None, finished_at=None,
            )
        ]

    async def fetch_finished_fixtures(
        self, *, league_id: int, season_year: int,
        date_from: date, date_to: date,
    ) -> list[FixtureDTO]:
        self.finished_call_count += 1
        return [
            FixtureDTO(
                external_id=_EXT_FIXTURE, competition_external_id=_EXT_COMP,
                season_external_id=_EXT_SEASON, season_year=_EXT_SEASON,
                home_team_external_id=_EXT_TEAM_HOME,
                away_team_external_id=_EXT_TEAM_AWAY,
                kickoff_time=datetime(2026, 8, 20, 19, 0, tzinfo=timezone.utc),
                kickoff_timezone=None,
                matchday=1, round="Regular Season - 1",
                venue=None,
                status="finished", status_short="FT",
                home_goals=2, away_goals=1,
                finished_at=datetime(2026, 8, 20, 21, 0, tzinfo=timezone.utc),
            )
        ]

    async def fetch_teams(
        self, *, league_id: int, season_year: int,
    ) -> list[TeamDTO]:
        return [
            TeamDTO(external_id=_EXT_TEAM_HOME, name="Arsenal FC",
                    country="England"),
            TeamDTO(external_id=_EXT_TEAM_AWAY, name="Chelsea FC",
                    country="England"),
        ]

    async def fetch_team_statistics(
        self, *, team_id: int, league_id: int, season_year: int,
        as_of_date: date,
    ) -> TeamStatisticsDTO | None:
        return TeamStatisticsDTO(
            team_external_id=team_id,
            competition_external_id=league_id,
            season_external_id=_EXT_SEASON,
            as_of_date=as_of_date,
            fixtures_played=3, wins=2, draws=1, losses=0,
            goals_for=6, goals_against=2, clean_sheets=2,
            failed_to_score=0, form="WWD",
            shots_total=12, shots_on_target=6,
            shots_inside_box=None, shots_outside_box=None,
            fouls=None, corners=None, offsides=None,
            possession_avg=58.5, yellow_cards=4, red_cards=0,
            passes_total=None, passes_accuracy=None,
            xg=1.7, xga=0.9,
        )

    async def fetch_players(
        self, *, team_id: int, season_year: int,
    ) -> list[PlayerDTO]:
        return [
            PlayerDTO(external_id=_EXT_PLAYER, name="Martin Odegaard",
                      position="MF", nationality="Norway"),
        ]

    async def fetch_player_statistics(
        self, *, player_id: int, league_id: int, season_year: int,
    ) -> "PlayerStatisticsDTO | None":
        # Imported lazily here to avoid top-of-file complexity.
        from app.providers.dto import PlayerStatisticsDTO
        return PlayerStatisticsDTO(
            player_external_id=player_id,
            team_external_id=_EXT_TEAM_HOME,
            competition_external_id=league_id,
            season_external_id=_EXT_SEASON,
            appearances=3, starts=3, minutes_played=270,
            goals=2, assists=1, yellow_cards=0, red_cards=0,
            rating=7.8,
        )

    async def fetch_injuries(
        self, *, league_id: int, season_year: int,
    ) -> list:  # noqa: ANN202  (test fixture)
        return []

    async def fetch_lineup(self, *, fixture_id: int) -> list:  # noqa: ANN202
        return []

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Settings fixture forcing a known target year/league
# ---------------------------------------------------------------------------
class _StubSettings:
    """Minimal duck-typed Settings usable by the sync services."""
    current_league_ids: list[int] = [_EXT_COMP]
    current_season_year: int = _EXT_SEASON
    sync_upcoming_days: int = 7
    sync_finished_window_hours: int = 6


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
async def test_competition_and_season_persisted_with_external_id(session) -> None:
    """The competition/season rows keep the provider's external_id while
    their internal ``id`` (BIGSERIAL) is auto-assigned."""
    provider = _FakeProvider()
    metrics = await sync_competitions_and_seasons(
        provider=provider, session=session, settings=_StubSettings(),
    )
    assert metrics.failed == 0

    comp = (await session.execute(
        select(Competition).where(Competition.external_id == _EXT_COMP)
    )).scalar_one()
    assert comp.id > 0  # internal BIGSERIAL, never equals _EXT_COMP
    assert comp.id != _EXT_COMP
    assert comp.name == "Premier League"

    season = (await session.execute(
        select(Season).where(Season.external_id == _EXT_SEASON)
    )).scalar_one()
    assert season.id > 0
    assert season.id != _EXT_SEASON
    assert season.competition_id == comp.id  # INTERNAL FK
    assert season.is_current is True


async def test_team_upsert_uses_internal_id_for_relations(session) -> None:
    """Teams are persisted keyed by ``external_id``; the repository
    automatically receives the internal ``id`` for any FK column required
    by subsequent syncs (e.g. fixtures)."""
    provider = _FakeProvider()
    settings = _StubSettings()
    # Pre-stage competition/season so sync_teams doesn't skip the league.
    await sync_competitions_and_seasons(
        provider=provider, session=session, settings=settings,
    )

    metrics = await sync_teams(provider=provider, session=session, settings=settings)
    assert metrics.failed == 0

    teams = (await session.execute(select(Team))).scalars().all()
    assert len(teams) == 2
    home = next(t for t in teams if t.external_id == _EXT_TEAM_HOME)
    assert home.id > 0
    assert home.id != _EXT_TEAM_HOME
    assert home.name == "Arsenal FC"


async def test_fixture_upsert_keys_by_external_id_not_internal(session) -> None:
    """Re-running the upcoming sync three times yields ONE fixture row,
    keyed by external_id. The internal id (BIGSERIAL) is unrelated to the
    external_id."""
    provider = _FakeProvider()
    settings = _StubSettings()
    await sync_competitions_and_seasons(
        provider=provider, session=session, settings=settings,
    )
    await sync_teams(provider=provider, session=session, settings=settings)

    for _ in range(3):
        m = await sync_upcoming_fixtures(
            provider=provider, session=session, settings=settings,
        )
        assert m.failed == 0
    assert provider.upcoming_call_count == 3

    fixtures = (await session.execute(
        select(Fixture).where(Fixture.external_id == _EXT_FIXTURE)
    )).scalars().all()
    assert len(fixtures) == 1  # idempotency: no duplicates
    fx = fixtures[0]
    assert fx.id > 0
    assert fx.id != _EXT_FIXTURE
    # FK columns must hold INTERNAL ids (the BIGSERIAL ones), NOT external.
    teams = {t.external_id: t.id for t in (await session.execute(select(Team))).scalars().all()}
    assert fx.home_team_id == teams[_EXT_TEAM_HOME]
    assert fx.away_team_id == teams[_EXT_TEAM_AWAY]
    assert fx.home_team_id != _EXT_TEAM_HOME
    assert fx.away_team_id != _EXT_TEAM_AWAY


async def test_finished_sync_updates_score_and_keeps_external_id(session) -> None:
    """After sync_finished, the same fixture row (same external_id, same
    internal id) now carries the final score and ``status='finished'``."""
    provider = _FakeProvider()
    settings = _StubSettings()
    await sync_competitions_and_seasons(
        provider=provider, session=session, settings=settings,
    )
    await sync_teams(provider=provider, session=session, settings=settings)
    await sync_upcoming_fixtures(
        provider=provider, session=session, settings=settings,
    )

    metrics = await sync_finished_fixtures(
        provider=provider, session=session, settings=settings,
    )
    assert metrics.failed == 0

    fixtures = (await session.execute(
        select(Fixture).where(Fixture.external_id == _EXT_FIXTURE)
    )).scalars().all()
    assert len(fixtures) == 1
    fx = fixtures[0]
    assert fx.status == "finished"
    assert fx.home_goals == 2
    assert fx.away_goals == 1
    assert fx.finished_at is not None


async def test_player_stats_use_internal_player_id(session) -> None:
    """``player_statistics.player_id`` MUST hold the internal ``players.id``;
    never the provider's external_id.

    This is the direct regression test for the bug reported: it forces
    future code to keep ``player_id = persisted_player.id`` (the internal
    PK) instead of leaking ``player.external_id`` into the FK column.
    """
    provider = _FakeProvider()
    settings = _StubSettings()
    await sync_competitions_and_seasons(
        provider=provider, session=session, settings=settings,
    )
    await sync_teams(provider=provider, session=session, settings=settings)

    metrics = await sync_player_statistics(
        provider=provider, session=session, settings=settings,
    )
    assert metrics.failed == 0

    player = (await session.execute(
        select(Player).where(Player.external_id == _EXT_PLAYER)
    )).scalar_one()
    assert player.id > 0
    assert player.id != _EXT_PLAYER

    stats = (await session.execute(
        select(PlayerStatistics).where(PlayerStatistics.player_id == player.id)
    )).scalar_one()
    # The FK reference MUST be the internal id, not the external one.
    assert stats.player_id == player.id
    assert stats.player_id != _EXT_PLAYER

    # Ensure NO row was accidentally created with the external_id as FK id.
    bogus = (await session.execute(
        select(PlayerStatistics).where(PlayerStatistics.player_id == _EXT_PLAYER)
    )).scalar_one_or_none()
    assert bogus is None, "external_id leaked into player_statistics.player_id"


async def test_team_statistics_use_internal_ids(session) -> None:
    """``team_statistics`` row's team_id / competition_id / season_id MUST
    be internal ids, NOT external."""
    provider = _FakeProvider()
    settings = _StubSettings()
    await sync_competitions_and_seasons(
        provider=provider, session=session, settings=settings,
    )
    await sync_teams(provider=provider, session=session, settings=settings)

    metrics = await sync_team_statistics(
        provider=provider, session=session, settings=settings,
    )
    assert metrics.failed == 0

    stats = (await session.execute(select(TeamStatistics))).scalars().all()
    assert len(stats) == 2
    for s in stats:
        assert s.team_id > 0
        assert s.team_id != _EXT_TEAM_HOME
        assert s.team_id != _EXT_TEAM_AWAY
        assert s.competition_id > 0
        assert s.competition_id != _EXT_COMP
        assert s.season_id > 0
        assert s.season_id != _EXT_SEASON


async def test_stubsettings_is_sufficiently_settings_shaped() -> None:
    """Smoke check that the duck-typed settings expose every attribute the
    sync implementations index."""
    s = _StubSettings()
    assert hasattr(s, "current_league_ids")
    assert hasattr(s, "current_season_year")
    assert hasattr(s, "sync_upcoming_days")
    assert hasattr(s, "sync_finished_window_hours")
