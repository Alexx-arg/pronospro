"""Sync: fixtures (upcoming + finished).

The flow:

1. Resolve each ``(league_id, season_year)`` to its internal
   ``Competition`` + ``Season`` ids (via the metadata sync that just ran,
   or the repository's upsert-cache). If either is missing the league is
   skipped (logged as ``failed`` so the operator can investigate).
2. For each league, call the provider with the configured window.
3. Resolve the two teams (home + away) via ``external_id``; teams are NOT
   upserted here — ``sync_teams`` is responsible for that. If a team is
   unknown to our DB, the fixture is logged as ``skipped`` (we'd rather
   skip it than auto-insert a half-known team with no stats attached).
4. Upsert the fixture via ``external_id``. Update mutable fields when the
   fixture already exists (kickoff_time, status, round, venue, status_short).
5. ``sync_finished`` additionally looks for finished fixtures in the
   provider response and calls ``mark_finished`` on the fixture if its
   score is missing or differs.

The service NEVER touches ``predictions`` or ``prediction_outcomes``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.logging import get_logger
from app.providers.base import DataProvider
from app.providers.dto import FixtureDTO
from app.repositories import (
    CompetitionRepository,
    FixtureRepository,
    SeasonRepository,
    TeamRepository,
)
from app.services.sync.metrics import SyncMetrics

_LOG = get_logger()


async def sync_upcoming_fixtures(
    *,
    provider: DataProvider,
    session: AsyncSession,
    settings: Settings,
    league_ids: list[int] | None = None,
    days: int | None = None,
) -> SyncMetrics:
    """Sync upcoming fixtures for each configured league."""
    return await _sync_fixtures_inner(
        mode="upcoming",
        provider=provider,
        session=session,
        settings=settings,
        league_ids=league_ids,
        days=days,
    )


async def sync_finished_fixtures(
    *,
    provider: DataProvider,
    session: AsyncSession,
    settings: Settings,
    league_ids: list[int] | None = None,
    window_hours: int | None = None,
) -> SyncMetrics:
    """Sync results of fixtures that finished recently."""
    return await _sync_fixtures_inner(
        mode="finished",
        provider=provider,
        session=session,
        settings=settings,
        league_ids=league_ids,
        window_hours=window_hours,
    )


async def _sync_fixtures_inner(
    *,
    mode: str,
    provider: DataProvider,
    session: AsyncSession,
    settings: Settings,
    league_ids: list[int] | None,
    days: int | None = None,
    window_hours: int | None = None,
) -> SyncMetrics:
    """Common machinery for both upcoming + finished fixture syncs."""
    metrics = SyncMetrics(job=f"sync_{mode}_fixtures")
    league_ids = list(league_ids or settings.current_league_ids())
    metrics.requested = len(league_ids)

    comp_repo = CompetitionRepository(session)
    season_repo = SeasonRepository(session)
    team_repo = TeamRepository(session)
    fixture_repo = FixtureRepository(session)

    now = datetime.now(timezone.utc)

    for league_id in league_ids:
        try:
            # Resolve internal ids once per league.
            comp = await comp_repo.find_by_external_id(league_id)
            if comp is None:
                metrics.failed += 1
                metrics.add_error(
                    f"league {league_id}: missing competition row — "
                    "run sync_competitions_and_seasons first"
                )
                _LOG.warning("sync_fixtures: league {} unknown in DB", league_id)
                continue

            season_year = await _resolve_season_year(
                provider=provider,
                session=session,
                settings=settings,
                league_id=league_id,
                season_repo=season_repo,
            )
            if season_year is None:
                metrics.skipped += 1
                metrics.add_error(f"league {league_id}: no season resolved")
                continue

            season = await season_repo.find_for_competition_and_year(
                competition_id=comp.id, year=season_year
            )
            if season is None:
                metrics.skipped += 1
                metrics.add_error(
                    f"league {league_id}: season {season_year} not in DB"
                )
                continue

            if mode == "upcoming":
                days_value = days or settings.sync_upcoming_days
                from_time = now
                to_time = _add_days(now, days_value)
                dtos = await provider.fetch_upcoming_fixtures(
                    league_id=league_id,
                    season_year=season_year,
                    date_from=from_time.date(),
                    date_to=to_time.date(),
                )
            else:
                hours = window_hours or settings.sync_finished_window_hours
                from_time = _add_hours(now, -hours)
                to_time = now
                dtos = await provider.fetch_finished_fixtures(
                    league_id=league_id,
                    season_year=season_year,
                    date_from=from_time.date(),
                    date_to=to_time.date(),
                )

            metrics.received += len(dtos)
            await _upsert_fixtures(
                dtos=dtos,
                fixture_repo=fixture_repo,
                team_repo=team_repo,
                comp_id=comp.id,
                season_id=season.id,
                metrics=metrics,
                mode=mode,
            )
        except Exception as exc:  # noqa: BLE001  (per-league fault isolation)
            metrics.failed += 1
            metrics.add_error(f"league {league_id}: {exc!r}")
            _LOG.warning("sync_fixtures: league {} failed {}", league_id, str(exc)[:256])

    _LOG.info("sync_{}_fixtures done: {}", mode, metrics.as_dict())
    return metrics


async def _resolve_season_year(
    *,
    provider: DataProvider,
    session: AsyncSession,
    settings: Settings,
    league_id: int,
    season_repo: SeasonRepository,
) -> int | None:
    """Resolve the season_year to use for a league.

    Prefers a previously-synced current season; falls back to the configured
    default; otherwise asks the provider for a list and picks the most
    recent.
    """
    existing = await season_repo.find_by_external_id(settings.current_season_year)
    if existing is not None:
        return settings.current_season_year
    # Fall back to a provider call.
    try:
        seasons = await provider.fetch_seasons(league_id=league_id)
    except Exception:  # noqa: BLE001
        return None
    for dto in seasons:
        if dto.is_current:
            return dto.year
    return seasons[-1].year if seasons else None


async def _upsert_fixtures(
    *,
    dtos: list[FixtureDTO],
    fixture_repo: FixtureRepository,
    team_repo: TeamRepository,
    comp_id: int,
    season_id: int,
    metrics: SyncMetrics,
    mode: str,
) -> None:
    """Upsert each fixture DTO, resolving home/away team ids."""
    for dto in dtos:
        try:
            home = await team_repo.find_by_external_id(dto.home_team_external_id)
            away = await team_repo.find_by_external_id(dto.away_team_external_id)
            if home is None or away is None:
                metrics.skipped += 1
                metrics.add_error(
                    f"fixture {dto.external_id}: unknown team "
                    f"(home={dto.home_team_external_id}, away={dto.away_team_external_id})"
                )
                continue

            existing = await fixture_repo.find_by_external_id(dto.external_id)
            was_inserted = existing is None

            # Walk the special-case: finished-mode syncs may encounter
            # fixtures whose DB row is still ``scheduled`` but the DTO has
            # the final score. We always use ``upsert`` for the mutable
            # fields and, when the DTO is finished, additionally call
            # ``mark_finished`` so ``finished_at`` is stamped.
            fixture = await fixture_repo.upsert(
                external_id=dto.external_id,
                competition_id=comp_id,
                season_id=season_id,
                home_team_id=home.id,
                away_team_id=away.id,
                kickoff_time=dto.kickoff_time,
                matchday=dto.matchday,
                round=dto.round,
                venue=dto.venue,
                status=dto.status,
                status_short=dto.status_short,
            )

            if dto.is_finished and (
                fixture.home_goals is None or fixture.away_goals is None
            ):
                # Stamps finished_at/finished status/score.
                await fixture_repo.mark_finished(
                    fixture_id=fixture.id,
                    home_goals=dto.home_goals or 0,
                    away_goals=dto.away_goals or 0,
                )
                metrics.updated += 1
            elif was_inserted:
                metrics.inserted += 1
            else:
                # Anything else: idempotent re-write (counts as updated only
                # when the DSTO actually changed; we count conservatively).
                metrics.updated += 1
        except Exception as exc:  # noqa: BLE001  (per-fixture fault isolation)
            metrics.failed += 1
            metrics.add_error(f"fixture {dto.external_id}: {exc!r}")
            _LOG.warning(
                "sync_fixtures ({}): fixture {} failed {}", mode,
                dto.external_id, str(exc)[:256]
            )


def _add_days(now: datetime, days: int) -> datetime:
    """Pure helper avoiding hidden tz conversions inside the service."""
    from datetime import timedelta

    return now + timedelta(days=days)


def _add_hours(now: datetime, hours: int) -> datetime:
    """Pure helper for time windows."""
    from datetime import timedelta

    return now + timedelta(hours=hours)


__all__ = ["sync_finished_fixtures", "sync_upcoming_fixtures"]
