"""Sync: competitions and seasons.

Always runs *first* (other syncs depend on the existence of competition +
season rows in PostgreSQL). The service:

1. Fetches leagues metadata for ``CURRENT_LEAGUES``.
2. For each league, fetches seasons.
3. Upserts competitions and seasons idempotently.

Idempotency: keyed by ``external_id`` (competitions) and
``(competition_id, year)`` (seasons). Re-running the job never drowns
existing rows: it just refreshes mutable fields.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.logging import get_logger
from app.providers.base import DataProvider
from app.repositories import (
    CompetitionRepository,
    SeasonRepository,
)
from app.services.sync.metrics import SyncMetrics

_LOG = get_logger()


async def sync_competitions_and_seasons(
    *,
    provider: DataProvider,
    session: AsyncSession,
    settings: Settings,
    league_ids: list[int] | None = None,
) -> SyncMetrics:
    """Idempotent upsert of competitions + seasons for the configured leagues.

    Args:
        provider: data provider instance.
        session: the SQLAlchemy async session (caller owns the transaction).
        settings: application settings; used only for the default league list.
        league_ids: override the configured league list when ``None``.

    Returns:
        :class:`SyncMetrics` with per-entity counters.
    """
    metrics = SyncMetrics(job="sync_competitions_and_seasons")
    league_ids = list(league_ids or settings.current_league_ids())
    metrics.requested = len(league_ids)

    comp_repo = CompetitionRepository(session)
    season_repo = SeasonRepository(session)

    competitions = await provider.fetch_leagues(league_ids=league_ids)
    metrics.received = len(competitions)

    for dto in competitions:
        try:
            await comp_repo.upsert(
                external_id=dto.external_id,
                name=dto.name,
                type=dto.type,
                country=dto.country,
                logo=dto.logo,
            )
            metrics.inserted += 1
        except Exception as exc:  # noqa: BLE001
            metrics.failed += 1
            metrics.add_error(f"competition {dto.external_id}: {exc!r}")
            _LOG.warning("sync_competitions: failed upsert {}", str(exc)[:256])

    for league_id in league_ids:
        try:
            seasons = await provider.fetch_seasons(league_id=league_id)
        except Exception as exc:  # noqa: BLE001
            metrics.add_error(f"seasons league={league_id}: {exc!r}")
            _LOG.warning("sync_competitions: fetch_seasons failed {}", str(exc)[:256])
            continue

        for dto in seasons:
            try:
                comp = await comp_repo.find_by_external_id(
                    dto.competition_external_id
                )
                if comp is None:
                    metrics.skipped += 1
                    continue
                # Mark only the requested season year as ``is_current`` to
                # avoid stale flags when the provider lists it (most
                # providers expose multiple "current" seasons per league).
                is_current_flag = dto.is_current and dto.year == settings.current_season_year
                await season_repo.upsert(
                    competition_id=comp.id,
                    external_id=dto.external_id,
                    year=dto.year,
                    start_date=dto.start_date,
                    end_date=dto.end_date,
                    is_current=is_current_flag,
                )
                metrics.updated += 1
            except Exception as exc:  # noqa: BLE001
                metrics.failed += 1
                metrics.add_error(
                    f"season {dto.external_id}/{dto.year}: {exc!r}"
                )
                _LOG.warning("sync_competitions: season upsert failed {}",
                             str(exc)[:256])

    _LOG.info("sync_competitions_and_seasons done: {}", metrics.as_dict())
    return metrics


async def current_season_for(
    *,
    provider: DataProvider,
    session: AsyncSession,
    settings: Settings,
    league_id: int,
) -> int | None:
    """Resolve the season year the configured job should target for a league.

    Returns ``settings.current_season_year`` if it's already known in the DB
    (matching it against any season by external_id, since ``external_id`` of
    a ``Season`` is the starting year — see SCHEMA.md §2). Falls back to
    asking the provider's ``fetch_seasons`` for the league.
    """
    season_repo = SeasonRepository(session)
    if await season_repo.find_by_external_id(settings.current_season_year) is not None:
        return settings.current_season_year
    try:
        seasons = await provider.fetch_seasons(league_id=league_id)
    except Exception:  # noqa: BLE001
        return None
    for dto in seasons:
        if dto.is_current:
            return dto.year
    return seasons[-1].year if seasons else None


__all__: list[str] = ["current_season_for", "sync_competitions_and_seasons"]
