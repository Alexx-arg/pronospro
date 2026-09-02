"""Sync: teams."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.logging import get_logger
from app.providers.base import DataProvider
from app.repositories import (
    CompetitionRepository,
    SeasonRepository,
    TeamRepository,
)
from app.services.sync.metrics import SyncMetrics

_LOG = get_logger()


async def sync_teams(
    *,
    provider: DataProvider,
    session: AsyncSession,
    settings: Settings,
    league_ids: list[int] | None = None,
) -> SyncMetrics:
    """Sync team rosters per league + season.

    Idempotent upsert keyed by ``external_id``. A team participating in two
    leagues is upserted twice but ends up as a single row.
    """
    metrics = SyncMetrics(job="sync_teams")
    league_ids = list(league_ids or settings.current_league_ids())
    metrics.requested = len(league_ids)

    comp_repo = CompetitionRepository(session)
    season_repo = SeasonRepository(session)
    team_repo = TeamRepository(session)

    for league_id in league_ids:
        try:
            comp = await comp_repo.find_by_external_id(league_id)
            if comp is None:
                metrics.failed += 1
                metrics.add_error(f"league {league_id}: missing competition row")
                _LOG.warning("sync_teams: league {} unknown", league_id)
                continue

            season = await season_repo.find_for_competition_and_year(
                competition_id=comp.id, year=settings.current_season_year
            )
            if season is None:
                metrics.skipped += 1
                metrics.add_error(
                    f"league {league_id}: season "
                    f"{settings.current_season_year} not in DB"
                )
                continue

            teams = await provider.fetch_teams(
                league_id=league_id, season_year=settings.current_season_year
            )
            metrics.received += len(teams)
            for dto in teams:
                try:
                    await team_repo.upsert(
                        external_id=dto.external_id,
                        name=dto.name,
                        short_name=dto.short_name,
                        code=dto.code,
                        country=dto.country,
                        logo=dto.logo,
                        venue=dto.venue,
                        founded=dto.founded,
                    )
                    metrics.updated += 1
                except Exception as exc:  # noqa: BLE001
                    metrics.failed += 1
                    metrics.add_error(f"team {dto.external_id}: {exc!r}")
                    _LOG.warning("sync_teams: team {} failed {}",
                                 dto.external_id, str(exc)[:256])
        except Exception as exc:  # noqa: BLE001
            metrics.failed += 1
            metrics.add_error(f"league {league_id}: {exc!r}")
            _LOG.warning("sync_teams: league {} failed {}", league_id, str(exc)[:256])

    _LOG.info("sync_teams done: {}", metrics.as_dict())
    return metrics


__all__ = ["sync_teams"]
