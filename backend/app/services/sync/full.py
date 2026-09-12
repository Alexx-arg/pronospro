"""Full bootstrap sync (PronosPro ingest).

Runs the ordered pipeline once to populate PostgreSQL with real data:
competitions+seasons → teams → upcoming fixtures (7 days) → team statistics.
Each stage reuses its standalone sync service so there is a single source of
truth for the sync logic; this module only orchestrates order + transaction.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.logging import get_logger
from app.providers.base import DataProvider
from app.services.sync import (
    sync_competitions_and_seasons,
    sync_team_statistics,
    sync_teams,
    sync_upcoming_fixtures,
)
from app.services.sync.metrics import SyncMetrics

_LOG = get_logger()


async def run_full_bootstrap(
    *,
    provider: DataProvider,
    session: AsyncSession,
    settings: Settings,
    league_ids: list[int] | None = None,
    days: int | None = None,
) -> dict[str, object]:
    """Execute every sync stage in dependency order.

    Returns a mapping of ``stage -> SyncMetrics.as_dict()`` so callers (admin
    endpoint, CLI) can surface per-stage outcomes.
    """
    leagues = list(league_ids or settings.current_league_ids())
    fixture_days = days if days is not None else settings.sync_upcoming_days

    report: dict[str, object] = {}

    _LOG.info("bootstrap: competitions+seasons for leagues={}", leagues)
    report["competitions_seasons"] = (
        await sync_competitions_and_seasons(
            provider=provider, session=session, settings=settings, league_ids=leagues
        )
    ).as_dict()

    _LOG.info("bootstrap: teams for leagues={}", leagues)
    report["teams"] = (
        await sync_teams(
            provider=provider, session=session, settings=settings, league_ids=leagues
        )
    ).as_dict()

    _LOG.info("bootstrap: upcoming fixtures ({} days)", fixture_days)
    report["upcoming_fixtures"] = (
        await sync_upcoming_fixtures(
            provider=provider,
            session=session,
            settings=settings,
            league_ids=leagues,
            days=fixture_days,
        )
    ).as_dict()

    _LOG.info("bootstrap: team statistics")
    report["team_statistics"] = (
        await sync_team_statistics(
            provider=provider, session=session, settings=settings, league_ids=leagues
        )
    ).as_dict()

    _LOG.info("bootstrap complete: {}", report)
    return report


async def run_full_bootstrap_metrics(
    report: dict[str, object],
) -> SyncMetrics:
    """Aggregate a bootstrap report into a single :class:`SyncMetrics`."""
    agg = SyncMetrics(job="full_bootstrap")
    for stage in report.values():
        if not isinstance(stage, dict):
            continue
        agg.requested += int(stage.get("requested", 0))
        agg.received += int(stage.get("received", 0))
        agg.inserted += int(stage.get("inserted", 0))
        agg.updated += int(stage.get("updated", 0))
        agg.skipped += int(stage.get("skipped", 0))
        agg.failed += int(stage.get("failed", 0))
        for err in stage.get("errors", []):
            agg.add_error(str(err))
    return agg


__all__ = ["run_full_bootstrap", "run_full_bootstrap_metrics"]