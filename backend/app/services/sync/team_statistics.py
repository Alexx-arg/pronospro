"""Sync: team statistics."""

from __future__ import annotations

from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.logging import get_logger
from app.providers.base import DataProvider
from app.repositories import (
    CompetitionRepository,
    SeasonRepository,
    TeamRepository,
    TeamStatisticsRepository,
)
from app.services.sync.metrics import SyncMetrics

_LOG = get_logger()


async def sync_team_statistics(
    *,
    provider: DataProvider,
    session: AsyncSession,
    settings: Settings,
    league_ids: list[int] | None = None,
) -> SyncMetrics:
    """Sync team statistics for each configured competition+season.

    Iterates the teams known to participate in each league (those already
    persisted by :func:`app.services.sync.teams.sync_teams`) and asks the
    provider for one stats payload per team. The fetched stats are stamped
    with today's ``as_of_date`` so future snapshots don't collide.
    """
    metrics = SyncMetrics(job="sync_team_statistics")
    league_ids = list(league_ids or settings.current_league_ids())
    metrics.requested = len(league_ids)

    comp_repo = CompetitionRepository(session)
    season_repo = SeasonRepository(session)
    team_repo = TeamRepository(session)
    stats_repo = TeamStatisticsRepository(session)

    today = date.today()

    for league_id in league_ids:
        try:
            comp = await comp_repo.find_by_external_id(league_id)
            if comp is None:
                metrics.failed += 1
                metrics.add_error(f"league {league_id}: missing competition row")
                continue

            season = await season_repo.find_for_competition_and_year(
                competition_id=comp.id, year=settings.current_season_year
            )
            if season is None:
                metrics.skipped += 1
                continue

            # Iterate teams via the Fixture rows: teams present as home/away
            # in this competition+season. Simpler & exhaustive than asking
            # the provider for the league roster again.
            team_external_ids = await _fetch_team_ids_for_competition(
                session=session,
                settings=settings,
                provider=provider,
                league_id=league_id,
            )
            if not team_external_ids:
                metrics.skipped += 1
                metrics.add_error(f"league {league_id}: no teams resolved")
                continue

            for ext_team_id in team_external_ids:
                try:
                    team = await team_repo.find_by_external_id(ext_team_id)
                    if team is None:
                        metrics.skipped += 1
                        continue

                    stats = await provider.fetch_team_statistics(
                        team_id=team.external_id,
                        league_id=league_id,
                        season_year=settings.current_season_year,
                        as_of_date=today,
                    )
                    if stats is None:
                        metrics.skipped += 1
                        continue
                    metrics.received += 1

                    # Idempotent upsert keyed by (team_id, comp_id, season_id,
                    # as_of_date). Always counts as updated when the (...)
                    # row was found; counts as inserted otherwise. We don't
                    # differentiate because it requires either a flush()'d
                    # identity map or a separate SELECT before.
                    existing = await stats_repo.latest_for(
                        team_id=team.id,
                        competition_id=comp.id,
                        season_id=season.id,
                        as_of_before=today,
                    )
                    await stats_repo.upsert(
                        team_id=team.id,
                        competition_id=comp.id,
                        season_id=season.id,
                        as_of_date=today,
                        fixtures_played=stats.fixtures_played,
                        wins=stats.wins,
                        draws=stats.draws,
                        losses=stats.losses,
                        goals_for=stats.goals_for,
                        goals_against=stats.goals_against,
                        clean_sheets=stats.clean_sheets,
                        failed_to_score=stats.failed_to_score,
                        form=stats.form,
                        shots_total=stats.shots_total,
                        shots_on_target=stats.shots_on_target,
                        shots_inside_box=stats.shots_inside_box,
                        shots_outside_box=stats.shots_outside_box,
                        fouls=stats.fouls,
                        corners=stats.corners,
                        offsides=stats.offsides,
                        possession_avg=stats.possession_avg,
                        yellow_cards=stats.yellow_cards,
                        red_cards=stats.red_cards,
                        passes_total=stats.passes_total,
                        passes_accuracy=stats.passes_accuracy,
                        xg=stats.xg,
                        xga=stats.xga,
                    )
                    if existing is None:
                        metrics.inserted += 1
                    else:
                        metrics.updated += 1
                except Exception as exc:  # noqa: BLE001
                    metrics.failed += 1
                    metrics.add_error(
                        f"team {ext_team_id} league {league_id}: {exc!r}"
                    )
                    _LOG.warning("sync_team_statistics: team {} failed {}",
                                 ext_team_id, str(exc)[:256])
        except Exception as exc:  # noqa: BLE001
            metrics.failed += 1
            metrics.add_error(f"league {league_id}: {exc!r}")
            _LOG.warning("sync_team_statistics: league {} failed {}",
                         league_id, str(exc)[:256])

    _LOG.info("sync_team_statistics done: {}", metrics.as_dict())
    return metrics


async def _fetch_team_ids_for_competition(
    *,
    session: AsyncSession,
    settings: Settings,
    provider: DataProvider,
    league_id: int,
) -> list[int]:
    """Return the external team ids participating in a league-season.

    We prefer asking the provider directly (one call) to avoid scanning
    fixtures. If that's empty (e.g. the list = None), fall back to
    fetching teams.
    """
    try:
        teams = await provider.fetch_teams(
            league_id=league_id, season_year=settings.current_season_year
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("_fetch_team_ids_for_competition: {}", str(exc)[:256])
        return []
    return [t.external_id for t in teams]


__all__ = ["sync_team_statistics"]
