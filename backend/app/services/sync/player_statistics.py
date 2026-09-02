"""Sync: players + player statistics."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.logging import get_logger
from app.providers.base import DataProvider
from app.providers.dto import PlayerDTO
from app.repositories import (
    CompetitionRepository,
    PlayerRepository,
    PlayerStatisticsRepository,
    SeasonRepository,
    TeamRepository,
)
from app.services.sync.metrics import SyncMetrics

_LOG = get_logger()


async def sync_player_statistics(
    *,
    provider: DataProvider,
    session: AsyncSession,
    settings: Settings,
    league_ids: list[int] | None = None,
) -> SyncMetrics:
    """Sync squads + per-player statistics for the configured leagues.

    Strategy:
    1. For each league, fetch its teams.
    2. For each team, fetch the squad and upsert partial ``players`` rows
       (name + external_id; the rest can be filled by a later enrichment).
    3. For each player we just upserted, fetch the player statistics for
       this league+season and upsert ``player_statistics``.

    Cost: 1 + N + (≤20*N) requests per league where N = number of teams.
    The HTTP client enforces the per-minute rate limit so this stays within
    the provider's plan.
    """
    metrics = SyncMetrics(job="sync_player_statistics")
    league_ids = list(league_ids or settings.current_league_ids())
    metrics.requested = len(league_ids)

    comp_repo = CompetitionRepository(session)
    season_repo = SeasonRepository(session)
    team_repo = TeamRepository(session)
    player_repo = PlayerRepository(session)
    stats_repo = PlayerStatisticsRepository(session)

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

            teams = await provider.fetch_teams(
                league_id=league_id, season_year=settings.current_season_year
            )
            for team_dto in teams:
                try:
                    team = await team_repo.find_by_external_id(team_dto.external_id)
                    if team is None:
                        metrics.skipped += 1
                        continue
                    players = await provider.fetch_players(
                        team_id=team.external_id,
                        season_year=settings.current_season_year,
                    )
                    metrics.received += len(players)
                    for player in players:
                        await _upsert_player_with_stats(
                            player=player,
                            player_repo=player_repo,
                            stats_repo=stats_repo,
                            team_id=team.id,
                            comp_id=comp.id,
                            season_id=season.id,
                            league_id=league_id,
                            settings=settings,
                            provider=provider,
                            metrics=metrics,
                        )
                except Exception as exc:  # noqa: BLE001
                    metrics.failed += 1
                    metrics.add_error(
                        f"team {team_dto.external_id}: {exc!r}"
                    )
                    _LOG.warning("sync_player_statistics: team {} failed {}",
                                 team_dto.external_id, str(exc)[:256])
        except Exception as exc:  # noqa: BLE001
            metrics.failed += 1
            metrics.add_error(f"league {league_id}: {exc!r}")
            _LOG.warning("sync_player_statistics: league {} failed {}",
                         league_id, str(exc)[:256])

    _LOG.info("sync_player_statistics done: {}", metrics.as_dict())
    return metrics


async def _upsert_player_with_stats(
    *,
    player: PlayerDTO,
    player_repo: PlayerRepository,
    stats_repo: PlayerStatisticsRepository,
    team_id: int,
    comp_id: int,
    season_id: int,
    league_id: int,
    settings: Settings,
    provider: DataProvider,
    metrics: SyncMetrics,
) -> None:
    """Upsert one player row + fetch + upsert the player's stats.

    The ``PlayerDTO`` only carries the provider's ``external_id``. After
    ``player_repo.upsert`` we get back the persisted :class:`Player` whose
    ``id`` is OUR internal primary key — that's the value the
    ``player_statistics.player_id`` FK column expects. Using
    ``player.external_id`` here would write the provider's id into a FK
    column and either violate the FK constraint or (worse) silently link
    the stats to the wrong internal row.
    """
    try:
        persisted_player = await player_repo.upsert(
            external_id=player.external_id,
            name=player.name,
            photo=player.photo,
            nationality=player.nationality,
            birth_date=player.birth_date,
            height_cm=player.height_cm,
            weight_kg=player.weight_kg,
            position=player.position,
        )
        # Sanity-check: a Player just-flushed MUST have an internal id.
        assert persisted_player.id is not None  # noqa: S101  (defensive)
        stats = await provider.fetch_player_statistics(
            player_id=player.external_id,
            league_id=league_id,
            season_year=settings.current_season_year,
        )
        if stats is None:
            metrics.skipped += 1
            return
        await stats_repo.upsert(
            player_id=persisted_player.id,
            team_id=team_id,
            competition_id=comp_id,
            season_id=season_id,
            appearances=stats.appearances,
            starts=stats.starts,
            minutes_played=stats.minutes_played,
            goals=stats.goals,
            assists=stats.assists,
            yellow_cards=stats.yellow_cards,
            red_cards=stats.red_cards,
            rating=stats.rating,
        )
        metrics.updated += 1
    except Exception as exc:  # noqa: BLE001
        metrics.failed += 1
        metrics.add_error(f"player {player.external_id}: {exc!r}")
        _LOG.warning("sync_player_statistics: player {} failed {}",
                     player.external_id, str(exc)[:256])


__all__ = ["sync_player_statistics"]
