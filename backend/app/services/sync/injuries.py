"""Sync: injuries.

The provider returns ``InjuryDTO`` carrying ``player_external_id`` and
``team_external_id``. The repository's ``injuries.player_id`` and
``injuries.team_id`` FK columns expect the **internal** PKs
(``players.id``, ``teams.id``). Resolving external → internal is done
once per DTO, so the upsert never sees a raw external id.

A secondary invariant: ``injuries.competition_id`` and
``injuries.fixture_id`` reference internal ``competitions.id`` and
``fixtures.id`` (both can be NULL when the provider doesn't include the
optional competition/fixture attribution). We resolve them when present;
we leave them NULL when absent (NEVER fabricate an id).
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.logging import get_logger
from app.providers.base import DataProvider
from app.repositories import (
    CompetitionRepository,
    FixtureRepository,
    InjuryRepository,
    PlayerRepository,
    TeamRepository,
)
from app.services.sync.metrics import SyncMetrics

_LOG = get_logger()


async def sync_injuries(
    *,
    provider: DataProvider,
    session: AsyncSession,
    settings: Settings,
    league_ids: list[int] | None = None,
) -> SyncMetrics:
    """Sync injuries for the configured leagues + current season."""
    metrics = SyncMetrics(job="sync_injuries")
    league_ids = list(league_ids or settings.current_league_ids())
    metrics.requested = len(league_ids)

    comp_repo = CompetitionRepository(session)
    player_repo = PlayerRepository(session)
    team_repo = TeamRepository(session)
    injury_repo = InjuryRepository(session)
    fixture_repo = FixtureRepository(session)

    for league_id in league_ids:
        try:
            injuries = await provider.fetch_injuries(
                league_id=league_id, season_year=settings.current_season_year
            )
            metrics.received += len(injuries)

            for dto in injuries:
                try:
                    # Resolve external → internal ids. Missing player/team
                    # skips the row rather than inserting half-known data.
                    player = await player_repo.find_by_external_id(
                        dto.player_external_id
                    )
                    team = await team_repo.find_by_external_id(
                        dto.team_external_id
                    )
                    if player is None or team is None:
                        metrics.skipped += 1
                        metrics.add_error(
                            f"injury {dto.external_id}: "
                            f"missing player/team "
                            f"(player_ext={dto.player_external_id}, "
                            f"team_ext={dto.team_external_id})"
                        )
                        continue

                    comp_internal_id: int | None = None
                    if dto.competition_external_id is not None:
                        comp = await comp_repo.find_by_external_id(
                            dto.competition_external_id
                        )
                        if comp is not None:
                            comp_internal_id = comp.id
                    fixture_internal_id: int | None = None
                    if dto.fixture_external_id is not None:
                        fixture = await fixture_repo.find_by_external_id(
                            dto.fixture_external_id
                        )
                        if fixture is not None:
                            fixture_internal_id = fixture.id

                    await injury_repo.upsert(
                        external_id=dto.external_id,
                        player_id=player.id,
                        team_id=team.id,
                        start_date=dto.start_date,
                        competition_id=comp_internal_id,
                        fixture_id=fixture_internal_id,
                        type=dto.type,
                        reason=dto.reason,
                        status=dto.status,
                        end_date=dto.end_date,
                        updated_external_at=dto.updated_external_at,
                    )
                    metrics.updated += 1
                except Exception as exc:  # noqa: BLE001
                    metrics.failed += 1
                    metrics.add_error(f"injury {dto.external_id}: {exc!r}")
                    _LOG.warning("sync_injuries: injury {} failed {}",
                                 dto.external_id, str(exc)[:256])
        except Exception as exc:  # noqa: BLE001
            metrics.failed += 1
            metrics.add_error(f"league {league_id}: {exc!r}")
            _LOG.warning("sync_injuries: league {} failed {}", league_id, str(exc)[:256])

    _LOG.info("sync_injuries done: {}", metrics.as_dict())
    return metrics


__all__ = ["sync_injuries"]
