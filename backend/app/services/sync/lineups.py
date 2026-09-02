"""Sync: lineups.

The provider returns home/away :class:`LineupDTO` instances that include
the fixture's ``external_id`` and each player's ``external_id``. The
``LineupRepository.upsert`` contract expects:

* ``fixture_id``  -> internal ``fixtures.id``
* ``team_id``     -> internal ``teams.id``
* each ``LineupPlayerSlot.player_id`` -> internal ``players.id``

The service resolves every external_id BEFORE calling the repository, so
the repository never sees the provider's ids.

We additionally resolve ``is_home`` based on the fixture row currently
stored in our DB (the DTO carries ``is_home=False`` placeholder; the
provider's ``/fixtures/lineups`` endpoint doesn't tell us which team is
home vs away — we look it up from ``fixtures.home_team_id``).
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.logging import get_logger
from app.models import Fixture
from app.providers.base import DataProvider
from app.providers.dto import LineupDTO
from app.repositories import (
    FixtureRepository,
    LineupRepository,
    PlayerRepository,
    TeamRepository,
)
from app.repositories.lineup import LineupPlayerSlot
from app.services.sync.metrics import SyncMetrics

_LOG = get_logger()


async def sync_lineups(
    *,
    provider: DataProvider,
    session: AsyncSession,
    settings: Settings,
    fixture_external_ids: list[int] | None = None,
) -> SyncMetrics:
    """Sync lineups for the requested fixtures.

    When ``fixture_external_ids`` is ``None`` we look up the upcoming
    fixtures (scheduled + in_play) within ``settings.sync_upcoming_days``
    days and ask the provider for each one's lineup. The provider returns
    an empty list when the lineup isn't published yet — we count it as
    ``skipped`` (not a failure).
    """
    metrics = SyncMetrics(job="sync_lineups")

    fixture_repo = FixtureRepository(session)
    lineup_repo = LineupRepository(session)
    player_repo = PlayerRepository(session)
    team_repo = TeamRepository(session)

    if fixture_external_ids is None:
        upcoming = await fixture_repo.list_upcoming(
            from_time=_now_utc(),
            limit=400,
        )
        fixtures: list[Fixture] = upcoming
    else:
        fixtures = []
        for ext_id in fixture_external_ids:
            fx = await fixture_repo.find_by_external_id(ext_id)
            if fx is None:
                metrics.skipped += 1
                metrics.add_error(f"fixture {ext_id}: not in DB")
                continue
            fixtures.append(fx)

    metrics.requested = len(fixtures)

    # Cache external_id → internal player id within this sync run.
    player_id_cache: dict[int, int | None] = {}

    for fx in fixtures:
        try:
            lineup_dtos = await provider.fetch_lineup(fx.external_id)
            metrics.received += len(lineup_dtos)

            if not lineup_dtos:
                metrics.skipped += 1
                continue

            # The DTO has ``is_home=False`` placeholder; resolve the
            # real side from the fixture's stored team ids.
            for dto in lineup_dtos:
                team = await team_repo.find_by_external_id(dto.team_external_id)
                if team is None:
                    metrics.skipped += 1
                    metrics.add_error(
                        f"lineup team {dto.team_external_id}: not in DB"
                    )
                    continue
                is_home = team.id == fx.home_team_id

                slots: list[LineupPlayerSlot] = []
                for slot_dto in dto.players:
                    internal_id = player_id_cache.get(slot_dto.player_external_id)
                    if internal_id is None and slot_dto.player_external_id not in player_id_cache:
                        player = await player_repo.find_by_external_id(
                            slot_dto.player_external_id
                        )
                        internal_id = player.id if player is not None else None
                        player_id_cache[slot_dto.player_external_id] = internal_id
                    if internal_id is None:
                        metrics.skipped += 1
                        metrics.add_error(
                            f"lineup player {slot_dto.player_external_id}: not in DB"
                        )
                        continue
                    slots.append(
                        LineupPlayerSlot(
                            player_id=internal_id,
                            position=slot_dto.position,
                            position_x=slot_dto.position_x,
                            position_y=slot_dto.position_y,
                            shirt_number=slot_dto.shirt_number,
                            is_starter=slot_dto.is_starter,
                        )
                    )

                await lineup_repo.upsert(
                    fixture_id=fx.id,
                    team_id=team.id,
                    is_home=is_home,
                    formation=dto.formation,
                    coach=dto.coach,
                    updated_external_at=dto.updated_external_at,
                    players=slots,
                )
                metrics.updated += 1
        except Exception as exc:  # noqa: BLE001
            metrics.failed += 1
            metrics.add_error(f"fixture {fx.external_id}: {exc!r}")
            _LOG.warning("sync_lineups: fixture {} failed {}",
                         fx.external_id, str(exc)[:256])

    _LOG.info("sync_lineups done: {}", metrics.as_dict())
    return metrics


def _now_utc() -> "datetime":  # type: ignore[name-defined]
    """Return the current UTC datetime (lazy import to keep top imports tidy)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


__all__ = ["sync_lineups"]
