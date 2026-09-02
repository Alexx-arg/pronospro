"""``Lineup`` and ``LineupPlayer`` repository.

The repository treats `player_id` (FK to ``players.id``) as an internal
primary key. The contract is intentionally narrow:

* Callers pass a :class:`LineupPlayerSlot` describing each player slot;
  ``player_id`` here is the **internal** ``players.id`` resolved upstream
  by the sync service via :meth:`PlayerRepository.find_by_external_id`.
* The repository itself NEVER translates ``external_id`` → ``id``: that
  belongs to the service layer (single responsibility).

This avoids the class of bug where a service accidentally passes a
provider ``external_id`` straight into a FK column. By typing the slot
payload with a clearly-named field ``player_id``, mypy strict can flag
mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, select

from app.models import Lineup, LineupPlayer
from app.repositories.base import BaseRepository


@dataclass(slots=True, frozen=True)
class LineupPlayerSlot:
    """A single player slot in a lineup, as the repository sees it.

    ``player_id`` MUST already be the internal ``players.id``. The sync
    service is responsible for resolving the provider's ``external_id``
    via :meth:`PlayerRepository.find_by_external_id` before constructing
    a slot.
    """

    player_id: int
    position: str | None = None
    position_x: int | None = None
    position_y: int | None = None
    shirt_number: int | None = None
    is_starter: bool = True


class LineupRepository(BaseRepository[Lineup]):
    """Repository for :class:`Lineup` and its players."""

    model = Lineup

    async def upsert(
        self,
        *,
        fixture_id: int,
        team_id: int,
        is_home: bool,
        formation: str | None = None,
        coach: str | None = None,
        updated_external_at: datetime | None = None,
        players: list[LineupPlayerSlot] | None = None,
    ) -> Lineup:
        """Insert-or-update a lineup keyed by ``(fixture_id, team_id)``.

        ``fixture_id`` and ``team_id`` MUST be the internal ``fixtures.id``
        and ``teams.id`` already resolved by the caller; see
        :mod:`app.services.sync.lineups` for an idiomatic resolution path.

        ``players`` is OPTIONAL: pass ``None`` to leave any existing player
        rows intact. When a list is provided, the existing rows for this
        lineup are deleted and replaced (the provider always returns the
        full lineup on each call, so a clean replace is idempotent and
        safe under retry).
        """
        stmt = select(Lineup).where(
            Lineup.fixture_id == fixture_id,
            Lineup.team_id == team_id,
        )
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing is None:
            existing = Lineup(
                fixture_id=fixture_id,
                team_id=team_id,
                is_home=is_home,
                formation=formation,
                coach=coach,
                updated_external_at=updated_external_at,
            )
            self.session.add(existing)
            await self.session.flush()
        else:
            existing.is_home = is_home
            existing.formation = formation
            existing.coach = coach
            existing.updated_external_at = updated_external_at
            await self.session.flush()

        if players is not None:
            # Atomic replace: delete slot rows then re-insert fresh copies.
            await self.session.execute(
                delete(LineupPlayer).where(LineupPlayer.lineup_id == existing.id)
            )
            for slot in players:
                self.session.add(self._build_lineup_player(existing.id, slot))
            await self.session.flush()
        return existing

    @staticmethod
    def _build_lineup_player(lineup_id: int, slot: LineupPlayerSlot) -> LineupPlayer:
        """Create a fresh :class:`LineupPlayer` ORM row from a slot DTO."""
        return LineupPlayer(
            lineup_id=lineup_id,
            player_id=slot.player_id,
            position=slot.position,
            position_x=slot.position_x,
            position_y=slot.position_y,
            shirt_number=slot.shirt_number,
            is_starter=slot.is_starter,
        )

    async def find_for_fixture(self, fixture_id: int) -> list[Lineup]:
        """Return the two lineups (home + away) for a fixture (or fewer)."""
        stmt = select(Lineup).where(Lineup.fixture_id == fixture_id)
        return list((await self.session.execute(stmt)).scalars().all())
