"""``Injury`` repository."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import select

from app.models import Injury
from app.repositories.base import BaseRepository


class InjuryRepository(BaseRepository[Injury]):
    """Repository for :class:`Injury`."""

    model = Injury

    async def upsert(
        self,
        *,
        external_id: int,
        player_id: int,
        team_id: int,
        start_date: date,
        competition_id: int | None = None,
        fixture_id: int | None = None,
        type: str | None = None,
        reason: str | None = None,
        status: str = "active",
        end_date: date | None = None,
        updated_external_at: datetime | None = None,
    ) -> Injury:
        """Insert-or-update an injury row keyed by ``external_id``.

        Provider payloads typically refresh ``status`` and ``end_date`` as
        the player progresses. We update mutable fields; ``start_date`` stays
        the same.
        """
        stmt = select(Injury).where(Injury.external_id == external_id)
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing is None:
            existing = Injury(
                external_id=external_id,
                player_id=player_id,
                team_id=team_id,
                competition_id=competition_id,
                fixture_id=fixture_id,
                type=type,
                reason=reason,
                status=status,
                start_date=start_date,
                end_date=end_date,
                updated_external_at=updated_external_at,
            )
            self.session.add(existing)
        else:
            existing.player_id = player_id
            existing.team_id = team_id
            existing.competition_id = competition_id
            existing.fixture_id = fixture_id
            existing.type = type
            existing.reason = reason
            existing.status = status
            existing.end_date = end_date
            existing.updated_external_at = updated_external_at
        await self.session.flush()
        return existing

    async def list_active(
        self,
        *,
        competition_id: int | None = None,
    ) -> list[Injury]:
        """Return injuries flagged ``active`` or ``doubtful``."""
        stmt = select(Injury).where(
            Injury.status.in_(("active", "doubtful")),
        )
        if competition_id is not None:
            stmt = stmt.where(Injury.competition_id == competition_id)
        return list((await self.session.execute(stmt)).scalars().all())

    async def find_by_external_id(self, external_id: int) -> Injury | None:
        """Find an injury by its provider external id."""
        stmt = select(Injury).where(Injury.external_id == external_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()
