"""``Player`` repository."""

from __future__ import annotations

from datetime import date

from sqlalchemy import select

from app.models import Player
from app.repositories.base import BaseRepository


class PlayerRepository(BaseRepository[Player]):
    """Repository for :class:`Player`."""

    model = Player

    async def upsert(
        self,
        *,
        external_id: int,
        name: str,
        photo: str | None = None,
        nationality: str | None = None,
        birth_date: date | None = None,
        height_cm: int | None = None,
        weight_kg: int | None = None,
        position: str | None = None,
    ) -> Player:
        """Insert-or-update a player keyed by ``external_id``.

        Partial updates are supported: callers (e.g. the lineup DTO builder)
        that only know ``(external_id, name)`` are safe — they will not erase
        previously known attributes because we only overwrite the fields they
        actually passed in.
        """
        stmt = select(Player).where(Player.external_id == external_id)
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing is None:
            existing = Player(
                external_id=external_id,
                name=name,
                photo=photo,
                nationality=nationality,
                birth_date=birth_date,
                height_cm=height_cm,
                weight_kg=weight_kg,
                position=position,
            )
            self.session.add(existing)
        else:
            # Update only the fields where the caller provided a value (not
            # None). The CHECK constraint on `position` requires a member of
            # {'GK','DF','MF','FW'} OR NULL so we pass through as-is.
            existing.name = name
            if photo is not None:
                existing.photo = photo
            if nationality is not None:
                existing.nationality = nationality
            if birth_date is not None:
                existing.birth_date = birth_date
            if height_cm is not None:
                existing.height_cm = height_cm
            if weight_kg is not None:
                existing.weight_kg = weight_kg
            if position is not None:
                existing.position = position
        await self.session.flush()
        return existing

    async def find_by_external_id(self, external_id: int) -> Player | None:
        """Find a player by its provider external id."""
        stmt = select(Player).where(Player.external_id == external_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()
