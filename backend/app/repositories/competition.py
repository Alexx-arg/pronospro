"""``Competition`` repository."""

from __future__ import annotations

from sqlalchemy import select

from app.models import Competition
from app.repositories.base import BaseRepository


class CompetitionRepository(BaseRepository[Competition]):
    """Repository for :class:`Competition`."""

    model = Competition

    async def upsert(
        self,
        *,
        external_id: int,
        name: str,
        type: str,
        country: str | None = None,
        logo: str | None = None,
    ) -> Competition:
        """Insert-or-update a competition keyed by ``external_id``.

        Update of an existing row is permitted here because ``competitions``
        is a mutable operational table (SCHEMA.md §14).
        """
        stmt = select(Competition).where(Competition.external_id == external_id)
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing is None:
            existing = Competition(
                external_id=external_id,
                name=name,
                type=type,
                country=country,
                logo=logo,
            )
            self.session.add(existing)
        else:
            existing.name = name
            existing.type = type
            existing.country = country
            existing.logo = logo
        await self.session.flush()
        return existing

    async def find_by_external_id(self, external_id: int) -> Competition | None:
        """Find a competition by its provider external id."""
        stmt = select(Competition).where(Competition.external_id == external_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_by_external_ids(
        self, external_ids: list[int]
    ) -> list[Competition]:
        """Bulk lookup of competitions by external ids (returned in any
        order)."""
        if not external_ids:
            return []
        stmt = select(Competition).where(Competition.external_id.in_(external_ids))
        return list((await self.session.execute(stmt)).scalars().all())
