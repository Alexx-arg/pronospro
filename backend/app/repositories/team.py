"""``Team`` repository."""

from __future__ import annotations

from sqlalchemy import select

from app.models import Team
from app.repositories.base import BaseRepository


class TeamRepository(BaseRepository[Team]):
    """Repository for :class:`Team`."""

    model = Team

    async def upsert(
        self,
        *,
        external_id: int,
        name: str,
        short_name: str | None = None,
        code: str | None = None,
        country: str | None = None,
        logo: str | None = None,
        venue: str | None = None,
        founded: int | None = None,
    ) -> Team:
        """Insert-or-update a team keyed by ``external_id``."""
        stmt = select(Team).where(Team.external_id == external_id)
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing is None:
            existing = Team(
                external_id=external_id,
                name=name,
                short_name=short_name,
                code=code,
                country=country,
                logo=logo,
                venue=venue,
                founded=founded,
            )
            self.session.add(existing)
        else:
            existing.name = name
            existing.short_name = short_name
            existing.code = code
            existing.country = country
            existing.logo = logo
            existing.venue = venue
            existing.founded = founded
        await self.session.flush()
        return existing

    async def find_by_external_id(self, external_id: int) -> Team | None:
        """Find a team by its provider external id."""
        stmt = select(Team).where(Team.external_id == external_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_by_external_ids(
        self, external_ids: list[int]
    ) -> list[Team]:
        """Bulk lookup of teams by external ids."""
        if not external_ids:
            return []
        stmt = select(Team).where(Team.external_id.in_(external_ids))
        return list((await self.session.execute(stmt)).scalars().all())
