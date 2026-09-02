"""``Season`` repository."""

from __future__ import annotations

from datetime import date

from sqlalchemy import select

from app.models import Season
from app.repositories.base import BaseRepository


class SeasonRepository(BaseRepository[Season]):
    """Repository for :class:`Season`."""

    model = Season

    async def upsert(
        self,
        *,
        competition_id: int,
        external_id: int,
        year: int,
        start_date: date | None = None,
        end_date: date | None = None,
        is_current: bool = False,
    ) -> Season:
        """Insert or update a season keyed by ``(competition_id, year)``."""
        stmt = select(Season).where(
            Season.competition_id == competition_id,
            Season.year == year,
        )
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing is None:
            existing = Season(
                competition_id=competition_id,
                external_id=external_id,
                year=year,
                start_date=start_date,
                end_date=end_date,
                is_current=is_current,
            )
            self.session.add(existing)
        else:
            existing.external_id = external_id
            existing.start_date = start_date
            existing.end_date = end_date
            existing.is_current = is_current
        await self.session.flush()
        return existing

    async def find_by_external_id(self, external_id: int) -> Season | None:
        """Find a season by its provider external id (the ``year``)."""
        stmt = select(Season).where(Season.external_id == external_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def find_current_for_competition(
        self, competition_id: int
    ) -> Season | None:
        """Return the current season for a competition (if any)."""
        stmt = select(Season).where(
            Season.competition_id == competition_id,
            Season.is_current.is_(True),
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def find_for_competition_and_year(
        self, *, competition_id: int, year: int
    ) -> Season | None:
        """Return the season of a competition for a given year (or None)."""
        stmt = select(Season).where(
            Season.competition_id == competition_id,
            Season.year == year,
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()
