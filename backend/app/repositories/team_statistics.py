"""``TeamStatistics`` repository."""

from __future__ import annotations

from datetime import date

from sqlalchemy import select

from app.models import TeamStatistics
from app.repositories.base import BaseRepository


class TeamStatisticsRepository(BaseRepository[TeamStatistics]):
    """Repository for :class:`TeamStatistics`."""

    model = TeamStatistics

    async def upsert(
        self,
        *,
        team_id: int,
        competition_id: int,
        season_id: int,
        as_of_date: date,
        **fields: object,
    ) -> TeamStatistics:
        """Insert-or-update a team statistics row.

        ``fields`` accepts the subset of column names defined in SCHEMA.md §6.
        Tables `team_statistics` is mutable so this method is allowed.
        """
        stmt = select(TeamStatistics).where(
            TeamStatistics.team_id == team_id,
            TeamStatistics.competition_id == competition_id,
            TeamStatistics.season_id == season_id,
            TeamStatistics.as_of_date == as_of_date,
        )
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing is None:
            existing = TeamStatistics(
                team_id=team_id,
                competition_id=competition_id,
                season_id=season_id,
                as_of_date=as_of_date,
                **fields,  # type: ignore[arg-type]
            )
            self.session.add(existing)
        else:
            for key, value in fields.items():
                setattr(existing, key, value)
        await self.session.flush()
        return existing

    async def latest_for(
        self,
        *,
        team_id: int,
        competition_id: int,
        season_id: int,
        as_of_before: date | None = None,
    ) -> TeamStatistics | None:
        """Return the most recent team stats row matching the tuple.

        ``as_of_before`` filters out stats strictly after the given date
        (so prediction services can use a snapshot anterior to kickoff).
        """
        stmt = select(TeamStatistics).where(
            TeamStatistics.team_id == team_id,
            TeamStatistics.competition_id == competition_id,
            TeamStatistics.season_id == season_id,
        )
        if as_of_before is not None:
            stmt = stmt.where(TeamStatistics.as_of_date <= as_of_before)
        stmt = stmt.order_by(TeamStatistics.as_of_date.desc()).limit(1)
        return (await self.session.execute(stmt)).scalar_one_or_none()
