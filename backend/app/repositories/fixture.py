"""``Fixture`` repository."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from app.models import Fixture
from app.repositories.base import BaseRepository


class FixtureRepository(BaseRepository[Fixture]):
    """Repository for :class:`Fixture`."""

    model = Fixture

    async def upsert(
        self,
        *,
        external_id: int,
        competition_id: int,
        season_id: int,
        home_team_id: int,
        away_team_id: int,
        kickoff_time: datetime,
        matchday: int | None = None,
        round: str | None = None,
        venue: str | None = None,
        status: str = "scheduled",
        status_short: str | None = None,
    ) -> Fixture:
        """Insert-or-update a fixture keyed by ``external_id``.

        Operational updates (new kickoff time, status changes, scores) are
        permitted because ``fixtures`` is mutable.
        """
        stmt = select(Fixture).where(Fixture.external_id == external_id)
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing is None:
            existing = Fixture(
                external_id=external_id,
                competition_id=competition_id,
                season_id=season_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                kickoff_time=kickoff_time,
                matchday=matchday,
                round=round,
                venue=venue,
                status=status,
                status_short=status_short,
            )
            self.session.add(existing)
        else:
            existing.kickoff_time = kickoff_time
            existing.matchday = matchday
            existing.round = round
            existing.venue = venue
            existing.status = status
            existing.status_short = status_short
            existing.synced_at = datetime.now(timezone.utc)
        await self.session.flush()
        return existing

    async def mark_finished(
        self,
        *,
        fixture_id: int,
        home_goals: int,
        away_goals: int,
    ) -> Fixture:
        """Set fixture status to ``finished`` with the final score."""
        fixture = await self.get(fixture_id)
        if fixture is None:
            raise LookupError(f"Fixture {fixture_id} not found")
        fixture.status = "finished"
        fixture.home_goals = home_goals
        fixture.away_goals = away_goals
        fixture.finished_at = datetime.now(timezone.utc)
        await self.session.flush()
        return fixture

    async def list_upcoming(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Fixture]:
        """Return upcoming fixtures ordered by kickoff_time."""
        stmt = select(Fixture).where(
            Fixture.status.in_(("scheduled", "in_play"))
        )
        if competition_id is not None:
            stmt = stmt.where(Fixture.competition_id == competition_id)
        if season_id is not None:
            stmt = stmt.where(Fixture.season_id == season_id)
        if from_time is not None:
            stmt = stmt.where(Fixture.kickoff_time >= from_time)
        if to_time is not None:
            stmt = stmt.where(Fixture.kickoff_time <= to_time)
        stmt = stmt.order_by(Fixture.kickoff_time).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_finished_without_outcome(self, limit: int = 200) -> list[Fixture]:
        """Return finished fixtures that may still need outcome evaluation.

        Implemented as a simple query here. The evaluation service (later
        phase) joins on ``prediction_outcomes`` to filter fixtures that
        already have all outcomes computed.
        """
        stmt = (
            select(Fixture)
            .where(Fixture.status == "finished")
            .order_by(Fixture.finished_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def find_by_external_id(self, external_id: int) -> Fixture | None:
        """Find a fixture by its provider external id."""
        stmt = select(Fixture).where(Fixture.external_id == external_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_finished_since(
        self,
        *,
        since: datetime,
        league_id: int | None = None,
        limit: int = 500,
    ) -> list[Fixture]:
        """Return finished fixtures whose ``finished_at`` >= ``since``.

        Used by the ``sync_finished`` job to refresh scores: it lists what
        we ALREADY know is finished, then asks the provider for the same
        window so we can compare and update scores when needed.
        """
        stmt = (
            select(Fixture)
            .where(
                Fixture.status == "finished",
                Fixture.finished_at >= since,
            )
            .order_by(Fixture.finished_at.desc())
            .limit(limit)
        )
        if league_id is not None:
            stmt = stmt.where(Fixture.competition_id == league_id)
        return list((await self.session.execute(stmt)).scalars().all())

    async def list_unfinished_in_window(
        self,
        *,
        from_time: datetime,
        to_time: datetime,
        league_id: int | None = None,
        limit: int = 500,
    ) -> list[Fixture]:
        """Return fixtures whose kickoff is in the window AND not finished.

        Used by ``sync_finished`` to detect fixtures that should be finished
        by now (kickoff < window end) but still marked scheduled in our DB.
        """
        stmt = (
            select(Fixture)
            .where(
                Fixture.kickoff_time >= from_time,
                Fixture.kickoff_time <= to_time,
                Fixture.status != "finished",
            )
            .order_by(Fixture.kickoff_time.desc())
            .limit(limit)
        )
        if league_id is not None:
            stmt = stmt.where(Fixture.competition_id == league_id)
        return list((await self.session.execute(stmt)).scalars().all())
