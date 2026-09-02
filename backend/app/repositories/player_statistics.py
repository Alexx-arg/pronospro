"""``PlayerStatistics`` repository."""

from __future__ import annotations

from sqlalchemy import select

from app.models import PlayerStatistics
from app.repositories.base import BaseRepository


class PlayerStatisticsRepository(BaseRepository[PlayerStatistics]):
    """Repository for :class:`PlayerStatistics`."""

    model = PlayerStatistics

    async def upsert(
        self,
        *,
        player_id: int,
        team_id: int,
        competition_id: int,
        season_id: int,
        appearances: int = 0,
        starts: int = 0,
        minutes_played: int = 0,
        goals: int = 0,
        assists: int = 0,
        yellow_cards: int = 0,
        red_cards: int = 0,
        rating: float | None = None,
    ) -> PlayerStatistics:
        """Insert-or-update a player statistics row keyed by
        ``(player_id, season_id, competition_id)``.
        """
        stmt = select(PlayerStatistics).where(
            PlayerStatistics.player_id == player_id,
            PlayerStatistics.season_id == season_id,
            PlayerStatistics.competition_id == competition_id,
        )
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing is None:
            existing = PlayerStatistics(
                player_id=player_id,
                team_id=team_id,
                competition_id=competition_id,
                season_id=season_id,
                appearances=appearances,
                starts=starts,
                minutes_played=minutes_played,
                goals=goals,
                assists=assists,
                yellow_cards=yellow_cards,
                red_cards=red_cards,
                rating=rating,
            )
            self.session.add(existing)
        else:
            existing.team_id = team_id
            existing.appearances = appearances
            existing.starts = starts
            existing.minutes_played = minutes_played
            existing.goals = goals
            existing.assists = assists
            existing.yellow_cards = yellow_cards
            existing.red_cards = red_cards
            existing.rating = rating
        await self.session.flush()
        return existing
