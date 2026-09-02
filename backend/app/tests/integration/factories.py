"""Shared factories producing domain entities for integration tests.

The factories use a small pseudo-random multiplier so multiple test runs
writing to the same DB don't collide on ``external_id`` unique constraints
(even though each test rolls back, some tests share a session lifecycle).
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

from app.models import (
    Competition,
    Fixture,
    ModelVersion,
    Player,
    PlayerStatistics,
    Prediction,
    Season,
    Team,
    TeamStatistics,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Factories:
    """Container exposing small factory helpers bound to a session.

    Factories ``flush`` (not ``commit``) the session so the test owns the
    transaction lifecycle. All objects created here are returned to allow
    further tweaking in the test body.
    """

    def __init__(self, session) -> None:  # noqa: ANN001  (avoid circular import)
        self.session = session
        self._seq = random.Random(0xBEEF)

    def _next_ext(self) -> int:
        # Random enough to avoid hard-coded clashes between tests.
        return self._seq.randint(1_000_000, 9_999_999)

    async def competition(self, **overrides) -> Competition:
        kwargs = dict(
            external_id=overrides.pop("external_id", self._next_ext()),
            name=overrides.pop("name", "Premier League"),
            type=overrides.pop("type", "league"),
            country=overrides.pop("country", "England"),
        )
        kwargs.update(overrides)
        entity = Competition(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def season(self, competition_id: int, **overrides) -> Season:
        kwargs = dict(
            competition_id=competition_id,
            external_id=overrides.pop("external_id", self._next_ext()),
            year=overrides.pop("year", 2025),
            is_current=overrides.pop("is_current", True),
        )
        kwargs.update(overrides)
        entity = Season(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def team(self, name: str = "Arsenal", **overrides) -> Team:
        kwargs = dict(
            external_id=overrides.pop("external_id", self._next_ext()),
            name=overrides.pop("name", name),
            country=overrides.pop("country", "England"),
        )
        kwargs.update(overrides)
        entity = Team(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def player(self, **overrides) -> Player:
        kwargs = dict(
            external_id=overrides.pop("external_id", self._next_ext()),
            name=overrides.pop("name", "John Doe"),
            position=overrides.pop("position", "MF"),
        )
        kwargs.update(overrides)
        entity = Player(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def fixture(
        self,
        *,
        competition_id: int,
        season_id: int,
        home_team_id: int,
        away_team_id: int,
        kickoff_time: datetime | None = None,
        **overrides,
    ) -> Fixture:
        kwargs = dict(
            external_id=overrides.pop("external_id", self._next_ext()),
            competition_id=competition_id,
            season_id=season_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            kickoff_time=kickoff_time or _now(),
        )
        kwargs.update(overrides)
        entity = Fixture(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def team_statistics(
        self,
        *,
        team_id: int,
        competition_id: int,
        season_id: int,
        **overrides,
    ) -> TeamStatistics:
        from datetime import date as _date

        kwargs = dict(
            team_id=team_id,
            competition_id=competition_id,
            season_id=season_id,
            as_of_date=overrides.pop("as_of_date", _date(2026, 1, 1)),
            fixtures_played=10,
            wins=5,
            draws=3,
            losses=2,
            goals_for=15,
            goals_against=8,
        )
        kwargs.update(overrides)
        entity = TeamStatistics(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def player_statistics(
        self,
        *,
        player_id: int,
        team_id: int,
        competition_id: int,
        season_id: int,
        **overrides,
    ) -> PlayerStatistics:
        kwargs = dict(
            player_id=player_id,
            team_id=team_id,
            competition_id=competition_id,
            season_id=season_id,
            appearances=10,
            goals=3,
        )
        kwargs.update(overrides)
        entity = PlayerStatistics(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def model_version(self, name: str = "elo", version: str = "v1.0.0",
                            is_active: bool = True) -> ModelVersion:
        entity = ModelVersion(name=name, version=version, is_active=is_active,
                              parameters={})
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def prediction(
        self,
        *,
        fixture_id: int,
        model_version_id: int,
        kickoff_time: datetime,
        home_probability: float = 0.5,
        draw_probability: float = 0.3,
        away_probability: float = 0.2,
        expected_home_goals: float = 1.5,
        expected_away_goals: float = 1.0,
        confidence: float = 0.6,
        features_snapshot: dict | None = None,
        explanation: dict | None = None,
    ) -> Prediction:
        # Force the probabilities to sum to 1 to satisfy the CHECK constraint.
        total = home_probability + draw_probability + away_probability
        if abs(total - 1.0) > 1e-6:
            raise AssertionError(f"probabilities must sum to 1, got {total}")
        entity = Prediction(
            fixture_id=fixture_id,
            model_version_id=model_version_id,
            kickoff_time=kickoff_time,
            home_probability=home_probability,
            draw_probability=draw_probability,
            away_probability=away_probability,
            expected_home_goals=expected_home_goals,
            expected_away_goals=expected_away_goals,
            confidence=confidence,
            features_snapshot=features_snapshot or {},
            explanation=explanation,
        )
        self.session.add(entity)
        await self.session.flush()
        return entity
