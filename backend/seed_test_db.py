"""Seeder for local testing with SQLite.

Creates a minimal set of fixtures/teams/predictions so the API returns
real data without Postgres. Run once: python seed_test_db.py
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_local.db")
os.environ.setdefault("POSTGRES_HOST", "localhost")

from app.db.base import Base
from app.db.session import create_engine
from app.models import Competition, Fixture, ModelVersion, Prediction, Season, Team, TeamStatistics
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker


async def seed():
    engine = create_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as s:
        # Competitions
        pl = Competition(external_id=39, name="Premier League", logo="https://media.api-sports.io/football/leagues/39.png", type="league")
        la = Competition(external_id=140, name="La Liga", logo="https://media.api-sports.io/football/leagues/140.png", type="league")
        s.add_all([pl, la])
        await s.flush()

        season = Season(competition_id=pl.id, external_id=2025, year=2025, is_current=True)
        s.add(season)
        await s.flush()

        # Teams
        arsenal = Team(external_id=50, name="Arsenal", short_name="ARS", code="ARS", logo="https://media.api-sports.io/football/teams/50.png")
        chelsea = Team(external_id=49, name="Chelsea", short_name="CHE", code="CHE", logo="https://media.api-sports.io/football/teams/49.png")
        city   = Team(external_id=51, name="Man City", short_name="MCI", code="MCI", logo="https://media.api-sports.io/football/teams/50.png")
        liver  = Team(external_id=42, name="Liverpool", short_name="LIV", code="LIV", logo="https://media.api-sports.io/football/teams/40.png")
        real   = Team(external_id=541, name="Real Madrid", short_name="RMA", code="RMA", logo="https://media.api-sports.io/football/teams/541.png")
        barca  = Team(external_id=529, name="Barcelona", short_name="BAR", code="BAR", logo="https://media.api-sports.io/football/teams/529.png")
        s.add_all([arsenal, chelsea, city, liver, real, barca])
        await s.flush()

        # Stats
        for team, form, xg, xga, corners, yellow, poss in [
            (arsenal, "WWDLW", 1.8, 0.9, 6.2, 2.1, 58.4),
            (chelsea, "WLWDW", 1.4, 1.1, 5.8, 2.8, 54.2),
            (city,    "WWDWW", 2.1, 0.8, 7.1, 1.6, 65.3),
            (liver,   "WWWWL", 2.3, 1.0, 6.8, 2.0, 62.1),
            (real,    "WWDLW", 2.0, 0.9, 6.5, 1.9, 60.0),
            (barca,   "WWWWL", 2.2, 0.7, 7.0, 1.5, 63.2),
        ]:
            s.add(TeamStatistics(
                team_id=team.id, competition_id=pl.id, season_id=season.id,
                as_of_date=datetime.now(timezone.utc).date(),
                fixtures_played=10, wins=7, draws=2, losses=1, goals_for=20, goals_against=8,
                clean_sheets=5, failed_to_score=1, form=form, shots_total=120, shots_on_target=45,
                corners=int(corners*10), possession_avg=poss, yellow_cards=int(yellow*10), red_cards=1,
                xg=xg, xga=xga,
            ))
        await s.flush()

        # Model version
        mv = ModelVersion(name="gradient_boosting", version="v1.0.0", description="LightGBM trained on historical fixtures", parameters={"model": "lightgbm"}, is_active=True)
        s.add(mv)
        await s.flush()

        # Fixtures
        now = datetime.now(timezone.utc)
        fixtures_data = [
            (1001, pl, season, arsenal, chelsea, now + timedelta(days=1, hours=14), "Emirates Stadium", 0.55, 0.25, 0.20),
            (1002, pl, season, city, liver, now + timedelta(days=1, hours=17), "Etihad Stadium", 0.48, 0.28, 0.24),
            (1003, la, season, real, barca, now + timedelta(days=2, hours=20), "Santiago Bernabéu", 0.42, 0.29, 0.29),
        ]
        for ext_id, comp, seas, home, away, kick, venue, ph, pd, pa in fixtures_data:
            fixture = Fixture(
                external_id=ext_id, competition_id=comp.id, season_id=seas.id,
                home_team_id=home.id, away_team_id=away.id,
                kickoff_time=kick, venue=venue, status="scheduled"
            )
            s.add(fixture)
            await s.flush()
            s.add(Prediction(
                fixture_id=fixture.id, model_version_id=mv.id,
                kickoff_time=kick, home_probability=ph, draw_probability=pd, away_probability=pa,
                expected_home_goals=1.6, expected_away_goals=1.1, confidence=0.72,
                features_snapshot={"dummy": True}
            ))

        await s.commit()
        print("Seed complete: 2 leagues, 6 teams, 3 fixtures, 3 predictions")

asyncio.run(seed())