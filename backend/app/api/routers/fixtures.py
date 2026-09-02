"""Fixtures endpoints backed by PostgreSQL.

Provides GET /fixtures/upcoming and GET /fixtures/{fixture_id}
with real data from the database including team logos from API-Football.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.security import require_api_key_dep
from app.db.session import get_session
from app.models import Competition, Fixture, Season, Team

router = APIRouter(prefix="/fixtures", tags=["fixtures"], dependencies=[Depends(require_api_key_dep)])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TeamInfo(BaseModel):
    id: int
    external_id: int
    name: str
    short_name: str | None = None
    code: str | None = None
    country: str | None = None
    logo: str | None = None
    venue: str | None = None


class CompetitionInfo(BaseModel):
    id: int
    external_id: int
    name: str
    logo: str | None = None


class SeasonInfo(BaseModel):
    id: int
    external_id: int
    year: int


class MatchMetrics(BaseModel):
    """Quantitative match metrics from TeamStatistics."""
    home_form: str | None = None
    away_form: str | None = None
    home_xg: float | None = None
    home_xga: float | None = None
    away_xg: float | None = None
    away_xga: float | None = None
    home_corners_avg: float | None = None
    away_corners_avg: float | None = None
    home_yellow_cards_avg: float | None = None
    away_yellow_cards_avg: float | None = None
    home_red_cards_avg: float | None = None
    away_red_cards_avg: float | None = None
    home_possession_avg: float | None = None
    away_possession_avg: float | None = None


class PredictionInfo(BaseModel):
    id: int
    model_version: str
    home_probability: float = Field(..., ge=0, le=1)
    draw_probability: float = Field(..., ge=0, le=1)
    away_probability: float = Field(..., ge=0, le=1)
    expected_home_goals: float | None = None
    expected_away_goals: float | None = None
    confidence: float | None = None
    created_at: datetime


class FixtureResponse(BaseModel):
    id: int
    external_id: int
    competition: CompetitionInfo
    season: SeasonInfo
    home_team: TeamInfo
    away_team: TeamInfo
    kickoff_time: datetime
    status: str
    status_short: str | None = None
    venue: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    metrics: MatchMetrics | None = None
    prediction: PredictionInfo | None = None


class PaginatedFixtures(BaseModel):
    items: list[FixtureResponse]
    limit: int
    offset: int
    total: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _get_latest_team_stats(
    db: AsyncSession,
    team_id: int,
    competition_id: int,
    season_id: int,
) -> dict | None:
    """Fetch the latest TeamStatistics row for a team/season as of today."""
    from app.models import TeamStatistics
    from datetime import date

    today = date.today()
    stmt = (
        select(TeamStatistics)
        .where(
            TeamStatistics.team_id == team_id,
            TeamStatistics.competition_id == competition_id,
            TeamStatistics.season_id == season_id,
            TeamStatistics.as_of_date <= today,
        )
        .order_by(TeamStatistics.as_of_date.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    stats = result.scalar_one_or_none()
    if stats is None:
        return None
    return {
        "form": stats.form,
        "xg": float(stats.xg) if stats.xg is not None else None,
        "xga": float(stats.xga) if stats.xga is not None else None,
        "corners": stats.corners,
        "fixtures_played": stats.fixtures_played,
        "yellow_cards": stats.yellow_cards,
        "red_cards": stats.red_cards,
        "possession_avg": float(stats.possession_avg) if stats.possession_avg is not None else None,
    }


async def _get_fixture_prediction(db: AsyncSession, fixture_id: int) -> PredictionInfo | None:
    """Fetch the latest prediction for a fixture if exists."""
    from app.models import ModelVersion, Prediction

    stmt = (
        select(Prediction, ModelVersion)
        .join(ModelVersion, Prediction.model_version_id == ModelVersion.id)
        .where(Prediction.fixture_id == fixture_id)
        .order_by(Prediction.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.first()
    if row is None:
        return None
    pred, mv = row
    return PredictionInfo(
        id=pred.id,
        model_version=f"{mv.name} v{mv.version}",
        home_probability=pred.home_probability,
        draw_probability=pred.draw_probability,
        away_probability=pred.away_probability,
        expected_home_goals=pred.expected_home_goals,
        expected_away_goals=pred.expected_away_goals,
        confidence=pred.confidence,
        created_at=pred.created_at,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/upcoming", response_model=PaginatedFixtures)
async def get_upcoming(
    competition_id: Optional[int] = Query(None),
    season_id: Optional[int] = Query(None),
    team_id: Optional[int] = Query(None),
    from_: Optional[datetime] = Query(None, alias="from"),
    to: Optional[datetime] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    include_metrics: bool = Query(False),
    include_prediction: bool = Query(False),
    db: AsyncSession = Depends(get_session),
) -> PaginatedFixtures:
    """Return upcoming fixtures ordered by kickoff_time from PostgreSQL."""
    stmt = select(Fixture).where(Fixture.status.in_(("scheduled", "in_play")))

    if competition_id is not None:
        stmt = stmt.where(Fixture.competition_id == competition_id)
    if season_id is not None:
        stmt = stmt.where(Fixture.season_id == season_id)
    if team_id is not None:
        stmt = stmt.where(
            (Fixture.home_team_id == team_id) | (Fixture.away_team_id == team_id)
        )
    if from_ is not None:
        stmt = stmt.where(Fixture.kickoff_time >= from_)
    if to is not None:
        stmt = stmt.where(Fixture.kickoff_time <= to)

    # Total count
    from sqlalchemy import func
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = stmt.order_by(Fixture.kickoff_time).limit(limit).offset(offset)
    result = await db.execute(stmt)
    fixtures = list(result.scalars().all())

    items = []
    for f in fixtures:
        # Load relationships
        await db.refresh(f, ["competition", "season", "home_team", "away_team"])

        comp = CompetitionInfo(
            id=f.competition.id,
            external_id=f.competition.external_id,
            name=f.competition.name,
            logo=f.competition.logo,
        )
        seas = SeasonInfo(
            id=f.season.id,
            external_id=f.season.external_id,
            year=f.season.year,
        )
        home = TeamInfo(
            id=f.home_team.id,
            external_id=f.home_team.external_id,
            name=f.home_team.name,
            short_name=f.home_team.short_name,
            code=f.home_team.code,
            country=f.home_team.country,
            logo=f.home_team.logo,
            venue=f.home_team.venue,
        )
        away = TeamInfo(
            id=f.away_team.id,
            external_id=f.away_team.external_id,
            name=f.away_team.name,
            short_name=f.away_team.short_name,
            code=f.away_team.code,
            country=f.away_team.country,
            logo=f.away_team.logo,
            venue=f.away_team.venue,
        )

        metrics = None
        if include_metrics:
            home_stats = await _get_latest_team_stats(db, f.home_team_id, f.competition_id, f.season_id)
            away_stats = await _get_latest_team_stats(db, f.away_team_id, f.competition_id, f.season_id)
            if home_stats or away_stats:
                metrics = MatchMetrics(
                    home_form=home_stats.get("form") if home_stats else None,
                    away_form=away_stats.get("form") if away_stats else None,
                    home_xg=home_stats.get("xg") if home_stats else None,
                    home_xga=home_stats.get("xga") if home_stats else None,
                    away_xg=away_stats.get("xg") if away_stats else None,
                    away_xga=away_stats.get("xga") if away_stats else None,
                    home_corners_avg=(home_stats["corners"] / home_stats["fixtures_played"]) if home_stats and home_stats["corners"] and home_stats["fixtures_played"] else None,
                    away_corners_avg=(away_stats["corners"] / away_stats["fixtures_played"]) if away_stats and away_stats["corners"] and away_stats["fixtures_played"] else None,
                    home_yellow_cards_avg=(home_stats["yellow_cards"] / home_stats["fixtures_played"]) if home_stats and home_stats["yellow_cards"] and home_stats["fixtures_played"] else None,
                    away_yellow_cards_avg=(away_stats["yellow_cards"] / away_stats["fixtures_played"]) if away_stats and away_stats["yellow_cards"] and away_stats["fixtures_played"] else None,
                    home_red_cards_avg=(home_stats["red_cards"] / home_stats["fixtures_played"]) if home_stats and home_stats["red_cards"] and home_stats["fixtures_played"] else None,
                    away_red_cards_avg=(away_stats["red_cards"] / away_stats["fixtures_played"]) if away_stats and away_stats["red_cards"] and away_stats["fixtures_played"] else None,
                    home_possession_avg=home_stats.get("possession_avg") if home_stats else None,
                    away_possession_avg=away_stats.get("possession_avg") if away_stats else None,
                )

        prediction = None
        if include_prediction:
            prediction = await _get_fixture_prediction(db, f.id)

        items.append(FixtureResponse(
            id=f.id,
            external_id=f.external_id,
            competition=comp,
            season=seas,
            home_team=home,
            away_team=away,
            kickoff_time=f.kickoff_time,
            status=f.status,
            status_short=f.status_short,
            venue=f.venue,
            home_goals=f.home_goals,
            away_goals=f.away_goals,
            metrics=metrics,
            prediction=prediction,
        ))

    return PaginatedFixtures(items=items, limit=limit, offset=offset, total=total)


@router.get("/{fixture_id}", response_model=FixtureResponse)
async def get_fixture(
    fixture_id: int,
    include_metrics: bool = Query(True),
    include_prediction: bool = Query(True),
    db: AsyncSession = Depends(get_session),
) -> FixtureResponse:
    """Return fixture detail with optional metrics and prediction."""
    stmt = select(Fixture).where(Fixture.id == fixture_id)
    result = await db.execute(stmt)
    f = result.scalar_one_or_none()
    if f is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Fixture not found")

    await db.refresh(f, ["competition", "season", "home_team", "away_team"])

    comp = CompetitionInfo(
        id=f.competition.id,
        external_id=f.competition.external_id,
        name=f.competition.name,
        logo=f.competition.logo,
    )
    seas = SeasonInfo(
        id=f.season.id,
        external_id=f.season.external_id,
        year=f.season.year,
    )
    home = TeamInfo(
        id=f.home_team.id,
        external_id=f.home_team.external_id,
        name=f.home_team.name,
        short_name=f.home_team.short_name,
        code=f.home_team.code,
        country=f.home_team.country,
        logo=f.home_team.logo,
        venue=f.home_team.venue,
    )
    away = TeamInfo(
        id=f.away_team.id,
        external_id=f.away_team.external_id,
        name=f.away_team.name,
        short_name=f.away_team.short_name,
        code=f.away_team.code,
        country=f.away_team.country,
        logo=f.away_team.logo,
        venue=f.away_team.venue,
    )

    metrics = None
    if include_metrics:
        home_stats = await _get_latest_team_stats(db, f.home_team_id, f.competition_id, f.season_id)
        away_stats = await _get_latest_team_stats(db, f.away_team_id, f.competition_id, f.season_id)
        if home_stats or away_stats:
            metrics = MatchMetrics(
                home_form=home_stats.get("form") if home_stats else None,
                away_form=away_stats.get("form") if away_stats else None,
                home_xg=home_stats.get("xg") if home_stats else None,
                home_xga=home_stats.get("xga") if home_stats else None,
                away_xg=away_stats.get("xg") if away_stats else None,
                away_xga=away_stats.get("xga") if away_stats else None,
                home_corners_avg=(home_stats["corners"] / home_stats["fixtures_played"]) if home_stats and home_stats["corners"] and home_stats["fixtures_played"] else None,
                away_corners_avg=(away_stats["corners"] / away_stats["fixtures_played"]) if away_stats and away_stats["corners"] and away_stats["fixtures_played"] else None,
                home_yellow_cards_avg=(home_stats["yellow_cards"] / home_stats["fixtures_played"]) if home_stats and home_stats["yellow_cards"] and home_stats["fixtures_played"] else None,
                away_yellow_cards_avg=(away_stats["yellow_cards"] / away_stats["fixtures_played"]) if away_stats and away_stats["yellow_cards"] and away_stats["fixtures_played"] else None,
                home_red_cards_avg=(home_stats["red_cards"] / home_stats["fixtures_played"]) if home_stats and home_stats["red_cards"] and home_stats["fixtures_played"] else None,
                away_red_cards_avg=(away_stats["red_cards"] / away_stats["fixtures_played"]) if away_stats and away_stats["red_cards"] and away_stats["fixtures_played"] else None,
                home_possession_avg=home_stats.get("possession_avg") if home_stats else None,
                away_possession_avg=away_stats.get("possession_avg") if away_stats else None,
            )

    prediction = None
    if include_prediction:
        prediction = await _get_fixture_prediction(db, f.id)

    return FixtureResponse(
        id=f.id,
        external_id=f.external_id,
        competition=comp,
        season=seas,
        home_team=home,
        away_team=away,
        kickoff_time=f.kickoff_time,
        status=f.status,
        status_short=f.status_short,
        venue=f.venue,
        home_goals=f.home_goals,
        away_goals=f.away_goals,
        metrics=metrics,
        prediction=prediction,
    )