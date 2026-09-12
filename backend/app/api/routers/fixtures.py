"""Fixtures endpoints backed by PostgreSQL.

Provides:
* ``GET /fixtures/upcoming``  — paginated, filterable by ``date`` (YYYY-MM-DD),
  ``from``/``to``, competition, season, team; returns markets (1X2, goals,
  metrics) per fixture.
* ``GET /fixtures/{fixture_id}`` — single fixture with markets.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.security import require_api_key_dep
from app.api.schemas.fixtures import (
    BttsMarket,
    CompetitionInfo,
    FixtureResponse,
    GoalsMarket,
    Markets,
    MatchMetrics,
    OverUnder25Market,
    PredictionInfo,
    SeasonInfo,
    TeamInfo,
    W1X2Market,
)
from app.db.session import get_session
from app.models import Competition, Fixture, ModelVersion, Prediction, Season, Team, TeamStatistics
from app.prediction.markets import btts, over_under_25

router = APIRouter(prefix="/fixtures", tags=["fixtures"], dependencies=[Depends(require_api_key_dep)])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _get_latest_team_stats(
    db: AsyncSession,
    team_id: int,
    competition_id: int,
    season_id: int,
) -> TeamStatistics | None:
    """Latest TeamStatistics row for a team/season as of today."""
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
    return (await db.execute(stmt)).scalar_one_or_none()


def _avg(stats: TeamStatistics | None, attr: str) -> float | None:
    """Per-match average of an aggregate column (None-safe)."""
    if stats is None:
        return None
    value = getattr(stats, attr, None)
    played = stats.fixtures_played or 0
    if value is None or played <= 0:
        return None
    return round(float(value) / float(played), 2)


def _to_metrics(
    home_stats: TeamStatistics | None,
    away_stats: TeamStatistics | None,
) -> MatchMetrics | None:
    if home_stats is None and away_stats is None:
        return None
    return MatchMetrics(
        home_form=getattr(home_stats, "form", None),
        away_form=getattr(away_stats, "form", None),
        home_xg=float(home_stats.xg) if home_stats and home_stats.xg is not None else None,
        home_xga=float(home_stats.xga) if home_stats and home_stats.xga is not None else None,
        away_xg=float(away_stats.xg) if away_stats and away_stats.xg is not None else None,
        away_xga=float(away_stats.xga) if away_stats and away_stats.xga is not None else None,
        home_corners_avg=_avg(home_stats, "corners"),
        away_corners_avg=_avg(away_stats, "corners"),
        home_yellow_cards_avg=_avg(home_stats, "yellow_cards"),
        away_yellow_cards_avg=_avg(away_stats, "yellow_cards"),
        home_red_cards_avg=_avg(home_stats, "red_cards"),
        away_red_cards_avg=_avg(away_stats, "red_cards"),
        home_possession_avg=(
            float(home_stats.possession_avg)
            if home_stats and home_stats.possession_avg is not None
            else None
        ),
        away_possession_avg=(
            float(away_stats.possession_avg)
            if away_stats and away_stats.possession_avg is not None
            else None
        ),
    )


async def _get_fixture_prediction(db: AsyncSession, fixture_id: int) -> PredictionInfo | None:
    """Latest prediction for a fixture if exists."""
    stmt = (
        select(Prediction, ModelVersion)
        .join(ModelVersion, Prediction.model_version_id == ModelVersion.id)
        .where(Prediction.fixture_id == fixture_id)
        .order_by(Prediction.created_at.desc())
        .limit(1)
    )
    row = (await db.execute(stmt)).first()
    if row is None:
        return None
    pred, mv = row
    return PredictionInfo(
        id=pred.id,
        model_version=f"{mv.name} v{mv.version}",
        home_probability=float(pred.home_probability),
        draw_probability=float(pred.draw_probability),
        away_probability=float(pred.away_probability),
        expected_home_goals=(
            float(pred.expected_home_goals) if pred.expected_home_goals is not None else None
        ),
        expected_away_goals=(
            float(pred.expected_away_goals) if pred.expected_away_goals is not None else None
        ),
        confidence=float(pred.confidence) if pred.confidence is not None else None,
        created_at=pred.created_at,
    )


def _build_markets(
    prediction: PredictionInfo | None,
    metrics: MatchMetrics | None,
) -> Markets | None:
    """Compose the markets block from the prediction + team metrics."""
    if prediction is None and metrics is None:
        return None

    w1x2 = None
    if prediction is not None:
        w1x2 = W1X2Market(
            home=prediction.home_probability,
            draw=prediction.draw_probability,
            away=prediction.away_probability,
            model_version=prediction.model_version,
        )

    # Goal markets: prefer the model's expected goals, else fall back to team xG.
    lambda_home = prediction.expected_home_goals if prediction and prediction.expected_home_goals else None
    lambda_away = prediction.expected_away_goals if prediction and prediction.expected_away_goals else None
    if lambda_home is None and metrics is not None:
        lambda_home = metrics.home_xg
    if lambda_away is None and metrics is not None:
        lambda_away = metrics.away_xg

    over_under = over_under_25(lambda_home, lambda_away) if (lambda_home and lambda_away) else None
    btts_mkt = btts(lambda_home, lambda_away) if (lambda_home and lambda_away) else None

    goals = None
    if over_under is not None or btts_mkt is not None:
        goals = GoalsMarket(
            over_under_2_5=(
                OverUnder25Market(over_2_5=over_under[0], under_2_5=over_under[1])
                if over_under
                else None
            ),
            btts=BttsMarket(yes=btts_mkt[0], no=btts_mkt[1]) if btts_mkt else None,
        )

    return Markets(w1x2=w1x2, goals=goals, metrics=metrics)


def _fixture_to_response(
    f: Fixture,
    prediction: PredictionInfo | None,
    metrics: MatchMetrics | None,
) -> FixtureResponse:
    return FixtureResponse(
        id=f.id,
        external_id=f.external_id,
        competition=CompetitionInfo(
            id=f.competition.id,
            external_id=f.competition.external_id,
            name=f.competition.name,
            logo=f.competition.logo,
        ),
        season=SeasonInfo(
            id=f.season.id,
            external_id=f.season.external_id,
            year=f.season.year,
        ),
        home_team=TeamInfo(
            id=f.home_team.id,
            external_id=f.home_team.external_id,
            name=f.home_team.name,
            short_name=f.home_team.short_name,
            code=f.home_team.code,
            country=f.home_team.country,
            logo=f.home_team.logo,
            venue=f.home_team.venue,
        ),
        away_team=TeamInfo(
            id=f.away_team.id,
            external_id=f.away_team.external_id,
            name=f.away_team.name,
            short_name=f.away_team.short_name,
            code=f.away_team.code,
            country=f.away_team.country,
            logo=f.away_team.logo,
            venue=f.away_team.venue,
        ),
        kickoff_time=f.kickoff_time,
        status=f.status,
        status_short=f.status_short,
        venue=f.venue,
        home_goals=f.home_goals,
        away_goals=f.away_goals,
        markets=_build_markets(prediction, metrics),
    )


async def _hydrate_fixture(
    db: AsyncSession,
    f: Fixture,
    *,
    include_metrics: bool,
    include_prediction: bool,
) -> FixtureResponse:
    await db.refresh(f, ["competition", "season", "home_team", "away_team"])

    metrics = None
    if include_metrics:
        home_stats = await _get_latest_team_stats(db, f.home_team_id, f.competition_id, f.season_id)
        away_stats = await _get_latest_team_stats(db, f.away_team_id, f.competition_id, f.season_id)
        metrics = _to_metrics(home_stats, away_stats)

    prediction = None
    if include_prediction:
        prediction = await _get_fixture_prediction(db, f.id)

    return _fixture_to_response(f, prediction, metrics)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/upcoming", response_model=dict)
async def get_upcoming(
    competition_id: Optional[int] = Query(None),
    season_id: Optional[int] = Query(None),
    team_id: Optional[int] = Query(None),
    date_: Optional[date] = Query(None, alias="date"),
    from_: Optional[datetime] = Query(None, alias="from"),
    to: Optional[datetime] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    include_metrics: bool = Query(True),
    include_prediction: bool = Query(True),
    db: AsyncSession = Depends(get_session),
) -> dict:
    """Return fixtures (scheduled/in_play/finished) filtered by day or window.

    When ``date`` is provided it bounds ``kickoff_time`` to that UTC day.
    Otherwise ``from``/``to`` (exclusive) are honoured; default window is
    the next 7 days.
    """
    stmt = select(Fixture)

    if date_ is not None:
        day_start = datetime.combine(date_, time.min, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        stmt = stmt.where(Fixture.kickoff_time >= day_start, Fixture.kickoff_time < day_end)
    else:
        now = datetime.now(timezone.utc)
        start = from_ if from_ is not None else now
        end = to if to is not None else now + timedelta(days=7)
        stmt = stmt.where(Fixture.kickoff_time >= start, Fixture.kickoff_time <= end)

    if competition_id is not None:
        stmt = stmt.where(Fixture.competition_id == competition_id)
    if season_id is not None:
        stmt = stmt.where(Fixture.season_id == season_id)
    if team_id is not None:
        stmt = stmt.where(
            (Fixture.home_team_id == team_id) | (Fixture.away_team_id == team_id)
        )

    from sqlalchemy import func

    total = (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar() or 0

    stmt = stmt.order_by(Fixture.kickoff_time).limit(limit).offset(offset)
    fixtures = list((await db.execute(stmt)).scalars().all())

    items = [
        await _hydrate_fixture(
            db, f, include_metrics=include_metrics, include_prediction=include_prediction
        )
        for f in fixtures
    ]

    return {"items": items, "limit": limit, "offset": offset, "total": total}


@router.get("/{fixture_id}", response_model=FixtureResponse)
async def get_fixture(
    fixture_id: int,
    include_metrics: bool = Query(True),
    include_prediction: bool = Query(True),
    db: AsyncSession = Depends(get_session),
) -> FixtureResponse:
    """Single fixture detail with markets."""
    f = (
        await db.execute(select(Fixture).where(Fixture.id == fixture_id))
    ).scalar_one_or_none()
    if f is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Fixture not found")

    return await _hydrate_fixture(
        db, f, include_metrics=include_metrics, include_prediction=include_prediction
    )