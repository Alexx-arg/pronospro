# ruff: noqa: N803, N806
"""Tests PredictionService — Sprint 6.2 (mock extremo, sin DB real)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.api.services.prediction_service import (
    FixtureNotFoundError,
    InsufficientHistoryError,
    PredictionService,
)
from app.prediction.contracts import MatchProbabilities


def _mock_model() -> MagicMock:
    m = MagicMock()
    m.predict.return_value = MatchProbabilities(p_home_win=0.5, p_draw=0.3, p_away_win=0.2)
    m.model_version = "test"
    return m


@pytest.mark.asyncio
async def test_camino_feliz() -> None:
    service = PredictionService()
    mock_session = AsyncMock()
    model = _mock_model()

    # Mock fixture ORM
    mock_fixture = MagicMock()
    mock_fixture.external_id = 123
    mock_fixture.competition_id = 39
    mock_fixture.season_id = 1
    mock_fixture.home_team_id = 10
    mock_fixture.away_team_id = 20
    mock_fixture.kickoff_time = __import__("datetime").datetime(2026, 1, 10, 15, 0, tzinfo=__import__("datetime").timezone.utc)
    mock_fixture.home_goals = None
    mock_fixture.away_goals = None
    mock_fixture.matchday = 1
    mock_fixture.venue = None

    # Mock internal _load_fixture
    service._load_fixture = AsyncMock(return_value=mock_fixture)  # type: ignore[method-assign]

    # Mock asof loaders via patch
    import app.api.services.prediction_service as svc_mod

    orig_build = svc_mod.build_example
    orig_load_finished = svc_mod.load_finished_fixtures_as_of
    orig_load_stats = svc_mod.load_team_stats_as_of
    orig_load_h2h = svc_mod.load_h2h_fixtures_as_of

    # Create a fake HistoricalMatchExample
    from app.features.example import HistoricalMatchExample

    fake_example = HistoricalMatchExample(
        fixture_id=123,
        kickoff=mock_fixture.kickoff_time,
        competition_id=39,
        season_id=1,
        home_team_id=10,
        away_team_id=20,
        features={name: 0.5 for name in __import__("app.features.example", fromlist=["FEATURE_NAMES"]).FEATURE_NAMES},
        targets={"home_win": 1, "draw": 0, "away_win": 0, "home_goals": 2, "away_goals": 0},
    )
    fake_example.features["home_elo_pre_match"] = 1500.0
    fake_example.features["away_elo_pre_match"] = 1500.0
    fake_example.features["elo_difference"] = 0.0

    svc_mod.build_example = MagicMock(return_value=fake_example)  # type: ignore[assignment]
    svc_mod.load_finished_fixtures_as_of = AsyncMock(return_value=[])  # type: ignore[assignment]
    svc_mod.load_team_stats_as_of = AsyncMock(return_value=[])  # type: ignore[assignment]
    svc_mod.load_h2h_fixtures_as_of = AsyncMock(return_value=[])  # type: ignore[assignment]

    try:
        resp = await service.predict_for_fixture(123, mock_session, model)
        assert resp.fixture_id == 123
        assert 0 <= resp.prob_home <= 1
        assert resp.prob_home + resp.prob_draw + resp.prob_away == pytest.approx(1.0, abs=1e-9)
    finally:
        svc_mod.build_example = orig_build  # type: ignore[assignment]
        svc_mod.load_finished_fixtures_as_of = orig_load_finished  # type: ignore[assignment]
        svc_mod.load_team_stats_as_of = orig_load_stats  # type: ignore[assignment]
        svc_mod.load_h2h_fixtures_as_of = orig_load_h2h  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_fixture_no_existe() -> None:
    service = PredictionService()
    service._load_fixture = AsyncMock(return_value=None)  # type: ignore[method-assign]
    with pytest.raises(FixtureNotFoundError):
        await service.predict_for_fixture(999, AsyncMock(), _mock_model())


@pytest.mark.asyncio
async def test_falla_construccion_features() -> None:
    service = PredictionService()
    mock_session = AsyncMock()
    model = _mock_model()
    mock_fixture = MagicMock()
    mock_fixture.external_id = 123
    mock_fixture.competition_id = 39
    mock_fixture.season_id = 1
    mock_fixture.home_team_id = 10
    mock_fixture.away_team_id = 20
    mock_fixture.kickoff_time = __import__("datetime").datetime(2026, 1, 10, tzinfo=__import__("datetime").timezone.utc)
    mock_fixture.home_goals = None
    mock_fixture.away_goals = None
    mock_fixture.matchday = None
    mock_fixture.venue = None
    service._load_fixture = AsyncMock(return_value=mock_fixture)  # type: ignore[method-assign]

    import app.api.services.prediction_service as svc_mod

    orig_build = svc_mod.build_example
    svc_mod.build_example = MagicMock(side_effect=ValueError("no history"))  # type: ignore[assignment]
    svc_mod.load_finished_fixtures_as_of = AsyncMock(return_value=[])  # type: ignore[assignment]
    svc_mod.load_team_stats_as_of = AsyncMock(return_value=[])  # type: ignore[assignment]
    svc_mod.load_h2h_fixtures_as_of = AsyncMock(return_value=[])  # type: ignore[assignment]
    try:
        with pytest.raises(InsufficientHistoryError):
            await service.predict_for_fixture(123, mock_session, model)
    finally:
        svc_mod.build_example = orig_build  # type: ignore[assignment]
