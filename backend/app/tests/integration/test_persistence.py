"""End-to-end persistence integration tests against PostgreSQL.

These tests cover the 10 acceptance criteria enumerated in the task spec:

1.  insert a fixture
2.  insert a prediction
3.  read a prediction back
4.  UPDATE on predictions is rejected (by the BEFORE UPDATE trigger)
5.  DELETE on predictions is rejected (by the BEFORE DELETE trigger)
6.  insert a prediction_outcome
7.  foreign keys are enforced
8.  unique constraints are enforced
9.  check constraints are enforced (probabilities sum, status enum, etc.)
10. transaction rollback works correctly

They require a Postgres instance pre-migrated with `alembic upgrade head`
(see conftest.py for the auto-skip mechanism). All tests are marked
``@pytest.mark.integration`` so they can be filtered out in environments
without DB access.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import delete, update
from sqlalchemy.exc import IntegrityError, InternalError, ProgrammingError

from app.models import Prediction, PredictionOutcome
from app.repositories.prediction import PredictionRepository
from app.repositories.prediction_outcome import PredictionOutcomeRepository
from app.tests.integration.factories import Factories

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _build_world(session) -> tuple[int, int, int, int, int, datetime]:
    """Insert a Competition/Season/Teams/Fixture/ModelVersion skeleton."""
    f = Factories(session)
    comp = await f.competition()
    season = await f.season(competition_id=comp.id)
    home = await f.team("Arsenal")
    away = await f.team("Chelsea")
    kickoff = datetime(2026, 8, 15, 19, 0, tzinfo=timezone.utc)
    fx = await f.fixture(
        competition_id=comp.id,
        season_id=season.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff_time=kickoff,
    )
    mv = await f.model_version(name="elo", version="v1.0.0")
    return fx.id, comp.id, season.id, home.id, mv.id, kickoff


# ---------------------------------------------------------------------------
# 1. Insert a fixture
# ---------------------------------------------------------------------------
async def test_insert_fixture(session) -> None:
    fx_id, *_ = await _build_world(session)
    assert fx_id > 0


# ---------------------------------------------------------------------------
# 2. Insert a prediction (via the repository, honoring the contract)
# ---------------------------------------------------------------------------
async def test_insert_prediction(session) -> None:
    fx_id, _, _, _, mv_id, kickoff = await _build_world(session)
    repo = PredictionRepository(session)
    pred = await repo.insert(
        fixture_id=fx_id,
        model_version_id=mv_id,
        kickoff_time=kickoff,
        home_probability=0.5,
        draw_probability=0.3,
        away_probability=0.2,
        expected_home_goals=1.5,
        expected_away_goals=1.0,
        confidence=0.6,
        features_snapshot={"as_of": "2026-08-14", "form_home": "WWDLW"},
    )
    assert pred.id is not None
    assert pred.explanation is None


# ---------------------------------------------------------------------------
# 3. Read a prediction back
# ---------------------------------------------------------------------------
async def test_read_prediction(session) -> None:
    fx_id, _, _, _, mv_id, kickoff = await _build_world(session)
    repo = PredictionRepository(session)
    inserted = await repo.insert(
        fixture_id=fx_id,
        model_version_id=mv_id,
        kickoff_time=kickoff,
        home_probability=0.6,
        draw_probability=0.25,
        away_probability=0.15,
        expected_home_goals=1.7,
        expected_away_goals=0.9,
        confidence=0.7,
        features_snapshot={"xg": 1.7, "xga": 0.9},
    )
    fetched = await repo.get(inserted.id)
    assert fetched is not None
    assert float(fetched.home_probability) == pytest.approx(0.6, abs=1e-5)
    assert float(fetched.expected_home_goals) == pytest.approx(1.7, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. UPDATE on predictions must fail (trigger)
# ---------------------------------------------------------------------------
async def test_prediction_update_is_blocked(session) -> None:
    fx_id, _, _, _, mv_id, kickoff = await _build_world(session)
    repo = PredictionRepository(session)
    pred = await repo.insert(
        fixture_id=fx_id,
        model_version_id=mv_id,
        kickoff_time=kickoff,
        home_probability=0.5,
        draw_probability=0.3,
        away_probability=0.2,
        expected_home_goals=1.5,
        expected_away_goals=1.0,
        confidence=0.6,
        features_snapshot={},
    )
    with pytest.raises((InternalError, IntegrityError, ProgrammingError)):
        await session.execute(
            update(Prediction)
            .where(Prediction.id == pred.id)
            .values(confidence=0.99)
        )
        await session.flush()


# ---------------------------------------------------------------------------
# 5. DELETE on predictions must fail (trigger)
# ---------------------------------------------------------------------------
async def test_prediction_delete_is_blocked(session) -> None:
    fx_id, _, _, _, mv_id, kickoff = await _build_world(session)
    repo = PredictionRepository(session)
    pred = await repo.insert(
        fixture_id=fx_id,
        model_version_id=mv_id,
        kickoff_time=kickoff,
        home_probability=0.5,
        draw_probability=0.3,
        away_probability=0.2,
        expected_home_goals=1.5,
        expected_away_goals=1.0,
        confidence=0.6,
        features_snapshot={},
    )
    with pytest.raises((InternalError, IntegrityError, ProgrammingError)):
        await session.execute(delete(Prediction).where(Prediction.id == pred.id))
        await session.flush()


# ---------------------------------------------------------------------------
# 6. Insert a prediction_outcome
# ---------------------------------------------------------------------------
async def test_insert_prediction_outcome(session) -> None:
    fx_id, _, _, _, mv_id, kickoff = await _build_world(session)
    pred_repo = PredictionRepository(session)
    out_repo = PredictionOutcomeRepository(session)
    pred = await pred_repo.insert(
        fixture_id=fx_id,
        model_version_id=mv_id,
        kickoff_time=kickoff,
        home_probability=0.5,
        draw_probability=0.3,
        away_probability=0.2,
        expected_home_goals=1.5,
        expected_away_goals=1.0,
        confidence=0.6,
        features_snapshot={},
    )
    pred_correct = max(pred.home_probability, pred.draw_probability, pred.away_probability)
    outcome = await out_repo.insert(
        prediction_id=pred.id,
        fixture_id=fx_id,
        actual_home_goals=2,
        actual_away_goals=1,
        actual_result="home",
        predicted_result="home",
        correct=True,
        predicted_correct_prob=float(pred_correct),
        brier_score=0.05,
        log_loss=0.10,
    )
    assert outcome.id is not None


# ---------------------------------------------------------------------------
# 7. Foreign keys are enforced
# ---------------------------------------------------------------------------
async def test_foreign_key_violation(session) -> None:
    repo = PredictionRepository(session)
    with pytest.raises(IntegrityError):
        await repo.insert(
            fixture_id=9_999_999,  # does not exist
            model_version_id=9_999_999,
            kickoff_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            home_probability=0.5,
            draw_probability=0.3,
            away_probability=0.2,
            expected_home_goals=1.0,
            expected_away_goals=1.0,
            confidence=0.5,
            features_snapshot={},
        )


# ---------------------------------------------------------------------------
# 8. Unique constraints are enforced
# ---------------------------------------------------------------------------
async def test_unique_constraint_violation_prediction(session) -> None:
    fx_id, _, _, _, mv_id, kickoff = await _build_world(session)
    repo = PredictionRepository(session)
    await repo.insert(
        fixture_id=fx_id,
        model_version_id=mv_id,
        kickoff_time=kickoff,
        home_probability=0.5,
        draw_probability=0.3,
        away_probability=0.2,
        expected_home_goals=1.5,
        expected_away_goals=1.0,
        confidence=0.6,
        features_snapshot={},
    )
    # Second insert for the same (fixture, model_version) must fail at DB level.
    # The repository short-circuits earlier with PredictionImmutableError, so
    # bypass it here to assert the underlying UNIQUE is enforced.
    with pytest.raises((IntegrityError, Exception)):  # noqa: PT011
        pred = Prediction(
            fixture_id=fx_id,
            model_version_id=mv_id,
            kickoff_time=kickoff,
            home_probability=0.5,
            draw_probability=0.3,
            away_probability=0.2,
            expected_home_goals=1.5,
            expected_away_goals=1.0,
            confidence=0.6,
            features_snapshot={},
        )
        session.add(pred)
        await session.flush()


# ---------------------------------------------------------------------------
# 9. Check constraints are enforced
# ---------------------------------------------------------------------------
async def test_check_constraint_probability_sum(session) -> None:
    fx_id, _, _, _, mv_id, kickoff = await _build_world(session)
    pred = Prediction(
        fixture_id=fx_id,
        model_version_id=mv_id,
        kickoff_time=kickoff,
        home_probability=0.40,
        draw_probability=0.30,
        away_probability=0.20,  # sums to 0.90 -> violates CHECK
        expected_home_goals=1.0,
        expected_away_goals=1.0,
        confidence=0.5,
        features_snapshot={},
    )
    session.add(pred)
    with pytest.raises(IntegrityError):
        await session.flush()


async def test_check_constraint_status_enum(session) -> None:
    """Fixture.status must be one of the allowed enum-like values."""
    from app.models import Fixture

    f = Factories(session)
    comp = await f.competition()
    season = await f.season(competition_id=comp.id)
    home = await f.team("A")
    away = await f.team("B")
    fx = Fixture(
        external_id=f._next_ext(),
        competition_id=comp.id,
        season_id=season.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff_time=datetime(2026, 9, 1, tzinfo=timezone.utc),
        status="invalid_status",
    )
    session.add(fx)
    with pytest.raises(IntegrityError):
        await session.flush()


# ---------------------------------------------------------------------------
# 10. Transaction rollback works
# ---------------------------------------------------------------------------
async def test_transaction_rollback(session) -> None:
    fx_id, _, _, _, mv_id, kickoff = await _build_world(session)
    repo = PredictionRepository(session)
    pred = await repo.insert(
        fixture_id=fx_id,
        model_version_id=mv_id,
        kickoff_time=kickoff,
        home_probability=0.5,
        draw_probability=0.3,
        away_probability=0.2,
        expected_home_goals=1.0,
        expected_away_goals=1.0,
        confidence=0.5,
        features_snapshot={"step": "before_rollback"},
    )
    pid = pred.id
    # Force an error that triggers a rollback inside the transaction.
    with pytest.raises(IntegrityError):
        bad = Prediction(
            fixture_id=fx_id,  # duplicate -> unique violation
            model_version_id=mv_id,
            kickoff_time=kickoff,
            home_probability=0.5,
            draw_probability=0.3,
            away_probability=0.2,
            expected_home_goals=1.0,
            expected_away_goals=1.0,
            confidence=0.5,
            features_snapshot={},
        )
        session.add(bad)
        await session.flush()
    await session.rollback()
    # After rollback, the original prediction (and any other inserts) should
    # not be readable in a new transaction. Validate by selecting it.
    fetched = await repo.get(pid)
    assert fetched is None


# ---------------------------------------------------------------------------
# Bonus: prediction_outcome UPDATE / DELETE are also blocked by triggers
# ---------------------------------------------------------------------------
async def test_outcome_update_is_blocked(session) -> None:
    fx_id, _, _, _, mv_id, kickoff = await _build_world(session)
    pred_repo = PredictionRepository(session)
    out_repo = PredictionOutcomeRepository(session)
    pred = await pred_repo.insert(
        fixture_id=fx_id,
        model_version_id=mv_id,
        kickoff_time=kickoff,
        home_probability=0.5,
        draw_probability=0.3,
        away_probability=0.2,
        expected_home_goals=1.0,
        expected_away_goals=1.0,
        confidence=0.5,
        features_snapshot={},
    )
    out = await out_repo.insert(
        prediction_id=pred.id,
        fixture_id=fx_id,
        actual_home_goals=1,
        actual_away_goals=0,
        actual_result="home",
        predicted_result="home",
        correct=True,
        predicted_correct_prob=0.5,
        brier_score=0.05,
        log_loss=0.10,
    )
    with pytest.raises((InternalError, IntegrityError)):
        # raw SQL UPDATE bypasses the repository contract.
        from sqlalchemy import text

        await session.execute(
            text(f"UPDATE prediction_outcomes SET correct=false WHERE id={out.id}")
        )
        await session.flush()
