"""``PredictionOutcome`` repository — INSERT-only.

``prediction_outcomes`` is governed by the same immutability principle
as ``predictions``: once the actual result is recorded for a prediction
the outcome row must never change. Two layers enforce this:

1. Python: this repository exposes ``insert`` and read methods only.
2. Database: ``trg_no_update_outcomes`` / ``trg_no_delete_outcomes`` (added
   by Alembic migration 0001) raise an exception on UPDATE or DELETE.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import PredictionImmutableError
from app.models import PredictionOutcome


class PredictionOutcomeRepository:
    """INSERT/SELECT-only repository for :class:`PredictionOutcome`."""

    model = PredictionOutcome

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, outcome_id: int) -> PredictionOutcome | None:
        """Fetch an outcome by id."""
        return await self.session.get(PredictionOutcome, outcome_id)

    async def find_by_prediction(self, prediction_id: int) -> PredictionOutcome | None:
        """Return the outcome for a prediction, if already evaluated."""
        stmt = select(PredictionOutcome).where(
            PredictionOutcome.prediction_id == prediction_id
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list(
        self,
        *,
        correct: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PredictionOutcome]:
        """List outcomes ordered by evaluation time (most recent first)."""
        stmt = select(PredictionOutcome).order_by(
            PredictionOutcome.evaluated_at.desc(),
            PredictionOutcome.id.desc(),
        )
        if correct is not None:
            stmt = stmt.where(PredictionOutcome.correct.is_(correct))
        stmt = stmt.limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def insert(
        self,
        *,
        prediction_id: int,
        fixture_id: int,
        actual_home_goals: int,
        actual_away_goals: int,
        actual_result: str,
        predicted_result: str,
        correct: bool,
        predicted_correct_prob: float,
        brier_score: float,
        log_loss: float,
        evaluated_at: datetime | None = None,
    ) -> PredictionOutcome:
        """Insert a new outcome.

        Raises :class:`PredictionImmutableError` if an outcome already
        exists for ``prediction_id``: re-evaluation is not permitted.
        """
        existing = await self.find_by_prediction(prediction_id)
        if existing is not None:
            raise PredictionImmutableError(
                f"An outcome already exists for prediction_id={prediction_id}. "
                "Outcomes are immutable; cannot re-evaluate."
            )
        outcome = PredictionOutcome(
            prediction_id=prediction_id,
            fixture_id=fixture_id,
            actual_home_goals=actual_home_goals,
            actual_away_goals=actual_away_goals,
            actual_result=actual_result,
            predicted_result=predicted_result,
            correct=correct,
            predicted_correct_prob=predicted_correct_prob,
            brier_score=brier_score,
            log_loss=log_loss,
            evaluated_at=evaluated_at if evaluated_at is not None else datetime.now(),
        )
        self.session.add(outcome)
        await self.session.flush()
        await self.session.refresh(outcome)
        return outcome

    async def update(self, *_args: object, **_kwargs: object) -> None:
        """Intentionally disabled. Outcomes are immutable."""
        raise PredictionImmutableError(
            "PredictionOutcomeRepository.update() is forbidden. "
            "Outcomes are insert-only."
        )

    async def delete(self, *_args: object, **_kwargs: object) -> None:
        """Intentionally disabled. Outcomes are immutable."""
        raise PredictionImmutableError(
            "PredictionOutcomeRepository.delete() is forbidden. "
            "Outcomes are insert-only."
        )
