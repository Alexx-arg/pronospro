"""``Prediction`` repository — INSERT-only.

This repository deliberately exposes NO ``update`` or ``delete`` method:
``predictions`` is immutable (SCHEMA.md §11.1 / §11.2). The Python layer
cannot mutate a prediction by accident. The database trigger
``trg_no_update_predictions`` (and ``trg_no_delete_predictions``) is a
defense-in-depth safety net that we exercise in integration tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import PredictionImmutableError
from app.models import Prediction
from app.models.prediction import PredictionExplanationPayload


class PredictionRepository:
    """INSERT/SELECT-only repository for :class:`Prediction`.

    Unlike :class:`BaseRepository`, this class does NOT inherit from
    :class:`BaseRepository` because doing so would expose the inherited
    ``session.delete`` / ``session.merge`` mechanics.  We keep the same
    constructor signature and re-declare the two safe read helpers.
    """

    model = Prediction

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    async def get(self, prediction_id: int) -> Prediction | None:
        """Fetch a prediction by id."""
        return await self.session.get(Prediction, prediction_id)

    async def find_by_fixture(self, fixture_id: int) -> list[Prediction]:
        """Return every prediction row stored for a fixture (any model)."""
        stmt = (
            select(Prediction)
            .where(Prediction.fixture_id == fixture_id)
            .order_by(Prediction.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def find_by_fixture_and_model(
        self,
        *,
        fixture_id: int,
        model_version_id: int,
    ) -> Prediction | None:
        """Return the unique prediction for a given (fixture, model) pair."""
        stmt = select(Prediction).where(
            Prediction.fixture_id == fixture_id,
            Prediction.model_version_id == model_version_id,
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list(
        self,
        *,
        model_version_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Prediction]:
        """Return a slice of predictions ordered by creation (most recent first)."""
        stmt = select(Prediction).order_by(
            Prediction.created_at.desc(), Prediction.id.desc()
        )
        if model_version_id is not None:
            stmt = stmt.where(Prediction.model_version_id == model_version_id)
        stmt = stmt.limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # writes (only INSERT)
    # ------------------------------------------------------------------
    async def insert(
        self,
        *,
        fixture_id: int,
        model_version_id: int,
        kickoff_time: datetime,
        home_probability: float,
        draw_probability: float,
        away_probability: float,
        expected_home_goals: float,
        expected_away_goals: float,
        confidence: float,
        features_snapshot: dict[str, Any],
        explanation: PredictionExplanationPayload | None = None,
    ) -> Prediction:
        """Insert a new prediction.

        Raises :class:`PredictionImmutableError` if a prediction for the
        same ``(fixture_id, model_version_id)`` already exists; updating it
        is forbidden — a new ``model_version`` should be created instead.
        """
        existing = await self.find_by_fixture_and_model(
            fixture_id=fixture_id, model_version_id=model_version_id
        )
        if existing is not None:
            raise PredictionImmutableError(
                f"A prediction already exists for fixture={fixture_id}, "
                f"model_version={model_version_id}. Create a new model "
                "version instead of mutating the immutable row."
            )

        prediction = Prediction(
            fixture_id=fixture_id,
            model_version_id=model_version_id,
            kickoff_time=kickoff_time,
            home_probability=home_probability,
            draw_probability=draw_probability,
            away_probability=away_probability,
            expected_home_goals=expected_home_goals,
            expected_away_goals=expected_away_goals,
            confidence=confidence,
            features_snapshot=features_snapshot,
            explanation=explanation,
        )
        self.session.add(prediction)
        await self.session.flush()
        await self.session.refresh(prediction)
        return prediction

    # ------------------------------------------------------------------
    # explicit non-existence: prevent accidental mutation
    # ------------------------------------------------------------------
    async def update(self, *_args: object, **_kwargs: object) -> None:
        """Intentionally disabled. Predictions are immutable."""
        raise PredictionImmutableError(
            "PredictionRepository.update() is forbidden: predictions are immutable."
        )

    async def delete(self, *_args: object, **_kwargs: object) -> None:
        """Intentionally disabled. Predictions are immutable."""
        raise PredictionImmutableError(
            "PredictionRepository.delete() is forbidden: predictions are immutable."
        )

    async def merge(self, *_args: object, **_kwargs: object) -> None:
        """Intentionally disabled. Merge could resurrect mutated state."""
        raise PredictionImmutableError(
            "PredictionRepository.merge() is forbidden: predictions are immutable."
        )
