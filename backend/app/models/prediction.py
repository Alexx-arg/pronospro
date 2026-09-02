"""``prediction`` entity (SCHEMA.md §11) — IMMUTABLE.

Hard contract enforced at three layers:

1. Python: ``PredictionRepository`` exposes only ``insert`` and read methods.
   There is intentionally no ``update`` or ``delete`` method.
2. Database permissions (installed by Alembic migration 0001): the
   ``app_user`` role only holds ``SELECT`` and ``INSERT`` on this table.
3. A ``BEFORE UPDATE`` trigger raises an exception even if a higher-privilege
   role attempts a direct ``UPDATE``.

``features_snapshot`` and ``explanation`` are JSONB. Per SCHEMA.md §11.2,
explanations that arrive *after* the INSERT (GLM latency) are NOT persisted
by mutating this row; they go to the sibling ``prediction_explanations``
table (1:1). The ``explanation`` column is set only when known at INSERT
time and is never updated afterwards.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Numeric,
    TIMESTAMP,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, CreatedAtMixin


class PredictionExplanationPayload(TypedDict, total=True):
    """Typed view of the optional ``explanation`` JSONB column.

    Mirrors the GLM service contract defined in
    ``docs/ARCHITECTURE.md`` (§4). The ``Prediction`` model stores this
    payload when known at INSERT time. It is **never** updated after INSERT.
    """

    summary: str
    main_factors: list[str]
    risk_factors: list[str]
    confidence_explanation: str


class Prediction(Base, CreatedAtMixin):
    """A prediction for a single fixture, produced by a model version.

    Immutable after INSERT (see module docstring). Note the absence of any
    ``updated_at`` column.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id", ondelete="RESTRICT"),
        nullable=False,
    )
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"),
        nullable=False,
    )
    kickoff_time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
    )
    home_probability: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    draw_probability: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    away_probability: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    expected_home_goals: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    expected_away_goals: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    features_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False
    )
    explanation: Mapped[PredictionExplanationPayload | None] = mapped_column(
        JSONB, nullable=True
    )

    fixture: Mapped["Fixture"] = relationship("Fixture", back_populates="predictions")  # noqa: F821
    model_version: Mapped["ModelVersion"] = relationship(  # noqa: F821
        "ModelVersion", back_populates="predictions"
    )
    explanation_row: Mapped["PredictionExplanation | None"] = relationship(  # noqa: F821
        "PredictionExplanation",
        back_populates="prediction",
        uselist=False,
        cascade="save-update, merge",
        passive_deletes=True,
    )
    outcome: Mapped["PredictionOutcome | None"] = relationship(  # noqa: F821
        "PredictionOutcome",
        back_populates="prediction",
        uselist=False,
        cascade="save-update, merge",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            "home_probability BETWEEN 0 AND 1",
            name="predictions_home_probability_check",
        ),
        CheckConstraint(
            "draw_probability BETWEEN 0 AND 1",
            name="predictions_draw_probability_check",
        ),
        CheckConstraint(
            "away_probability BETWEEN 0 AND 1",
            name="predictions_away_probability_check",
        ),
        CheckConstraint(
            "expected_home_goals >= 0", name="predictions_expected_home_goals_check"
        ),
        CheckConstraint(
            "expected_away_goals >= 0", name="predictions_expected_away_goals_check"
        ),
        CheckConstraint(
            "confidence BETWEEN 0 AND 1", name="predictions_confidence_check"
        ),
        CheckConstraint(
            "home_probability + draw_probability + away_probability BETWEEN 0.999 AND 1.001",
            name="predictions_probabilities_sum_check",
        ),
        UniqueConstraint(
            "fixture_id", "model_version_id", name="predictions_fixture_model_key"
        ),
        Index("idx_predictions_model", "model_version_id", "created_at"),
        Index("idx_predictions_fixture", "fixture_id"),
        Index(
            "idx_predictions_created",
            "created_at",
            postgresql_using="gin",
        ),
    )
