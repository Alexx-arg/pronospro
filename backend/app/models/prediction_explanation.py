"""``prediction_explanation`` entity (SCHEMA.md §11.2) — append-only.

Used ONLY when an explanation arrives after the prediction has already been
INSERTed (GLM latency). The immutable ``predictions`` row is never mutated;
this row is created with a fresh INSERT and joined back via
``prediction_id``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, CreatedAtMixin


class PredictionExplanation(Base, CreatedAtMixin):
    """Textual explanation produced by the GLM service for a prediction.

    One-to-one with :class:`Prediction`. Created **only** when the
    explanation arrives after the prediction was persisted (numbers-first
    principle). No ``updated_at`` column: this row is also append-only.
    """

    __tablename__ = "prediction_explanations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    main_factors: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    risk_factors: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    confidence_explanation: Mapped[str] = mapped_column(Text, nullable=False)
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)

    prediction: Mapped["Prediction"] = relationship(  # noqa: F821
        "Prediction", back_populates="explanation_row"
    )

    __table_args__ = (
        UniqueConstraint("prediction_id", name="prediction_explanations_prediction_key"),
    )
