"""``prediction_outcome`` entity (SCHEMA.md §12) — insert-only.

Once a fixture is finished, the actual result is recorded against each
existing prediction. The outcome is immutable: no subsequent ``UPDATE`` /
``DELETE`` is permitted by the ``app_user`` role (installed by the initial
Alembic migration). The trigger-based ``updated_at`` audit refresh is
intentionally disabled for this table.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    TIMESTAMP,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class PredictionOutcome(Base, TimestampMixin):
    """Evaluated outcome of a prediction once the fixture finished.

    Although this model uses :class:`TimestampMixin` (because SCHEMA.md
    defines both ``created_at`` and ``updated_at``), the database-level
    permissions installed by Alembic 0001 prevent ``UPDATE`` and ``DELETE``
    by the ``app_user`` role. The ``updated_at`` column therefore only
    moves on revisions performed by the ``migrator`` role (which is excluded
    from normal traffic).
    """

    __tablename__ = "prediction_outcomes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id", ondelete="RESTRICT"),
        nullable=False,
    )
    actual_home_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_away_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_result: Mapped[str] = mapped_column(String(5), nullable=False)
    predicted_result: Mapped[str] = mapped_column(String(5), nullable=False)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    predicted_correct_prob: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    brier_score: Mapped[float] = mapped_column(Numeric(8, 5), nullable=False)
    log_loss: Mapped[float] = mapped_column(Numeric(8, 5), nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    prediction: Mapped["Prediction"] = relationship(  # noqa: F821
        "Prediction", back_populates="outcome"
    )

    __table_args__ = (
        CheckConstraint(
            "actual_home_goals >= 0", name="prediction_outcomes_actual_home_goals_check"
        ),
        CheckConstraint(
            "actual_away_goals >= 0", name="prediction_outcomes_actual_away_goals_check"
        ),
        CheckConstraint(
            "actual_result IN ('home','draw','away')",
            name="prediction_outcomes_actual_result_check",
        ),
        CheckConstraint(
            "predicted_result IN ('home','draw','away')",
            name="prediction_outcomes_predicted_result_check",
        ),
        UniqueConstraint(
            "prediction_id",
            name="prediction_outcomes_prediction_key",
        ),
        Index("idx_outcomes_model", "fixture_id"),
        Index("idx_outcomes_correct", "correct"),
    )
