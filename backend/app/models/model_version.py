"""``model_version`` entity (SCHEMA.md §10).

Only one version of a given model name may be ``is_active`` at any time.
This is enforced at the database level via a partial unique index so that
even a buggy service-layer cannot enable two versions of the same model
simultaneously.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class ModelVersion(Base, TimestampMixin):
    """A versioned prediction model (elo / poisson / gradient_boosting)."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    parameters: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    predictions: Mapped[list["Prediction"]] = relationship(  # noqa: F821
        "Prediction",
        back_populates="model_version",
        cascade="save-update, merge",
    )

    __table_args__ = (
        CheckConstraint(
            "name IN ('elo','poisson','gradient_boosting')",
            name="model_versions_name_check",
        ),
        UniqueConstraint("name", "version", name="model_versions_name_version_key"),
        Index(
            "uq_model_active_per_name",
            "name",
            unique=True,
            postgresql_where="is_active",
        ),
    )
