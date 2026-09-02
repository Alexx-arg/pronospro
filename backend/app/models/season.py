"""``season`` entity (SCHEMA.md §2)."""

from __future__ import annotations

from datetime import date

from sqlalchemy import Boolean, Date, ForeignKey, Index, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Season(Base, TimestampMixin):
    """A season within a competition (e.g. Premier League 2025/2026)."""

    __tablename__ = "seasons"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    competition_id: Mapped[int] = mapped_column(
        ForeignKey("competitions.id", ondelete="CASCADE"),
        nullable=False,
    )
    external_id: Mapped[int] = mapped_column(nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    competition: Mapped["Competition"] = relationship(  # noqa: F821
        "Competition", back_populates="seasons"
    )
    fixtures: Mapped[list["Fixture"]] = relationship(  # noqa: F821
        "Fixture", back_populates="season", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("competition_id", "year", name="seasons_competition_id_year_key"),
        UniqueConstraint("external_id", "year", name="seasons_external_id_year_key"),
        Index(
            "idx_seasons_current",
            "is_current",
            postgresql_where="is_current",
        ),
    )
