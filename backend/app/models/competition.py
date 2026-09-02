"""``competition`` entity (SCHEMA.md §1)."""

from __future__ import annotations

from sqlalchemy import CheckConstraint, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Competition(Base, TimestampMixin):
    """Football competition (league/cup/playoff/super_cup)."""

    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    external_id: Mapped[int] = mapped_column(unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    country: Mapped[str | None] = mapped_column(String(100), nullable=True)
    logo: Mapped[str | None] = mapped_column(Text, nullable=True)
    type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    seasons: Mapped[list["Season"]] = relationship(  # noqa: F821 (forward ref)
        "Season",
        back_populates="competition",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "type IN ('league','cup','playoff','super_cup')",
            name="competitions_type_check",
        ),
        Index("idx_competitions_country", "country"),
    )
