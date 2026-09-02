"""``team`` entity (SCHEMA.md §3)."""

from __future__ import annotations

from sqlalchemy import Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Team(Base, TimestampMixin):
    """Football team."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    external_id: Mapped[int] = mapped_column(unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    short_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    code: Mapped[str | None] = mapped_column(String(10), nullable=True)
    country: Mapped[str | None] = mapped_column(String(100), nullable=True)
    logo: Mapped[str | None] = mapped_column(Text, nullable=True)
    venue: Mapped[str | None] = mapped_column(String(200), nullable=True)
    founded: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Reverse relationships used by other models (not strictly required by
    # SCHEMA.md but harmless and useful for repositories).
    home_fixtures: Mapped[list["Fixture"]] = relationship(  # noqa: F821
        "Fixture",
        foreign_keys="Fixture.home_team_id",
        back_populates="home_team",
    )
    away_fixtures: Mapped[list["Fixture"]] = relationship(  # noqa: F821
        "Fixture",
        foreign_keys="Fixture.away_team_id",
        back_populates="away_team",
    )

    __table_args__ = (
        Index(
            "idx_teams_name_lower",
            func.lower("name"),
        ),
    )
