"""``injury`` entity (SCHEMA.md §8)."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint,
    Date,
    ForeignKey,
    Index,
    Integer,
    String,
    TIMESTAMP,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Injury(Base, TimestampMixin):
    """Player injury / suspension record."""

    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    external_id: Mapped[int] = mapped_column(unique=True, nullable=False)
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False,
    )
    team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"),
        nullable=False,
    )
    competition_id: Mapped[int | None] = mapped_column(
        ForeignKey("competitions.id"),
        nullable=True,
    )
    fixture_id: Mapped[int | None] = mapped_column(
        ForeignKey("fixtures.id"),
        nullable=True,
    )
    type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        server_default="active",
    )
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    updated_external_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('active','doubtful','recovered','suspended')",
            name="injuries_status_check",
        ),
        Index(
            "idx_injuries_active",
            "status",
            "end_date",
            postgresql_where="status IN ('active','doubtful')",
        ),
        Index("idx_injuries_player", "player_id", "start_date"),
        Index("idx_injuries_fixture", "fixture_id"),
    )
