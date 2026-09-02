"""``lineup`` and ``lineup_players`` entities (SCHEMA.md §9)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    TIMESTAMP,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Lineup(Base, TimestampMixin):
    """A team's lineup for a fixture."""

    __tablename__ = "lineups"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id", ondelete="CASCADE"),
        nullable=False,
    )
    team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"),
        nullable=False,
    )
    is_home: Mapped[bool] = mapped_column(Boolean, nullable=False)
    formation: Mapped[str | None] = mapped_column(String(10), nullable=True)
    coach: Mapped[str | None] = mapped_column(String(200), nullable=True)
    updated_external_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    players: Mapped[list["LineupPlayer"]] = relationship(
        "LineupPlayer",
        back_populates="lineup",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("fixture_id", "team_id", name="lineups_fixture_team_key"),
    )


class LineupPlayer(Base):
    """A single player slot in a lineup."""

    __tablename__ = "lineup_players"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    lineup_id: Mapped[int] = mapped_column(
        ForeignKey("lineups.id", ondelete="CASCADE"),
        nullable=False,
    )
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[str | None] = mapped_column(String(20), nullable=True)
    position_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shirt_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_starter: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    lineup: Mapped[Lineup] = relationship("Lineup", back_populates="players")

    __table_args__ = (
        CheckConstraint(
            "position IN ('GK','DF','MF','FW') OR position IS NULL",
            name="lineup_players_position_check",
        ),
        UniqueConstraint(
            "lineup_id", "player_id", name="lineup_players_lineup_player_key"
        ),
        Index("idx_lineup_players_lineup", "lineup_id", "is_starter"),
    )
