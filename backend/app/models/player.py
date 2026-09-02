"""``player`` entity and ``player_team_seasons`` association (SCHEMA.md §4)."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint,
    Date,
    ForeignKey,
    Index,
    Integer,
    Boolean,
    String,
    Text,
    TIMESTAMP,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Player(Base, TimestampMixin):
    """Football player."""

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    external_id: Mapped[int] = mapped_column(unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    photo: Mapped[str | None] = mapped_column(Text, nullable=True)
    nationality: Mapped[str | None] = mapped_column(String(100), nullable=True)
    birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    height_cm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    weight_kg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position: Mapped[str | None] = mapped_column(String(20), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "position IN ('GK','DF','MF','FW') OR position IS NULL",
            name="players_position_check",
        ),
    )


class PlayerTeamSeason(Base):
    """Player-team association for a given season/competition.

    Tracks loans and transfers. Creates a normalized many-to-many link so a
    player's team at the moment of a fixture can be looked up without
    embedding team ids inside player statistics.
    """

    __tablename__ = "player_team_seasons"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False,
    )
    team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"),
        nullable=False,
    )
    competition_id: Mapped[int] = mapped_column(
        ForeignKey("competitions.id"),
        nullable=False,
    )
    season_id: Mapped[int] = mapped_column(
        ForeignKey("seasons.id", ondelete="CASCADE"),
        nullable=False,
    )
    is_on_loan: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    # Note: only `created_at` is needed per SCHEMA.md; TimestampMixin would
    # add `updated_at` which is intentionally not part of this association.

    __table_args__ = (
        UniqueConstraint(
            "player_id", "season_id", "competition_id",
            name="player_team_seasons_player_season_competition_key",
        ),
        Index("idx_pts_team", "team_id", "season_id"),
        Index("idx_pts_player", "player_id", "season_id"),
    )
