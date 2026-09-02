"""``fixture`` entity (SCHEMA.md §5)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    TIMESTAMP,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Fixture(Base, TimestampMixin):
    """A scheduled or played football match."""

    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    external_id: Mapped[int] = mapped_column(unique=True, nullable=False)
    competition_id: Mapped[int] = mapped_column(
        ForeignKey("competitions.id"),
        nullable=False,
    )
    season_id: Mapped[int] = mapped_column(
        ForeignKey("seasons.id", ondelete="CASCADE"),
        nullable=False,
    )
    home_team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id"),
        nullable=False,
    )
    away_team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id"),
        nullable=False,
    )
    matchday: Mapped[int | None] = mapped_column(Integer, nullable=True)
    round: Mapped[str | None] = mapped_column(String(50), nullable=True)
    venue: Mapped[str | None] = mapped_column(String(200), nullable=True)
    kickoff_time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="scheduled",
        server_default="scheduled",
    )
    status_short: Mapped[str | None] = mapped_column(String(5), nullable=True)
    home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    synced_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    season: Mapped["Season"] = relationship("Season", back_populates="fixtures")  # noqa: F821
    competition: Mapped["Competition"] = relationship("Competition")  # noqa: F821
    home_team: Mapped["Team"] = relationship(  # noqa: F821
        "Team",
        foreign_keys=[home_team_id],
        back_populates="home_fixtures",
    )
    away_team: Mapped["Team"] = relationship(  # noqa: F821
        "Team",
        foreign_keys=[away_team_id],
        back_populates="away_fixtures",
    )
    predictions: Mapped[list["Prediction"]] = relationship(  # noqa: F821
        "Prediction",
        back_populates="fixture",
        cascade="save-update, merge",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('scheduled','in_play','finished','postponed','cancelled','suspended')",
            name="fixtures_status_check",
        ),
        CheckConstraint(
            "(home_goals IS NULL AND away_goals IS NULL) "
            "OR (home_goals IS NOT NULL AND away_goals IS NOT NULL)",
            name="fixtures_goals_paired_check",
        ),
        Index("idx_fixtures_kickoff", "kickoff_time"),
        Index("idx_fixtures_status", "status", "kickoff_time"),
        Index("idx_fixtures_comp_season", "competition_id", "season_id"),
        Index("idx_fixtures_home", "home_team_id", "kickoff_time"),
        Index("idx_fixtures_away", "away_team_id", "kickoff_time"),
    )
