"""``team_statistics`` entity (SCHEMA.md §6)."""

from __future__ import annotations

from datetime import date

from sqlalchemy import (
    Date,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class TeamStatistics(Base, TimestampMixin):
    """Aggregated team statistics for a season at a given ``as_of_date``."""

    __tablename__ = "team_statistics"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
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
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)

    fixtures_played: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    draws: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    goals_for: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    goals_against: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    clean_sheets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_to_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    form: Mapped[str | None] = mapped_column(String(20), nullable=True)
    shots_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shots_inside_box: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shots_outside_box: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    corners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    offsides: Mapped[int | None] = mapped_column(Integer, nullable=True)
    possession_avg: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    yellow_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    red_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    passes_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    passes_accuracy: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    xg: Mapped[float | None] = mapped_column(Numeric(6, 3), nullable=True)
    xga: Mapped[float | None] = mapped_column(Numeric(6, 3), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "team_id",
            "competition_id",
            "season_id",
            "as_of_date",
            name="team_statistics_team_comp_season_asof_key",
        ),
        Index("idx_team_stats_lkp", "season_id", "team_id", "as_of_date"),
    )
