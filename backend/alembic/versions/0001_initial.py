"""Initial schema, immutable predictions policy, app role.

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-15 00:00:00.000000

Creates every table defined in docs/SCHEMA.md with the same constraint and
index names so that the schema stays verifiable against the design doc.

Also installs:

- A generic ``set_updated_at()`` trigger function applied to every **mutable**
  table (except ``predictions`` and ``prediction_explanations`` per §14).
- The ``prevent_prediction_update()`` trigger on ``predictions`` (§11.1) and a
  mirror ``prevent_prediction_outcome_update()`` on ``prediction_outcomes``
  to discourage manual edits through roles with elevated privileges.
- The ``app_user`` role with minimal grants per §15:
    * SELECT/INSERT on almost everything,
    * UPDATE on ``fixtures``, ``team_statistics``, ``player_statistics``,
      ``injuries``, ``lineups`` (operational upserts),
    * never UPDATE/DELETE on ``predictions`` / ``prediction_outcomes``.

Because Alembic autogenerate emits RAW ``create_table`` statements with JSONB
columns and CHECK constraints, this migration is hand-tuned to match the
schema document (constraint names, index names, partial indexes and the GIN
index on predictions.created_at that Alembic cannot recreate reliably).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Tables that SHOULD auto-refresh updated_at (per §14).
_MUTABLE_TABLES = (
    "competitions",
    "seasons",
    "teams",
    "player_team_seasons",
    "fixtures",
    "team_statistics",
    "player_statistics",
    "injuries",
    "lineups",
    "lineup_players",
    "model_versions",
    "prediction_outcomes",
)


def _jsonb_default() -> sa.TextClause:
    return sa.text("'{}'::jsonb")


def upgrade() -> None:
    # ============================================================
    # 1. competitions
    # ============================================================
    op.create_table(
        "competitions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("external_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("country", sa.String(length=100), nullable=True),
        sa.Column("logo", sa.Text(), nullable=True),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "type IN ('league','cup','playoff','super_cup')",
            name="competitions_type_check",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_competitions"),
        sa.UniqueConstraint(
            "external_id", name="uq_competitions_external_id"
        ),
    )
    op.create_index("idx_competitions_country", "competitions", ["country"])

    # ============================================================
    # 2. seasons
    # ============================================================
    op.create_table(
        "seasons",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("competition_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Integer(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column(
            "is_current",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["competition_id"],
            ["competitions.id"],
            name="fk_seasons_competition_id_competitions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_seasons"),
        sa.UniqueConstraint(
            "competition_id", "year", name="seasons_competition_id_year_key"
        ),
        sa.UniqueConstraint(
            "external_id", "year", name="seasons_external_id_year_key"
        ),
    )
    op.create_index(
        "idx_seasons_current",
        "seasons",
        ["is_current"],
        postgresql_where=sa.text("is_current"),
    )

    # ============================================================
    # 3. teams
    # ============================================================
    op.create_table(
        "teams",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("external_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("short_name", sa.String(length=50), nullable=True),
        sa.Column("code", sa.String(length=10), nullable=True),
        sa.Column("country", sa.String(length=100), nullable=True),
        sa.Column("logo", sa.Text(), nullable=True),
        sa.Column("venue", sa.String(length=200), nullable=True),
        sa.Column("founded", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_teams"),
        sa.UniqueConstraint("external_id", name="uq_teams_external_id"),
    )
    op.create_index(
        "idx_teams_name_lower", "teams", [sa.text("LOWER(name)")]
    )

    # ============================================================
    # 4. players + player_team_seasons
    # ============================================================
    op.create_table(
        "players",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("external_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("photo", sa.Text(), nullable=True),
        sa.Column("nationality", sa.String(length=100), nullable=True),
        sa.Column("birth_date", sa.Date(), nullable=True),
        sa.Column("height_cm", sa.Integer(), nullable=True),
        sa.Column("weight_kg", sa.Integer(), nullable=True),
        sa.Column("position", sa.String(length=20), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "position IN ('GK','DF','MF','FW') OR position IS NULL",
            name="players_position_check",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_players"),
        sa.UniqueConstraint("external_id", name="uq_players_external_id"),
    )

    op.create_table(
        "player_team_seasons",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("competition_id", sa.BigInteger(), nullable=False),
        sa.Column("season_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "is_on_loan",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"],
            name="fk_player_team_seasons_player_id_players",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"],
            name="fk_player_team_seasons_team_id_teams",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["competition_id"], ["competitions.id"],
            name="fk_player_team_seasons_competition_id_competitions",
        ),
        sa.ForeignKeyConstraint(
            ["season_id"], ["seasons.id"],
            name="fk_player_team_seasons_season_id_seasons",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_player_team_seasons"),
        sa.UniqueConstraint(
            "player_id", "season_id", "competition_id",
            name="player_team_seasons_player_season_competition_key",
        ),
    )
    op.create_index("idx_pts_team", "player_team_seasons", ["team_id", "season_id"])
    op.create_index("idx_pts_player", "player_team_seasons", ["player_id", "season_id"])

    # ============================================================
    # 5. fixtures
    # ============================================================
    op.create_table(
        "fixtures",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("external_id", sa.Integer(), nullable=False),
        sa.Column("competition_id", sa.BigInteger(), nullable=False),
        sa.Column("season_id", sa.BigInteger(), nullable=False),
        sa.Column("home_team_id", sa.BigInteger(), nullable=False),
        sa.Column("away_team_id", sa.BigInteger(), nullable=False),
        sa.Column("matchday", sa.Integer(), nullable=True),
        sa.Column("round", sa.String(length=50), nullable=True),
        sa.Column("venue", sa.String(length=200), nullable=True),
        sa.Column("kickoff_time", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'scheduled'"),
            nullable=False,
        ),
        sa.Column("status_short", sa.String(length=5), nullable=True),
        sa.Column("home_goals", sa.Integer(), nullable=True),
        sa.Column("away_goals", sa.Integer(), nullable=True),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("synced_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('scheduled','in_play','finished','postponed','cancelled','suspended')",
            name="fixtures_status_check",
        ),
        sa.CheckConstraint(
            "(home_goals IS NULL AND away_goals IS NULL) "
            "OR (home_goals IS NOT NULL AND away_goals IS NOT NULL)",
            name="fixtures_goals_paired_check",
        ),
        sa.ForeignKeyConstraint(
            ["competition_id"], ["competitions.id"],
            name="fk_fixtures_competition_id_competitions",
        ),
        sa.ForeignKeyConstraint(
            ["season_id"], ["seasons.id"],
            name="fk_fixtures_season_id_seasons",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["home_team_id"], ["teams.id"],
            name="fk_fixtures_home_team_id_teams",
        ),
        sa.ForeignKeyConstraint(
            ["away_team_id"], ["teams.id"],
            name="fk_fixtures_away_team_id_teams",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_fixtures"),
        sa.UniqueConstraint("external_id", name="uq_fixtures_external_id"),
    )
    op.create_index("idx_fixtures_kickoff", "fixtures", ["kickoff_time"])
    op.create_index("idx_fixtures_status", "fixtures", ["status", "kickoff_time"])
    op.create_index(
        "idx_fixtures_comp_season", "fixtures", ["competition_id", "season_id"]
    )
    op.create_index(
        "idx_fixtures_home", "fixtures", ["home_team_id", "kickoff_time"]
    )
    op.create_index(
        "idx_fixtures_away", "fixtures", ["away_team_id", "kickoff_time"]
    )

    # ============================================================
    # 6. team_statistics
    # ============================================================
    op.create_table(
        "team_statistics",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("competition_id", sa.BigInteger(), nullable=False),
        sa.Column("season_id", sa.BigInteger(), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("fixtures_played", sa.Integer(), server_default="0", nullable=False),
        sa.Column("wins", sa.Integer(), server_default="0", nullable=False),
        sa.Column("draws", sa.Integer(), server_default="0", nullable=False),
        sa.Column("losses", sa.Integer(), server_default="0", nullable=False),
        sa.Column("goals_for", sa.Integer(), server_default="0", nullable=False),
        sa.Column("goals_against", sa.Integer(), server_default="0", nullable=False),
        sa.Column("clean_sheets", sa.Integer(), server_default="0", nullable=False),
        sa.Column("failed_to_score", sa.Integer(), server_default="0", nullable=False),
        sa.Column("form", sa.String(length=20), nullable=True),
        sa.Column("shots_total", sa.Integer(), nullable=True),
        sa.Column("shots_on_target", sa.Integer(), nullable=True),
        sa.Column("shots_inside_box", sa.Integer(), nullable=True),
        sa.Column("shots_outside_box", sa.Integer(), nullable=True),
        sa.Column("fouls", sa.Integer(), nullable=True),
        sa.Column("corners", sa.Integer(), nullable=True),
        sa.Column("offsides", sa.Integer(), nullable=True),
        sa.Column("possession_avg", sa.Numeric(5, 2), nullable=True),
        sa.Column("yellow_cards", sa.Integer(), nullable=True),
        sa.Column("red_cards", sa.Integer(), nullable=True),
        sa.Column("passes_total", sa.Integer(), nullable=True),
        sa.Column("passes_accuracy", sa.Numeric(5, 2), nullable=True),
        sa.Column("xg", sa.Numeric(6, 3), nullable=True),
        sa.Column("xga", sa.Numeric(6, 3), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"],
            name="fk_team_statistics_team_id_teams",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["competition_id"], ["competitions.id"],
            name="fk_team_statistics_competition_id_competitions",
        ),
        sa.ForeignKeyConstraint(
            ["season_id"], ["seasons.id"],
            name="fk_team_statistics_season_id_seasons",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_team_statistics"),
        sa.UniqueConstraint(
            "team_id", "competition_id", "season_id", "as_of_date",
            name="team_statistics_team_comp_season_asof_key",
        ),
    )
    op.create_index(
        "idx_team_stats_lkp",
        "team_statistics",
        ["season_id", "team_id", "as_of_date"],
    )

    # ============================================================
    # 7. player_statistics
    # ============================================================
    op.create_table(
        "player_statistics",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("competition_id", sa.BigInteger(), nullable=False),
        sa.Column("season_id", sa.BigInteger(), nullable=False),
        sa.Column("appearances", sa.Integer(), server_default="0", nullable=False),
        sa.Column("starts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("minutes_played", sa.Integer(), server_default="0", nullable=False),
        sa.Column("goals", sa.Integer(), server_default="0", nullable=False),
        sa.Column("assists", sa.Integer(), server_default="0", nullable=False),
        sa.Column("yellow_cards", sa.Integer(), server_default="0", nullable=False),
        sa.Column("red_cards", sa.Integer(), server_default="0", nullable=False),
        sa.Column("rating", sa.Numeric(4, 2), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"],
            name="fk_player_statistics_player_id_players",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"],
            name="fk_player_statistics_team_id_teams",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["competition_id"], ["competitions.id"],
            name="fk_player_statistics_competition_id_competitions",
        ),
        sa.ForeignKeyConstraint(
            ["season_id"], ["seasons.id"],
            name="fk_player_statistics_season_id_seasons",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_player_statistics"),
        sa.UniqueConstraint(
            "player_id", "season_id", "competition_id",
            name="player_statistics_player_season_competition_key",
        ),
    )
    op.create_index(
        "idx_player_stats_team", "player_statistics", ["team_id", "season_id"]
    )

    # ============================================================
    # 8. injuries
    # ============================================================
    op.create_table(
        "injuries",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("external_id", sa.Integer(), nullable=False),
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("competition_id", sa.BigInteger(), nullable=True),
        sa.Column("fixture_id", sa.BigInteger(), nullable=True),
        sa.Column("type", sa.String(length=100), nullable=True),
        sa.Column("reason", sa.String(length=255), nullable=True),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'active'"),
            nullable=False,
        ),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("updated_external_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('active','doubtful','recovered','suspended')",
            name="injuries_status_check",
        ),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"],
            name="fk_injuries_player_id_players",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"],
            name="fk_injuries_team_id_teams",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["competition_id"], ["competitions.id"],
            name="fk_injuries_competition_id_competitions",
        ),
        sa.ForeignKeyConstraint(
            ["fixture_id"], ["fixtures.id"],
            name="fk_injuries_fixture_id_fixtures",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_injuries"),
        sa.UniqueConstraint("external_id", name="uq_injuries_external_id"),
    )
    op.create_index(
        "idx_injuries_active",
        "injuries",
        ["status", "end_date"],
        postgresql_where=sa.text("status IN ('active','doubtful')"),
    )
    op.create_index(
        "idx_injuries_player", "injuries", ["player_id", "start_date"]
    )
    op.create_index("idx_injuries_fixture", "injuries", ["fixture_id"])

    # ============================================================
    # 9. lineups + lineup_players
    # ============================================================
    op.create_table(
        "lineups",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fixture_id", sa.BigInteger(), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("is_home", sa.Boolean(), nullable=False),
        sa.Column("formation", sa.String(length=10), nullable=True),
        sa.Column("coach", sa.String(length=200), nullable=True),
        sa.Column("updated_external_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["fixture_id"], ["fixtures.id"],
            name="fk_lineups_fixture_id_fixtures",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"],
            name="fk_lineups_team_id_teams",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_lineups"),
        sa.UniqueConstraint("fixture_id", "team_id", name="lineups_fixture_team_key"),
    )

    op.create_table(
        "lineup_players",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("lineup_id", sa.BigInteger(), nullable=False),
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("position", sa.String(length=20), nullable=True),
        sa.Column("position_x", sa.Integer(), nullable=True),
        sa.Column("position_y", sa.Integer(), nullable=True),
        sa.Column("shirt_number", sa.Integer(), nullable=True),
        sa.Column(
            "is_starter",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "position IN ('GK','DF','MF','FW') OR position IS NULL",
            name="lineup_players_position_check",
        ),
        sa.ForeignKeyConstraint(
            ["lineup_id"], ["lineups.id"],
            name="fk_lineup_players_lineup_id_lineups",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"],
            name="fk_lineup_players_player_id_players",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_lineup_players"),
        sa.UniqueConstraint(
            "lineup_id", "player_id", name="lineup_players_lineup_player_key"
        ),
    )
    op.create_index(
        "idx_lineup_players_lineup", "lineup_players", ["lineup_id", "is_starter"]
    )

    # ============================================================
    # 10. model_versions
    # ============================================================
    op.create_table(
        "model_versions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "parameters",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "name IN ('elo','poisson','gradient_boosting')",
            name="model_versions_name_check",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_model_versions"),
        sa.UniqueConstraint(
            "name", "version", name="model_versions_name_version_key"
        ),
    )
    op.create_index(
        "uq_model_active_per_name",
        "model_versions",
        ["name"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )

    # ============================================================
    # 11. predictions (immutable)
    # ============================================================
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fixture_id", sa.BigInteger(), nullable=False),
        sa.Column("model_version_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("kickoff_time", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("home_probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("draw_probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("away_probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("expected_home_goals", sa.Numeric(5, 2), nullable=False),
        sa.Column("expected_away_goals", sa.Numeric(5, 2), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 2), nullable=False),
        sa.Column("features_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("explanation", postgresql.JSONB(), nullable=True),
        sa.CheckConstraint(
            "home_probability BETWEEN 0 AND 1",
            name="predictions_home_probability_check",
        ),
        sa.CheckConstraint(
            "draw_probability BETWEEN 0 AND 1",
            name="predictions_draw_probability_check",
        ),
        sa.CheckConstraint(
            "away_probability BETWEEN 0 AND 1",
            name="predictions_away_probability_check",
        ),
        sa.CheckConstraint(
            "expected_home_goals >= 0",
            name="predictions_expected_home_goals_check",
        ),
        sa.CheckConstraint(
            "expected_away_goals >= 0",
            name="predictions_expected_away_goals_check",
        ),
        sa.CheckConstraint(
            "confidence BETWEEN 0 AND 1",
            name="predictions_confidence_check",
        ),
        sa.CheckConstraint(
            "home_probability + draw_probability + away_probability "
            "BETWEEN 0.999 AND 1.001",
            name="predictions_probabilities_sum_check",
        ),
        sa.ForeignKeyConstraint(
            ["fixture_id"], ["fixtures.id"],
            name="fk_predictions_fixture_id_fixtures",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"], ["model_versions.id"],
            name="fk_predictions_model_version_id_model_versions",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_predictions"),
        sa.UniqueConstraint(
            "fixture_id", "model_version_id", name="predictions_fixture_model_key"
        ),
    )
    op.create_index(
        "idx_predictions_model", "predictions", ["model_version_id", "created_at"]
    )
    op.create_index("idx_predictions_fixture", "predictions", ["fixture_id"])
    op.create_index(
        "idx_predictions_created",
        "predictions",
        ["created_at"],
        postgresql_using="gin",
    )

    # ============================================================
    # 11.2 prediction_explanations (append-only)
    # ============================================================
    op.create_table(
        "prediction_explanations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("prediction_id", sa.BigInteger(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("main_factors", postgresql.JSONB(), nullable=False),
        sa.Column("risk_factors", postgresql.JSONB(), nullable=False),
        sa.Column("confidence_explanation", sa.Text(), nullable=False),
        sa.Column("model_used", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"], ["predictions.id"],
            name="fk_prediction_explanations_prediction_id_predictions",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_prediction_explanations"),
        sa.UniqueConstraint(
            "prediction_id",
            name="prediction_explanations_prediction_key",
        ),
    )

    # ============================================================
    # 12. prediction_outcomes (insert-only)
    # ============================================================
    op.create_table(
        "prediction_outcomes",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("prediction_id", sa.BigInteger(), nullable=False),
        sa.Column("fixture_id", sa.BigInteger(), nullable=False),
        sa.Column("actual_home_goals", sa.Integer(), nullable=False),
        sa.Column("actual_away_goals", sa.Integer(), nullable=False),
        sa.Column("actual_result", sa.String(length=5), nullable=False),
        sa.Column("predicted_result", sa.String(length=5), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=False),
        sa.Column("predicted_correct_prob", sa.Numeric(6, 5), nullable=False),
        sa.Column("brier_score", sa.Numeric(8, 5), nullable=False),
        sa.Column("log_loss", sa.Numeric(8, 5), nullable=False),
        sa.Column(
            "evaluated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "actual_home_goals >= 0",
            name="prediction_outcomes_actual_home_goals_check",
        ),
        sa.CheckConstraint(
            "actual_away_goals >= 0",
            name="prediction_outcomes_actual_away_goals_check",
        ),
        sa.CheckConstraint(
            "actual_result IN ('home','draw','away')",
            name="prediction_outcomes_actual_result_check",
        ),
        sa.CheckConstraint(
            "predicted_result IN ('home','draw','away')",
            name="prediction_outcomes_predicted_result_check",
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"], ["predictions.id"],
            name="fk_prediction_outcomes_prediction_id_predictions",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["fixture_id"], ["fixtures.id"],
            name="fk_prediction_outcomes_fixture_id_fixtures",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_prediction_outcomes"),
        sa.UniqueConstraint(
            "prediction_id",
            name="prediction_outcomes_prediction_key",
        ),
    )
    op.create_index("idx_outcomes_model", "prediction_outcomes", ["fixture_id"])
    op.create_index("idx_outcomes_correct", "prediction_outcomes", ["correct"])

    # ============================================================
    # §11.1 / §11.2 / §15 — triggers and roles
    # ============================================================
    _install_triggers_and_roles()


def _install_triggers_and_roles() -> None:
    """Run the post-DDL plumbing documented in SCHEMA.md."""
    bind = op.get_bind()

    # --- Generic updated_at trigger function + per-table trigger ---
    bind.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION set_updated_at()
            RETURNS TRIGGER LANGUAGE plpgsql AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$;
            """
        )
    )
    for tbl in _MUTABLE_TABLES:
        bind.execute(
            sa.text(
                f"""
                CREATE TRIGGER trg_set_updated_at
                BEFORE UPDATE ON {tbl}
                FOR EACH ROW EXECUTE FUNCTION set_updated_at();
                """
            )
        )

    # --- predictions immutable trigger (BEFORE UPDATE) ---
    bind.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION prevent_prediction_update()
            RETURNS TRIGGER LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION 'predictions is immutable: cannot UPDATE row id=%', OLD.id;
            END;
            $$;
            """
        )
    )
    bind.execute(
        sa.text(
            """
            CREATE TRIGGER trg_no_update_predictions
            BEFORE UPDATE ON predictions
            FOR EACH ROW EXECUTE FUNCTION prevent_prediction_update();
            """
        )
    )

    # --- predictions DELETE block (BEFORE DELETE) ---
    # SCHEMA.md §11.1 specifies INSERT/SELECT only. We add a DELETE guard
    # at the trigger level too (defense-in-depth) so that even the
    # migrator/superuser sloppiness can't accidentally delete history.
    bind.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION prevent_prediction_delete()
            RETURNS TRIGGER LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION 'predictions is immutable: cannot DELETE row id=%', OLD.id;
            END;
            $$;
            """
        )
    )
    bind.execute(
        sa.text(
            """
            CREATE TRIGGER trg_no_delete_predictions
            BEFORE DELETE ON predictions
            FOR EACH ROW EXECUTE FUNCTION prevent_prediction_delete();
            """
        )
    )

    # --- prediction_outcomes insert-only (BEFORE UPDATE/DELETE) ---
    bind.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION prevent_prediction_outcome_mutation()
            RETURNS TRIGGER LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION
                    'prediction_outcomes is insert-only: cannot % row id=%',
                    TG_OP, COALESCE(OLD.id, NEW.id);
            END;
            $$;
            """
        )
    )
    bind.execute(
        sa.text(
            """
            CREATE TRIGGER trg_no_update_outcomes
            BEFORE UPDATE ON prediction_outcomes
            FOR EACH ROW EXECUTE FUNCTION prevent_prediction_outcome_mutation();
            """
        )
    )
    bind.execute(
        sa.text(
            """
            CREATE TRIGGER trg_no_delete_outcomes
            BEFORE DELETE ON prediction_outcomes
            FOR EACH ROW EXECUTE FUNCTION prevent_prediction_outcome_mutation();
            """
        )
    )

    # --- Roles and grants (§15) ---
    # Idempotent: roles may already exist when running against a
    # developer-managed cluster that doesn't recreate the DB on every boot.
    bind.execute(
        sa.text(
            "DO $$ BEGIN "
            "  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN "
            "    CREATE ROLE app_user LOGIN PASSWORD 'changeme'; "
            "  END IF; "
            "END $$;"
        )
    )

    # Defaults
    bind.execute(sa.text("REVOKE ALL ON predictions FROM PUBLIC;"))
    bind.execute(sa.text("REVOKE ALL ON prediction_outcomes FROM PUBLIC;"))
    bind.execute(sa.text("REVOKE ALL ON prediction_explanations FROM PUBLIC;"))

    # grants: SELECT/INSERT on every table belonging to app_user
    for tbl in {
        "competitions",
        "seasons",
        "teams",
        "players",
        "player_team_seasons",
        "fixtures",
        "team_statistics",
        "player_statistics",
        "injuries",
        "lineups",
        "lineup_players",
        "model_versions",
        "predictions",
        "prediction_outcomes",
        "prediction_explanations",
    }:
        # Public schema sequences (the BIGSERIAL PKs) need USAGE for INSERT.
        bind.execute(sa.text("GRANT USAGE, SELECT ON SCHEMA public TO app_user;"))
        bind.execute(sa.text(f"GRANT SELECT, INSERT ON {tbl} TO app_user;"))
        # Sequence for SERIAL/BIGSERIAL PK autoincrement.
        bind.execute(sa.text(f"GRANT USAGE, SELECT ON SEQUENCE {tbl}_id_seq TO app_user;"))

    # Operational UPDATEs only on a few tables (SCHEMA.md §15).
    for tbl in (
        "fixtures",
        "team_statistics",
        "player_statistics",
        "injuries",
        "lineups",
    ):
        bind.execute(sa.text(f"GRANT UPDATE ON {tbl} TO app_user;"))

    # Hard-deny UPDATE/DELETE on the immutable tables even though we granted
    # INSERT/SELECT only. With this explicit REVOKE, accidental role grants
    # (e.g. granting ALL to PUBLIC by migrations) cannot bypass the policy.
    bind.execute(sa.text("REVOKE UPDATE, DELETE ON predictions FROM app_user;"))
    bind.execute(sa.text("REVOKE UPDATE, DELETE ON prediction_outcomes FROM app_user;"))
    bind.execute(
        sa.text("REVOKE UPDATE, DELETE ON prediction_explanations FROM app_user;")
    )


def downgrade() -> None:
    """Drop triggers, functions, roles grants and every table in reverse."""
    bind = op.get_bind()
    # Triggers / functions
    for stmt in (
        "DROP TRIGGER IF EXISTS trg_set_updated_at ON competitions",
        "DROP TRIGGER IF EXISTS trg_set_updated_at ON seasons",
        "DROP TRIGGER IF EXISTS trg_set_updated_at ON teams",
        "DROP TRIGGER IF EXISTS trg_set_updated_at ON player_team_seasons",
        "DROP TRIGGER IF EXISTS trg_set_updated_at ON fixtures",
        "DROP TRIGGER IF EXISTS trg_set_updated_at ON team_statistics",
        "DROP TRIGGER IF EXISTS trg_set_updated_at ON player_statistics",
        "DROP TRIGGER IF EXISTS trg_set_updated_at ON injuries",
        "DROP TRIGGER IF EXISTS trg_set_updated_at ON lineups",
        "DROP TRIGGER IF EXISTS trg_set_updated_at ON lineup_players",
        "DROP TRIGGER IF EXISTS trg_set_updated_at ON model_versions",
        "DROP TRIGGER IF EXISTS trg_set_updated_at ON prediction_outcomes",
        "DROP TRIGGER IF EXISTS trg_no_update_predictions ON predictions",
        "DROP TRIGGER IF EXISTS trg_no_delete_predictions ON predictions",
        "DROP TRIGGER IF EXISTS trg_no_update_outcomes ON prediction_outcomes",
        "DROP TRIGGER IF EXISTS trg_no_delete_outcomes ON prediction_outcomes",
        "DROP FUNCTION IF EXISTS set_updated_at()",
        "DROP FUNCTION IF EXISTS prevent_prediction_update()",
        "DROP FUNCTION IF EXISTS prevent_prediction_delete()",
        "DROP FUNCTION IF EXISTS prevent_prediction_outcome_mutation()",
    ):
        try:
            bind.execute(sa.text(stmt))
        except Exception:
            pass  # ignore missing-object errors on downgrade
    op.drop_index("idx_outcomes_correct", table_name="prediction_outcomes")
    op.drop_index("idx_outcomes_model", table_name="prediction_outcomes")
    op.drop_table("prediction_outcomes")
    op.drop_index(
        "prediction_explanations_prediction_key", table_name="prediction_explanations"
    )
    op.drop_table("prediction_explanations")
    op.drop_index("idx_predictions_created", table_name="predictions")
    op.drop_index("idx_predictions_fixture", table_name="predictions")
    op.drop_index("idx_predictions_model", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("uq_model_active_per_name", table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_index("idx_lineup_players_lineup", table_name="lineup_players")
    op.drop_table("lineup_players")
    op.drop_table("lineups")
    op.drop_index("idx_injuries_fixture", table_name="injuries")
    op.drop_index("idx_injuries_player", table_name="injuries")
    op.drop_index("idx_injuries_active", table_name="injuries")
    op.drop_table("injuries")
    op.drop_index("idx_player_stats_team", table_name="player_statistics")
    op.drop_table("player_statistics")
    op.drop_index("idx_team_stats_lkp", table_name="team_statistics")
    op.drop_table("team_statistics")
    op.drop_index("idx_fixtures_away", table_name="fixtures")
    op.drop_index("idx_fixtures_home", table_name="fixtures")
    op.drop_index("idx_fixtures_comp_season", table_name="fixtures")
    op.drop_index("idx_fixtures_status", table_name="fixtures")
    op.drop_index("idx_fixtures_kickoff", table_name="fixtures")
    op.drop_table("fixtures")
    op.drop_index("idx_pts_player", table_name="player_team_seasons")
    op.drop_index("idx_pts_team", table_name="player_team_seasons")
    op.drop_table("player_team_seasons")
    op.drop_table("players")
    op.drop_index("idx_teams_name_lower", table_name="teams")
    op.drop_table("teams")
    op.drop_index("idx_seasons_current", table_name="seasons")
    op.drop_table("seasons")
    op.drop_index("idx_competitions_country", table_name="competitions")
    op.drop_table("competitions")
