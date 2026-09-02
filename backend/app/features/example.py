"""Canonical feature/target names + the immutable dataset example DTO.

Naming
------
Feature names follow a single convention:

    <scope>_<metric>[_<modifier>]_last_<N>

where:

* ``scope`` is one of ``home / away`` (perspective of the home or away
  team), ``h2h`` (head-to-head), ``table`` (reconstructed standings),
  or a custom prefix explained in docs/FEATURES.md.
* ``metric`` names the underlying quantity (``goals_for``, ``points``,
  ``wins``, ``xg_for``, ``table_position``, ``rest_days``).
* ``modifier`` (optional) disambiguates variants (e.g.
  ``home_performance`` vs ``away_performance``).
* ``last_<N>`` is the rolling window size when the feature is a window
  aggregate. Standings / rest / ELO / H2H don't use ``last_<N>``.

The names are exposed both as constants (``FEATURE_HOME_WINS_LAST_5``)
and as the ordered list :data:`FEATURE_NAMES`. The dataset builder writes
exactly :data:`FEATURE_NAMES` into ``metadata.json`` so a downstream
consumer can assert "this version had feature X at position Y".

Features are always float|None|None-int-on-the-rare-baseline (Phase 4
intentionally uses ``None`` for "missing" and reserves ``0`` for cases
where the math is genuinely zero — see docs/FEATURES.md §missing).

Targets live on a separate list (:data:`TARGET_NAMES`). Feature math
can never reach them: the assembler takes the post-match FixtureRow and
writes the targets in its own block, fully isolated from the features
dict during computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Feature name constants
# ---------------------------------------------------------------------------
# Grouped for readability. The order of the constants below defines the
# canonical column order of the produced dataset.csv (see FEATURE_NAMES).

# ---- FORM (last 3 / 5 / 10 — results sequence as W/D/L count) ----
# Number of wins / draws / losses in the last N fixtures of the team.
HOME_WINS_LAST_3 = "home_wins_last_3"
HOME_DRAWS_LAST_3 = "home_draws_last_3"
HOME_LOSSES_LAST_3 = "home_losses_last_3"

HOME_WINS_LAST_5 = "home_wins_last_5"
HOME_DRAWS_LAST_5 = "home_draws_last_5"
HOME_LOSSES_LAST_5 = "home_losses_last_5"

HOME_WINS_LAST_10 = "home_wins_last_10"
HOME_DRAWS_LAST_10 = "home_draws_last_10"
HOME_LOSSES_LAST_10 = "home_losses_last_10"

AWAY_WINS_LAST_3 = "away_wins_last_3"
AWAY_DRAWS_LAST_3 = "away_draws_last_3"
AWAY_LOSSES_LAST_3 = "away_losses_last_3"

AWAY_WINS_LAST_5 = "away_wins_last_5"
AWAY_DRAWS_LAST_5 = "away_draws_last_5"
AWAY_LOSSES_LAST_5 = "away_losses_last_5"

AWAY_WINS_LAST_10 = "away_wins_last_10"
AWAY_DRAWS_LAST_10 = "away_draws_last_10"
AWAY_LOSSES_LAST_10 = "away_losses_last_10"

# Points accrued across N most recent fixtures (3 for win, 1 for draw).
HOME_POINTS_LAST_5 = "home_points_last_5"
HOME_POINTS_LAST_10 = "home_points_last_10"
AWAY_POINTS_LAST_5 = "away_points_last_5"
AWAY_POINTS_LAST_10 = "away_points_last_10"

# ---- GOALS (last 5 / 10 — totals for the rolling match slate) ----
HOME_GOALS_FOR_LAST_5 = "home_goals_for_last_5"
HOME_GOALS_AGAINST_LAST_5 = "home_goals_against_last_5"
HOME_GOALS_FOR_LAST_10 = "home_goals_for_last_10"
HOME_GOALS_AGAINST_LAST_10 = "home_goals_against_last_10"

AWAY_GOALS_FOR_LAST_5 = "away_goals_for_last_5"
AWAY_GOALS_AGAINST_LAST_5 = "away_goals_against_last_5"
AWAY_GOALS_FOR_LAST_10 = "away_goals_for_last_10"
AWAY_GOALS_AGAINST_LAST_10 = "away_goals_against_last_10"

# Mean goals over the last N (mean returned when full window present).
HOME_GOALS_FOR_MEAN_LAST_5 = "home_goals_for_mean_last_5"
HOME_GOALS_AGAINST_MEAN_LAST_5 = "home_goals_against_mean_last_5"
HOME_GOALS_FOR_MEAN_LAST_10 = "home_goals_for_mean_last_10"
HOME_GOALS_AGAINST_MEAN_LAST_10 = "home_goals_against_mean_last_10"
AWAY_GOALS_FOR_MEAN_LAST_5 = "away_goals_for_mean_last_5"
AWAY_GOALS_AGAINST_MEAN_LAST_5 = "away_goals_against_mean_last_5"
AWAY_GOALS_FOR_MEAN_LAST_10 = "away_goals_for_mean_last_10"
AWAY_GOALS_AGAINST_MEAN_LAST_10 = "away_goals_against_mean_last_10"

# ---- HOME / AWAY specific splits ----
# Performance restricted to home matches (home team) / away matches (away).
HOME_HOME_POINTS_LAST_5 = "home_home_points_last_5"
HOME_HOME_GOALS_FOR_LAST_5 = "home_home_goals_for_last_5"
HOME_HOME_GOALS_AGAINST_LAST_5 = "home_home_goals_against_last_5"
AWAY_AWAY_POINTS_LAST_5 = "away_away_points_last_5"
AWAY_AWAY_GOALS_FOR_LAST_5 = "away_away_goals_for_last_5"
AWAY_AWAY_GOALS_AGAINST_LAST_5 = "away_away_goals_against_last_5"

# ---- H2H (head-to-head available strictly before kickoff) ----
H2H_HOME_WINS = "h2h_home_wins"
H2H_AWAY_WINS = "h2h_away_wins"
H2H_DRAWS = "h2h_draws"
H2H_TOTAL_GOALS_MEAN = "h2h_total_goals_mean"
H2H_SAMPLE_SIZE = "h2h_sample_size"

# ---- TABLE (reconstructed standings strictly before kickoff) ----
HOME_TABLE_POSITION = "home_table_position"
AWAY_TABLE_POSITION = "away_table_position"
HOME_POINTS_TABLE = "home_points_table"
AWAY_POINTS_TABLE = "away_points_table"
HOME_GOAL_DIFFERENCE_TABLE = "home_goal_difference_table"
AWAY_GOAL_DIFFERENCE_TABLE = "away_goal_difference_table"
TABLE_POINTS_DIFFERENCE = "table_points_difference"
TABLE_GOAL_DIFFERENCE_DIFFERENCE = "table_goal_difference_difference"

# ---- REST DAYS ----
HOME_REST_DAYS = "home_rest_days"
AWAY_REST_DAYS = "away_rest_days"

# ---- ELO (reconstructed from strictly-past results) ----
HOME_ELO_PRE_MATCH = "home_elo_pre_match"
AWAY_ELO_PRE_MATCH = "away_elo_pre_match"
ELO_DIFFERENCE = "elo_difference"

# ---- xG (snapshot strictly before kickoff_date) ----
# Phase 4 limitation: xG-per-match is NOT persisted by the sync layer
# (the ``fixtures`` table has no xg_home/xg_away columns). The only xG
# source today is the ``team_statistics`` seasonal snapshot (xg / xga
# cumulative up to ``as_of_date``). We therefore expose the latest xG /
# xGA **season-as-of** strictly before kickoff, NOT a rolling per-match
# window. The names below intentionally avoid "last_5" / "last_10"
# suffixes so that downstream consumers cannot mistake them for
# per-match rollings. Adding per-match xG later requires extending the
# fixtures model + a new sync branch.
HOME_XG_SEASON_ASOF = "home_xg_season_asof"
HOME_XGA_SEASON_ASOF = "home_xga_season_asof"
AWAY_XG_SEASON_ASOF = "away_xg_season_asof"
AWAY_XGA_SEASON_ASOF = "away_xga_season_asof"

# ---------------------------------------------------------------------------
# Ordered registry
# ---------------------------------------------------------------------------
FEATURE_NAMES: tuple[str, ...] = (
    # Form counts (last 3/5/10)
    HOME_WINS_LAST_3, HOME_DRAWS_LAST_3, HOME_LOSSES_LAST_3,
    HOME_WINS_LAST_5, HOME_DRAWS_LAST_5, HOME_LOSSES_LAST_5,
    HOME_WINS_LAST_10, HOME_DRAWS_LAST_10, HOME_LOSSES_LAST_10,
    AWAY_WINS_LAST_3, AWAY_DRAWS_LAST_3, AWAY_LOSSES_LAST_3,
    AWAY_WINS_LAST_5, AWAY_DRAWS_LAST_5, AWAY_LOSSES_LAST_5,
    AWAY_WINS_LAST_10, AWAY_DRAWS_LAST_10, AWAY_LOSSES_LAST_10,
    HOME_POINTS_LAST_5, HOME_POINTS_LAST_10,
    AWAY_POINTS_LAST_5, AWAY_POINTS_LAST_10,
    # Goals totals
    HOME_GOALS_FOR_LAST_5, HOME_GOALS_AGAINST_LAST_5,
    HOME_GOALS_FOR_LAST_10, HOME_GOALS_AGAINST_LAST_10,
    AWAY_GOALS_FOR_LAST_5, AWAY_GOALS_AGAINST_LAST_5,
    AWAY_GOALS_FOR_LAST_10, AWAY_GOALS_AGAINST_LAST_10,
    # Goals means
    HOME_GOALS_FOR_MEAN_LAST_5, HOME_GOALS_AGAINST_MEAN_LAST_5,
    HOME_GOALS_FOR_MEAN_LAST_10, HOME_GOALS_AGAINST_MEAN_LAST_10,
    AWAY_GOALS_FOR_MEAN_LAST_5, AWAY_GOALS_AGAINST_MEAN_LAST_5,
    AWAY_GOALS_FOR_MEAN_LAST_10, AWAY_GOALS_AGAINST_MEAN_LAST_10,
    # Home/away performance splits
    HOME_HOME_POINTS_LAST_5,
    HOME_HOME_GOALS_FOR_LAST_5, HOME_HOME_GOALS_AGAINST_LAST_5,
    AWAY_AWAY_POINTS_LAST_5,
    AWAY_AWAY_GOALS_FOR_LAST_5, AWAY_AWAY_GOALS_AGAINST_LAST_5,
    # H2H
    H2H_HOME_WINS, H2H_AWAY_WINS, H2H_DRAWS,
    H2H_TOTAL_GOALS_MEAN, H2H_SAMPLE_SIZE,
    # Standings
    HOME_TABLE_POSITION, AWAY_TABLE_POSITION,
    HOME_POINTS_TABLE, AWAY_POINTS_TABLE,
    HOME_GOAL_DIFFERENCE_TABLE, AWAY_GOAL_DIFFERENCE_TABLE,
    TABLE_POINTS_DIFFERENCE, TABLE_GOAL_DIFFERENCE_DIFFERENCE,
    # Rest
    HOME_REST_DAYS, AWAY_REST_DAYS,
    # ELO
    HOME_ELO_PRE_MATCH, AWAY_ELO_PRE_MATCH, ELO_DIFFERENCE,
    # xG season-as-of snapshots (NOT per-match rollings — see module doc.)
    HOME_XG_SEASON_ASOF, HOME_XGA_SEASON_ASOF,
    AWAY_XG_SEASON_ASOF, AWAY_XGA_SEASON_ASOF,
)

# ---------------------------------------------------------------------------
# Target names (kept in a separate list — feature math must never read)
# ---------------------------------------------------------------------------
HOME_WIN = "home_win"
DRAW = "draw"
AWAY_WIN = "away_win"
HOME_GOALS_TARGET = "home_goals"
AWAY_GOALS_TARGET = "away_goals"

TARGET_NAMES: tuple[str, ...] = (
    HOME_WIN, DRAW, AWAY_WIN,
    HOME_GOALS_TARGET, AWAY_GOALS_TARGET,
)


# ---------------------------------------------------------------------------
# DTO
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HistoricalMatchExample:
    """A single row of the historical dataset.

    ``features`` and ``targets`` are **separate dicts**: computation code
    receives only ``features`` plus the FixtureRow projection; the
    assembler writes ``targets`` once at the very end. There is no shared
    container, no pair of dicts aliasing — physically impossible for the
    feature math to read the target vector.

    The example is frozen so once built it cannot be mutated downstream.
    """

    fixture_id: int
    kickoff: datetime
    competition_id: int
    season_id: int
    home_team_id: int
    away_team_id: int
    features: dict[str, float | int | None] = field(default_factory=dict)
    targets: dict[str, int | None] = field(default_factory=dict)
    # Per-feature missing-data note: an entry per feature that was None
    # because of insufficient data (NOT for math-zero cases). Kept here so
    # the builder can expose a per-row "missing report" for auditability.
    missing_report: dict[str, str] = field(default_factory=dict)

    def as_row_dict(self) -> dict[str, Any]:
        """Flat row for CSV serialisation.

        Identity columns first, then features in :data:`FEATURE_NAMES`
        order, then targets in :data:`TARGET_NAMES` order. Missing
        features become ``""`` (CSV-string-NaN) — the loader
        (:func:`app.dataset.loader.load_csv`) restores them to ``None``
        rather than imputing ``0``.
        """
        out: dict[str, Any] = {
            "fixture_id": self.fixture_id,
            "kickoff": self.kickoff.isoformat(),
            "competition_id": self.competition_id,
            "season_id": self.season_id,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
        }
        for name in FEATURE_NAMES:
            out[name] = self.features.get(name)
        for name in TARGET_NAMES:
            out[name] = self.targets.get(name)
        return out


__all__ = [
    "AWAY_AWAY_GOALS_AGAINST_LAST_5",
    "AWAY_AWAY_GOALS_FOR_LAST_5",
    "AWAY_AWAY_POINTS_LAST_5",
    "AWAY_DRAWS_LAST_10",
    "AWAY_DRAWS_LAST_3",
    "AWAY_DRAWS_LAST_5",
    "AWAY_GOALS_AGAINST_LAST_10",
    "AWAY_GOALS_AGAINST_LAST_5",
    "AWAY_GOALS_AGAINST_MEAN_LAST_10",
    "AWAY_GOALS_AGAINST_MEAN_LAST_5",
    "AWAY_GOALS_FOR_LAST_10",
    "AWAY_GOALS_FOR_LAST_5",
    "AWAY_GOALS_FOR_MEAN_LAST_10",
    "AWAY_GOALS_FOR_MEAN_LAST_5",
    "AWAY_GOALS_TARGET",
    "AWAY_LOSSES_LAST_10",
    "AWAY_LOSSES_LAST_3",
    "AWAY_LOSSES_LAST_5",
    "AWAY_POINTS_LAST_10",
    "AWAY_POINTS_LAST_5",
    "AWAY_POINTS_TABLE",
    "AWAY_TABLE_POSITION",
    "AWAY_WIN",
    "AWAY_WINS_LAST_10",
    "AWAY_WINS_LAST_3",
    "AWAY_WINS_LAST_5",
    "AWAY_XG_SEASON_ASOF",
    "AWAY_XGA_SEASON_ASOF",
    "DRAW",
    "ELO_DIFFERENCE",
    "FEATURE_NAMES",
    "H2H_AWAY_WINS",
    "H2H_DRAWS",
    "H2H_HOME_WINS",
    "H2H_SAMPLE_SIZE",
    "H2H_TOTAL_GOALS_MEAN",
    "HOME_DRAWS_LAST_10",
    "HOME_DRAWS_LAST_3",
    "HOME_DRAWS_LAST_5",
    "HOME_ELO_PRE_MATCH",
    "HOME_GOAL_DIFFERENCE_TABLE",
    "HOME_GOALS_AGAINST_LAST_10",
    "HOME_GOALS_AGAINST_LAST_5",
    "HOME_GOALS_AGAINST_MEAN_LAST_10",
    "HOME_GOALS_AGAINST_MEAN_LAST_5",
    "HOME_GOALS_FOR_LAST_10",
    "HOME_GOALS_FOR_LAST_5",
    "HOME_GOALS_FOR_MEAN_LAST_10",
    "HOME_GOALS_FOR_MEAN_LAST_5",
    "HOME_GOALS_TARGET",
    "HOME_HOME_GOALS_AGAINST_LAST_5",
    "HOME_HOME_GOALS_FOR_LAST_5",
    "HOME_HOME_POINTS_LAST_5",
    "HOME_LOSSES_LAST_10",
    "HOME_LOSSES_LAST_3",
    "HOME_LOSSES_LAST_5",
    "HOME_POINTS_LAST_10",
    "HOME_POINTS_LAST_5",
    "HOME_POINTS_TABLE",
    "HOME_REST_DAYS",
    "HOME_TABLE_POSITION",
    "HOME_WIN",
    "HOME_WINS_LAST_10",
    "HOME_WINS_LAST_3",
    "HOME_WINS_LAST_5",
    "HOME_XG_SEASON_ASOF",
    "HOME_XGA_SEASON_ASOF",
    "HistoricalMatchExample",
    "TABLE_GOAL_DIFFERENCE_DIFFERENCE",
    "TABLE_POINTS_DIFFERENCE",
    "TARGET_NAMES",
]
