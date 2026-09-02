"""Reconstructed standings strictly before a kickoff (as-of table).

Per the Phase 4 design decision (Standings: rebuild from fixtures), we
reconstruct the league table at the moment of the target fixture's
kickoff **entirely from the finished fixtures of the same season**. No
``standings`` table exists, no API-Football ``/standings`` call is
made: the table is a deterministic function of the fixtures.

Reproduction of the table:

* Only ``FixtureRow`` entries whose ``kickoff_time < T`` qualify, *and*
  only those belonging to the same ``(competition, season)`` — the
  loader guarantees both.
* Points: ``W = 3``, ``D = 1``, ``L = 0``.
* Goals-for / goals-against are summed across the team's own fixtures.
* Goal-difference = goals-for - goals-against.
* Sorting key: ``(points DESC, goal_difference DESC, goals_for DESC,
  team_id ASC)`` — the team_id ASC tie-breaker keeps results
  deterministic when the natural keys draw and the actual competition's
  tiebreaker rules (e.g. head-to-head, fair-play) are unknown. See
  docs/FEATURES.md §standings for the caveat: real leagues use
  extra tiebreakers we cannot reproduce from fixtures alone; we pick
  the documented deterministic order.

The function returns the full standings (sorted, enriched with
``position``) so the assembler can look up the home and away teams'
rows. Performance: O(M log M) for sorting where M = number of teams in
the season; called once per target fixture. The dataset builder caches
the result across fixtures that share the same kickoff (matches playing
simultaneously share the same table ASCII).
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.features import example as ex
from app.features.rows import FixtureRow


class TeamStandingRow:
    """One team's accumulated record strictly before ``kickoff``.

    Kept as a plain class (no dataclass slots) because we mutate it as
    we stream the season fixtures; the public snapshot returned by
    :func:`standings_as_of` is the :class:`TeamStandingSnapshot` below.
    """

    __slots__ = ("team_id", "played", "w", "d", "l", "gf", "ga")

    def __init__(self, team_id: int) -> None:
        self.team_id = team_id
        self.played = 0
        self.w = 0
        self.d = 0
        self.l = 0
        self.gf = 0
        self.ga = 0

    @property
    def points(self) -> int:
        return 3 * self.w + 1 * self.d

    @property
    def goal_difference(self) -> int:
        return self.gf - self.ga


def _ensure_aware_tz(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def standings_as_of(
    *,
    season_fixtures: list[FixtureRow],
    kickoff: datetime,
    exclude_fixture_id: int | None = None,
) -> list[tuple[int, int, int, int, int]]:
    """Reconstruct standings strictly before ``kickoff``.

    Args:
        season_fixtures: the season's finished fixtures, as produced by
            :func:`app.features.asof.load_finished_fixtures_as_of` (any
            order — we re-sort internally).
        kickoff: target fixture's kickoff (tz-aware normalised).
        exclude_fixture_id: the target fixture itself, dropped from the
            table even if it would be eligible by kickoff ordering (it
            shouldn't be — its kickoff_time == cutoff and the boundary
            is strict ``<`` — but defensive against fp-rounding).

    Returns:
        Sorted list of ``(team_id, position, points, gf, ga, gd)`` tuples
        highest-ranked first.

    Sorting key: ``(points, gd, gf, -team_id)`` inversed via reverse
    on the first three; team_id ascending is the deterministic tiebreak.
    """
    cutoff = _ensure_aware_tz(kickoff)
    rows: dict[int, TeamStandingRow] = {}

    # Stream fixtures in chronological order — the table is a running
    # state machine. Strictly less-than cutoff.
    eligible = sorted(
        (
            fx
            for fx in season_fixtures
            if fx.kickoff_time < cutoff
            and (exclude_fixture_id is None or fx.fixture_id != exclude_fixture_id)
        ),
        key=lambda fx: fx.kickoff_time,
    )

    for fx in eligible:
        for side, team_id in (("home", fx.home_team_id), ("away", fx.away_team_id)):
            row = rows.setdefault(team_id, TeamStandingRow(team_id))
            row.played += 1
            gf = fx.goals_for(team_id)
            ga = fx.goals_against(team_id)
            row.gf += gf
            row.ga += ga
            if gf > ga:
                row.w += 1
            elif gf == ga:
                row.d += 1
            else:
                row.l += 1

    # Sort: points DESC, GD DESC, GF DESC, team_id ASC (tiebreak).
    ranked = sorted(
        rows.values(),
        key=lambda r: (-r.points, -r.goal_difference, -r.gf, r.team_id),
    )
    return [
        (r.team_id, idx + 1, r.points, r.gf, r.ga, r.goal_difference)
        for idx, r in enumerate(ranked)
    ]


def compute_standings_features(
    *,
    season_fixtures: list[FixtureRow],
    home_team_id: int,
    away_team_id: int,
    kickoff: datetime,
    exclude_fixture_id: int | None,
) -> tuple[dict[str, float | int | None], dict[str, str]]:
    """Return the standings-related features for the home and away team."""
    table = standings_as_of(
        season_fixtures=season_fixtures,
        kickoff=kickoff,
        exclude_fixture_id=exclude_fixture_id,
    )
    # Build a lookup for the two teams of interest:
    # team_id -> (position, points, goal_difference).
    lookup: dict[int, tuple[int, int, int]] = {}
    for team_id, position, pts, gf, ga, gd in table:
        lookup[team_id] = (position, pts, gd)

    features: dict[str, float | int | None] = {}
    missing: dict[str, str] = {}

    if home_team_id in lookup:
        pos, pts, gd = lookup[home_team_id]
        features[ex.HOME_TABLE_POSITION] = pos
        features[ex.HOME_POINTS_TABLE] = pts
        features[ex.HOME_GOAL_DIFFERENCE_TABLE] = gd
    else:
        features[ex.HOME_TABLE_POSITION] = None
        features[ex.HOME_POINTS_TABLE] = None
        features[ex.HOME_GOAL_DIFFERENCE_TABLE] = None
        missing[ex.HOME_TABLE_POSITION] = "home team has no finished fixture before T"

    if away_team_id in lookup:
        pos, pts, gd = lookup[away_team_id]
        features[ex.AWAY_TABLE_POSITION] = pos
        features[ex.AWAY_POINTS_TABLE] = pts
        features[ex.AWAY_GOAL_DIFFERENCE_TABLE] = gd
    else:
        features[ex.AWAY_TABLE_POSITION] = None
        features[ex.AWAY_POINTS_TABLE] = None
        features[ex.AWAY_GOAL_DIFFERENCE_TABLE] = None
        missing[ex.AWAY_TABLE_POSITION] = "away team has no finished fixture before T"

    if home_team_id in lookup and away_team_id in lookup:
        features[ex.TABLE_POINTS_DIFFERENCE] = (
            lookup[home_team_id][1] - lookup[away_team_id][1]
        )
        features[ex.TABLE_GOAL_DIFFERENCE_DIFFERENCE] = (
            lookup[home_team_id][2] - lookup[away_team_id][2]
        )
    else:
        features[ex.TABLE_POINTS_DIFFERENCE] = None
        features[ex.TABLE_GOAL_DIFFERENCE_DIFFERENCE] = None
        missing[ex.TABLE_POINTS_DIFFERENCE] = (
            "at least one team has no finished fixture before T"
        )

    return features, missing


__all__ = [
    "TeamStandingRow",
    "compute_standings_features",
    "standings_as_of",
]
