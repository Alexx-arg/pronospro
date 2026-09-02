"""Deterministic factories for feature-engineering tests.

Every factory in this module is **pure** and **deterministic** — no
randomness, no time-of-test dependency. Failures in anti-leakage tests
must be reproducible byte-for-byte across runs.

The factories construct :class:`app.features.rows.FixtureRow` and
:class:`app.features.rows.TeamStatsRow` projections directly (bypassing
the ORM), so unit tests do NOT need a Postgres fixture to assert
feature-math correctness. They only need the schemas we project on.

Layout
------
* :func:`make_fixture`   — one finished fixture, deterministic ids.
* :func:`make_team_stats` — one team_statistics snapshot row.
* :func:`team_history`   — a chronological ladder of fixtures all
  involving ``team_id`` (or optionally the same opponent), so window
  tests have a stable "last N matches" sequence.
* :func:`season_ladder`  — a chain of finished fixtures between two
  fixed teams for H2H / form / standings verification.

Helpers
-------
* :data:`FIXED_TEAMS`     — canonical ids reused across tests.
* :data:`FIXED_COMPETITION`, :data:`FIXED_SEASON_ID`,
  :data:`FIXED_KICKOFF` — anchor values tests index by.

Real DB ids are BIGSERIAL ints (>0); to keep tests readable the
factories use small ids (1, 2, ...) — distinguishable from external_ids
that the API-Football sync would carry (large numbers).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from app.features.rows import FixtureRow, TeamStatsRow

#: Canonical team ids reused across the anti-leakage suite.
FIXED_TEAMS: dict[str, int] = {
    "HOME": 1,
    "AWAY": 2,
    "OTHER": 3,
}
#: Anchor competition / season; tests override only when needed.
FIXED_COMPETITION: int = 39
FIXED_SEASON_ID: int = 1
#: Anchor kickoff; tests offset by timedelta relative to this.
FIXED_KICKOFF: datetime = datetime(2026, 3, 20, 19, 0, tzinfo=timezone.utc)


def make_fixture(
    *,
    fixture_id: int,
    home_team_id: int,
    away_team_id: int,
    kickoff: datetime,
    home_goals: int,
    away_goals: int,
    competition_id: int = FIXED_COMPETITION,
    season_id: int = FIXED_SEASON_ID,
    matchday: int | None = None,
    venue: str | None = None,
) -> FixtureRow:
    """Build a finished :class:`FixtureRow` deterministically."""
    return FixtureRow(
        fixture_id=fixture_id,
        competition_id=competition_id,
        season_id=season_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        kickoff_time=kickoff,
        home_goals=home_goals,
        away_goals=away_goals,
        matchday=matchday,
        venue=venue,
    )


def make_team_stats(
    *,
    team_id: int,
    as_of_date: date,
    xg: float | None,
    xga: float | None,
    competition_id: int = FIXED_COMPETITION,
    season_id: int = FIXED_SEASON_ID,
) -> TeamStatsRow:
    """Build a :class:`TeamStatsRow` deterministically."""
    return TeamStatsRow(
        team_id=team_id,
        competition_id=competition_id,
        season_id=season_id,
        as_of_date=as_of_date,
        xg=xg,
        xga=xga,
    )


def team_history(
    *,
    team_id: int,
    goals_scored: list[int],
    goals_conceded: list[int] | None = None,
    opponent_id: int | None = None,
    start_kickoff: datetime = FIXED_KICKOFF,
    step_days: int = 7,
    is_home_alternating: bool = False,
    fixtures_signed: bool = False,
) -> list[FixtureRow]:
    """Build a chronological ladder of fixtures involving ``team_id``.

    The first fixture in the returned list kicks off earliest; the LAST
    is the most recent. All fixtures finish strictly before any sensible
    "target" the caller is going to build later.

    Args:
        team_id: the team whose recent form is being simulated.
        goals_scored: list of goals scored by ``team_id`` in each
            historical fixture, oldest-to-newest.
        goals_conceded: same shape, goals conceded. Defaults to a
            constant ``0`` per match if not provided.
        opponent_id: a fixed opponent (e.g. for H2H tests). When
            ``None``, a synthetic opponent id is used.
        start_kickoff: kickoff of the OLDEST fixture; subsequent
            fixtures advance by ``step_days`` each.
        step_days: gap between consecutive fixtures.
        is_home_alternating: when True, ``team_id`` alternates between
            home and away across the ladder.
        fixtures_signed: when True the returned list is newest-first (as
            expected by the rolling helpers via
            :func:`app.features.asof.team_history_before`). When False
            it's oldest-first — useful for building the season universe
            and passing to loaders that filter-by-kickoff themselves.
    """
    n = len(goals_scored)
    if goals_conceded is None:
        goals_conceded = [0] * n
    if len(goals_conceded) != n:
        raise ValueError(
            "goals_scored and goals_conceded must have the same length"
        )
    opp = opponent_id if opponent_id is not None else FIXED_TEAMS["OTHER"]

    out: list[FixtureRow] = []
    found_ids: set[int] = set()
    fid = 1000
    for i in range(n):
        # pick a unique id
        while fid in found_ids:
            fid += 1
        found_ids.add(fid)

        is_home = (i % 2 == 0) if is_home_alternating else True
        home = team_id if is_home else opp
        away = opp if is_home else team_id
        gf = goals_scored[i]
        ga = goals_conceded[i]
        kickoff = start_kickoff + timedelta(days=step_days * i)
        fx = make_fixture(
            fixture_id=fid,
            home_team_id=home,
            away_team_id=away,
            kickoff=kickoff,
            home_goals=gf if is_home else ga,
            away_goals=ga if is_home else gf,
        )
        out.append(fx)
        fid += 1
    if fixtures_signed:
        out.reverse()
    return out


def season_ladder(
    *,
    home_id: int = FIXED_TEAMS["HOME"],
    away_id: int = FIXED_TEAMS["AWAY"],
    pairs: list[tuple[int, int]],
    # (home_goals, away_goals) — chronological, oldest first
    start_kickoff: datetime = FIXED_KICKOFF,
    step_days: int = 1,
) -> list[FixtureRow]:
    """Build a season fixture universe where every match is one fixed
    home/away pair.

    Used by standings / elo tests where the home & away team of the
    target fixture recurrently plays each other.
    """
    out: list[FixtureRow] = []
    fid = 500
    for i, (hg, ag) in enumerate(pairs):
        out.append(
            make_fixture(
                fixture_id=fid + i,
                home_team_id=home_id,
                away_team_id=away_id,
                kickoff=start_kickoff + timedelta(days=step_days * i),
                home_goals=hg,
                away_goals=ag,
            )
        )
    return out


__all__ = [
    "FIXED_COMPETITION",
    "FIXED_KICKOFF",
    "FIXED_SEASON_ID",
    "FIXED_TEAMS",
    "make_fixture",
    "make_team_stats",
    "season_ladder",
    "team_history",
]
