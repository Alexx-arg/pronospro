"""As-of loaders: the single point where the temporal filter is applied.

Two contracts matter here:

1. **The fixture under analysis is excluded.** When we build features for
   a fixture ``F`` with kickoff at time ``T``, every loader returns only
   rows whose temporal key is **strictly less than** ``T``. We never
   return ``F`` itself or any match that kicked off at-or-after ``T``.
2. **No future rows, ever.** The DB may contain finished fixtures with
   ``kickoff_time`` after ``T`` (history written later by the sync
   services) or ``team_statistics`` snapshots whose
   ``as_of_date >= DATE(T)``. Those rows would leak future information
   into the feature math; the loaders refuse to emit them.

All loaders are async and project ORM rows onto frozen dataclasses
(:class:`~app.features.rows.FixtureRow`, :class:`~app.features.rows.TeamStatsRow`)
before returning. The rest of the feature layer therefore CANNOT trigger
DB round-trips while aggregating.

Two access patterns:

* :func:`load_finished_fixtures_as_of` — bulk version, for the dataset
  builder: one query per ``(competition_id, season_id)``, returns
  everything strictly before ``T``. The builder then iterates fixtures
  in chronological order and slices the pre-fetched list per kickoff.
* :func:`load_finished_fixtures_before` — single-fixture version: returns
  the rows of one ``(competition, season)`` strictly before ``T``, used
  by feature code paths that need a per-target slice (e.g. tests).

The bulk path is preferred for large dataset builds because it satisfies
the Phase 4 performance requirement: O(N) queries for the whole build
instead of O(N²).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.features.rows import FixtureRow, TeamStatsRow
from app.models import Fixture, TeamStatistics

_LOG = get_logger()


def _ensure_aware(value: datetime) -> datetime:
    """Normalise to tz-aware so comparisons are unambiguous."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _project_fixture(fx: Fixture) -> FixtureRow:
    """Project a :class:`Fixture` ORM row into a :class:`FixtureRow`.

    Only ``status == 'finished'`` and ``(home_goals, away_goals)`` not
    NULL are emitted; otherwise the caller skips the row. We assert the
    invariant explicitly here so the feature math can rely on it.
    """
    if fx.status != "finished" or fx.home_goals is None or fx.away_goals is None:
        raise ValueError(
            f"fixture {fx.id} is not finished (status={fx.status!r}); "
            "loader must not emit it"
        )
    return FixtureRow(
        fixture_id=fx.id,
        competition_id=fx.competition_id,
        season_id=fx.season_id,
        home_team_id=fx.home_team_id,
        away_team_id=fx.away_team_id,
        kickoff_time=_ensure_aware(fx.kickoff_time),
        home_goals=fx.home_goals,
        away_goals=fx.away_goals,
        matchday=fx.matchday,
        venue=fx.venue,
    )


def _project_team_stats(ts: TeamStatistics) -> TeamStatsRow:
    """Project a ``TeamStatistics`` ORM row into a :class:`TeamStatsRow`.

    ``xg`` / ``xga`` are ``Numeric`` in SQL and can be ``Decimal`` at the
    Python level; we cast to ``float`` so the rolling helpers stay
    arithmetic-light.
    """
    xg = float(ts.xg) if ts.xg is not None else None
    xga = float(ts.xga) if ts.xga is not None else None
    return TeamStatsRow(
        team_id=ts.team_id,
        competition_id=ts.competition_id,
        season_id=ts.season_id,
        as_of_date=ts.as_of_date,
        xg=xg,
        xga=xga,
    )


# ---------------------------------------------------------------------------
# Finished fixtures loader
# ---------------------------------------------------------------------------
async def load_finished_fixtures_as_of(
    session: AsyncSession,
    *,
    competition_id: int,
    season_id: int,
    exclude_fixture_id: int | None = None,
) -> list[FixtureRow]:
    """Load all finished fixtures of a ``(competition, season)``.

    "As of" semantics:

    * Only ``status='finished'`` rows.
    * ``(home_goals, away_goals)`` non-NULL (enforced by the DB
      immutability check anyway, but defensive here).
    * Optionally excludes ``exclude_fixture_id`` (the fixture under
      analysis, before its own kickoff). The temporal ``< T`` filter
      is applied by the feature layer per-feature via
      :func:`fixtures_before` to avoid O(N) copies of the list.
    * The result is **NOT** filtered here by kickoff: returning the full
      season lets the builder reuse the same list across the whole run.

    Returns a :class:`list` ordered ascending by ``kickoff_time`` so
    chronological processing is trivial.
    """
    stmt = (
        select(Fixture)
        .where(
            Fixture.competition_id == competition_id,
            Fixture.season_id == season_id,
            Fixture.status == "finished",
            Fixture.home_goals.is_not(None),
            Fixture.away_goals.is_not(None),
        )
        .order_by(Fixture.kickoff_time.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()

    out: list[FixtureRow] = []
    for fx in rows:
        if exclude_fixture_id is not None and fx.id == exclude_fixture_id:
            # Defensive: never return the target fixture itself.
            continue
        out.append(_project_fixture(fx))
    return out


def fixtures_before(
    season_fixtures: Iterable[FixtureRow],
    *,
    kickoff: datetime,
    exclude_fixture_id: int | None = None,
) -> list[FixtureRow]:
    """Slice a season's finished fixtures to those strictly before ``kickoff``.

    Pure (synchronous) helper — the temporal boundary belongs in this
    module even when it operates on already-loaded rows. The single
    boundary point makes the anti-leakage behaviour auditable: any
    fixture that ends up in feature math was emitted here, and every row
    emitted here has ``kickoff_time < kickoff``.

    Args:
        season_fixtures: rows returned by
            :func:`load_finished_fixtures_as_of` (already asc-ordered).
        kickoff: the cutoff T (tz-aware normalised). Strictly less-than.
        exclude_fixture_id: optional, drops the target fixture in case
            it had the same kickoff as some other match.
    """
    cutoff = _ensure_aware(kickoff)
    out: list[FixtureRow] = []
    for fx in season_fixtures:
        if fx.kickoff_time >= cutoff:
            # Season fixtures are asc-ordered; once we cross the cutoff
            # nothing past this can be eligible.
            break
        if exclude_fixture_id is not None and fx.fixture_id == exclude_fixture_id:
            continue
        out.append(fx)
    # Return newest-first for the rolling helpers (which take "first N").
    out.reverse()
    return out


def fixtures_before_unordered(
    season_fixtures: Iterable[FixtureRow],
    *,
    kickoff: datetime,
    exclude_fixture_id: int | None = None,
) -> list[FixtureRow]:
    """Same as :func:`fixtures_before` but works on unsorted input.

    Used by the H2H loader where the input is already filtered by team
    pair — we cannot assume ascending order. Redundant comparison but
    keeps the single-boundary contract readable.
    """
    cutoff = _ensure_aware(kickoff)
    out = [
        fx
        for fx in season_fixtures
        if fx.kickoff_time < cutoff
        and (exclude_fixture_id is None or fx.fixture_id != exclude_fixture_id)
    ]
    out.sort(key=lambda fx: fx.kickoff_time, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Per-team history (convenience wrapper used by feature families)
# ---------------------------------------------------------------------------
def team_history_before(
    *,
    team_id: int,
    season_fixtures: Iterable[FixtureRow],
    kickoff: datetime,
    exclude_fixture_id: int | None = None,
) -> list[FixtureRow]:
    """Per-team finished matches strictly before ``kickoff``.

    Returns newest-first. The ``team_id`` is the *internal* teams.id.
    """
    out = [
        fx
        for fx in season_fixtures
        if (fx.home_team_id == team_id or fx.away_team_id == team_id)
        and fx.kickoff_time < _ensure_aware(kickoff)
        and (exclude_fixture_id is None or fx.fixture_id != exclude_fixture_id)
    ]
    out.sort(key=lambda fx: fx.kickoff_time, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Head-to-head loader (cross-season pair history)
# ---------------------------------------------------------------------------
async def load_h2h_fixtures_as_of(
    session: AsyncSession,
    *,
    home_team_id: int,
    away_team_id: int,
    competition_id: int | None = None,
) -> list[FixtureRow]:
    """Load every finished fixture between ``home_team_id`` and
    ``away_team_id``, optionally restricted to one ``competition_id``.

    Cross-season by design: H2H between two historic clubs is most
    informative when accumulated across all seasons they played each
    other. No temporal cutoff is applied here — the per-fixture
    boundary ``kickoff < T`` is enforced by
    :func:`app.features.asof.fixtures_before_unordered` inside
    :func:`app.features.h2h.compute_h2h_features`. Returning the
    full pair history once lets the dataset builder reuse it for
    every fixture of that pair across the build.

    Args:
        session: SQLAlchemy async session.
        home_team_id: internal ``teams.id`` of one side of the pair.
        away_team_id: internal ``teams.id`` of the other side. The two
            teams are interchangeable (the loader emits fixtures in
            either venue assignment).
        competition_id: optional scope (e.g. restrict to a single
            league). ``None`` means "any competition" (default — the
            cross-season contract).

    Returns:
        Asc-ordered ``list[FixtureRow]`` of finished fixtures between
        the two teams. Each row has ``(home_goals, away_goals)`` non
        NULL (enforced by the loader projection; the DB CHECK
        constraint on the ``fixtures`` table would have rejected a
        finished row with NULL goals anyway).
    """
    from sqlalchemy import or_

    pair_clause = or_(
        (Fixture.home_team_id == home_team_id)
        & (Fixture.away_team_id == away_team_id),
        (Fixture.home_team_id == away_team_id)
        & (Fixture.away_team_id == home_team_id),
    )
    stmt = (
        select(Fixture)
        .where(
            pair_clause,
            Fixture.status == "finished",
            Fixture.home_goals.is_not(None),
            Fixture.away_goals.is_not(None),
        )
        .order_by(Fixture.kickoff_time.asc())
    )
    if competition_id is not None:
        stmt = stmt.where(Fixture.competition_id == competition_id)

    rows = (await session.execute(stmt)).scalars().all()
    return [_project_fixture(fx) for fx in rows]


# ---------------------------------------------------------------------------
# team_statistics loader (snapshots)
# ---------------------------------------------------------------------------
async def load_team_stats_as_of(
    session: AsyncSession,
    *,
    competition_id: int,
    season_id: int,
    team_ids: Iterable[int] | None = None,
) -> list[TeamStatsRow]:
    """Load every ``team_statistics`` snapshot for a season.

    The loader does NOT slice by ``as_of_date`` here because the slice
    depends on the match's kickoff (one snapshot per match). The
    per-match slice is done by :mod:`app.features.xg` via
    :func:`latest_team_stats_before`. Returning the full season once is
    what makes the bulk build O(1) overhead per fixture.

    ``team_ids`` optionally restricts which teams are loaded.
    """
    stmt = (
        select(TeamStatistics)
        .where(
            TeamStatistics.competition_id == competition_id,
            TeamStatistics.season_id == season_id,
        )
        .order_by(TeamStatistics.as_of_date.asc())
    )
    if team_ids is not None:
        ids = list(team_ids)
        if not ids:
            return []
        stmt = stmt.where(TeamStatistics.team_id.in_(ids))
    rows = (await session.execute(stmt)).scalars().all()
    return [_project_team_stats(ts) for ts in rows]


__all__ = [
    "fixtures_before",
    "fixtures_before_unordered",
    "load_finished_fixtures_as_of",
    "load_h2h_fixtures_as_of",
    "load_team_stats_as_of",
    "team_history_before",
]
