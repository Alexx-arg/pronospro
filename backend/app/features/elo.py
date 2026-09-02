"""ELO temporal reconstruction.

For each fixture we want the ELO ratings that the two teams held
**immediately before** that fixture's kickoff, computed from fixtures
that kicked off **strictly earlier**.

The classic ELO recurrence (per-match):

    R_home_new = R_home + K * (actual_home - E_home)
    R_away_new = R_away + K * (actual_away - E_away)

where:

* ``K`` is the development coefficient (20 here, the standard "FIFA" value
  used in many football models — configurable per call).
* ``E_home`` = expected score for the home team given the rating
  difference adjusted by a home-field-advantage constant ``HFA``
  (default 65 rating points, roughly equivalent to ~0.5 goals).
* ``actual_home`` = 1 (home win), 0.5 (draw), 0 (home loss).
* Symmetrically for away.

Each team starts at ``BASE_RATING`` (1500 by convention). We process
the season's finished fixtures in chronological order; for each fixture
we first record the **pre-match** ratings of its two teams (these are
the features), then update both ratings using the actual result. The
target fixture is processed last by the builder when it asks for the
pre-match ratings of its teams — this means a fixture's own result
never participates in its own pre-match ELO.

Why chronological single-pass?
-----------------------------
The dataset builder needs the pre-match ELO of ~thousand fixtures.
Pythonically recomputing from scratch for each target would be O(N²);
the single-pass chronological sweep published here is O(N). Calls
that only need one fixture's ELO (e.g. runtime prediction features in
Phase 5) can use the same :func:`elo_pre_match` helper that also works
as a per-fixture convenience wrapper around :func:`elo_stream`.

The shape published intentionally matches :data:`FEATURE_NAMES`:

* ``home_elo_pre_match``  — the home team's rating pre-match
* ``away_elo_pre_match``  — the away team's rating pre-match
* ``elo_difference``      — home_elo_pre_match - away_elo_pre_match
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import NamedTuple

from app.features import example as ex
from app.features.rows import FixtureRow

#: Standard FIFA-style development coefficient.
DEFAULT_K: float = 20.0
#: Default home-field-advantage in ELO points (~0.5 goals worth).
DEFAULT_HFA: float = 65.0
#: Default starting rating. Higher than 0 to avoid negative output.
BASE_RATING: float = 1500.0


class EloPreMatch(NamedTuple):
    """Snapshot of the two teams' ELO ratings before a fixture."""

    fixture_id: int
    home_team_id: int
    away_team_id: int
    home_elo: float
    away_elo: float


def _expected_score(rating_a: float, rating_b: float, hfa: float) -> float:
    """Standard ELO expected score for A vs B with HFA applied to A.

    ``hfa`` is added to A's effective rating — used when A is the home
    team (we always make A the home team at call sites).
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - (rating_a + hfa)) / 400.0))


def _actual_score(home_goals: int, away_goals: int) -> tuple[float, float]:
    """Return (home_score, away_score) on the 1 / 0.5 / 0 scale."""
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def elo_stream(
    *,
    season_fixtures: Iterable[FixtureRow],
    k: float = DEFAULT_K,
    base_rating: float = BASE_RATING,
    hfa: float = DEFAULT_HFA,
) -> Iterator[EloPreMatch]:
    """Yield pre-match ELO snapshots for every fixture of the season.

    Pre-match ratings are yielded **before** the fixture's own result
    is folded into the ratings, so ``Y[t].home_elo`` is precisely the
    rating the home team held one instant before kickoff.

    The caller controls fixture ordering: when used by the dataset
    builder, ``season_fixtures`` should be sorted ascending by kickoff
    so that stepping the stream yields snapshots in chronological order
    and matches played on the same date use pre-match ratings that
    exclude each other (a simultaneous-match detail; see
    docs/FEATURES.md §elo for the documented subtlety).
    """
    ratings: dict[int, float] = {}

    def rating_of(team_id: int) -> float:
        return ratings.get(team_id, base_rating)

    for fx in season_fixtures:
        r_home = rating_of(fx.home_team_id)
        r_away = rating_of(fx.away_team_id)
        yield EloPreMatch(
            fixture_id=fx.fixture_id,
            home_team_id=fx.home_team_id,
            away_team_id=fx.away_team_id,
            home_elo=r_home,
            away_elo=r_away,
        )

        e_home = _expected_score(r_home, r_away, hfa)
        actual_home, actual_away = _actual_score(fx.home_goals, fx.away_goals)
        ratings[fx.home_team_id] = r_home + k * (actual_home - e_home)
        ratings[fx.away_team_id] = r_away + k * (actual_away - (1.0 - e_home))


def elo_pre_match(
    *,
    fixture: FixtureRow,
    season_fixtures: list[FixtureRow],
    k: float = DEFAULT_K,
    base_rating: float = BASE_RATING,
    hfa: float = DEFAULT_HFA,
) -> tuple[float, float]:
    """Convenience helper: pre-match ELOs for one fixture.

    Streams the fixtures strictly before ``fixture`` (already asc-ordered
    by the caller), then appends ``fixture`` itself so :func:`elo_stream`
    yields a snapshot for it BEFORE folding it in. Returns
    ``(home_elo, away_elo)``.
    """
    qualified = [
        fx
        for fx in season_fixtures
        if fx.kickoff_time < fixture.kickoff_time and fx.fixture_id != fixture.fixture_id
    ]
    qualified.sort(key=lambda fx: fx.kickoff_time)
    snap_iter = elo_stream(
        season_fixtures=list(qualified) + [fixture],
        k=k, base_rating=base_rating, hfa=hfa,
    )
    for snap in snap_iter:
        if snap.fixture_id == fixture.fixture_id:
            return snap.home_elo, snap.away_elo
    return base_rating, base_rating


def compute_elo_features(
    *,
    fixture: FixtureRow,
    season_fixtures: list[FixtureRow],
    k: float = DEFAULT_K,
    base_rating: float = BASE_RATING,
    hfa: float = DEFAULT_HFA,
) -> tuple[dict[str, float | int | None], dict[str, str]]:
    """Return ELO features for a single fixture.

    Used by the assembler. Internally:
    1. Filter the season's fixtures to those strictly before ``fixture``
       (defensive duplicate check, in case the caller passes the whole
       season including the target).
    2. Sort asc by kickoff.
    3. Inject ``fixture`` itself at the end with kickoff_time unchanged
       so the stream yields a snapshot for it BEFORE folding it in.
    4. Take the snapshot of ``fixture`` and return.
    """
    qualified = [
        fx
        for fx in season_fixtures
        if fx.kickoff_time < fixture.kickoff_time and fx.fixture_id != fixture.fixture_id
    ]
    qualified.sort(key=lambda fx: fx.kickoff_time)
    # The synthetic injection: append ``fixture`` at the very end and
    # synthesize a snapshot of its own pre-match ELOs.
    target_iterator = elo_stream(
        season_fixtures=list(qualified) + [fixture],
        k=k, base_rating=base_rating, hfa=hfa,
    )
    home_elo = base_rating
    away_elo = base_rating
    found = False
    for snap in target_iterator:
        if snap.fixture_id == fixture.fixture_id:
            home_elo, away_elo = snap.home_elo, snap.away_elo
            found = True
            break
    if not found:
        # Defensive — shouldn't happen. Means the fixture wasn't in
        # the stream we just composed.
        return (
            {ex.HOME_ELO_PRE_MATCH: None, ex.AWAY_ELO_PRE_MATCH: None,
             ex.ELO_DIFFERENCE: None},
            {ex.HOME_ELO_PRE_MATCH: "fixture not found in elo_stream"},
        )
    diff = home_elo - away_elo
    return (
        {
            ex.HOME_ELO_PRE_MATCH: float(home_elo),
            ex.AWAY_ELO_PRE_MATCH: float(away_elo),
            ex.ELO_DIFFERENCE: float(diff),
        },
        {},
    )


__all__ = [
    "BASE_RATING",
    "DEFAULT_HFA",
    "DEFAULT_K",
    "EloPreMatch",
    "compute_elo_features",
    "elo_pre_match",
    "elo_stream",
]
