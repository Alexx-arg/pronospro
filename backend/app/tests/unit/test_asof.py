"""Anti-leakage tests on :mod:`app.features.asof` (the boundary layer).

The asof module is the **only** place where the temporal filter
``kickoff < T`` is applied for in-memory slicing.feature math can only
see rows emitted by it. Tests here validate that:

* the boundary is strict ``<``, never ``<=``;
* the target fixture itself is excluded even if it shares a kickoff;
* per-team history is sliced correctly.

A second batch (``test_anti_leakage_features``) tests the full assembler
with deliberately-injected future data and asserts features don't
change.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.features.asof import (
    fixtures_before,
    fixtures_before_unordered,
    team_history_before,
)
from app.tests.unit._factories import (
    FIXED_KICKOFF,
    make_fixture,
)


def test_fixtures_before_is_strict() -> None:
    """A fixture kicking off exactly at ``T`` is excluded.

    Even if the float point matches (`T == kickoff`), the boundary is
    ``<``, never ``<=``. This is what blocks the target fixture itself
    from feeding its own future into its own features.
    """
    earlier = make_fixture(
        fixture_id=10, home_team_id=1, away_team_id=2,
        kickoff=FIXED_KICKOFF - timedelta(days=1), home_goals=1, away_goals=0,
    )
    same_time = make_fixture(
        fixture_id=11, home_team_id=1, away_team_id=3,
        kickoff=FIXED_KICKOFF, home_goals=2, away_goals=2,
    )
    later = make_fixture(
        fixture_id=12, home_team_id=2, away_team_id=3,
        kickoff=FIXED_KICKOFF + timedelta(hours=1),
        home_goals=0, away_goals=0,
    )
    universe = [earlier, same_time, later]
    # Strict cutoff at FIXED_KICKOFF: only the earlier fixture qualifies.
    out = fixtures_before(universe, kickoff=FIXED_KICKOFF)
    assert [fx.fixture_id for fx in out] == [10]


def test_fixtures_before_excludes_target_fixture_id() -> None:
    """Even if a fixture had the same kickoff as ours (race), the
    target fixture is dropped via ``exclude_fixture_id``."""
    earlier = make_fixture(
        fixture_id=10, home_team_id=1, away_team_id=2,
        kickoff=FIXED_KICKOFF - timedelta(days=1),
        home_goals=1, away_goals=0,
    )
    target = make_fixture(
        fixture_id=42, home_team_id=1, away_team_id=2,
        kickoff=FIXED_KICKOFF - timedelta(days=1),
        home_goals=3, away_goals=3,
    )
    universe = [earlier, target]
    out = fixtures_before(
        universe, kickoff=FIXED_KICKOFF, exclude_fixture_id=42,
    )
    assert [fx.fixture_id for fx in out] == [10]


def test_fixtures_before_returns_newest_first() -> None:
    """Rolling helpers ``_take_n`` expect newest-first input."""
    fxs = [
        make_fixture(fixture_id=i, home_team_id=1, away_team_id=2,
                     kickoff=FIXED_KICKOFF - timedelta(days=i),
                     home_goals=0, away_goals=0)
        for i in (10, 5, 1)
    ]  # already asc by kickoff so universe order matches
    out = fixtures_before(fxs, kickoff=FIXED_KICKOFF)
    # The earliest kicked off is "i=10 days ago"; on the newest-first
    # ordering it should appear LAST; the most recent ("i=1") first.
    assert out[0].fixture_id == 1
    assert out[-1].fixture_id == 10


def test_fixtures_before_unordered_skips_target() -> None:
    """``fixtures_before_unordered`` is the per-team / per-pair variant
    used by the H2H loader; it must also observe the boundary + drop the
    target fixture.
    """
    target = make_fixture(
        fixture_id=99, home_team_id=1, away_team_id=2,
        kickoff=FIXED_KICKOFF - timedelta(days=10),
        home_goals=0, away_goals=0,
    )
    pair1 = make_fixture(
        fixture_id=12, home_team_id=1, away_team_id=2,
        kickoff=FIXED_KICKOFF - timedelta(days=20),
        home_goals=1, away_goals=0,
    )
    later = make_fixture(
        fixture_id=15, home_team_id=1, away_team_id=2,
        kickoff=FIXED_KICKOFF + timedelta(days=1),
        home_goals=0, away_goals=1,
    )
    out = fixtures_before_unordered(
        [target, pair1, later],
        kickoff=FIXED_KICKOFF,
        exclude_fixture_id=99,
    )
    assert [fx.fixture_id for fx in out] == [12]


def test_team_history_before_filters_team_and_time() -> None:
    """Only fixtures involving the specific team AND strictly before T."""
    fx_h_a = make_fixture(
        fixture_id=1, home_team_id=1, away_team_id=2,
        kickoff=FIXED_KICKOFF - timedelta(days=5),
        home_goals=2, away_goals=0,
    )
    fx_h_b = make_fixture(
        fixture_id=2, home_team_id=1, away_team_id=3,
        kickoff=FIXED_KICKOFF - timedelta(days=3),
        home_goals=1, away_goals=1,
    )
    fx_other = make_fixture(
        fixture_id=3, home_team_id=2, away_team_id=3,
        kickoff=FIXED_KICKOFF - timedelta(days=2),
        home_goals=3, away_goals=2,
    )
    fx_future_a = make_fixture(
        fixture_id=4, home_team_id=1, away_team_id=3,
        kickoff=FIXED_KICKOFF + timedelta(days=1),
        home_goals=0, away_goals=0,
    )
    # Target fixture: must be excluded by id.
    fx_target = make_fixture(
        fixture_id=99, home_team_id=1, away_team_id=2,
        kickoff=FIXED_KICKOFF - timedelta(days=1),
        home_goals=4, away_goals=4,
    )

    universe = [fx_h_a, fx_h_b, fx_other, fx_future_a, fx_target]

    out = team_history_before(
        team_id=1,
        season_fixtures=universe,
        kickoff=FIXED_KICKOFF,
        exclude_fixture_id=99,
    )
    # Expected: fx_h_b (kickoff 3 days before T, newest-first) and fx_h_a.
    # fx_other involved team 2&3, fx_future_a kicks off after T,
    # fx_target is excluded by id.
    assert [fx.fixture_id for fx in out] == [2, 1]


def test_fixtures_before_tz_naive_cutoff_is_normalised() -> None:
    """A naive cutoff is implicitly UTC-normalised. The strict-less
    comparison must still work; if it silently swapped UTC for the row's
    tz the leak would be subtle.
    """
    cutoff_naive = datetime(2026, 3, 20, 19, 0)  # naive!
    earlier = make_fixture(
        fixture_id=10, home_team_id=1, away_team_id=2,
        kickoff=datetime(2026, 3, 19, 19, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=0,
    )
    same_utc = make_fixture(
        fixture_id=11, home_team_id=1, away_team_id=2,
        kickoff=datetime(2026, 3, 20, 19, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=0,
    )
    out = fixtures_before([earlier, same_utc], kickoff=cutoff_naive)
    assert [fx.fixture_id for fx in out] == [10]
