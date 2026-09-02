"""Anti-leakage + cross-season tests for :mod:`app.features.h2h`.

The H2H feature aggregates matches played between the two teams of
the target fixture. Phase 4 contract:

* Strict ``kickoff < T`` boundary — post-target fixtures never count.
* Target fixture itself is excluded by id even if it sits in the H2H
  universe at the same kickoff instant.
* **Cross-season**: H2H spans every finished match between the two
  teams, not only the current season's fixtures. The Phase 4 dataset
  builder forwards the cross-season ``h2h_fixtures`` universe into
  :func:`app.features.assembler.build_example`; when called directly
  the helper accepts the same argument.

These tests exercise ``compute_h2h_features`` directly (no DB), with
synthesised cross-season :class:`FixtureRow` lists crafted via
:func:`app.tests.unit._factories.make_fixture`. The seven required
scenarios are tagged so a regression is easy to localise.
"""

from __future__ import annotations

from datetime import timedelta

from app.features import example as ex
from app.features.h2h import MIN_SAMPLE, compute_h2h_features
from app.tests.unit._factories import FIXED_KICKOFF, FIXED_TEAMS, make_fixture

HOME = FIXED_TEAMS["HOME"]    # 1
AWAY = FIXED_TEAMS["AWAY"]    # 2


def _h2h_pair(kickoff_delta: timedelta, *, fx_id: int, hg: int, ag: int):
    """One H2H match: HOME vs AWAY at offset from FIXED_KICKOFF."""
    return make_fixture(
        fixture_id=fx_id, home_team_id=HOME, away_team_id=AWAY,
        kickoff=FIXED_KICKOFF + kickoff_delta,
        home_goals=hg, away_goals=ag,
    )


def _h2h_pair_away(kickoff_delta: timedelta, *, fx_id: int, hg: int, ag: int):
    """One H2H match with AWAY as the home team (venue alternation
    still counts as a pair fixture).
    """
    return make_fixture(
        fixture_id=fx_id, home_team_id=AWAY, away_team_id=HOME,
        kickoff=FIXED_KICKOFF + kickoff_delta,
        home_goals=hg, away_goals=ag,
    )


# ---------------------------------------------------------------------------
# 1. H2H from a previous season IS used (cross-season)
# ---------------------------------------------------------------------------
def test_h2h_from_previous_season_is_used() -> None:
    """An H2H fixture finishing over a year before the target kickoff
    must count toward the H2H aggregate — Phase 4's cross-season
    contract.
    """
    universe = [
        _h2h_pair(timedelta(days=-600), fx_id=1001, hg=2, ag=0),  # HOME W
        _h2h_pair(timedelta(days=-400), fx_id=1002, hg=1, ag=1),  # D
        _h2h_pair(timedelta(days=-200), fx_id=1003, hg=0, ag=2),  # AWAY W
    ]
    feats, missing = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=universe,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert feats[ex.H2H_SAMPLE_SIZE] == 3
    assert feats[ex.H2H_HOME_WINS] == 1
    assert feats[ex.H2H_AWAY_WINS] == 1
    assert feats[ex.H2H_DRAWS] == 1
    # Total goals across the three matches, then mean over the sample count.
    total_goals = sum(fx.home_goals + fx.away_goals for fx in universe)
    assert feats[ex.H2H_TOTAL_GOALS_MEAN] == total_goals / 3
    assert ex.H2H_HOME_WINS not in missing


# ---------------------------------------------------------------------------
# 2. H2H after kickoff is NOT used
# ---------------------------------------------------------------------------
def test_h2h_after_kickoff_not_used() -> None:
    """An H2H match finishing AFTER the target kickoff must not
    influence the H2H features even when it is in the h2h_fixtures
    universe passed to the assembler.
    """
    universe = [
        _h2h_pair(timedelta(days=-100), fx_id=1, hg=2, ag=0),   # HOME W (prior)
        _h2h_pair(timedelta(days=-50), fx_id=2, hg=0, ag=1),   # AWAY W (prior)
        _h2h_pair(timedelta(days=-20), fx_id=3, hg=1, ag=1),   # D (prior)
        # Post-kickoff H2H (must be excluded by the strict < boundary):
        _h2h_pair(timedelta(days=+10), fx_id=4, hg=10, ag=0),  # HOME W (future)
        _h2h_pair(timedelta(days=+30), fx_id=5, hg=0, ag=10),  # AWAY W (future)
    ]
    feats, missing = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=universe,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    # Only the three prior matchups contribute.
    assert feats[ex.H2H_SAMPLE_SIZE] == 3
    assert feats[ex.H2H_HOME_WINS] == 1
    assert feats[ex.H2H_AWAY_WINS] == 1
    assert feats[ex.H2H_DRAWS] == 1


# ---------------------------------------------------------------------------
# 3. Target fixture NOT counted in H2H
# ---------------------------------------------------------------------------
def test_target_fixture_excluded_from_h2h() -> None:
    """The target fixture itself (excluded by ``exclude_fixture_id``)
    NEVER counts toward its own H2H, even if it sits in the universe.

    Sanity check: without excluding, the 7-0 would have changed the
    H2H aggregate.
    """
    universe = [
        _h2h_pair(timedelta(days=-100), fx_id=1, hg=1, ag=1),
        _h2h_pair(timedelta(days=-50), fx_id=2, hg=0, ag=0),
        _h2h_pair(timedelta(days=-25), fx_id=3, hg=2, ag=2),
        # The target itself, kicking off a day before FIXED_KICKOFF
        # with a HUGE 7-0 scoreline:
        _h2h_pair(timedelta(days=-1), fx_id=42, hg=7, ag=0),
    ]
    feats_excluded, _ = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=universe,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=42,
    )
    # Only the three priors count: total goals = 2 + 0 + 4 = 6 over 3
    assert feats_excluded[ex.H2H_SAMPLE_SIZE] == 3
    assert feats_excluded[ex.H2H_TOTAL_GOALS_MEAN] == 6 / 3

    # Sanity: WITHOUT exclusion the 7-0 WOULD have leaked in.
    feats_inclusive, _ = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=universe,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert feats_inclusive[ex.H2H_SAMPLE_SIZE] == 4
    assert feats_inclusive[ex.H2H_HOME_WINS] == 1
    assert feats_inclusive[ex.H2H_TOTAL_GOALS_MEAN] == (6 + 7) / 4


# ---------------------------------------------------------------------------
# 4. Adding a future fixture does NOT change H2H
# ---------------------------------------------------------------------------
def test_adding_future_fixture_does_not_change_h2h() -> None:
    base = [
        _h2h_pair(timedelta(days=-100), fx_id=1, hg=1, ag=0),   # HOME W
        _h2h_pair(timedelta(days=-70), fx_id=2, hg=1, ag=2),   # AWAY W
        _h2h_pair(timedelta(days=-30), fx_id=3, hg=2, ag=2),   # D
    ]
    base_feats, _ = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=base,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    # Inject a future H2H with a massive, polluting scoreline.
    future = _h2h_pair(timedelta(days=+15), fx_id=999, hg=20, ag=0)
    feats_after, _ = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=base + [future],
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert feats_after == base_feats, (
        "future H2H fixture leaked into pre-match H2H features"
    )


# ---------------------------------------------------------------------------
# 5. No prior H2H -> missing per existing policy
# ---------------------------------------------------------------------------
def test_no_prior_h2h_returns_missing_per_policy() -> None:
    """When the pair has zero prior meetings, ``h2h_sample_size = 0``
    (the real count — never imputed) and the metric features are
    ``None`` (never ``0``) because the sample is below ``MIN_SAMPLE``.
    """
    feats, missing = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=[],  # no prior H2H at all
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert feats[ex.H2H_SAMPLE_SIZE] == 0
    for name in (ex.H2H_HOME_WINS, ex.H2H_AWAY_WINS, ex.H2H_DRAWS,
                 ex.H2H_TOTAL_GOALS_MEAN):
        assert feats[name] is None
        assert name in missing
    assert str(MIN_SAMPLE) in missing[ex.H2H_HOME_WINS]


def test_insufficient_h2h_returns_missing_with_real_count() -> None:
    """When the pair has between 1 and ``MIN_SAMPLE-1`` meetings, the
    ``h2h_sample_size`` exposes the real partial count but the metrics
    stay ``None`` (sample policy).
    """
    universe = [
        _h2h_pair(timedelta(days=-100), fx_id=1, hg=3, ag=0),
    ]
    feats, missing = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=universe,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert feats[ex.H2H_SAMPLE_SIZE] == 1
    assert feats[ex.H2H_HOME_WINS] is None
    assert feats[ex.H2H_TOTAL_GOALS_MEAN] is None


# ---------------------------------------------------------------------------
# 6. Determinism — same inputs -> same outputs
# ---------------------------------------------------------------------------
def test_h2h_features_are_deterministic() -> None:
    """Two calls over the SAME universe (same list, same order) produce
    exactly identical feature dicts.
    """
    universe = [
        _h2h_pair(timedelta(days=-100), fx_id=1, hg=2, ag=1),
        _h2h_pair(timedelta(days=-60), fx_id=2, hg=1, ag=2),
        _h2h_pair(timedelta(days=-30), fx_id=3, hg=0, ag=0),
    ]
    f1, m1 = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=universe,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    f2, m2 = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=list(universe),  # fresh list, same order
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert f1 == f2
    assert m1 == m2


def test_h2h_features_order_independent_under_permutation() -> None:
    """The H2H order of intake does not change the aggregate values
    (the helper internally re-sorts by kickoff desc).
    """
    a = [_h2h_pair(timedelta(days=-100), fx_id=1, hg=2, ag=0),
         _h2h_pair(timedelta(days=-50), fx_id=2, hg=1, ag=1),
         _h2h_pair(timedelta(days=-10), fx_id=3, hg=0, ag=3)]
    b = list(reversed(a))  # reverse order
    f1, _ = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=a,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    f2, _ = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=b,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert f1 == f2


# ---------------------------------------------------------------------------
# 7. Builder NO LONGER passes ``h2h_fixtures=None``
# ---------------------------------------------------------------------------
def test_assembler_uses_h2h_fixtures_argument_when_provided() -> None:
    """When the assembler receives a populated ``h2h_fixtures`` list
    it MUST use it INSIDEAD of the same-season ``season_fixtures``.

    This is the unit-level guarantee that the builder, having loaded
    the cross-season H2H universe, sees its input win over the
    season-scoped fallback. We construct the test so the two
    universes are DIFFERENT and the H2H result matches the
    cross-season one; if the assembler fell back to season_fixtures
    the test would fail.
    """
    from app.features.assembler import build_example

    # season_fixtures only contains a single non-H2H match (away side
    # is the OTHER team, not AWAY); so if H2H fell back to
    # season_fixtures the sample would be 0.
    season_other = [
        make_fixture(
            fixture_id=2, home_team_id=HOME, away_team_id=FIXED_TEAMS["OTHER"],
            kickoff=FIXED_KICKOFF - timedelta(days=20),
            home_goals=1, away_goals=0,
        )
    ]
    # Cross-season H2H (3 prior meetings, all before FIXED_KICKOFF):
    h2h_rows = [
        _h2h_pair(timedelta(days=-400), fx_id=10, hg=1, ag=0),  # HOME W
        _h2h_pair(timedelta(days=-300), fx_id=11, hg=2, ag=2),  # D
        _h2h_pair(timedelta(days=-100), fx_id=12, hg=0, ag=1),  # AWAY W
    ]
    target = make_fixture(
        fixture_id=99, home_team_id=HOME, away_team_id=AWAY,
        kickoff=FIXED_KICKOFF, home_goals=0, away_goals=0,
    )
    ex_out = build_example(
        fixture=target,
        season_fixtures=season_other,
        stats_rows=[],
        h2h_fixtures=h2h_rows,
    )
    feats = ex_out.features
    assert feats[ex.H2H_SAMPLE_SIZE] == 3
    assert feats[ex.H2H_HOME_WINS] == 1
    assert feats[ex.H2H_AWAY_WINS] == 1
    assert feats[ex.H2H_DRAWS] == 1
    # The 5-0 finished in season_fixtures's "other" match never
    # enters H2H (wrong opponent). If the assembler used
    # season_fixtures as the H2H source, sample size would be 0.


def test_assembler_falls_back_to_season_fixtures_when_h2h_none() -> None:
    """Contract documentation: when ``h2h_fixtures=None`` the assembler
    reuses the same-season ``season_fixtures`` for H2H. This is kept
    for callers without the loader (e.g. tests, prediction-time).
    """
    from app.features.assembler import build_example

    season_h2h = [
        _h2h_pair(timedelta(days=-50), fx_id=10, hg=1, ag=0),
        _h2h_pair(timedelta(days=-40), fx_id=11, hg=1, ag=1),
        _h2h_pair(timedelta(days=-30), fx_id=12, hg=0, ag=2),
    ]
    target = make_fixture(
        fixture_id=99, home_team_id=HOME, away_team_id=AWAY,
        kickoff=FIXED_KICKOFF, home_goals=0, away_goals=0,
    )
    ex_out = build_example(
        fixture=target,
        season_fixtures=season_h2h,
        stats_rows=[],
        h2h_fixtures=None,  # explicit None triggers the season fallback
    )
    feats = ex_out.features
    assert feats[ex.H2H_SAMPLE_SIZE] == 3
    assert feats[ex.H2H_HOME_WINS] == 1
    assert feats[ex.H2H_AWAY_WINS] == 1
    assert feats[ex.H2H_DRAWS] == 1


# ---------------------------------------------------------------------------
# Bonus: venue alternation — AWAY as home team still counts as pair H2H
# ---------------------------------------------------------------------------
def test_h2h_counts_pair_meetings_regardless_of_venue() -> None:
    """An H2H match where AWAY acts as the home team still tracks as a
    pair fixture — cross-season H2H aggregation is venue-agnostic.
    This mirrors what :func:`app.features.asof.load_h2h_fixtures_as_of`
    returns in production (both clauses of the OR).
    """
    universe = [
        _h2h_pair(timedelta(days=-100), fx_id=1, hg=2, ag=0),    # HOME/host W
        _h2h_pair_away(timedelta(days=-50), fx_id=2, hg=1, ag=1),  # AWAY/host D
        _h2h_pair(timedelta(days=-10), fx_id=3, hg=0, ag=1),    # HOME/host AWAY W
    ]
    feats, _ = compute_h2h_features(
        home_team_id=HOME, away_team_id=AWAY,
        season_fixtures=universe,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    # fixture 1 (HOME host): HOME 2 - 0 AWAY   -> HOME win
    # fixture 2 (AWAY host): the AWAY acts as home team; home_goals=1 (by AWAY),
    #                       away_goals=1 (by HOME)  -> draw
    # fixture 3 (HOME host): HOME 0 - 1 AWAY   -> AWAY win
    assert feats[ex.H2H_SAMPLE_SIZE] == 3
    assert feats[ex.H2H_HOME_WINS] == 1   # fixture 1
    assert feats[ex.H2H_DRAWS] == 1       # fixture 2
    assert feats[ex.H2H_AWAY_WINS] == 1   # fixture 3
