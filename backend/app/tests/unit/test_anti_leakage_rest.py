"""Anti-leakage tests for :mod:`app.features.rest`.

Rest days = ``int((T - last_match.kickoff_time).total_seconds() // 86400)``
where ``last_match`` is the most recent FINISHED match of the team
strictly before T. A future match must NOT change the rest-day count
because it never supplants ``last_match``.

Test pattern: build base rest_days over a known prior run, then inject
a future finished match and assert the rest_days stay equal.
"""

from __future__ import annotations

from datetime import timedelta

from app.features import example as ex
from app.features.rest import compute_rest_features
from app.tests.unit._factories import FIXED_KICKOFF, FIXED_TEAMS, make_fixture

HOME = FIXED_TEAMS["HOME"]
OTHER = FIXED_TEAMS["OTHER"]


def test_future_match_does_not_change_rest_days() -> None:
    prior = [
        make_fixture(
            fixture_id=1, home_team_id=HOME, away_team_id=OTHER,
            kickoff=FIXED_KICKOFF - timedelta(days=20),
            home_goals=1, away_goals=0,
        ),
        # Most recent: 7 days before T -> rest_days = 7.
        make_fixture(
            fixture_id=2, home_team_id=HOME, away_team_id=OTHER,
            kickoff=FIXED_KICKOFF - timedelta(days=7),
            home_goals=2, away_goals=0,
        ),
    ]
    base_feats, _ = compute_rest_features(
        season_fixtures=prior, team_id=HOME,
        scope_prefix="home", kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert base_feats[ex.HOME_REST_DAYS] == 7

    # Inject a future played match 1 day after T — if it leaked in,
    # the "most recent" wiring WOULD change and rest_days would become 1
    # (or whatever the new wiring chose). The strict < boundary must
    # drop it.
    future_after_t = make_fixture(
        fixture_id=998, home_team_id=HOME, away_team_id=OTHER,
        kickoff=FIXED_KICKOFF + timedelta(days=1),
        home_goals=10, away_goals=0,
    )
    feats, _ = compute_rest_features(
        season_fixtures=prior + [future_after_t],
        team_id=HOME,
        scope_prefix="home", kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert feats == base_feats, (
        "future fixture leaked into rest_days computation"
    )


def test_rest_days_returns_missing_when_no_prior_match() -> None:
    """No prior fixture for this team before T -> rest_days is None
    (not 0). 0 is reserved for the valid edge case where two matches
    kick off the same day.
    """
    feats, missing = compute_rest_features(
        season_fixtures=[], team_id=HOME,
        scope_prefix="home", kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert feats[ex.HOME_REST_DAYS] is None
    assert ex.HOME_REST_DAYS in missing


def test_zero_rest_days_when_two_matches_same_day() -> None:
    """When the most recent match is on the SAME calendar day (T - 0
    days), rest_days = 0 — the math-zero, NOT a missing value.
    """
    prior = [
        make_fixture(
            fixture_id=1, home_team_id=HOME, away_team_id=OTHER,
            kickoff=FIXED_KICKOFF - timedelta(hours=3),  # same day
            home_goals=1, away_goals=0,
        ),
    ]
    feats, missing = compute_rest_features(
        season_fixtures=prior, team_id=HOME,
        scope_prefix="home", kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert feats[ex.HOME_REST_DAYS] == 0
    assert ex.HOME_REST_DAYS not in missing
