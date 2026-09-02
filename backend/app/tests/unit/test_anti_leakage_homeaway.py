"""Anti-leakage tests for :mod:`app.features.homeaway`.

The home/away split features are conditioned on each team's venue
performance — but anti-leakage is identical to the form/goals family:
any fixture strictly AFTER the target kickoff must NOT count.

Test pattern: build features over a controlled "prior" history, then
inject a future finished match and assert the features are unchanged.
"""

from __future__ import annotations

from datetime import timedelta

from app.features import example as ex
from app.features.homeaway import compute_homeaway_features
from app.tests.unit._factories import FIXED_KICKOFF, FIXED_TEAMS, make_fixture

HOME = FIXED_TEAMS["HOME"]
AWAY = FIXED_TEAMS["AWAY"]
OTHER = FIXED_TEAMS["OTHER"]


def _home_history_for_home_team(goals: list[int]) -> list:
    """Build N prior HOME matches of the home team, all at HOME."""
    out = []
    for i, g in enumerate(goals, start=1):
        out.append(make_fixture(
            fixture_id=100 + i,
            home_team_id=HOME, away_team_id=OTHER,
            kickoff=FIXED_KICKOFF - timedelta(days=(len(goals) - i + 1) * 7),
            home_goals=g, away_goals=0,
        ))
    return out


def test_future_match_does_not_change_home_split_features() -> None:
    history = _home_history_for_home_team([1, 2, 3, 0, 2])  # 5 home matches
    base_feats, base_missing = compute_homeaway_features(
        scope_prefix="home", team_id=HOME,
        season_fixtures=history, kickoff=FIXED_KICKOFF,
        exclude_fixture_id=None, is_home_role=True,
    )
    # Sanity: full window -> features populated
    assert base_feats[ex.HOME_HOME_POINTS_LAST_5] is not None

    future = make_fixture(
        fixture_id=999, home_team_id=HOME, away_team_id=OTHER,
        kickoff=FIXED_KICKOFF + timedelta(days=10),
        home_goals=10, away_goals=0,  # huge, would change sums
    )
    feats, missing = compute_homeaway_features(
        scope_prefix="home", team_id=HOME,
        season_fixtures=history + [future],
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None, is_home_role=True,
    )
    assert feats == base_feats, "future fixture leaked into home/away split"
    assert missing == base_missing


def test_future_match_does_not_change_away_split_features() -> None:
    # Prior: 5 AWAY matches of the AWAY team vs OTHER, alternating guest/host.
    history = [
        make_fixture(
            fixture_id=200 + i,
            home_team_id=OTHER, away_team_id=AWAY,
            kickoff=FIXED_KICKOFF - timedelta(days=(6 - i) * 7),
            home_goals=0, away_goals=g,
        )
        for i, g in enumerate([0, 1, 2, 0, 1], start=1)
    ]
    base_feats, _ = compute_homeaway_features(
        scope_prefix="away", team_id=AWAY,
        season_fixtures=history, kickoff=FIXED_KICKOFF,
        exclude_fixture_id=None, is_home_role=False,
    )
    assert base_feats[ex.AWAY_AWAY_POINTS_LAST_5] is not None

    future = make_fixture(
        fixture_id=888, home_team_id=OTHER, away_team_id=AWAY,
        kickoff=FIXED_KICKOFF + timedelta(days=20),
        home_goals=0, away_goals=10,
    )
    feats, _ = compute_homeaway_features(
        scope_prefix="away", team_id=AWAY,
        season_fixtures=history + [future],
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None, is_home_role=False,
    )
    assert feats == base_feats
