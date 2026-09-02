"""Anti-leakage tests for :mod:`app.features.standings`.

Standings are reconstructed **purely from finished fixtures of the
season**, strictly before the target kickoff. No ``standings`` table
participates — see :mod:`app.features.standings` for the design rationale.

Tests here assert the two anti-leakage guarantees explicitly via
``compute_standings_features``:

1. **Future fixture injection**: adding a finished fixture whose
   ``kickoff_time`` is strictly after the target's kickoff must not
   change any standings feature of the target match.
2. **Target's own exclusion**: the target fixture itself (by id) is
   excluded from the table even if its kickoff happened before T.

The before/after pattern is explicit so a regression that lets a
future match leak in shows up as a feature diff.
"""

from __future__ import annotations

from datetime import timedelta

from app.features import example as ex
from app.features.standings import compute_standings_features
from app.tests.unit._factories import FIXED_KICKOFF, FIXED_TEAMS, make_fixture

HOME = FIXED_TEAMS["HOME"]   # 1
AWAY = FIXED_TEAMS["AWAY"]   # 2
OTHER = FIXED_TEAMS["OTHER"]  # 3


def _season_priors() -> list:
    """Three finished fixtures BEFORE FIXED_KICKOFF between the
    canonical teams so the table has rows to rank.

    Fixtures (chronological, oldest first):
        fx_10: HOME 2 - 1 AWAY     (HOME win)
        fx_11: HOME 3 - 0 OTHER   (HOME win, OTHER joins table)
        fx_12: AWAY 1 - 0 OTHER    (AWAY win)
    Final table standings strictly before FIXED_KICKOFF:
        HOME: 2P  2W  0D  0L  +5/-1   6 pts   gd=+4
        AWAY: 2P  1W  0D  1L  +1/-3   3 pts   gd=-2
        OTHER:2P  0W  0D  2L  +0/-4   0 pts   gd=-4
    """
    k1 = FIXED_KICKOFF - timedelta(days=30)
    k2 = FIXED_KICKOFF - timedelta(days=23)
    k3 = FIXED_KICKOFF - timedelta(days=16)
    return [
        make_fixture(fixture_id=10, home_team_id=HOME, away_team_id=AWAY,
                     kickoff=k1, home_goals=2, away_goals=1),
        make_fixture(fixture_id=11, home_team_id=HOME, away_team_id=OTHER,
                     kickoff=k2, home_goals=3, away_goals=0),
        make_fixture(fixture_id=12, home_team_id=AWAY, away_team_id=OTHER,
                     kickoff=k3, home_goals=1, away_goals=0),
    ]


# ---------------------------------------------------------------------------
# 1. Future fixture does not affect table
# ---------------------------------------------------------------------------
def test_future_fixture_does_not_change_standings_as_of() -> None:
    priors = _season_priors()
    base_feats, base_missing = compute_standings_features(
        season_fixtures=priors,
        home_team_id=HOME, away_team_id=AWAY,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    # Sanity: both teams exist in the table -> features populated.
    assert base_feats[ex.HOME_TABLE_POSITION] == 1
    assert base_feats[ex.AWAY_TABLE_POSITION] == 2
    assert base_feats[ex.HOME_POINTS_TABLE] == 6
    assert base_feats[ex.AWAY_POINTS_TABLE] == 3
    assert base_feats[ex.TABLE_POINTS_DIFFERENCE] == 3
    assert base_feats[ex.TABLE_GOAL_DIFFERENCE_DIFFERENCE] == 6  # +4 - (-2)

    # Inject a FUTURE finished fixture: AWAY hammers HOME 7-0 (which
    # WOULD change the table if leaked in: AWAY gets +3 and HOME gets
    # 0 for this match, vaulting the standings).
    future = make_fixture(
        fixture_id=777, home_team_id=AWAY, away_team_id=HOME,
        kickoff=FIXED_KICKOFF + timedelta(days=10),
        home_goals=7, away_goals=0,
    )
    feats_after, missing_after = compute_standings_features(
        season_fixtures=priors + [future],
        home_team_id=HOME, away_team_id=AWAY,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert feats_after == base_feats, (
        "future fixture leaked into standings"
    )
    assert missing_after == base_missing


# ---------------------------------------------------------------------------
# 2. Target fixture excluded from its own pre-match table
# ---------------------------------------------------------------------------
def test_target_fixture_excluded_from_pre_match_standings() -> None:
    """Even when the target fixture is present in the season universe
    (kicking off strictly before FIXED_KICKOFF by a day, as if it had
    finished earlier the same matchday), it must NOT contribute to its
    own pre-match standings.
    """
    priors = _season_priors()
    # Synthesize the target itself, sitting right before FIXED_KICKOFF.
    target = make_fixture(
        fixture_id=42, home_team_id=HOME, away_team_id=AWAY,
        kickoff=FIXED_KICKOFF - timedelta(hours=1),
        home_goals=5, away_goals=0,  # huge result — should NOT pollute
    )
    universe = priors + [target]

    feats_excluded, _ = compute_standings_features(
        season_fixtures=universe,
        home_team_id=HOME, away_team_id=AWAY,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=42,
    )
    # The 5-0 must NOT be in: table should match the priors-only one.
    assert feats_excluded[ex.HOME_POINTS_TABLE] == 6
    assert feats_excluded[ex.AWAY_POINTS_TABLE] == 3

    # Sanity — if we DIDN'T exclude the target it WOULD leak in.
    feats_inclusive, _ = compute_standings_features(
        season_fixtures=universe,
        home_team_id=HOME, away_team_id=AWAY,
        kickoff=FIXED_KICKOFF, exclude_fixture_id=None,
    )
    assert feats_inclusive[ex.HOME_POINTS_TABLE] == 9, (
        "without exclusion the 5-0 result would add 3 points to HOME"
    )
    assert feats_inclusive[ex.HOME_POINTS_TABLE] != feats_excluded[ex.HOME_POINTS_TABLE]
