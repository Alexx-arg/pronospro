"""Anti-leakage tests for :mod:`app.features.elo` (temporal ELO reconstruction).

ELO pre-match is rebuilt from the season's finished fixtures in
chronological order; for each fixture we capture the pre-match
ratings BEFORE folding its own result. Two leakage channels tested:

1. **Future results injection**: adding a finished fixture whose
   ``kickoff_time`` is strictly after the target's kickoff must not
   change the target's ``home_elo_pre_match`` / ``away_elo_pre_match`` /
   ``elo_difference``.
2. **Target's own result isolation**: the target fixture itself
   (by id) cannot contribute to its own pre-match ELO even if it
   appears earlier in the season universe than another match that
   shares the same kickoff instant.

Pattern: build features over a "base universe", then add a future
fixture with an extreme result and assert feature equality.
"""

from __future__ import annotations

from datetime import timedelta

from app.features import example as ex
from app.features.elo import compute_elo_features
from app.tests.unit._factories import FIXED_KICKOFF, FIXED_TEAMS, make_fixture

HOME = FIXED_TEAMS["HOME"]
AWAY = FIXED_TEAMS["AWAY"]
OTHER = FIXED_TEAMS["OTHER"]


def _priors() -> list:
    """A small base universe strictly before FIXED_KICKOFF.

    fx_1: HOME 1-0 OTHER   (HOME plays first)
    fx_2: AWAY 3-0 OTHER   (AWAY catches up)
    Both fixtures finish strictly before our target's kickoff.
    """
    k1 = FIXED_KICKOFF - timedelta(days=20)
    k2 = FIXED_KICKOFF - timedelta(days=10)
    return [
        make_fixture(fixture_id=1, home_team_id=HOME, away_team_id=OTHER,
                     kickoff=k1, home_goals=1, away_goals=0),
        make_fixture(fixture_id=2, home_team_id=AWAY, away_team_id=OTHER,
                     kickoff=k2, home_goals=3, away_goals=0),
    ]


def _target() -> "make_fixture":
    return make_fixture(
        fixture_id=99, home_team_id=HOME, away_team_id=AWAY,
        kickoff=FIXED_KICKOFF, home_goals=0, away_goals=0,
    )


# ---------------------------------------------------------------------------
# 1. Future result does not change Elo pre-match
# ---------------------------------------------------------------------------
def test_future_result_does_not_change_elo_pre_match() -> None:
    target = _target()
    priors = _priors()

    base_feats, _ = compute_elo_features(
        fixture=target, season_fixtures=priors,
    )
    assert base_feats[ex.HOME_ELO_PRE_MATCH] is not None
    assert base_feats[ex.AWAY_ELO_PRE_MATCH] is not None
    base_home = base_feats[ex.HOME_ELO_PRE_MATCH]
    base_away = base_feats[ex.AWAY_ELO_PRE_MATCH]
    base_diff = base_feats[ex.ELO_DIFFERENCE]

    # Inject a future finished match: HOME destroys AWAY 10-0 tomorrow.
    future = make_fixture(
        fixture_id=555, home_team_id=HOME, away_team_id=AWAY,
        kickoff=FIXED_KICKOFF + timedelta(days=1),
        home_goals=10, away_goals=0,
    )
    feats_after, _ = compute_elo_features(
        fixture=target, season_fixtures=priors + [future],
    )

    assert feats_after[ex.HOME_ELO_PRE_MATCH] == base_home, (
        "future result leaked into home ELO pre-match"
    )
    assert feats_after[ex.AWAY_ELO_PRE_MATCH] == base_away, (
        "future result leaked into away ELO pre-match"
    )
    assert feats_after[ex.ELO_DIFFERENCE] == base_diff


# ---------------------------------------------------------------------------
# 2. The target's own result is NOT folded into its pre-match ELO
# ---------------------------------------------------------------------------
def test_target_fixture_excluded_from_own_elo() -> None:
    """Even if the target fixture is in ``season_fixtures``, it must
    yield its PRE-match ratings — never the post-match ratings.

    The target is rebuilt from a synthesised target whose goal line is
    fed only post-match into the ELO stream; :func:`compute_elo_features`
    re-injects the target at the end of the stream so its snapshot is
    captured BEFORE its own result is folded. If the exclusion code
    breaks, the target's own result would be folded before yielding its
    snapshot and HOME would gain rating points (home_goals=10).

    We construct the test so that the target has a HUGE goal count:
    a passing implementation yields snapshot-pre-match ratings that
    DO NOT reflect the 10-0 win.
    """
    target = make_fixture(
        fixture_id=99, home_team_id=HOME, away_team_id=AWAY,
        kickoff=FIXED_KICKOFF, home_goals=10, away_goals=0,
    )

    # WITHOUT the target in the universe: baseline of priors only.
    priors = _priors()
    base_feats, _ = compute_elo_features(
        fixture=target, season_fixtures=priors,
    )
    base_home = base_feats[ex.HOME_ELO_PRE_MATCH]

    # WITH the target in the universe (defensive duplicate). It MUST
    # remain excluded because ``fixture.kickoff_time < fixture.kickoff_time``
    # is impossible — the qualifier filter only sees the priors.
    feats_with_target, _ = compute_elo_features(
        fixture=target, season_fixtures=priors + [target],
    )
    assert feats_with_target[ex.HOME_ELO_PRE_MATCH] == base_home, (
        "the target's own 10-0 result polluted its pre-match ELO"
    )

    # Sanity evidence: the "post-match" rating for HOME would be higher
    # than the pre-match one because the home team just won 10-0. If the
    # implementation accidentally folded the target's result first, the
    # value would not equal ``base_home``.
    assert feats_with_target[ex.HOME_ELO_PRE_MATCH] is not None
    assert feats_with_target[ex.AWAY_ELO_PRE_MATCH] is not None
