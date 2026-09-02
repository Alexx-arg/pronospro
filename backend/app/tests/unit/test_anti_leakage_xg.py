"""Anti-leakage tests for :mod:`app.features.xg`.

Phase 4 cannot emit per-match xG rollings (the ``fixtures`` table
does not persist xG columns). xG features come from the seasonal
``team_statistics`` snapshot via
:func:`app.features.xg.latest_team_stats_before`, which selects the
LATEST snapshot strictly before the target fixture's kickoff date.

Tests cover:

* Post-kickoff snapshots do NOT contribute to the as-of feature.
* The "latest strictly before" rule correctly excludes same-day and
  later snapshots.
* When no eligible snapshot exists, the feature is ``None`` and the
  missing_report records the reason.
* A snapshot xG/xGA NULL while the snapshot itself exists still
  surfaces a missing entry per-feature (per
  :mod:`app.features.xg`'s two-level "no snapshot" vs "snapshot with
  NULL value" report).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from app.features import example as ex
from app.features.xg import (
    compute_xg_features,
    latest_team_stats_before,
)
from app.tests.unit._factories import (
    FIXED_KICKOFF,
    FIXED_TEAMS,
    make_team_stats,
)

HOME = FIXED_TEAMS["HOME"]


def _stats(team_id: int, as_of: date, *, xg: float | None, xga: float | None):
    return make_team_stats(team_id=team_id, as_of_date=as_of, xg=xg, xga=xga)


# ---------------------------------------------------------------------------
# 1. Snapshots AFTER the kickoff date do NOT contribute
# ---------------------------------------------------------------------------
def test_post_kickoff_snapshot_does_not_contribute_to_xg() -> None:
    kd = FIXED_KICKOFF.date()  # the kickoff DATE — boundary is < kickoff.date()
    # A pre-kickoff snapshot (HEAD strict-before).
    snap_before = _stats(HOME, kd - timedelta(days=10), xg=12.5, xga=8.0)

    # A same-day snapshot (boundary == kickoff.date() -> excluded;
    # this is the "snapshot synced later on the matchday" risk
    # documented in :mod:`app.features.xg`).
    snap_same_day = _stats(HOME, kd, xg=99.0, xga=99.0)

    # A post-kickoff snapshot (would replace everything if it leaked
    # in — both numbers are huge).
    snap_after = _stats(HOME, kd + timedelta(days=1), xg=1000.0, xga=500.0)

    stats_rows = [snap_after, snap_same_day, snap_before]  # shuffled order

    feats, missing = compute_xg_features(
        stats_rows=stats_rows,
        home_team_id=HOME,
        away_team_id=FIXED_TEAMS["AWAY"],
        kickoff=FIXED_KICKOFF,
    )
    # The pre-kickoff snapshot wins.
    assert feats[ex.HOME_XG_SEASON_ASOF] == 12.5
    assert feats[ex.HOME_XGA_SEASON_ASOF] == 8.0
    assert ex.HOME_XG_SEASON_ASOF not in missing
    assert ex.HOME_XGA_SEASON_ASOF not in missing

    # The post-kickoff snapshot is NEVER the selected one: re-run with
    # the post-kickoff snapshot ALONE and confirm feature is missing.
    feats_post_only, missing_post = compute_xg_features(
        stats_rows=[snap_after],
        home_team_id=HOME,
        away_team_id=FIXED_TEAMS["AWAY"],
        kickoff=FIXED_KICKOFF,
    )
    assert feats_post_only[ex.HOME_XG_SEASON_ASOF] is None
    assert ex.HOME_XG_SEASON_ASOF in missing_post


# ---------------------------------------------------------------------------
# 2. ``latest_team_stats_before`` direct boundary behavior
# ---------------------------------------------------------------------------
def test_latest_team_stats_before_strict_boundary() -> None:
    kd = FIXED_KICKOFF.date()
    s_pre = _stats(HOME, kd - timedelta(days=15), xg=5.0, xga=2.0)
    s_same = _stats(HOME, kd, xg=99.0, xga=99.0)
    s_post = _stats(HOME, kd + timedelta(days=5), xg=1000.0, xga=500.0)

    snap = latest_team_stats_before(
        stats_rows=[s_pre, s_same, s_post],
        team_id=HOME, kickoff=FIXED_KICKOFF,
    )
    assert snap is s_pre


def test_latest_team_stats_before_returns_none_when_only_post() -> None:
    kd = FIXED_KICKOFF.date()
    snap = latest_team_stats_before(
        stats_rows=[_stats(HOME, kd, xg=1.0, xga=1.0),  # same day, excluded
                    _stats(HOME, kd + timedelta(days=1), xg=2.0, xga=2.0)],
        team_id=HOME, kickoff=FIXED_KICKOFF,
    )
    assert snap is None


# ---------------------------------------------------------------------------
# 3. No snapshot -> missing xG feature
# ---------------------------------------------------------------------------
def test_no_snapshot_returns_missing_xg() -> None:
    feats, missing = compute_xg_features(
        stats_rows=[],
        home_team_id=HOME,
        away_team_id=FIXED_TEAMS["AWAY"],
        kickoff=FIXED_KICKOFF,
    )
    assert feats[ex.HOME_XG_SEASON_ASOF] is None
    assert feats[ex.HOME_XGA_SEASON_ASOF] is None
    assert feats[ex.AWAY_XG_SEASON_ASOF] is None
    assert feats[ex.AWAY_XGA_SEASON_ASOF] is None
    for name in (ex.HOME_XG_SEASON_ASOF, ex.AWAY_XG_SEASON_ASOF):
        assert name in missing


# ---------------------------------------------------------------------------
# 4. Snapshot exists but xG/xGA are NULL — two-level missing report
# ---------------------------------------------------------------------------
def test_snapshot_with_null_xg_is_missing_not_zero() -> None:
    kd = FIXED_KICKOFF.date()
    # Snapshot exists strictly before kickoff date but xG/xGA columns
    # are NULL (the provider didn't return them).
    snap = _stats(HOME, kd - timedelta(days=2), xg=None, xga=None)

    feats, missing = compute_xg_features(
        stats_rows=[snap],
        home_team_id=HOME,
        away_team_id=FIXED_TEAMS["AWAY"],
        kickoff=FIXED_KICKOFF,
    )
    # The policy: snapshot found is NOT None — the missing report says
    # "snapshot exists but xG is NULL", not "no snapshot before T".
    assert feats[ex.HOME_XG_SEASON_ASOF] is None
    assert feats[ex.HOME_XGA_SEASON_ASOF] is None
    assert ex.HOME_XG_SEASON_ASOF in missing
    assert ex.HOME_XGA_SEASON_ASOF in missing
    # Reason must mention the snapshot's NULL state, not the as-of gap.
    assert "NULL" in missing[ex.HOME_XG_SEASON_ASOF]


# ---------------------------------------------------------------------------
# 5. Determinism — same snapshots, any order, same selection.
# ---------------------------------------------------------------------------
def test_xg_features_are_deterministic_under_permutation() -> None:
    kd = FIXED_KICKOFF.date()
    rows = [
        _stats(HOME, kd - timedelta(days=30), xg=1.0, xga=2.0),
        _stats(HOME, kd - timedelta(days=10), xg=3.0, xga=4.0),  # latest
        _stats(HOME, kd - timedelta(days=20), xg=2.0, xga=3.0),
    ]
    f1, _ = compute_xg_features(
        stats_rows=rows,
        home_team_id=HOME, away_team_id=FIXED_TEAMS["AWAY"],
        kickoff=FIXED_KICKOFF,
    )
    f2, _ = compute_xg_features(
        stats_rows=list(reversed(rows)),
        home_team_id=HOME, away_team_id=FIXED_TEAMS["AWAY"],
        kickoff=FIXED_KICKOFF,
    )
    assert f1 == f2
    # Latest strict-before wins (10 days before kickoff).
    assert f1[ex.HOME_XG_SEASON_ASOF] == 3.0


# ---------------------------------------------------------------------------
# 6. Sanity: tz-aware kickoff also uses .date() correctly
# ---------------------------------------------------------------------------
def test_xg_features_with_tz_aware_kickoff_use_date_part() -> None:
    """The xG snapshot is taken from the team_statistics.as_of_date
    (a ``date``). The helper uses ``kickoff.date()`` so a tz-aware
    kickoff whose UTC date differs from the local kickoff date is
    compared by its UTC date component (the provider's date is taken
    in the team's locale; this contract is intentional).
    """
    aware_kickoff = datetime(2026, 3, 20, 23, 30, tzinfo=timezone.utc)
    kd = aware_kickoff.date()  # 2026-03-20 UTC

    rows = [
        _stats(HOME, kd - timedelta(days=1), xg=10.0, xga=10.0),  # eligible
        _stats(HOME, kd + timedelta(days=1), xg=20.0, xga=20.0),  # excluded
    ]
    feats, _ = compute_xg_features(
        stats_rows=rows,
        home_team_id=HOME,
        away_team_id=FIXED_TEAMS["AWAY"],
        kickoff=aware_kickoff,
    )
    assert feats[ex.HOME_XG_SEASON_ASOF] == 10.0
