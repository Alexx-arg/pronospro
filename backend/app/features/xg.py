"""xG / xGA features taken from the season-snapshot ``team_statistics``.

**Important**: see :mod:`app.features.example` — Phase 4 cannot expose
``xg_for_last_N`` as a per-match rolling because the ``fixtures`` table
does NOT persist per-match xG. The only reliable source is the
``team_statistics`` seasonal snapshot whose ``xg`` / ``xga`` are
cumulative tallies up to ``as_of_date``.

We therefore emit ``home_xg_season_asof`` / ``home_xga_season_asof`` +
away counterparts: the xG/xGA totals from the LATEST snapshot whose
``as_of_date`` is **strictly before** the target kickoff's date.

Boundary rule
--------------
* The latest snapshot with ``as_of_date < kickoff.date()`` qualifies.
* If the only available snapshot is on or after the kickoff date, the
  feature is ``None`` and a missing_report entry is recorded.
* When no snapshot exists for the (team, competition, season) trio at
  all, also ``None``.

The "latest snapshot strictly before" rule protects against a snapshot
synced later by ``sync_team_statistics`` — which stamps ``as_of_date =
today`` — leaking information created AFTER the target match into the
features (e.g. a 2024-09 fixture's features must not see the snapshot
dated 2024-12-31).
"""

from __future__ import annotations

from datetime import date, datetime

from app.features import example as ex
from app.features.rows import TeamStatsRow


def _kickoff_date(value: datetime) -> date:
    return value.date()


def latest_team_stats_before(
    *,
    stats_rows: list[TeamStatsRow],
    team_id: int,
    kickoff: datetime,
) -> TeamStatsRow | None:
    """Return the most recent snapshot strictly before ``kickoff`` date."""
    kd = _kickoff_date(kickoff)
    eligible = [
        row
        for row in stats_rows
        if row.team_id == team_id and row.as_of_date < kd
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda r: r.as_of_date, reverse=True)
    return eligible[0]


def compute_xg_features(
    *,
    stats_rows: list[TeamStatsRow],
    home_team_id: int,
    away_team_id: int,
    kickoff: datetime,
) -> tuple[dict[str, float | int | None], dict[str, str]]:
    """Return ``(home_xg/away_xg season-as-of features, missing_report)``."""
    features: dict[str, float | int | None] = {}
    missing: dict[str, str] = {}

    home_snap = latest_team_stats_before(
        stats_rows=stats_rows, team_id=home_team_id, kickoff=kickoff,
    )
    if home_snap is None:
        features[ex.HOME_XG_SEASON_ASOF] = None
        features[ex.HOME_XGA_SEASON_ASOF] = None
        missing[ex.HOME_XG_SEASON_ASOF] = (
            "no team_statistics snapshot strictly before kickoff date"
        )
    else:
        features[ex.HOME_XG_SEASON_ASOF] = home_snap.xg
        features[ex.HOME_XGA_SEASON_ASOF] = home_snap.xga
        if home_snap.xg is None:
            missing[ex.HOME_XG_SEASON_ASOF] = (
                "snapshot exists but xG is NULL (provider did not return)"
            )
        if home_snap.xga is None:
            missing[ex.HOME_XGA_SEASON_ASOF] = (
                "snapshot exists but xGA is NULL (provider did not return)"
            )

    away_snap = latest_team_stats_before(
        stats_rows=stats_rows, team_id=away_team_id, kickoff=kickoff,
    )
    if away_snap is None:
        features[ex.AWAY_XG_SEASON_ASOF] = None
        features[ex.AWAY_XGA_SEASON_ASOF] = None
        missing[ex.AWAY_XG_SEASON_ASOF] = (
            "no team_statistics snapshot strictly before kickoff date"
        )
    else:
        features[ex.AWAY_XG_SEASON_ASOF] = away_snap.xg
        features[ex.AWAY_XGA_SEASON_ASOF] = away_snap.xga
        if away_snap.xg is None:
            missing[ex.AWAY_XG_SEASON_ASOF] = (
                "snapshot exists but xG is NULL (provider did not return)"
            )
        if away_snap.xga is None:
            missing[ex.AWAY_XGA_SEASON_ASOF] = (
                "snapshot exists but xGA is NULL (provider did not return)"
            )

    return features, missing


__all__ = [
    "compute_xg_features",
    "latest_team_stats_before",
]
