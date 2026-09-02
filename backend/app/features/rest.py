"""Rest days: time since the team's most recent finished match before T.

The metric is whole days (integer, floored). When a team has played no
fixture before T (season opener at the very start), the feature is
``None`` — not 0: "zero days" would mean two matches on the same day,
a valid value when it really happens, so we keep 0 reserved for that
edge case (see docs/FEATURES.md).
"""

from __future__ import annotations

from datetime import datetime

from app.features import example as ex
from app.features.asof import team_history_before
from app.features.rows import FixtureRow


def compute_rest_features(
    *,
    season_fixtures: list[FixtureRow],
    team_id: int,
    scope_prefix: str,  # "home" | "away"
    kickoff: datetime,
    exclude_fixture_id: int | None,
) -> tuple[dict[str, float | int | None], dict[str, str]]:
    """Return ``{scope}_rest_days`` for one team.

    Uses the team's most recent finished match strictly before T — even
    if it happened across competitions/seasons, but the dataset builder
    currently restricts ``season_fixtures`` to the league/season the
    target fixture belongs to. Cross-season rest is therefore only
    available when the caller passes a wider history; see the builder
    docs.
    """
    history = team_history_before(
        team_id=team_id,
        season_fixtures=season_fixtures,
        kickoff=kickoff,
        exclude_fixture_id=exclude_fixture_id,
    )
    feat_name = ex.HOME_REST_DAYS if scope_prefix == "home" else ex.AWAY_REST_DAYS

    if not history:
        return {feat_name: None}, {feat_name: "no previous fixture before T"}

    last = history[0]  # newest-first
    delta = kickoff - last.kickoff_time
    days = int(delta.total_seconds() // 86400)
    return {feat_name: days}, {}


__all__ = ["compute_rest_features"]
