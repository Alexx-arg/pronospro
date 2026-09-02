"""Home-field advantage features.

For the **home team** of the target fixture: how it has performed AT HOME
in its last 5 home matches. For the **away team**: how it has performed
AWAY in its last 5 away matches.

Knowing "team X is great at home but poor away" is a major predictor of
match outcome and is usually more informative than the unconditioned
form. Phase 4 supports only the last-5 split; per docs/FEATURES.md the
missing value semantics require a full 5-sample window, otherwise the
feature is None.
"""

from __future__ import annotations

from app.features.asof import team_history_before
from app.features.rows import FixtureRow
from app.features.rolling import rolling_sum

_WINDOW = 5


def _split_history(
    history: list[FixtureRow],
    *,
    team_id: int,
    is_home: bool,
) -> list[FixtureRow]:
    """Slice to only home or away matches of ``team_id``."""
    if is_home:
        return [fx for fx in history if fx.home_team_id == team_id]
    return [fx for fx in history if fx.away_team_id == team_id]


def compute_homeaway_features(
    *,
    scope_prefix: str,  # "home" (means "home-side split") or "away"
    team_id: int,
    season_fixtures: list[FixtureRow],
    kickoff: object,
    exclude_fixture_id: int | None,
    is_home_role: bool,  # True for the fixture's home team, False for away
) -> tuple[dict[str, float | int | None], dict[str, str]]:
    """Return per-side (home/away) recent performance.

    Naming follows:

    * ``home_home_*``  — home team, restricted to its own home matches
    * ``away_away_*``  — away team, restricted to its own away matches
    """
    history = team_history_before(
        team_id=team_id,
        season_fixtures=season_fixtures,
        kickoff=kickoff,  # type: ignore[arg-type]
        exclude_fixture_id=exclude_fixture_id,
    )
    sliced = _split_history(history, team_id=team_id, is_home=is_home_role)
    role_tag = "home" if is_home_role else "away"

    features: dict[str, float | int | None] = {}
    missing: dict[str, str] = {}

    pts = rolling_sum(sliced, _WINDOW, lambda fx: fx.points_for(team_id))
    if pts is None:
        features[f"{scope_prefix}_{role_tag}_points_last_{_WINDOW}"] = None
        missing[f"{scope_prefix}_{role_tag}_points_last_{_WINDOW}"] = (
            f"fewer than {_WINDOW} {role_tag} matches before T"
        )
    else:
        features[f"{scope_prefix}_{role_tag}_points_last_{_WINDOW}"] = int(pts)

    gf = rolling_sum(sliced, _WINDOW, lambda fx: fx.goals_for(team_id))
    if gf is None:
        features[f"{scope_prefix}_{role_tag}_goals_for_last_{_WINDOW}"] = None
        missing[f"{scope_prefix}_{role_tag}_goals_for_last_{_WINDOW}"] = (
            f"fewer than {_WINDOW} {role_tag} matches before T"
        )
    else:
        features[f"{scope_prefix}_{role_tag}_goals_for_last_{_WINDOW}"] = int(gf)

    ga = rolling_sum(sliced, _WINDOW, lambda fx: fx.goals_against(team_id))
    if ga is None:
        features[f"{scope_prefix}_{role_tag}_goals_against_last_{_WINDOW}"] = None
        missing[f"{scope_prefix}_{role_tag}_goals_against_last_{_WINDOW}"] = (
            f"fewer than {_WINDOW} {role_tag} matches before T"
        )
    else:
        features[f"{scope_prefix}_{role_tag}_goals_against_last_{_WINDOW}"] = int(ga)

    return features, missing


__all__ = ["compute_homeaway_features"]
