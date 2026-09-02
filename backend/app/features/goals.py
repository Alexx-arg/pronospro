"""Goals features: scored / conceded totals and means over the last N."""

from __future__ import annotations

from app.features.asof import team_history_before
from app.features.rows import FixtureRow
from app.features.rolling import rolling_mean, rolling_sum

_WINDOWS: tuple[int, ...] = (5, 10)


def compute_goals_features(
    *,
    scope_prefix: str,  # "home" | "away"
    team_id: int,
    season_fixtures: list[FixtureRow],
    kickoff: object,
    exclude_fixture_id: int | None,
) -> tuple[dict[str, float | int | None], dict[str, str]]:
    """Return (features, missing_report) for goal totals and means.

    Totals (``goals_for_last_N``, ``goals_against_last_N``) use
    :func:`~app.features.rolling.rolling_sum`, returning ``None`` when
    the window is partial.

    Means use :func:`~app.features.rolling.rolling_mean`, also ``None``
    when the window is partial.

    A fixture contributes 0 to the sum/mean only when its actual goal
    count was 0 — the rolling helpers treat ``None`` (missing data row)
    as a non-contribution, never as 0. See docs/FEATURES.md.
    """
    history = team_history_before(
        team_id=team_id,
        season_fixtures=season_fixtures,
        kickoff=kickoff,  # type: ignore[arg-type]
        exclude_fixture_id=exclude_fixture_id,
    )

    features: dict[str, float | int | None] = {}
    missing: dict[str, str] = {}

    for n in _WINDOWS:
        gf_sum = rolling_sum(history, n, lambda fx: fx.goals_for(team_id))
        ga_sum = rolling_sum(history, n, lambda fx: fx.goals_against(team_id))
        gf_mean = rolling_mean(history, n, lambda fx: fx.goals_for(team_id))
        ga_mean = rolling_mean(history, n, lambda fx: fx.goals_against(team_id))

        if gf_sum is None:
            features[f"{scope_prefix}_goals_for_last_{n}"] = None
            missing[f"{scope_prefix}_goals_for_last_{n}"] = (
                f"fewer than {n} finished matches before T"
            )
        else:
            features[f"{scope_prefix}_goals_for_last_{n}"] = int(gf_sum)

        if ga_sum is None:
            features[f"{scope_prefix}_goals_against_last_{n}"] = None
            missing[f"{scope_prefix}_goals_against_last_{n}"] = (
                f"fewer than {n} finished matches before T"
            )
        else:
            features[f"{scope_prefix}_goals_against_last_{n}"] = int(ga_sum)

        if gf_mean is None:
            features[f"{scope_prefix}_goals_for_mean_last_{n}"] = None
            missing[f"{scope_prefix}_goals_for_mean_last_{n}"] = (
                f"fewer than {n} finished matches before T"
            )
        else:
            features[f"{scope_prefix}_goals_for_mean_last_{n}"] = float(gf_mean)

        if ga_mean is None:
            features[f"{scope_prefix}_goals_against_mean_last_{n}"] = None
            missing[f"{scope_prefix}_goals_against_mean_last_{n}"] = (
                f"fewer than {n} finished matches before T"
            )
        else:
            features[f"{scope_prefix}_goals_against_mean_last_{n}"] = float(ga_mean)

    return features, missing


__all__ = ["compute_goals_features"]
