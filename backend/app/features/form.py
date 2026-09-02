"""Form features: counts of W / D / L and points over the last N finished
matches for a single team.

Both home and away teams use this module; only the per-team history
differs. Output uses :mod:`app.features.rolling` so the window-exclusion
contract is enforced in a single place.
"""

from __future__ import annotations

from app.features.asof import team_history_before
from app.features.rows import FixtureRow
from app.features.rolling import rolling_count, rolling_sum

_WINDOWS: tuple[int, ...] = (3, 5, 10)


def compute_form_features(
    *,
    scope_prefix: str,  # "home" | "away"
    team_id: int,
    season_fixtures: list[FixtureRow],
    kickoff: object,
    exclude_fixture_id: int | None,
) -> tuple[dict[str, float | int | None], dict[str, str]]:
    """Return (features, missing_report) for form for one team.

    ``kickoff`` is passed as ``object`` because the loaders signature
    takes datetime and we want to avoid forcing callers to import it
    just for type hygiene; it is forwarded to
    :func:`app.features.asof.team_history_before` which enforces the
    temporal boundary.
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
        wins = rolling_count(history, n, lambda fx: fx.outcome_for(team_id) == "W")
        draws = rolling_count(history, n, lambda fx: fx.outcome_for(team_id) == "D")
        losses = rolling_count(history, n, lambda fx: fx.outcome_for(team_id) == "L")
        # ``rolling_count`` already returns a non-negative int even for
        # partial windows (per design of count semantics). But the form
        # features published in FEATURES.md REQUIRE a full window — anything
        # less is "missing". Guard that here so downstream consumers see
        # consistent None semantics.
        if len(history) >= n:
            features[f"{scope_prefix}_wins_last_{n}"] = wins
            features[f"{scope_prefix}_draws_last_{n}"] = draws
            features[f"{scope_prefix}_losses_last_{n}"] = losses
        else:
            features[f"{scope_prefix}_wins_last_{n}"] = None
            features[f"{scope_prefix}_draws_last_{n}"] = None
            features[f"{scope_prefix}_losses_last_{n}"] = None
            missing[f"{scope_prefix}_wins_last_{n}"] = (
                f"fewer than {n} finished matches before T"
            )

    # Points over 5 and 10 only (per FEATURES.md / spec).
    for n in (5, 10):
        pts = rolling_sum(history, n, lambda fx: fx.points_for(team_id))
        if pts is None:
            features[f"{scope_prefix}_points_last_{n}"] = None
            missing[f"{scope_prefix}_points_last_{n}"] = (
                f"fewer than {n} finished matches before T"
            )
        else:
            features[f"{scope_prefix}_points_last_{n}"] = int(pts)

    return features, missing


__all__ = [
    "compute_form_features",
]
