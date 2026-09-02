"""Head-to-head features across the fixtures played between two teams.

Important anti-leakage properties:

* All qualifying fixtures kicked off **strictly before** the target
  fixture's kickoff (boundary enforced via
  :func:`app.features.asof.fixtures_before_unordered`).
* The window is pair-specific (only fixtures between these two teams).
* When fewer than ``MIN_SAMPLE`` H2H matches are available we emit
  ``h2h_sample_size`` as that count and set the metric features to
  ``None`` — never to ``0`` — so downstream consumers can choose
  whether to impute or skip.
"""

from __future__ import annotations

from app.features import example as ex
from app.features.asof import fixtures_before_unordered
from app.features.rows import FixtureRow

#: Minimum sample size for H2H metric features. Smaller sample sizes
#: arguably make the H2H rate unrealiable; we still expose the raw
#: ``h2h_sample_size`` so a downstream model can decide.
MIN_SAMPLE: int = 3


def _is_pair_match(fx: FixtureRow, *, team_a: int, team_b: int) -> bool:
    """True iff ``fx`` is between ``team_a`` and ``team_b`` (any venue)."""
    teams = {fx.home_team_id, fx.away_team_id}
    return team_a in teams and team_b in teams


def compute_h2h_features(
    *,
    home_team_id: int,
    away_team_id: int,
    season_fixtures: list[FixtureRow],
    kickoff: object,
    exclude_fixture_id: int | None,
) -> tuple[dict[str, float | int | None], dict[str, str]]:
    """Return H2H features between ``home_team_id`` and ``away_team_id``.

    H2H spans multiple seasons typically (teams face each other across
    years) — but the dataset builder operates per (competition, season)
    for performance. The builder is expected to **pre-load the global
    pair fixture list** (all seasons) before invoking this helper; we
    don't fetch inside it.

    The sample-size policy is explicit: when ``sample >= MIN_SAMPLE`` we
    emit ``h2h_sample_size = sample`` and the metric values; otherwise we
    emit ``h2h_sample_size = sample`` (real count) and the metrics are
    ``None`` (NOT 0). This preserves the "real count can be 0, missing
    metric is None" contract from docs/FEATURES.md.
    """
    eligible = [
        fx
        for fx in season_fixtures
        if _is_pair_match(fx, team_a=home_team_id, team_b=away_team_id)
    ]
    ordered = fixtures_before_unordered(
        eligible,
        kickoff=kickoff,  # type: ignore[arg-type]
        exclude_fixture_id=exclude_fixture_id,
    )

    sample = len(ordered)
    features: dict[str, float | int | None] = {ex.H2H_SAMPLE_SIZE: sample}
    missing: dict[str, str] = {}

    if sample < MIN_SAMPLE:
        features[ex.H2H_HOME_WINS] = None
        features[ex.H2H_AWAY_WINS] = None
        features[ex.H2H_DRAWS] = None
        features[ex.H2H_TOTAL_GOALS_MEAN] = None
        missing[ex.H2H_HOME_WINS] = (
            f"only {sample} H2H fixtures before T (< MIN_SAMPLE={MIN_SAMPLE})"
        )
        return features, missing

    home_wins = sum(1 for fx in ordered if fx.outcome_for(home_team_id) == "W")
    away_wins = sum(1 for fx in ordered if fx.outcome_for(away_team_id) == "W")
    draws = sum(1 for fx in ordered if fx.outcome_for(home_team_id) == "D")
    total_goals = sum(fx.home_goals + fx.away_goals for fx in ordered)

    features[ex.H2H_HOME_WINS] = home_wins
    features[ex.H2H_AWAY_WINS] = away_wins
    features[ex.H2H_DRAWS] = draws
    features[ex.H2H_TOTAL_GOALS_MEAN] = float(total_goals) / float(sample)
    return features, missing


__all__ = [
    "MIN_SAMPLE",
    "compute_h2h_features",
]
