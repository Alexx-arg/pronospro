"""Assembler: combines every feature family into a complete
:class:`~app.features.example.HistoricalMatchExample`.

The assembler is the SINGLE entry point used by the dataset builder so
the rest of the codebase never directly wires feature families
together. That keeps the "features must never read targets" guarantee
explicit: the assembler builds the features dict FIRST (from a
:class:`FixtureRow` whose goals are populated only because the row is
already finished — but we never read them for the feature math; the
goals are read solely when writing the target block at the end).
"""

from __future__ import annotations

from app.features.elo import compute_elo_features
from app.features.example import (
    AWAY_GOALS_TARGET,
    AWAY_WIN,
    DRAW,
    HOME_GOALS_TARGET,
    HOME_WIN,
    HistoricalMatchExample,
)
from app.features.form import compute_form_features
from app.features.goals import compute_goals_features
from app.features.h2h import compute_h2h_features
from app.features.homeaway import compute_homeaway_features
from app.features.rest import compute_rest_features
from app.features.rows import FixtureRow, TeamStatsRow
from app.features.standings import compute_standings_features
from app.features.xg import compute_xg_features


def _merge(
    target_dicts: list[tuple[dict[str, float | int | None], dict[str, str]]],
) -> tuple[dict[str, float | int | None], dict[str, str]]:
    """Flatten a list of (features, missing) pairs into two dicts."""
    feats: dict[str, float | int | None] = {}
    miss: dict[str, str] = {}
    for f, m in target_dicts:
        feats.update(f)
        miss.update(m)
    return feats, miss


def build_example(
    *,
    fixture: FixtureRow,
    season_fixtures: list[FixtureRow],
    stats_rows: list[TeamStatsRow],
    h2h_fixtures: list[FixtureRow] | None = None,
) -> HistoricalMatchExample:
    """Build a full :class:`HistoricalMatchExample` for ``fixture``.

    Args:
        fixture: the target fixture as a :class:`FixtureRow` projection.
            Must be ``status='finished'`` so the targets (goals / result
            flags) are well-defined. Feature math never reads ``fixture``
            itself — only the teams of surrounding fixtures; the
            ``fixture`` argument is only used to (a) pick the four team
            ids, (b) define ``kickoff`` and ``exclude_fixture_id``, and
            (c) populate the target block at the end.
        season_fixtures: the season's finished fixtures the assembler
            pulls historical form / goals / standings / elo from. The
            loader's :func:`fixtures_before` boundary helper is applied
            inside each feature module, with ``kickoff = fixture.kickoff_time``
            and ``exclude_fixture_id = fixture.fixture_id``. The caller
            is responsible for matching ``(competition, season)``.
        stats_rows: every ``team_statistics`` snapshot for the same
            season (for xG features).
        h2h_fixtures: optional cross-season pair history between the
            target's two teams. When set, H2H features are computed
            exclusively from this universe (the boundary
            ``kickoff < T`` and the target exclusion are applied inside
            :func:`app.features.h2h.compute_h2h_features` via
            :func:`app.features.asof.fixtures_before_unordered`). When
            ``None``, H2H falls back to the season-scoped
            ``season_fixtures`` — a cross-season-oblivious default kept
            for callers without the bulk loader (e.g. tests). The
            Phase 4 dataset builder always passes a populated
            ``h2h_fixtures``.

    Returns:
        Frozen example with features and targets populated.
    """
    kickoff = fixture.kickoff_time
    exclude = fixture.fixture_id

    # ---- Features ---------------------------------------------------------
    home_form, m1 = compute_form_features(
        scope_prefix="home", team_id=fixture.home_team_id,
        season_fixtures=season_fixtures, kickoff=kickoff,
        exclude_fixture_id=exclude,
    )
    away_form, m2 = compute_form_features(
        scope_prefix="away", team_id=fixture.away_team_id,
        season_fixtures=season_fixtures, kickoff=kickoff,
        exclude_fixture_id=exclude,
    )
    home_goals, m3 = compute_goals_features(
        scope_prefix="home", team_id=fixture.home_team_id,
        season_fixtures=season_fixtures, kickoff=kickoff,
        exclude_fixture_id=exclude,
    )
    away_goals, m4 = compute_goals_features(
        scope_prefix="away", team_id=fixture.away_team_id,
        season_fixtures=season_fixtures, kickoff=kickoff,
        exclude_fixture_id=exclude,
    )
    home_split, m5 = compute_homeaway_features(
        scope_prefix="home", team_id=fixture.home_team_id,
        season_fixtures=season_fixtures, kickoff=kickoff,
        exclude_fixture_id=exclude, is_home_role=True,
    )
    away_split, m6 = compute_homeaway_features(
        scope_prefix="away", team_id=fixture.away_team_id,
        season_fixtures=season_fixtures, kickoff=kickoff,
        exclude_fixture_id=exclude, is_home_role=False,
    )
    h2h, m7 = compute_h2h_features(
        home_team_id=fixture.home_team_id, away_team_id=fixture.away_team_id,
        season_fixtures=(h2h_fixtures if h2h_fixtures is not None else season_fixtures),
        kickoff=kickoff, exclude_fixture_id=exclude,
    )
    standings, m8 = compute_standings_features(
        season_fixtures=season_fixtures,
        home_team_id=fixture.home_team_id, away_team_id=fixture.away_team_id,
        kickoff=kickoff, exclude_fixture_id=exclude,
    )
    home_rest, m9 = compute_rest_features(
        season_fixtures=season_fixtures, team_id=fixture.home_team_id,
        scope_prefix="home", kickoff=kickoff, exclude_fixture_id=exclude,
    )
    away_rest, m10 = compute_rest_features(
        season_fixtures=season_fixtures, team_id=fixture.away_team_id,
        scope_prefix="away", kickoff=kickoff, exclude_fixture_id=exclude,
    )
    elo, m11 = compute_elo_features(
        fixture=fixture, season_fixtures=season_fixtures,
    )
    xg, m12 = compute_xg_features(
        stats_rows=stats_rows,
        home_team_id=fixture.home_team_id, away_team_id=fixture.away_team_id,
        kickoff=kickoff,
    )

    features, missing = _merge(
        [home_form, away_form, home_goals, away_goals,
         home_split, away_split, h2h, standings,
         home_rest, away_rest, elo, xg],
    )

    # ---- Targets ---------------------------------------------------------
    # Built from the same FixtureRow but written into the SEPARATE
    # ``targets`` dict so feature math cannot read them post-hoc.
    targets = _build_targets(fixture)

    return HistoricalMatchExample(
        fixture_id=fixture.fixture_id,
        kickoff=fixture.kickoff_time,
        competition_id=fixture.competition_id,
        season_id=fixture.season_id,
        home_team_id=fixture.home_team_id,
        away_team_id=fixture.away_team_id,
        features=features,
        targets=targets,
        missing_report=missing,
    )


def _build_targets(fixture: FixtureRow) -> dict[str, int | None]:
    """Compute the post-match targets from a finished fixture."""
    if fixture.home_goals is None or fixture.away_goals is None:
        # Shouldn't happen — loaders only emit finished rows — but we
        # keep the targets missing rather than letting a None leak through.
        return {HOME_WIN: None, DRAW: None, AWAY_WIN: None,
                HOME_GOALS_TARGET: None, AWAY_GOALS_TARGET: None}

    home_won = 1 if fixture.home_goals > fixture.away_goals else 0
    away_won = 1 if fixture.away_goals > fixture.home_goals else 0
    draw = 1 if fixture.home_goals == fixture.away_goals else 0
    return {
        HOME_WIN: home_won,
        DRAW: draw,
        AWAY_WIN: away_won,
        HOME_GOALS_TARGET: fixture.home_goals,
        AWAY_GOALS_TARGET: fixture.away_goals,
    }


__all__ = ["build_example"]
