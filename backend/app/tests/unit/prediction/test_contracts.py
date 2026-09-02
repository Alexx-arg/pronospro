"""Sprint 5.0 — contracts import cleanly and expose the expected API."""

from __future__ import annotations

from app.prediction import contracts


def test_contracts_module_imports() -> None:
    """Sanity: importing the module does not raise. mypy strict + ruff
    are the actual gate; this test guards against accidental runtime
    errors that would slip through ``import-from-name`` checks.
    """
    assert contracts is not None


def test_model_name_exposes_three_canonical_models() -> None:
    members = {m.name for m in contracts.ModelName}
    assert members == {"ELO_BASELINE", "POISSON", "GRADIENT_BOOSTING"}


def test_model_name_str_values_match_design() -> None:
    # StrEnum exposes the string value as the canonical form for
    # JSON serialisation. The values must match PHASE_5.md §3 exactly
    # so the persistence layout (storage/layout.py) doesn't drift.
    assert contracts.ModelName.ELO_BASELINE == "elo_baseline"
    assert contracts.ModelName.POISSON == "poisson"
    assert contracts.ModelName.GRADIENT_BOOSTING == "gradient_boosting"


def test_match_probabilities_represents_valid_distribution() -> None:
    """A point on the 2-simplex admitted by the NamedTuple."""
    p = contracts.MatchProbabilities(
        p_home_win=0.5,
        p_draw=0.3,
        p_away_win=0.2,
    )
    assert p.p_home_win + p.p_draw + p.p_away_win == 1.0
    assert 0.0 <= p.p_home_win <= 1.0
    assert 0.0 <= p.p_draw <= 1.0
    assert 0.0 <= p.p_away_win <= 1.0


def test_match_probabilities_optional_goals_default_to_none() -> None:
    p = contracts.MatchProbabilities(0.5, 0.3, 0.2)
    assert p.p_home_goals is None
    assert p.p_away_goals is None


def test_calibrator_kind_is_three_approved_methods() -> None:
    members = {k.name for k in contracts.CalibratorKind}
    assert members == {"IDENTITY", "TEMPERATURE", "DIRICHLET"}


def test_nan_policy_exposes_two_approved_modes() -> None:
    members = {n.name for n in contracts.NanPolicy}
    assert members == {"DROP_ROW", "IMPUTE_TRAIN_MEAN"}


def test_walk_forward_mode_exposes_two_modes() -> None:
    members = {w.name for w in contracts.WalkForwardMode}
    assert members == {"EXPANDING", "SLIDING"}


def test_protocols_are_runtime_checkable() -> None:
    # Structural typing requires runtime_checkable to allow
    # isinstance(); the only point of this is that subsequent sprints
    # can write `isinstance(p, Predictor)` in tests.
    assert hasattr(contracts.Predictor, "_is_protocol")
    assert hasattr(contracts.Trainer, "_is_protocol")
    assert hasattr(contracts.Calibrator, "_is_protocol")
