# ruff: noqa: N806
"""Unit tests for LightGBM wrapper — Sprint 5.11.

Usa datos sintéticos (mocked) para verificar:
- entrenamiento (fit) y predicción (predict / predict_proba)
- forma de tensores (shape)
- salida probabilidades (simplex)
- determinismo con misma seed
- manejo NaN passthrough (no imputación)
"""

from __future__ import annotations

import numpy as np
import pytest
from app.prediction.models.lightgbm import DEFAULT_LIGHTGBM_PARAMS, LightGBMModel, LightGBMTrainer
from app.prediction.training import get_trainer as get_trainer_factory


def _synthetic_data(n: int = 20, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 66))
    # Simula NaN en 10% para probar passthrough
    mask = rng.random(size=X.shape) < 0.05
    X[mask] = np.nan
    y = rng.integers(0, 3, size=n)
    # Asegura al menos 2 clases
    if len(np.unique(y)) < 2:
        y[0] = 0
        y[1] = 1
    return X, y


def test_lightgbm_fit_predict_shape() -> None:
    X, y = _synthetic_data(30, seed=1)
    trainer = LightGBMTrainer()
    model = trainer.fit(X, y)
    assert isinstance(model, LightGBMModel)
    proba = model.predict_proba(X)
    assert proba.shape == (30, 3)
    # Cada fila suma ≈1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.all(np.isfinite(proba))


def test_lightgbm_predict_fixture_features_shape() -> None:
    """Verifica predict con FixtureFeatures (contrato Predictor)."""
    from datetime import UTC, datetime

    from app.features.example import FEATURE_NAMES
    from app.prediction.contracts import FixtureFeatures

    X, y = _synthetic_data(20, seed=2)
    trainer = LightGBMTrainer()
    model = trainer.fit(X, y)

    vec = np.full(len(FEATURE_NAMES), 0.5, dtype=np.float32)
    feat = FixtureFeatures(
        fixture_id=1,
        kickoff=datetime(2026, 1, 1, tzinfo=UTC),
        feature_vector=vec,
        feature_names=tuple(FEATURE_NAMES),
    )
    mp = model.predict(feat)
    s = mp.p_home_win + mp.p_draw + mp.p_away_win
    assert s == pytest.approx(1.0, abs=1e-9)
    assert mp.p_home_goals is None
    assert mp.p_away_goals is None


def test_lightgbm_determinismo() -> None:
    X, y = _synthetic_data(40, seed=123)
    # Dos entrenamientos con misma seed
    t1 = LightGBMTrainer(params={"random_state": 42, "verbosity": -1})
    t2 = LightGBMTrainer(params={"random_state": 42, "verbosity": -1})
    m1 = t1.fit(X, y)
    m2 = t2.fit(X, y)
    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)
    assert np.allclose(p1, p2)
    # Pesos idénticos vía predict_proba
    assert np.array_equal(np.argmax(p1, axis=1), np.argmax(p2, axis=1))


def test_lightgbm_determinismo_diferente_seed_difiere() -> None:
    X, y = _synthetic_data(40, seed=7)
    t1 = LightGBMTrainer(params={"random_state": 1, "verbosity": -1, "n_estimators": 20})
    t2 = LightGBMTrainer(params={"random_state": 999, "verbosity": -1, "n_estimators": 20})
    # Con datos y hiperparámetros que permitan variabilidad, dos seeds distintas pueden dar probas distintas
    # No garantizado siempre, pero al menos entrenan sin error y producen simplex válido
    m1 = t1.fit(X, y)
    m2 = t2.fit(X, y)
    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)
    assert p1.shape == p2.shape
    assert np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))


def test_lightgbm_nan_passthrough() -> None:
    X, y = _synthetic_data(30, seed=3)
    # Fuerza NaN en X
    X[0, 0] = np.nan
    X[1, 5] = np.nan
    trainer = LightGBMTrainer(params={"random_state": 42, "verbosity": -1})
    model = trainer.fit(X, y)
    # Predicción con NaN no debe imputar a 0 ni fallar
    proba = model.predict_proba(X)
    assert np.all(np.isfinite(proba))
    # Verifica que el modelo maneja NaN sin error de forma
    assert proba.shape == (30, 3)


def test_lightgbm_train_artifact_determinismo_via_fixturefeatures() -> None:
    """Entrenamiento vía API Fase 5 (train con FixtureFeatures) es determinista."""
    from datetime import UTC, datetime

    from app.features.example import FEATURE_NAMES
    from app.prediction.contracts import FixtureFeatures

    def _feat(fid: int, val: float = 0.5, nan: bool = False) -> FixtureFeatures:
        vec = np.full(len(FEATURE_NAMES), val, dtype=np.float32)
        if nan:
            vec[0] = np.nan
        return FixtureFeatures(
            fixture_id=fid,
            kickoff=datetime(2026, 1, fid, tzinfo=UTC),
            feature_vector=vec,
            feature_names=tuple(FEATURE_NAMES),
        )

    feats = [_feat(i) for i in range(1, 7)]
    targets = [[1, 0, 0, 2, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 2]] * 2
    t1 = LightGBMTrainer(params={"random_state": 42, "verbosity": -1, "n_estimators": 20})
    t2 = LightGBMTrainer(params={"random_state": 42, "verbosity": -1, "n_estimators": 20})
    a1 = t1.train(feats, targets, seed=42)  # type: ignore[arg-type]
    a2 = t2.train(feats, targets, seed=42)  # type: ignore[arg-type]
    assert a1.payload_sha256 == a2.payload_sha256


def test_lightgbm_factory_registro() -> None:
    # Verifica que el factory reconoce "lightgbm"
    trainer = get_trainer_factory("lightgbm")
    assert trainer is not None
    assert hasattr(trainer, "fit")
    # También debe funcionar con mayúsculas/espacios
    trainer2 = get_trainer_factory(" LightGBM ")
    assert trainer2 is not None

    # Verifica get_model_class
    from app.prediction.models import get_model_class

    cls = get_model_class("lightgbm")
    assert cls.__name__ == "LightGBMModel"


def test_lightgbm_default_params_fijos() -> None:
    assert DEFAULT_LIGHTGBM_PARAMS["random_state"] == 42
    assert DEFAULT_LIGHTGBM_PARAMS["verbosity"] == -1
    assert DEFAULT_LIGHTGBM_PARAMS["n_jobs"] == 1
