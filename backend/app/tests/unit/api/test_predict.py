# ruff: noqa: N803, N806
"""Tests API predict — Sprint 6.1 (D8) + Sprint 6.3 (D10)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from app.api.main import create_app
from app.features.example import FEATURE_NAMES
from fastapi.testclient import TestClient

TEST_API_KEY = "test-secret-key-for-ci"
HEADERS = {"X-API-Key": TEST_API_KEY}


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_SECRET_KEY", TEST_API_KEY)
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    monkeypatch.delenv("API_SECRET_KEY", raising=False)
    get_settings.cache_clear()


def _make_request_payload() -> dict:
    vec = [0.5 if i % 3 else None for i in range(66)]  # incluye None → NaN
    return {
        "fixture_id": 1,
        "kickoff": datetime(2026, 1, 1, 12, 0, tzinfo=UTC).isoformat(),
        "home_team_id": 10,
        "away_team_id": 20,
        "competition_id": 39,
        "season_id": 1,
        "feature_vector": vec,
        "feature_names": list(FEATURE_NAMES),
    }


def _make_synthetic_model(tmp_path: Path):
    """Crea LightGBMModel sintético y lo guarda para lifespan, retorna modelo."""
    from app.prediction.models.lightgbm import LightGBMTrainer
    from app.prediction.models.persistence import save_model

    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 66))
    y = rng.integers(0, 3, size=30)
    y[0] = 0
    y[1] = 1
    y[2] = 2
    trainer = LightGBMTrainer(params={"random_state": 42, "verbosity": -1, "n_estimators": 20})
    model = trainer.fit(X, y)
    p = tmp_path / "lightgbm_production.joblib"
    save_model(model, p)
    return model, p


def test_predict_valid_retorna_200(tmp_path: Path, monkeypatch) -> None:
    model, model_path = _make_synthetic_model(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    app = create_app()
    app.state.model = model  # type: ignore[attr-defined]
    app.state.model_version = "test"  # type: ignore[attr-defined]

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/v1/predict", json=_make_request_payload(), headers=HEADERS)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "prob_home" in data and "prob_draw" in data and "prob_away" in data
    s = data["prob_home"] + data["prob_draw"] + data["prob_away"]
    assert s == pytest.approx(1.0, abs=1e-6)  # type: ignore[name-defined]
    assert 0 <= data["prob_home"] <= 1
    assert data["model_version"] == "test"
    assert "predicted_at" in data


def test_predict_503_si_modelo_no_cargado() -> None:
    app = create_app()
    app.state.model = None  # type: ignore[attr-defined]
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/v1/predict", json=_make_request_payload(), headers=HEADERS)
    # Sin modelo, debe ser 503 incluso con API key válida (prioridad 503 sobre 401)
    # Pero si API key falta, sería 401; aquí probamos sin modelo pero con key
    assert resp.status_code == 503
    assert resp.json()["detail"] == "model not loaded"


def test_predict_lifespan_carga_unica(tmp_path: Path, monkeypatch) -> None:
    """Verifica D8.2: lifespan carga una vez, endpoint no hace I/O."""
    _, model_path = _make_synthetic_model(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    from fastapi.testclient import TestClient

    app = create_app()
    with TestClient(app) as client:
        assert getattr(client.app.state, "model", None) is not None
        resp = client.post("/api/v1/predict", json=_make_request_payload(), headers=HEADERS)
        assert resp.status_code == 200
    assert getattr(app.state, "model", None) is None


def test_predict_pydantic_valida_entrada() -> None:
    app = create_app()
    import numpy as np
    from app.prediction.models.lightgbm import LightGBMTrainer

    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 66))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    trainer = LightGBMTrainer(params={"random_state": 42, "verbosity": -1, "n_estimators": 10})
    model = trainer.fit(X, y)
    app.state.model = model  # type: ignore[attr-defined]
    app.state.model_version = "test"  # type: ignore[attr-defined]
    client = TestClient(app, raise_server_exceptions=False)
    bad = _make_request_payload()
    bad.pop("feature_vector")
    resp = client.post("/api/v1/predict", json=bad, headers=HEADERS)
    assert resp.status_code == 422


def test_predict_fixture_valid(monkeypatch) -> None:
    """GET /predict/fixture/{id} camino feliz (mock DB + builder)."""
    from unittest.mock import AsyncMock

    from app.api.main import create_app
    from app.db.session import get_session

    import numpy as np
    from app.prediction.models.lightgbm import LightGBMTrainer

    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 66))
    y = rng.integers(0, 3, size=20)
    y[0] = 0
    y[1] = 1
    y[2] = 2
    trainer = LightGBMTrainer(params={"random_state": 42, "verbosity": -1, "n_estimators": 10})
    model = trainer.fit(X, y)

    app = create_app()
    app.state.model = model  # type: ignore[attr-defined]
    app.state.model_version = "test"  # type: ignore[attr-defined]

    mock_session = AsyncMock()
    app.dependency_overrides[get_session] = lambda: mock_session  # type: ignore[assignment]

    from app.api.services.prediction_service import PredictionService
    from app.api.schemas.prediction import MatchPredictionResponse
    from datetime import UTC, datetime

    async def _fake_predict(*args, **kwargs):
        return MatchPredictionResponse(
            fixture_id=123,
            prob_home=0.5,
            prob_draw=0.3,
            prob_away=0.2,
            model_version="test",
            predicted_at=datetime.now(UTC),
        )

    monkeypatch.setattr(PredictionService, "predict_for_fixture", _fake_predict)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/v1/predict/fixture/123", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["fixture_id"] == 123
    assert data["prob_home"] == 0.5
    app.dependency_overrides.clear()


def test_predict_fixture_404(monkeypatch) -> None:
    from unittest.mock import AsyncMock

    from app.api.main import create_app
    from app.api.services.prediction_service import FixtureNotFoundError, PredictionService
    from app.db.session import get_session

    import numpy as np
    from app.prediction.models.lightgbm import LightGBMTrainer

    rng = np.random.default_rng(2)
    X = rng.normal(size=(10, 66))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    trainer = LightGBMTrainer(params={"random_state": 42, "verbosity": -1, "n_estimators": 10})
    model = trainer.fit(X, y)
    app = create_app()
    app.state.model = model  # type: ignore[attr-defined]
    app.dependency_overrides[get_session] = lambda: AsyncMock()  # type: ignore[assignment]

    async def _raise_not_found(*args, **kwargs):
        raise FixtureNotFoundError("Fixture 999 not found")

    monkeypatch.setattr(PredictionService, "predict_for_fixture", _raise_not_found)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/v1/predict/fixture/999", headers=HEADERS)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
    app.dependency_overrides.clear()


def test_predict_fixture_422_insufficient_history(monkeypatch) -> None:
    from unittest.mock import AsyncMock

    from app.api.main import create_app
    from app.api.services.prediction_service import InsufficientHistoryError, PredictionService
    from app.db.session import get_session

    import numpy as np
    from app.prediction.models.lightgbm import LightGBMTrainer

    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 66))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    trainer = LightGBMTrainer(params={"random_state": 42, "verbosity": -1, "n_estimators": 10})
    model = trainer.fit(X, y)
    app = create_app()
    app.state.model = model  # type: ignore[attr-defined]
    app.dependency_overrides[get_session] = lambda: AsyncMock()  # type: ignore[assignment]

    async def _raise_insufficient(*args, **kwargs):
        raise InsufficientHistoryError("Not enough historical data for feature generation")

    monkeypatch.setattr(PredictionService, "predict_for_fixture", _raise_insufficient)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/v1/predict/fixture/123", headers=HEADERS)
    assert resp.status_code == 422
    assert "historical" in resp.json()["detail"].lower()
    app.dependency_overrides.clear()


def test_predict_401_sin_api_key(tmp_path: Path, monkeypatch) -> None:
    from app.api.main import create_app

    app = create_app()
    import numpy as np
    from app.prediction.models.lightgbm import LightGBMTrainer

    rng = np.random.default_rng(9)
    X = rng.normal(size=(10, 66))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    trainer = LightGBMTrainer(params={"random_state": 42, "verbosity": -1, "n_estimators": 10})
    model = trainer.fit(X, y)
    app.state.model = model  # type: ignore[attr-defined]
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/v1/predict", json=_make_request_payload())
    assert resp.status_code == 401


def test_predict_401_api_key_incorrect(tmp_path: Path, monkeypatch) -> None:
    from app.api.main import create_app

    import numpy as np
    from app.prediction.models.lightgbm import LightGBMTrainer

    rng = np.random.default_rng(10)
    X = rng.normal(size=(10, 66))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    trainer = LightGBMTrainer(params={"random_state": 42, "verbosity": -1, "n_estimators": 10})
    model = trainer.fit(X, y)
    app = create_app()
    app.state.model = model  # type: ignore[attr-defined]
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/v1/predict", json=_make_request_payload(), headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401
