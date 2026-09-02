"""FastAPI app con lifespan de carga única — Sprint 6.1."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from app.prediction.models.persistence import load_model
from fastapi import FastAPI


def _resolve_model_path() -> Path | None:
    """Resuelve ruta del .joblib sin tocar disco en cada request (D8.2)."""
    # 1. Env var explícita
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
    # 2. Ubicaciones canónicas
    candidates = [
        Path("data/models/final/lightgbm_production.joblib"),
        Path("backend/data/models/final/lightgbm_production.joblib"),
        Path(__file__).resolve().parents[3] / "data" / "models" / "final" / "lightgbm_production.joblib",
        Path(__file__).resolve().parents[2] / "lightgbm_production.joblib",
        Path("lightgbm_production.joblib"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    # Startup: carga única (D8.2)
    model_path = _resolve_model_path()
    if model_path is not None:
        try:
            model = load_model(model_path)
            app.state.model = model
            app.state.model_path = str(model_path)
            app.state.model_version = getattr(model, "model_version", "production")
        except Exception:
            app.state.model = None
            app.state.model_path = None
    else:
        app.state.model = None
        app.state.model_path = None
    yield
    # Shutdown: limpia memoria
    app.state.model = None


def create_app() -> FastAPI:
    """Factory testeable (permite inyección de modelo mock en tests)."""
    app = FastAPI(title="Football Prediction API — Sprint 6.1", lifespan=lifespan)

    # Sprint 6.3 — CORS (D10, env-driven, default restrictivo)
    try:  # noqa: I001
        from fastapi.middleware.cors import CORSMiddleware  # noqa: I001

        from app.config import get_settings  # noqa: I001

        settings = get_settings()
        origins = settings.cors_origins_list()
        # En desarrollo sin CORS configurado, permitir "*" solo si debug
        if not origins and settings.env == "development":
            origins = ["*"]
        if origins:
            # Si allow_origins es ["*"], no usar credentials con "*"
            allow_credentials = origins != ["*"]
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=allow_credentials,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    except Exception:
        # No bloquear startup si CORS falla
        pass

    from app.api.routers.predict import router as predict_router
    from app.api.routers.fixtures import router as fixtures_router

    app.include_router(predict_router, prefix="/api/v1")
    app.include_router(fixtures_router, prefix="/api/v1")

    from app.api.routers.explain import router as explain_router

    app.include_router(explain_router, prefix="/api/v1")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_loaded": getattr(app.state, "model", None) is not None}

    return app


app = create_app()

__all__ = ["app", "create_app", "lifespan"]
