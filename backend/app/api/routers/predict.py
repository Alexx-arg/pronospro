"""Router de inferencia — Sprint 6.1 (D8.1-8.3)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from app.api.dependencies.security import require_api_key_dep
from app.api.schemas.prediction import MatchPredictionRequest, MatchPredictionResponse
from app.api.services.prediction_service import (
    FixtureNotFoundError,
    InsufficientHistoryError,
    PredictionService,
)
from app.db.session import get_session
from app.prediction.contracts import FixtureFeatures
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(tags=["predict"], dependencies=[Depends(require_api_key_dep)])


def _request_to_features(req: MatchPredictionRequest) -> FixtureFeatures:
    """Transforma request Pydantic → FixtureFeatures sin DB ni iterador (D8.1)."""
    # feature_vector puede contener None (missing) → se mantiene como None para vector.py
    # Pero FixtureFeatures espera np.ndarray con NaN; aquí construimos directo
    # para evitar pasar por vector.py (no tenemos LoadedExample). Mantenemos NaN.
    vec_list: list[float] = []
    for v in req.feature_vector:
        if v is None:
            vec_list.append(float("nan"))
        else:
            vec_list.append(float(v))
    arr = np.asarray(vec_list, dtype=np.float32)
    return FixtureFeatures(
        fixture_id=req.fixture_id,
        kickoff=req.kickoff,
        feature_vector=arr,
        feature_names=tuple(req.feature_names),
    )


@router.post("/predict", response_model=MatchPredictionResponse)
def predict(req: MatchPredictionRequest, request: Request) -> Any:
    """Inferencia pura: request → modelo en memoria → response (D8.1)."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "model not loaded"},
        )
    # Transformación sin DB/I/O por request (D8.2: no load_model aquí)
    try:
        features = _request_to_features(req)
        prob = model.predict(features)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )

    # Validación estricta D8.3 ya hecha por Pydantic en response_model
    # Asegura simplex si el modelo es correcto; si no, Pydantic validará rangos
    return MatchPredictionResponse(
        fixture_id=req.fixture_id,
        prob_home=float(prob.p_home_win),
        prob_draw=float(prob.p_draw),
        prob_away=float(prob.p_away_win),
        model_version=getattr(request.app.state, "model_version", "production"),
        predicted_at=datetime.now(UTC),
    )


@router.get("/predict/fixture/{fixture_id}", response_model=MatchPredictionResponse)
async def predict_fixture(
    fixture_id: int,
    request: Request,
    db_session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    """Orquestación por fixture_id — delega a PredictionService (D9.1)."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "model not loaded"},
        )
    service = PredictionService()
    try:
        return await service.predict_for_fixture(fixture_id, db_session, model)
    except FixtureNotFoundError as exc:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )
    except InsufficientHistoryError as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )


__all__ = ["router"]
