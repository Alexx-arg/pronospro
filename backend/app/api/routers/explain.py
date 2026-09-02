"""Explanation endpoint using NVIDIA NIM (Llama 3.1 70B).

Replaces the temporary BzzOiro integration.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.dependencies.security import require_api_key_dep
from app.api.services.nvidia_explanation import NvidiaExplanationService

router = APIRouter(tags=["explain"])


class ExplainRequest(BaseModel):
    fixture_id: int
    prob_home: float = Field(..., ge=0, le=1)
    prob_draw: float = Field(..., ge=0, le=1)
    prob_away: float = Field(..., ge=0, le=1)
    home_team: str | None = None
    away_team: str | None = None
    metrics: dict[str, Any] | None = None


class ExplainResponse(BaseModel):
    explanation: str
    model_used: str = "NVIDIA NIM (meta/llama-3.1-70b-instruct)"


# Singleton service instance
_nvidia_service: NvidiaExplanationService | None = None


def get_nvidia_service() -> NvidiaExplanationService:
    global _nvidia_service
    if _nvidia_service is None:
        _nvidia_service = NvidiaExplanationService()
    return _nvidia_service


@router.post("/explain", response_model=ExplainResponse, dependencies=[Depends(require_api_key_dep)])
async def explain_prediction(
    body: ExplainRequest,
    request: Request,
    nvidia: NvidiaExplanationService = Depends(get_nvidia_service),
) -> ExplainResponse:
    """Generate tactical analysis using NVIDIA NIM.

    Takes LightGBM probabilities and optional match metrics,
    returns LLM-generated explanation.
    """
    if not nvidia.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NVIDIA_API_KEY no configurada. Explicación IA no disponible.",
        )

    explanation = await nvidia.explain_match(
        fixture_id=body.fixture_id,
        home_team=body.home_team or "Local",
        away_team=body.away_team or "Visitante",
        prob_home=body.prob_home,
        prob_draw=body.prob_draw,
        prob_away=body.prob_away,
        metrics=body.metrics,
    )

    return ExplainResponse(explanation=explanation)