"""NVIDIA NIM Explanation Service.

Provides LLM-powered tactical analysis using NVIDIA's hosted models
via OpenAI-compatible API.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from app.config import get_settings


class NvidiaExplanationService:
    """Service for generating match explanations via NVIDIA NIM."""

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.nvidia_api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = settings.nvidia_base_url or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.model = settings.nvidia_model or os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
        self.timeout = 30.0

    def is_configured(self) -> bool:
        """Check if NVIDIA API key is available."""
        return bool(self.api_key)

    async def explain_match(
        self,
        fixture_id: int,
        home_team: str,
        away_team: str,
        prob_home: float,
        prob_draw: float,
        prob_away: float,
        metrics: dict[str, Any] | None = None,
    ) -> str:
        """Generate a tactical analysis for a match.

        Args:
            fixture_id: Fixture identifier
            home_team: Home team name
            away_team: Away team name
            prob_home: Home win probability
            prob_draw: Draw probability
            prob_away: Away win probability
            metrics: Optional match metrics (xG, form, corners, cards, possession)

        Returns:
            Explanation text (max ~150 words)
        """
        if not self.is_configured():
            return "Explicación IA no disponible: NVIDIA_API_KEY no configurada."

        # Build metrics context
        metrics_text = ""
        if metrics:
            parts = []
            if metrics.get("home_form"):
                parts.append(f"{home_team} forma: {metrics['home_form']}")
            if metrics.get("away_form"):
                parts.append(f"{away_team} forma: {metrics['away_form']}")
            if metrics.get("home_xg") is not None:
                parts.append(f"xG {home_team}: {metrics['home_xg']:.2f} / {away_team}: {metrics.get('away_xg', 0):.2f}")
            if metrics.get("home_xga") is not None:
                parts.append(f"xGA {home_team}: {metrics['home_xga']:.2f} / {away_team}: {metrics.get('away_xga', 0):.2f}")
            if metrics.get("home_corners_avg") is not None:
                parts.append(f"Córners prom. {home_team}: {metrics['home_corners_avg']:.1f} / {away_team}: {metrics.get('away_corners_avg', 0):.1f}")
            if metrics.get("home_yellow_cards_avg") is not None:
                parts.append(f"Tarjetas amarillas prom. {home_team}: {metrics['home_yellow_cards_avg']:.1f} / {away_team}: {metrics.get('away_yellow_cards_avg', 0):.1f}")
            if metrics.get("home_possession_avg") is not None:
                parts.append(f"Posesión prom. {home_team}: {metrics['home_possession_avg']:.1f}% / {away_team}: {metrics.get('away_possession_avg', 0):.1f}%")
            if parts:
                metrics_text = "Métricas del partido: " + "; ".join(parts) + "."

        # Determine favorite
        max_prob = max(prob_home, prob_draw, prob_away)
        if max_prob == prob_home:
            favorite = home_team
        elif max_prob == prob_away:
            favorite = away_team
        else:
            favorite = "empate"

        prompt = (
            f"Eres un analista táctico de fútbol experto. Proporciona un análisis conciso "
            f"(máximo 150 palabras, en español) para el partido {home_team} vs {away_team}. "
            f"Predicción del modelo LightGBM: Local {prob_home:.1%}, Empate {prob_draw:.1%}, Visitante {prob_away:.1%}. "
            f"Favorito según modelo: {favorite}. "
            f"{metrics_text} "
            f"No inventes datos, no menciones formaciones hipotéticas, enfócate en lo que indican "
            f"las probabilidades y métricas reales. Sé directo y usa lenguaje técnico apropiado."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Eres un analista táctico de fútbol conciso y preciso. Máximo 150 palabras."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 250,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                if "choices" in data and data["choices"]:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    return content.strip()
                return "No se pudo generar la explicación."
        except httpx.HTTPStatusError as e:
            return f"Error NVIDIA NIM ({e.response.status_code}): {e.response.text[:200]}"
        except Exception as e:
            return f"Error generando explicación: {str(e)[:200]}"