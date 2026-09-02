# ruff: noqa: I001
"""Prediction orchestrator — Sprint 6.2.

Fat Service, Skinny Controller (D9.1). Reusa estrictamente FeatureBuilder
y políticas as-of de Fase 4 (D9.2). No reinventa construcción de features.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.prediction import MatchPredictionResponse
from app.dataset.loader import LoadedExample  # noqa: F401
from app.features.example import FEATURE_NAMES
from app.prediction.contracts import FixtureFeatures, MatchProbabilities

_import_error: Exception | None = None
try:
    from app.features.asof import (
        load_finished_fixtures_as_of,
        load_h2h_fixtures_as_of,
        load_team_stats_as_of,
    )
    from app.features.assembler import build_example
except Exception as exc:  # pragma: no cover — fallback for test collection with incompatible SQLAlchemy/Python
    _import_error = exc
    load_finished_fixtures_as_of = None  # type: ignore[assignment]
    load_h2h_fixtures_as_of = None  # type: ignore[assignment]
    load_team_stats_as_of = None  # type: ignore[assignment]

    def build_example(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        if _import_error is not None:
            raise _import_error
        raise RuntimeError("build_example not available")

# ---------------------------------------------------------------------------
# Exceptions — mapeadas a HTTP en el router
# ---------------------------------------------------------------------------


class FixtureNotFoundError(ValueError):
    """Fixture no existe en DB → 404."""


class InsufficientHistoryError(ValueError):
    """No hay historial mínimo para generar features → 422."""

    def __init__(self, msg: str = "Not enough historical data for feature generation") -> None:
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class PredictionService:
    """Orquestador de predicción por fixture_id."""

    async def predict_for_fixture(
        self,
        fixture_id: int,
        db_session: AsyncSession,
        model: Any,
    ) -> MatchPredictionResponse:
        """Pipeline completo: DB → FeatureBuilder → model.predict.

        Args:
            fixture_id: external_id del fixture a predecir.
            db_session: AsyncSession inyectada vía Depends.
            model: Predictor cargado en app.state.model (LightGBMModel, etc.)

        Returns:
            MatchPredictionResponse con probabilidades.

        Raises:
            FixtureNotFoundError: si la DB no encuentra el fixture.
            InsufficientHistoryError: si build_example falla por falta de historial.
        """
        # 1. Cargar fixture crudo
        fixture_orm = await self._load_fixture(db_session, fixture_id)
        if fixture_orm is None:
            raise FixtureNotFoundError(f"Fixture {fixture_id} not found")

        # 2. Cargar contexto histórico as-of kickoff (Fase 4)
        try:
            season_fixtures = await load_finished_fixtures_as_of(
                db_session,
                competition_id=fixture_orm.competition_id,
                season_id=fixture_orm.season_id,
            )
            stats_rows = await load_team_stats_as_of(
                db_session,
                competition_id=fixture_orm.competition_id,
                season_id=fixture_orm.season_id,
            )
            h2h_fixtures = await load_h2h_fixtures_as_of(
                db_session,
                home_team_id=fixture_orm.home_team_id,
                away_team_id=fixture_orm.away_team_id,
            )
        except Exception as exc:
            raise InsufficientHistoryError(str(exc)) from exc

        # 3. Construir HistoricalMatchExample vía FeatureBuilder (as-of)
        # Necesitamos proyectar fixture_orm a FixtureRow. Reusamos la proyección
        # mínima necesaria para build_example (no reinventamos).
        from app.features.rows import FixtureRow

        fixture_row = FixtureRow(
            fixture_id=fixture_orm.external_id,
            competition_id=fixture_orm.competition_id,
            season_id=fixture_orm.season_id,
            home_team_id=fixture_orm.home_team_id,
            away_team_id=fixture_orm.away_team_id,
            kickoff_time=fixture_orm.kickoff_time,
            home_goals=getattr(fixture_orm, "home_goals", 0) or 0,
            away_goals=getattr(fixture_orm, "away_goals", 0) or 0,
            matchday=getattr(fixture_orm, "matchday", None),
            venue=getattr(fixture_orm, "venue", None),
        )

        try:
            example = build_example(
                fixture=fixture_row,
                season_fixtures=season_fixtures,
                stats_rows=stats_rows,
                h2h_fixtures=h2h_fixtures,
            )
        except Exception as exc:
            raise InsufficientHistoryError(f"Feature generation failed: {exc}") from exc

        # 4. Convertir HistoricalMatchExample → FixtureFeatures (vector 66)
        # Reutiliza lógica de vector.py: None→NaN, orden FEATURE_NAMES
        vec = np.array(
            [example.features.get(name) if example.features.get(name) is not None else float("nan") for name in FEATURE_NAMES],
            dtype=np.float32,
        )
        # Validación mínima: si faltan datos core (ej. elo), consideramos insuficiente
        # Si todas son NaN, es señal de historial nulo
        if np.all(np.isnan(vec)):
            raise InsufficientHistoryError("Not enough historical data for feature generation: all features are NaN")

        features = FixtureFeatures(
            fixture_id=example.fixture_id,
            kickoff=example.kickoff,
            feature_vector=vec,
            feature_names=tuple(FEATURE_NAMES),
        )

        # 5. Inferencia pura (modelo ya en RAM, sin I/O)
        try:
            prob: MatchProbabilities = model.predict(features)
        except Exception as exc:
            raise InsufficientHistoryError(f"Model prediction failed: {exc}") from exc

        # 6. Respuesta Pydantic
        from datetime import UTC

        return MatchPredictionResponse(
            fixture_id=features.fixture_id,
            prob_home=float(prob.p_home_win),
            prob_draw=float(prob.p_draw),
            prob_away=float(prob.p_away_win),
            model_version=getattr(model, "model_version", "production"),
            predicted_at=datetime.now(UTC),
        )

    async def _load_fixture(self, db_session: AsyncSession, fixture_id: int) -> Any | None:
        """Carga fixture por external_id. Extraíble para mock en tests."""
        from app.repositories.fixture import FixtureRepository

        repo = FixtureRepository(db_session)
        return await repo.find_by_external_id(fixture_id)


__all__ = ["FixtureNotFoundError", "InsufficientHistoryError", "PredictionService"]
