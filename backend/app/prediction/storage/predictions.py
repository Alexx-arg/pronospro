"""PredictionStore — append-only per fixture (Sprint 5.8)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.prediction.artifacts import PredictionRecord
from app.prediction.contracts import ModelName
from app.prediction.storage import layout as layout_mod

logger = logging.getLogger(__name__)


class PredictionStore:
    """Persist PredictionRecord per (model_name, model_version, fixture_id)."""

    def __init__(self, base_path: Path) -> None:
        self.base_path: Path = Path(base_path)

    def save(self, record: PredictionRecord, *, overwrite: bool = False) -> Path:
        """Save record. If exists and overwrite=False, log warning and return existing path."""
        path = layout_mod.prediction_record_path(
            self.base_path,
            model_name=record.model_name,
            model_version=record.model_version,
            fixture_id=record.fixture_id,
        )
        if path.exists() and not overwrite:
            logger.warning("prediction already exists (immutable): %s", path)
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "model_name": str(record.model_name),
            "model_version": record.model_version,
            "fixture_id": record.fixture_id,
            "kickoff": record.kickoff.isoformat(),
            "probabilities": {
                "p_home_win": record.probabilities.p_home_win,
                "p_draw": record.probabilities.p_draw,
                "p_away_win": record.probabilities.p_away_win,
                "p_home_goals": record.probabilities.p_home_goals,
                "p_away_goals": record.probabilities.p_away_goals,
            },
            "artifact_sha256": record.artifact_sha256,
            "predicted_at": record.predicted_at.isoformat(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def load(
        self, model_name: ModelName, model_version: str, fixture_id: int
    ) -> dict[str, Any]:
        path = layout_mod.prediction_record_path(
            self.base_path,
            model_name=model_name,
            model_version=model_version,
            fixture_id=fixture_id,
        )
        if not path.is_file():
            raise FileNotFoundError(f"prediction not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


__all__ = ["PredictionStore"]
