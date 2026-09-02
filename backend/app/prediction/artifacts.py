"""Immutable artifacts describing a sealed trained model.

These dataclasses are the **persistent certificate** of one model
training. Like Phase 4's :class:`app.dataset.manifest.DatasetManifest`,
they are frozen so once a trainer seals one it cannot be mutated
downstream — every Sprint 5.x component reads them by attribute access
only.

Two contracts live here:

* :class:`ModelArtifact` — describes one trained model. Persisted as
  ``data/models/<model_name>/<model_version>/manifest.json`` with the
  payload (weights/parameters) alongside it.
* :class:`PredictionRecord` — describes one prediction made by an
  artifact against one fixture. Persisted as
  ``data/predictions/<model_name>/<model_version>/<fixture_id>.json``.
  Immutable per contract: re-running with the same key is a no-op +
  WARNING (Sprint 5.8).

The lists / dict fields are intentionally typed as ``tuple`` and
``MappingProxyType`` respectively to make them unhashable-on-write so
even a copy-through cannot mutate the artifact.

See ``docs/PHASE_5.md`` §3 and §9 for the design rationale.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from app.prediction.contracts import MatchProbabilities, ModelName


@dataclass(frozen=True, slots=True)
class ModelArtifactInputs:
    """Explicit declaration of which features the artifact consumes.

    Defends against drift between Phase 4 and a future Phase-5
    iteration that moves a feature. Recorded on the artifact; the
    ``storage.artifacts`` load path can reject a request whose
    incoming ``FixtureFeatures.feature_names`` does not match.

    * ``feature_names`` — the tuple of feature names consumed (must
      be a subset of the dataset's ``FEATURE_NAMES``).
    * ``feature_definition_version`` — bucked from the Phase 4
      dataset manifest so a re-bump obsoletes the artifact loudly.
    * ``head_features`` — for Poisson (the only model with two heads):
      ``{"home": [...], "away": [...]}``. ``None`` for Elo baseline
      and Gradient Boosting.
    """

    feature_names: tuple[str, ...]
    feature_definition_version: str
    head_features: tuple[tuple[str, ...], tuple[str, ...]] | None = None


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Immutable certificate of one trained model.

    Once sealed by a trainer, instances are immutable. The field
    ``payload_sha256`` is the SHA-256 hex digest of the on-disk
    ``payload_ref`` bytes; ``payload_ref`` is a relative path under
    the model directory (NOT an absolute path — keeps artifacts
    relocatable).

    See ``docs/PHASE_5.md`` §3 for the manifest shape in JSON.
    """

    model_name: ModelName
    model_version: str
    training_data_version: str
    feature_definition_version: str
    inputs: ModelArtifactInputs
    hyperparameters: Mapping[str, Any]
    training_cutoff: datetime
    created_at: datetime
    metrics: Mapping[str, float]
    fitted_seed: int | None
    payload_ref: str
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (caller persists with ``json.dumps``).

        Datetimes are ISO-formatted (tz-aware by construction since
        the builder enforces UTC); mappings are deep-copied into
        plain dicts to keep the frozen original untouched.
        """
        return {
            "model_name": str(self.model_name),
            "model_version": self.model_version,
            "training_data_version": self.training_data_version,
            "feature_definition_version": self.feature_definition_version,
            "inputs": {
                "feature_names": list(self.inputs.feature_names),
                "feature_definition_version": self.inputs.feature_definition_version,
                "head_features": (
                    None
                    if self.inputs.head_features is None
                    else [
                        list(self.inputs.head_features[0]),
                        list(self.inputs.head_features[1]),
                    ]
                ),
            },
            "hyperparameters": dict(self.hyperparameters),
            "training_cutoff": self.training_cutoff.isoformat(),
            "created_at": self.created_at.isoformat(),
            "metrics": dict(self.metrics),
            "fitted_seed": self.fitted_seed,
            "payload_ref": self.payload_ref,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One prediction made by one artifact against one fixture.

    Persistence rule (Sprint 5.8 enforce): for the triple
    ``(model_name, model_version, fixture_id)`` there is at most one
    record on disk. Re-prediction is a no-op + WARNING.
    """

    model_name: ModelName
    model_version: str
    fixture_id: int
    kickoff: datetime
    probabilities: MatchProbabilities
    artifact_sha256: str
    predicted_at: datetime


def make_hyperparameters(**kwargs: Any) -> Mapping[str, Any]:
    """Factory helper that wraps kwargs in a frozen mapping view.

    Used by trainers to seal their hyperparameters at construction.
    The mapping is **read-only**: any attempt to mutate raises
    ``TypeError`` — a structural guard against silent drift between
    a trained artifact and a persisted one.
    """
    return MappingProxyType(dict(kwargs))


def frozen_metrics(**kwargs: float) -> Mapping[str, float]:
    """Like :func:`make_hyperparameters` but for the metrics dict.

    Kept separate so the type-checker can reject float|None values
    slipping in (every recorded metric is a real ``float``; when a
    fold has ``n_predictions == 0`` the runner MUST omit the metric
    rather than emit a ``None`` — that would ambiguity the manifest).
    """
    return MappingProxyType(dict(kwargs))


__all__ = [
    "ModelArtifact",
    "ModelArtifactInputs",
    "PredictionRecord",
    "frozen_metrics",
    "make_hyperparameters",
]
