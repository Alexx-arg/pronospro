"""ModelArtifactStore — bin + manifest, base_path injectable (Sprint 5.8)."""

from __future__ import annotations

import json
from pathlib import Path

from app.prediction.artifacts import ModelArtifact
from app.prediction.contracts import ModelName
from app.prediction.storage import layout as layout_mod


class ModelArtifactStore:
    """Persist / load ModelArtifact + payload bytes."""

    def __init__(self, base_path: Path) -> None:
        self.base_path: Path = Path(base_path)

    def save(
        self,
        artifact: ModelArtifact,
        payload_bytes: bytes,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist artifact manifest + payload.bin.

        Returns manifest path. Raises FileExistsError if exists and not overwrite.
        """
        manifest_path = layout_mod.artifact_manifest_path(
            self.base_path, model_name=artifact.model_name, model_version=artifact.model_version
        )
        bin_path = layout_mod.artifact_bin_path(
            self.base_path, model_name=artifact.model_name, model_version=artifact.model_version
        )
        if not overwrite and manifest_path.exists():
            raise FileExistsError(f"artifact already exists: {manifest_path}")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # Write payload first so manifest sha can be validated on load
        bin_path.write_bytes(payload_bytes)
        manifest_path.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")
        return manifest_path

    def load(self, model_name: ModelName, model_version: str) -> tuple[ModelArtifact, bytes]:
        manifest_path = layout_mod.artifact_manifest_path(
            self.base_path, model_name=model_name, model_version=model_version
        )
        bin_path = layout_mod.artifact_bin_path(
            self.base_path, model_name=model_name, model_version=model_version
        )
        if not manifest_path.is_file() or not bin_path.is_file():
            raise FileNotFoundError(f"artifact not found: {model_name}/{model_version}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Reconstruct ModelArtifact (light — we don't validate all fields strictly here)
        # Use from_dict-like via manual construction

        # Simpler: reconstruct via ModelArtifact fields directly from dict
        # Since ModelArtifact is frozen, we need to handle datetime parsing
        from datetime import datetime

        def _dt(s: str) -> datetime:
            return datetime.fromisoformat(s)

        from types import MappingProxyType

        from app.prediction.artifacts import ModelArtifactInputs

        inputs_raw = data["inputs"]
        inputs = ModelArtifactInputs(
            feature_names=tuple(inputs_raw["feature_names"]),
            feature_definition_version=inputs_raw["feature_definition_version"],
            head_features=(
                None
                if inputs_raw["head_features"] is None
                else (tuple(inputs_raw["head_features"][0]), tuple(inputs_raw["head_features"][1]))
            ),
        )
        artifact = ModelArtifact(
            model_name=ModelName(data["model_name"]),
            model_version=data["model_version"],
            training_data_version=data["training_data_version"],
            feature_definition_version=data["feature_definition_version"],
            inputs=inputs,
            hyperparameters=MappingProxyType(dict(data["hyperparameters"])),
            training_cutoff=_dt(data["training_cutoff"]),
            created_at=_dt(data["created_at"]),
            metrics=MappingProxyType(dict(data["metrics"])),
            fitted_seed=data["fitted_seed"],
            payload_ref=data["payload_ref"],
            payload_sha256=data["payload_sha256"],
        )
        payload = bin_path.read_bytes()
        return artifact, payload


__all__ = ["ModelArtifactStore"]
