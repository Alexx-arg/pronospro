# ruff: noqa: N806
"""Gradient Boosting trainer — HistGradientBoostingClassifier (Sprint 5.7)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]

from app.features.example import FEATURE_NAMES
from app.prediction.artifacts import (
    ModelArtifact,
    ModelArtifactInputs,
    frozen_metrics,
    make_hyperparameters,
)
from app.prediction.contracts import FixtureFeatures, ModelName
from app.prediction.metrics.classification import log_loss
from app.prediction.models.gradient_boosting import GradientBoostingModel

# v1 fixed hyperparameters (serializable, deterministic)
GB_V1_HYPERPARAMS: dict[str, Any] = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "l2_regularization": 0.0,
    "early_stopping": False,
    "loss": "log_loss",
}


def _extract_labels(targets_seq: Any) -> Any:
    """Extract 0/1/2 labels from trainer protocol targets."""
    labels: list[int] = []
    for t in targets_seq:
        if isinstance(t, dict):
            if t.get("home_win") == 1:
                labels.append(0)
            elif t.get("draw") == 1:
                labels.append(1)
            else:
                labels.append(2)
        else:
            try:
                arr = np.asarray(t, dtype=np.float64)
                if arr.shape[0] >= 3:
                    # One-hot
                    hw, dr = float(arr[0]), float(arr[1])
                    if hw == 1:
                        labels.append(0)
                    elif dr == 1:
                        labels.append(1)
                    else:
                        labels.append(2)
                    continue
                if arr.size == 1:
                    labels.append(int(arr[0]))
                    continue
                labels.append(int(t))
            except Exception:
                labels.append(int(t))
    return np.asarray(labels, dtype=np.int64)


def _build_matrix(features_list: list[FixtureFeatures]) -> Any:
    """Build (n,66) float64 matrix preserving NaN."""
    n = len(features_list)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    for i, f in enumerate(features_list):
        # Assume feature_names == FEATURE_NAMES (validated in model)
        # Copy in order of FEATURE_NAMES
        for j, name in enumerate(FEATURE_NAMES):
            try:
                idx = f.feature_names.index(name)
            except ValueError:
                X[i, j] = np.nan
                continue
            v = float(f.feature_vector[idx])  # type: ignore[index]
            X[i, j] = v
    return X


class GradientBoostingTrainer:
    """Trainer for Gradient Boosting — HistGradientBoostingClassifier."""

    name: ModelName = ModelName.GRADIENT_BOOSTING

    def __init__(
        self,
        *,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        base = dict(GB_V1_HYPERPARAMS)
        if hyperparameters:
            base.update(hyperparameters)
        self.hyperparameters: dict[str, Any] = base

    def train(
        self,
        train_block: list[FixtureFeatures],
        train_targets: list[Any],
        hyperparameters: dict[str, Any] | None = None,
        seed: int | None = None,
        *,
        val_block: list[FixtureFeatures] | None = None,
        val_targets: list[Any] | None = None,
        model_version: str = "v001",
        training_data_version: str = "v001",
        feature_definition_version: str = "fd_v1",
        training_cutoff: datetime | None = None,
    ) -> ModelArtifact:
        hyper = dict(self.hyperparameters)
        if hyperparameters:
            hyper.update(hyperparameters)
        # Seed handling for determinism
        random_state = int(seed) if seed is not None else int(hyper.get("random_state", 42))
        # Build matrices
        X_train = _build_matrix(train_block)
        y_train = _extract_labels(train_targets)
        if X_train.shape[0] == 0:
            raise ValueError("train block empty")
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"train labels must contain at least 2 classes, got {np.unique(y_train)}")

        # HistGradientBoostingClassifier with fixed v1 hyperparams
        clf = HistGradientBoostingClassifier(
            learning_rate=float(hyper.get("learning_rate", 0.1)),
            max_iter=int(hyper.get("max_iter", 100)),
            max_leaf_nodes=int(hyper.get("max_leaf_nodes", 31)) if hyper.get("max_leaf_nodes") is not None else 31,
            max_depth=hyper.get("max_depth"),
            l2_regularization=float(hyper.get("l2_regularization", 0.0)),
            loss="log_loss",
            early_stopping=False,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        # Optional val health-check (no selection, just metric)
        val_log_loss: float | None = None
        if val_block is not None and val_targets is not None:
            X_val = _build_matrix(val_block)
            y_val = _extract_labels(val_targets)
            proba = clf.predict_proba(X_val)
            # Ensure proba order matches classes [0,1,2] — HistGB returns sorted classes
            # If missing class, need to align
            classes = list(clf.classes_)
            # Build aligned proba (n,3)
            aligned = np.zeros((proba.shape[0], 3), dtype=np.float64)
            for idx, c in enumerate(classes):
                aligned[:, int(c)] = proba[:, idx]
            # For missing classes, prob remains 0 — but need to renormalize?
            # If only 2 classes seen in train, missing class prob 0 is okay for log_loss? It would be 0 for true class that is missing → log(0) inf
            # So we smooth missing class to small epsilon before log_loss
            # Instead, we compute log_loss with central validator which will handle, but missing class will cause sum !=1?
            # For health-check, if train missing a class, we skip val log_loss
            if set(np.unique(y_train)) == {0, 1, 2}:
                try:
                    # Ensure rows sum to 1 (missing class 0 prob case would sum <1)
                    row_sums = aligned.sum(axis=1)
                    # If sums <1 due missing, distribute remainder uniformly to missing?
                    # For now, renormalize
                    mask = row_sums > 0
                    aligned[mask] = aligned[mask] / row_sums[mask, None]
                    val_log_loss = float(log_loss(y_val, aligned))
                except Exception:
                    val_log_loss = None

        # Artifact
        now = datetime.now(UTC)
        cutoff = training_cutoff or (max(f.kickoff for f in train_block) if train_block else now)

        hyper_final = {
            "learning_rate": float(hyper.get("learning_rate", 0.1)),
            "max_iter": int(hyper.get("max_iter", 100)),
            "max_leaf_nodes": int(hyper.get("max_leaf_nodes", 31)) if hyper.get("max_leaf_nodes") is not None else 31,
            "max_depth": hyper.get("max_depth"),
            "l2_regularization": float(hyper.get("l2_regularization", 0.0)),
            "loss": "log_loss",
            "early_stopping": False,
            "random_state": int(random_state),
        }

        payload = {
            "model_name": str(self.name),
            "hyperparameters": hyper_final,
            "feature_names": list(FEATURE_NAMES),
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        sha = hashlib.sha256(payload_bytes).hexdigest()

        inputs = ModelArtifactInputs(
            feature_names=tuple(FEATURE_NAMES),
            feature_definition_version=feature_definition_version,
            head_features=None,
        )
        metrics_dict: dict[str, float] = {
            "train_n": float(X_train.shape[0]),
        }
        if val_log_loss is not None:
            metrics_dict["val_log_loss"] = float(val_log_loss)
        # Feature importance top10 placeholder (not computed here)
        artifact = ModelArtifact(
            model_name=self.name,
            model_version=model_version,
            training_data_version=training_data_version,
            feature_definition_version=feature_definition_version,
            inputs=inputs,
            hyperparameters=make_hyperparameters(**hyper_final),
            training_cutoff=cutoff,
            created_at=now,
            metrics=frozen_metrics(**metrics_dict),
            fitted_seed=seed,
            payload_ref="gradient_boosting.json",
            payload_sha256=sha,
        )
        # Keep last model for tests
        self._last_model = GradientBoostingModel(clf, feature_names=tuple(FEATURE_NAMES))
        self._last_artifact_sha = sha
        return artifact

    def get_model(self, artifact: ModelArtifact | None = None) -> GradientBoostingModel:
        m = getattr(self, "_last_model", None)
        if m is not None:
            return m  # type: ignore[no-any-return]
        raise ValueError("no GradientBoostingModel fitted yet (call train first)")


__all__ = ["GB_V1_HYPERPARAMS", "GradientBoostingTrainer"]
