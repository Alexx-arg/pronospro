# ruff: noqa: N806
"""Poisson trainer — two PoissonRegressor heads (Sprint 5.6)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.linear_model import PoissonRegressor  # type: ignore[import-untyped]

from app.features.example import FEATURE_NAMES
from app.prediction.artifacts import (
    ModelArtifact,
    ModelArtifactInputs,
    frozen_metrics,
    make_hyperparameters,
)
from app.prediction.contracts import FixtureFeatures, ModelName
from app.prediction.metrics.classification import log_loss
from app.prediction.models.poisson import (
    HEAD_AWAY_FEATURES,
    HEAD_HOME_FEATURES,
    PoissonModel,
)


def _extract_targets_goals(targets_seq: Any) -> tuple[Any, Any, Any]:
    """Extract home_goals, away_goals, and class label 0/1/2.

    Supports protocol list-of-lists [[hw,dr,aw,hg,ag]] or dicts.
    """
    hg_list: list[float] = []
    ag_list: list[float] = []
    labels: list[int] = []
    for t in targets_seq:
        if isinstance(t, dict):
            hg = float(t.get("home_goals", 0) or 0)
            ag = float(t.get("away_goals", 0) or 0)
            hw = t.get("home_win")
            dr = t.get("draw")
            if hw == 1:
                lbl = 0
            elif dr == 1:
                lbl = 1
            else:
                lbl = 2
        else:
            # Assume sequence [hw,dr,aw,hg,ag] or ndarray
            try:
                arr = np.asarray(t, dtype=np.float64)
                if arr.shape[0] >= 5:
                    hw, dr, _aw = float(arr[0]), float(arr[1]), float(arr[2])
                    hg = float(arr[3])
                    ag = float(arr[4])
                    # One-hot to label
                    if hw == 1:
                        lbl = 0
                    elif dr == 1:
                        lbl = 1
                    else:
                        lbl = 2
                    # Also handle case where t is already label int
                    if arr.shape[0] == 1:
                        lbl = int(arr[0])
                        hg = 0
                        ag = 0
                    hg_list.append(hg)
                    ag_list.append(ag)
                    labels.append(lbl)
                    continue
                # Fallback: t is int label
                lbl = int(arr[0]) if arr.size == 1 else int(t)
                hg = 0
                ag = 0
            except Exception:
                lbl = int(t)
                hg = 0
                ag = 0
        hg_list.append(hg)
        ag_list.append(ag)
        labels.append(lbl)
    return (
        np.asarray(hg_list, dtype=np.float64),
        np.asarray(ag_list, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )


def _build_matrices(
    features_list: list[FixtureFeatures],
    targets_goals_home: Any,
    targets_goals_away: Any,
    *,
    head: str,
    train_means: dict[str, float] | None = None,
    is_train: bool = True,
) -> tuple[Any, Any, dict[str, float], int]:
    """Build X, y for a head, handling missing.

    Returns (X, y, means, n_dropped).
    For train: drop rows with NaN in head's features.
    For test/val: impute NaN with train_means.
    """
    wanted = HEAD_HOME_FEATURES if head == "home" else HEAD_AWAY_FEATURES
    n_features = len(wanted)
    # Map feature name -> column index in wanted
    # Compute train means if not provided
    if train_means is None:
        # Compute means from available data (for train)
        means: dict[str, float] = {}
        # Collect per-feature values
        for _j, name in enumerate(wanted):
            vals: list[float] = []
            for f in features_list:
                try:
                    idx = f.feature_names.index(name)
                except ValueError:
                    continue
                v = float(f.feature_vector[idx])  # type: ignore[index]
                if np.isfinite(v):
                    vals.append(v)
            if vals:
                means[name] = float(np.mean(vals))
            else:
                means[name] = 0.0
    else:
        means = train_means

    rows: list[list[float]] = []
    ys: list[float] = []
    dropped = 0
    goals_arr = targets_goals_home if head == "home" else targets_goals_away
    for i, f in enumerate(features_list):
        # Build row
        row: list[float] = []
        has_nan = False
        for name in wanted:
            try:
                idx = f.feature_names.index(name)
            except ValueError:
                has_nan = True
                break
            v = float(f.feature_vector[idx])  # type: ignore[index]
            if not np.isfinite(v):
                has_nan = True
                if is_train:
                    break
                # For test: impute
                v = float(means.get(name, 0.0))
            row.append(v)
        if has_nan and is_train:
            dropped += 1
            continue
        # If is_train and we broke early, skip
        if len(row) != n_features:
            if is_train:
                continue
            # For test we already imputed, so len should be n_features
            # Pad with means if missing
            while len(row) < n_features:
                # need name for missing index
                name = wanted[len(row)]
                row.append(float(means.get(name, 0.0)))
        rows.append(row)
        ys.append(float(goals_arr[i]))

    if rows:
        X = np.asarray(rows, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
    else:
        X = np.empty((0, n_features), dtype=np.float64)
        y = np.empty((0,), dtype=np.float64)
    return X, y, means, dropped


class PoissonTrainer:
    """Trainer for Poisson model — two heads, fixed L2."""

    name: ModelName = ModelName.POISSON

    def __init__(
        self,
        *,
        regularization_l2: float = 0.01,
        max_goals: int = 10,
        max_iter: int = 1000,
    ) -> None:
        self.regularization_l2: float = float(regularization_l2)
        self.max_goals: int = int(max_goals)
        self.max_iter: int = int(max_iter)

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
        hyper = dict(hyperparameters or {})
        reg_l2 = float(hyper.get("regularization_l2", self.regularization_l2))
        max_goals = int(hyper.get("max_goals", self.max_goals))

        # Extract goals
        hg_train, ag_train, _ = _extract_targets_goals(train_targets)

        # Build matrices for train (drop)
        X_home, y_home, means_home, drop_home = _build_matrices(
            train_block, hg_train, ag_train, head="home", is_train=True
        )
        X_away, y_away, means_away, drop_away = _build_matrices(
            train_block, hg_train, ag_train, head="away", is_train=True
        )
        # For reporting, total dropouts is max of both? Use union count:
        # Count rows where either head would drop (global)
        # Compute global dropped as rows where either head had NaN
        # Already we have per-head drops, but global is at least max
        # We'll compute union by checking each original row for any head NaN
        # Simpler: use max
        n_drop_train = int(max(drop_home, drop_away))

        if X_home.shape[0] == 0 or X_away.shape[0] == 0:
            raise ValueError("train block empty after dropping NaN Poisson rows")

        # Fit regressors (deterministic, sklearn PoissonRegressor)
        reg_home = PoissonRegressor(
            alpha=reg_l2, max_iter=self.max_iter, fit_intercept=True
        )
        reg_away = PoissonRegressor(
            alpha=reg_l2, max_iter=self.max_iter, fit_intercept=True
        )
        reg_home.fit(X_home, y_home)
        reg_away.fit(X_away, y_away)

        # Optional val health-check (log_loss) — not for selection (D2 fixed)
        val_log_loss: float | None = None
        if val_block is not None and val_targets is not None:
            # Build val matrices with impute using train means
            hg_val, ag_val, labels_val = _extract_targets_goals(val_targets)
            # For val, we need to predict via model with impute
            # Create temporary model for prediction
            tmp_model = PoissonModel(
                reg_home, reg_away, means_home, means_away, max_goals=max_goals
            )
            # Predict on val with imputation
            proba = tmp_model.predict_proba_array(val_block)
            try:
                val_log_loss = float(log_loss(labels_val, proba))
            except Exception:
                val_log_loss = None

        # Build artifact
        now = datetime.now(UTC)
        cutoff = training_cutoff or (max(f.kickoff for f in train_block) if train_block else now)

        hyper_final = {
            "regularization_l2": float(reg_l2),
            "max_goals": int(max_goals),
            "nan_policy_train": "drop_row",
            "nan_policy_test": "impute_train_mean",
        }
        # Include fitted intercept/coef for reproducibility (not required but informative)
        # We store only hyperparams per spec, not full coef, but we need payload for artifact
        payload = {
            "model_name": str(self.name),
            "hyperparameters": hyper_final,
            "feature_names": list(FEATURE_NAMES),
            "head_home_features": list(HEAD_HOME_FEATURES),
            "head_away_features": list(HEAD_AWAY_FEATURES),
            "train_means_home": means_home,
            "train_means_away": means_away,
            # Store coef for debugging
            "coef_home": reg_home.coef_.tolist() if hasattr(reg_home, "coef_") else [],
            "coef_away": reg_away.coef_.tolist() if hasattr(reg_away, "coef_") else [],
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        sha = hashlib.sha256(payload_bytes).hexdigest()

        inputs = ModelArtifactInputs(
            feature_names=tuple(FEATURE_NAMES),
            feature_definition_version=feature_definition_version,
            head_features=(HEAD_HOME_FEATURES, HEAD_AWAY_FEATURES),
        )
        metrics_dict: dict[str, float] = {
            "train_block_dropouts": float(n_drop_train),
            "train_n": float(X_home.shape[0]),
        }
        if val_log_loss is not None:
            metrics_dict["val_log_loss"] = float(val_log_loss)

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
            payload_ref="poisson.json",
            payload_sha256=sha,
        )
        # Keep last fitted model for tests (artifact is frozen/slots, cannot attach)
        self._last_model = PoissonModel(reg_home, reg_away, means_home, means_away, max_goals=max_goals)
        self._last_artifact_sha = sha
        return artifact

    # Helper to retrieve the fitted PoissonModel from artifact (for tests)
    def get_model(self, artifact: ModelArtifact | None = None) -> PoissonModel:
        m = getattr(self, "_last_model", None)
        if m is not None:
            return m  # type: ignore[no-any-return]
        raise ValueError("no PoissonModel fitted yet (call train first)")


__all__ = ["PoissonTrainer"]
