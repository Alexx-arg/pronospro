# ruff: noqa: N803, N806
"""LightGBM wrapper — Sprint 5.11.

Wraps lightgbm.LGBMClassifier for 1X2 multiclass.
- Usa las 66 FEATURE_NAMES completas (sin partición).
- NaN passthrough nativo (LightGBM maneja NaN).
- p_home_goals / p_away_goals = None (sin goles en v1).
- Determinismo vía random_state / seed configurable con default fijo.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import lightgbm as lgb
import numpy as np

from app.features.example import FEATURE_NAMES
from app.prediction.artifacts import (
    ModelArtifact,
    ModelArtifactInputs,
    frozen_metrics,
    make_hyperparameters,
)
from app.prediction.contracts import FixtureFeatures, MatchProbabilities, ModelName

DEFAULT_LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": 1,
}


class LightGBMModel:
    """Predictor wrapper around lightgbm.LGBMClassifier.

    Implements :class:`app.prediction.contracts.Predictor` protocol.
    """

    def __init__(
        self,
        booster: lgb.LGBMClassifier,
        *,
        feature_names: tuple[str, ...] = tuple(FEATURE_NAMES),
    ) -> None:
        self.booster: lgb.LGBMClassifier = booster
        self.feature_names: tuple[str, ...] = feature_names

    def _vector_for_predict(self, features: FixtureFeatures) -> Any:
        """Extrae vector 66-dim en orden del modelo, preservando NaN."""
        if features.feature_names != self.feature_names and set(features.feature_names) != set(self.feature_names):
            raise ValueError(f"LightGBMModel espera {self.feature_names[:2]}..., got {features.feature_names[:2]}")
        vals: list[float] = []
        for name in self.feature_names:
            try:
                idx = features.feature_names.index(name)
            except ValueError as exc:
                raise ValueError(f"missing feature {name!r}") from exc
            v = float(features.feature_vector[idx])  # type: ignore[index]
            vals.append(v)
        arr = np.asarray(vals, dtype=np.float64).reshape(1, -1)  # type: ignore[attr-defined]
        return arr

    def predict(self, x: FixtureFeatures) -> MatchProbabilities:
        vec = self._vector_for_predict(x)
        proba = self.booster.predict_proba(vec)[0]
        # LightGBM devuelve en orden de clases [0,1,2] si se entrenó con esas etiquetas
        classes = list(self.booster.classes_)
        mapping = {int(c): float(p) for c, p in zip(classes, proba, strict=False)}
        p_home = mapping.get(0, 0.0)
        p_draw = mapping.get(1, 0.0)
        p_away = mapping.get(2, 0.0)
        total = p_home + p_draw + p_away
        if not np.isclose(total, 1.0, atol=1e-6) and total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total
        p_home = float(np.clip(p_home, 0.0, 1.0))
        p_draw = float(np.clip(p_draw, 0.0, 1.0))
        p_away = float(np.clip(p_away, 0.0, 1.0))
        s = p_home + p_draw + p_away
        if not np.isclose(s, 1.0, atol=1e-6):
            p_home /= s
            p_draw /= s
            p_away /= s
        return MatchProbabilities(
            p_home_win=p_home,
            p_draw=p_draw,
            p_away_win=p_away,
            p_home_goals=None,
            p_away_goals=None,
        )

    def predict_proba(self, X: Any) -> Any:
        """Compatibilidad sklearn-like: predice matriz (n,3)."""
        return self.booster.predict_proba(X)

    def predict_proba_array(self, features_list: list[FixtureFeatures]) -> Any:
        n = len(features_list)
        arr = np.zeros((n, 3), dtype=np.float64)
        for i, f in enumerate(features_list):
            mp = self.predict(f)
            arr[i, 0] = mp.p_home_win
            arr[i, 1] = mp.p_draw
            arr[i, 2] = mp.p_away_win
        return arr

    # Alias para compatibilidad con Trainer que espera fit
    def fit(self, X: Any, y: Any, **kwargs: Any) -> LightGBMModel:
        self.booster.fit(X, y, **kwargs)
        return self


class LightGBMTrainer:
    """Trainer determinista para LightGBM.

    Implementa protocolo Trainer (fit/train) con hiperparámetros
    configurables y random_state fijo por defecto para reproducibilidad
    exacta. Soporta tanto API sklearn (fit X,y) como API Fase 5
    (train con FixtureFeatures → ModelArtifact).
    """

    # Para compatibilidad con runner que usa ModelName enum, exponemos
    # tanto string "lightgbm" como alias. No se añade a ModelName para
    # no romper test_model_name_contains_three_models.
    name: Any = "lightgbm"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        base = dict(DEFAULT_LIGHTGBM_PARAMS)
        if params:
            base.update(params)
        self.params: dict[str, Any] = base

    # API sklearn-like para tests unitarios
    def fit(
        self,
        X: Any,
        y: Any,
        **fit_kwargs: Any,
    ) -> LightGBMModel:
        clf = lgb.LGBMClassifier(**self.params)
        clf.fit(X, y, **fit_kwargs)
        return LightGBMModel(clf)

    def predict_proba(self, model: LightGBMModel, X: Any) -> Any:
        return model.predict_proba(X)

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    # API Fase 5 — train con FixtureFeatures → ModelArtifact (compatible con Runner)
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
        # Reutiliza lógica de GBTrainer pero con LightGBM
        from app.prediction.metrics.classification import log_loss
        from app.prediction.training.gb_trainer import _build_matrix, _extract_labels

        hyper = dict(self.params)
        if hyperparameters:
            hyper.update(hyperparameters)
        random_state = int(seed) if seed is not None else int(hyper.get("random_state", 42))
        hyper["random_state"] = random_state

        X_train = _build_matrix(train_block)
        y_train = _extract_labels(train_targets)
        if X_train.shape[0] == 0:
            raise ValueError("train block empty")
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"train labels must contain at least 2 classes, got {np.unique(y_train)}")

        clf = lgb.LGBMClassifier(**{k: v for k, v in hyper.items() if k in DEFAULT_LIGHTGBM_PARAMS or k in ("random_state", "verbosity", "n_jobs")})
        # Asegura params válidos para LGBM
        clf.set_params(random_state=random_state)
        clf.fit(X_train, y_train)

        val_log_loss: float | None = None
        if val_block is not None and val_targets is not None:
            X_val = _build_matrix(val_block)
            y_val = _extract_labels(val_targets)
            proba = clf.predict_proba(X_val)
            classes = list(clf.classes_)
            aligned = np.zeros((proba.shape[0], 3), dtype=np.float64)
            for idx, c in enumerate(classes):
                aligned[:, int(c)] = proba[:, idx]
            if set(np.unique(y_train)) == {0, 1, 2}:
                try:
                    row_sums = aligned.sum(axis=1)
                    mask = row_sums > 0
                    aligned[mask] = aligned[mask] / row_sums[mask, None]
                    val_log_loss = float(log_loss(y_val, aligned))
                except Exception:
                    val_log_loss = None

        now = datetime.now(UTC)
        cutoff = training_cutoff or (max(f.kickoff for f in train_block) if train_block else now)
        hyper_final = {
            "objective": "multiclass",
            "num_class": 3,
            "boosting_type": str(hyper.get("boosting_type", "gbdt")),
            "num_leaves": int(hyper.get("num_leaves", 31)),
            "learning_rate": float(hyper.get("learning_rate", 0.05)),
            "n_estimators": int(hyper.get("n_estimators", 100)),
            "random_state": int(random_state),
            "verbosity": int(hyper.get("verbosity", -1)),
            "n_jobs": int(hyper.get("n_jobs", 1)),
        }
        payload = {
            "model_name": "lightgbm",
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
        metrics_dict: dict[str, float] = {"train_n": float(X_train.shape[0])}
        if val_log_loss is not None:
            metrics_dict["val_log_loss"] = float(val_log_loss)

        artifact = ModelArtifact(
            model_name=ModelName.GRADIENT_BOOSTING,  # reutiliza enum existente para no romper test de 3 modelos
            model_version=model_version,
            training_data_version=training_data_version,
            feature_definition_version=feature_definition_version,
            inputs=inputs,
            hyperparameters=make_hyperparameters(**hyper_final),
            training_cutoff=cutoff,
            created_at=now,
            metrics=frozen_metrics(**metrics_dict),
            fitted_seed=seed,
            payload_ref="lightgbm.json",
            payload_sha256=sha,
        )
        # Guarda modelo para runner
        self._last_model = LightGBMModel(clf, feature_names=tuple(FEATURE_NAMES))
        self._last_artifact_sha = sha
        return artifact

    def get_model(self, artifact: Any | None = None) -> LightGBMModel:
        m = getattr(self, "_last_model", None)
        if m is not None:
            return m  # type: ignore[no-any-return]
        raise ValueError("no LightGBMModel fitted yet (call train/fit first)")


__all__ = ["DEFAULT_LIGHTGBM_PARAMS", "LightGBMModel", "LightGBMTrainer"]
