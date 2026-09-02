"""Elo baseline trainer with deterministic grid search (Sprint 5.5)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np

from app.features.example import (
    AWAY_ELO_PRE_MATCH,
    ELO_DIFFERENCE,
    HOME_ELO_PRE_MATCH,
)
from app.prediction.artifacts import (
    ModelArtifact,
    ModelArtifactInputs,
    frozen_metrics,
    make_hyperparameters,
)
from app.prediction.contracts import FixtureFeatures, ModelName
from app.prediction.metrics.classification import log_loss
from app.prediction.models.elo_baseline import EloBaselineModel, EloBaselineParams

# Grids per spec §11
GRID_K: tuple[int, ...] = (15, 20, 25, 30)
GRID_HFA: tuple[int, ...] = (50, 65, 80)
GRID_BETA1: tuple[float, ...] = (0.0015, 0.0020, 0.0025)

ELO_FEATURES: tuple[str, ...] = (
    HOME_ELO_PRE_MATCH,
    AWAY_ELO_PRE_MATCH,
    ELO_DIFFERENCE,
)


def _extract_goals(
    targets_seq: Any,
) -> tuple[Any, Any]:
    """Extract home_goals (index 3) and away_goals (index 4) from targets.

    Accepts either Sequence[Sequence[int]] (trainer protocol) or
    list[dict] style. For Sprint 5.5 we assume protocol order:
    [home_win, draw, away_win, home_goals, away_goals].
    """
    # Try protocol list-of-list form
    try:
        arr = np.asarray(targets_seq, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 5:
            # Use columns 3 and 4
            hg = arr[:, 3]
            ag = arr[:, 4]
            return hg, ag
    except Exception:
        pass
    # Fallback: assume list of dicts with keys
    hg_list: list[float] = []
    ag_list: list[float] = []
    for t in targets_seq:
        if isinstance(t, dict):
            hg_list.append(float(t.get("home_goals", 0) or 0))
            ag_list.append(float(t.get("away_goals", 0) or 0))
        else:
            # Sequence form but not 2-D? Try index
            try:
                hg_list.append(float(t[3]))
                ag_list.append(float(t[4]))
            except Exception:
                hg_list.append(0.0)
                ag_list.append(0.0)
    return np.asarray(hg_list, dtype=np.float64), np.asarray(ag_list, dtype=np.float64)


def _compute_beta0s(
    train_targets: Any,
) -> tuple[float, float]:
    """Compute β0_home/β0_away as mean goals in train (global)."""
    hg, ag = _extract_goals(train_targets)
    # Filter finite
    hg = hg[np.isfinite(hg)]
    ag = ag[np.isfinite(ag)]
    if hg.size == 0 or ag.size == 0:
        # Default to league-average ~1.3 if no data
        return 0.2, 0.0
    beta0_home = float(np.log(max(float(hg.mean()), 0.1)))
    beta0_away = float(np.log(max(float(ag.mean()), 0.1)))
    return beta0_home, beta0_away


def _filter_valid_elo(
    features_list: list[FixtureFeatures],
    targets_list: list[Any] | None = None,
) -> tuple[list[FixtureFeatures], list[Any] | None, int]:
    """Drop rows where any Elo feature is NaN.

    Returns (filtered_features, filtered_targets_or_None, n_dropped).
    """
    kept_f: list[FixtureFeatures] = []
    kept_t: list[Any] = [] if targets_list is not None else []
    dropped = 0
    for idx, f in enumerate(features_list):
        try:
            i_home = f.feature_names.index(HOME_ELO_PRE_MATCH)
            i_away = f.feature_names.index(AWAY_ELO_PRE_MATCH)
            i_diff = f.feature_names.index(ELO_DIFFERENCE)
        except ValueError:
            dropped += 1
            continue
        vals = [float(f.feature_vector[i_home]), float(f.feature_vector[i_away]), float(f.feature_vector[i_diff])]  # type: ignore[index]
        if any(not np.isfinite(v) for v in vals):  # NaN or inf
            dropped += 1
            continue
        kept_f.append(f)
        if targets_list is not None:
            kept_t.append(targets_list[idx])
    if targets_list is None:
        return kept_f, None, dropped
    return kept_f, kept_t, dropped


class EloBaselineTrainer:
    """Trainer for Elo baseline — grid search over (K, HFA, β1) on val."""

    name: ModelName = ModelName.ELO_BASELINE

    def __init__(
        self,
        *,
        max_goals: int = 10,
        grid_k: tuple[int, ...] = GRID_K,
        grid_hfa: tuple[int, ...] = GRID_HFA,
        grid_beta1: tuple[float, ...] = GRID_BETA1,
    ) -> None:
        self.max_goals: int = int(max_goals)
        self.grid_k: tuple[int, ...] = grid_k
        self.grid_hfa: tuple[int, ...] = grid_hfa
        self.grid_beta1: tuple[float, ...] = grid_beta1

    def search(
        self,
        train_features: list[FixtureFeatures],
        train_targets: list[Any],
        val_features: list[FixtureFeatures],
        val_targets: list[Any],
    ) -> tuple[EloBaselineParams, float, dict[str, int]]:
        """Exhaustive lexicographic grid search, returns (best_params, best_logloss, info)."""
        # Compute beta0s from train (after dropping NaN)
        filt_train_f, filt_train_t, n_drop_train = _filter_valid_elo(train_features, train_targets)
        filt_val_f, filt_val_t, n_drop_val = _filter_valid_elo(val_features, val_targets)
        if len(filt_train_f) == 0:
            raise ValueError("train block empty after dropping NaN Elo rows")
        if len(filt_val_f) == 0:
            raise ValueError("val block empty after dropping NaN Elo rows")

        beta0_home, beta0_away = _compute_beta0s(filt_train_t)

        # Prepare val targets for log_loss: need 0/1/2 ints
        # val_targets are in protocol form [home_win,draw,away_win,home_goals,away_goals]
        # Convert to class label 0/1/2 via argmax of first 3
        val_labels: list[int] = []
        for t in filt_val_t:  # type: ignore[union-attr]
            if isinstance(t, (list, tuple, np.ndarray)):  # noqa: UP038
                # First 3 are one-hot
                arr = np.asarray(t[:3], dtype=np.float64)
                val_labels.append(int(np.argmax(arr)))
            elif isinstance(t, dict):
                if t.get("home_win") == 1:
                    val_labels.append(0)
                elif t.get("draw") == 1:
                    val_labels.append(1)
                else:
                    val_labels.append(2)
            else:
                val_labels.append(int(t))

        best_params: EloBaselineParams | None = None
        best_loss = float("inf")
        # Lexicographic order K → HFA → beta1
        for k in self.grid_k:
            for hfa in self.grid_hfa:
                for b1 in self.grid_beta1:
                    params = EloBaselineParams(
                        K=int(k),
                        HFA=float(hfa),
                        beta_0_home=float(beta0_home),
                        beta_0_away=float(beta0_away),
                        beta_1=float(b1),
                        max_goals=int(self.max_goals),
                    )
                    model = EloBaselineModel(params)
                    proba = model.predict_proba_array(filt_val_f)
                    # proba is (n,3)
                    loss = log_loss(val_labels, proba)
                    if loss < best_loss - 1e-12:  # strict < with tie → first wins
                        best_loss = float(loss)
                        best_params = params
        if best_params is None:
            raise ValueError("grid search found no valid params")
        info = {
            "train_dropouts": int(n_drop_train),
            "val_dropouts": int(n_drop_val),
            "train_n": int(len(filt_train_f)),
            "val_n": int(len(filt_val_f)),
        }
        return best_params, float(best_loss), info

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
        """Trainer entry point compatible with contracts.Trainer.

        If ``val_block`` is provided, performs grid search; otherwise fits
        with hyperparameters dict (expects K,HFA,beta_1) or defaults.
        """
        hyper = dict(hyperparameters or {})
        # If val provided, run search and override hyper
        if val_block is not None and val_targets is not None:
            best_params, best_loss, info = self.search(
                train_block, train_targets, val_block, val_targets
            )
            beta0_home = best_params.beta_0_home  # noqa: N806
            beta0_away = best_params.beta_0_away  # noqa: N806
            K = best_params.K  # noqa: N806
            HFA = best_params.HFA  # noqa: N806
            beta1 = best_params.beta_1  # noqa: N806
            metrics_extra = {"val_log_loss": float(best_loss)}
            metrics_extra.update({k: float(v) for k, v in info.items()})
        else:
            # No val → use provided hyper or defaults (first grid values)
            # Compute beta0s
            filt_train_f, filt_train_t, n_drop = _filter_valid_elo(train_block, train_targets)
            if len(filt_train_f) == 0:
                raise ValueError("train block empty after dropping NaN")
            beta0_home, beta0_away = _compute_beta0s(filt_train_t)  # noqa: N806
            K = int(hyper.get("K", GRID_K[0]))  # noqa: N806
            HFA = float(hyper.get("HFA", GRID_HFA[0]))  # noqa: N806
            beta1 = float(hyper.get("beta_1", GRID_BETA1[1]))  # noqa: N806
            best_params = EloBaselineParams(
                K=K, HFA=HFA, beta_0_home=beta0_home, beta_0_away=beta0_away, beta_1=beta1, max_goals=self.max_goals
            )
            metrics_extra = {"train_dropouts": float(n_drop)}

        hyper_final = {
            "K": int(best_params.K),
            "HFA": float(best_params.HFA),
            "beta_0_home": float(best_params.beta_0_home),
            "beta_0_away": float(best_params.beta_0_away),
            "beta_1": float(best_params.beta_1),
            "max_goals": int(best_params.max_goals),
        }

        # Build artifact
        now = datetime.now(UTC)
        cutoff = training_cutoff or (max(f.kickoff for f in train_block) if train_block else now)
        payload = {
            "model_name": str(self.name),
            "hyperparameters": hyper_final,
            "feature_names": list(ELO_FEATURES),
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        sha = hashlib.sha256(payload_bytes).hexdigest()

        inputs = ModelArtifactInputs(
            feature_names=ELO_FEATURES,
            feature_definition_version=feature_definition_version,
            head_features=None,
        )
        metrics = frozen_metrics(
            **{k: float(v) for k, v in metrics_extra.items()},
        )
        artifact = ModelArtifact(
            model_name=self.name,
            model_version=model_version,
            training_data_version=training_data_version,
            feature_definition_version=feature_definition_version,
            inputs=inputs,
            hyperparameters=make_hyperparameters(**hyper_final),
            training_cutoff=cutoff,
            created_at=now,
            metrics=metrics,
            fitted_seed=seed,
            payload_ref="elo_baseline.json",
            payload_sha256=sha,
        )
        # Keep last model for runner
        self._last_model = EloBaselineModel(best_params)
        self._last_artifact_sha = sha
        return artifact

    def get_model(self, artifact: ModelArtifact | None = None) -> EloBaselineModel:
        m = getattr(self, "_last_model", None)
        if m is not None:
            return m  # type: ignore[no-any-return]
        raise ValueError("no EloBaselineModel fitted yet (call train first)")


__all__ = ["ELO_FEATURES", "GRID_BETA1", "GRID_HFA", "GRID_K", "EloBaselineTrainer"]
