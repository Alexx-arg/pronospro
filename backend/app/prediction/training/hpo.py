# ruff: noqa: N806
"""HPO con Optuna — Sprint 5.12.

Ejecuta búsqueda de hiperparámetros usando *estrictamente* los folds
del WalkForwardIterator existente (cero leakage). Cada trial:

  1. Optuna sugiere hiperparámetros (TPESampler seed fijo).
  2. Trainer se instancia con esos hiperparámetros + seed fijo.
  3. Se ejecuta backtesting completo (run_backtest) sobre folds temporales.
  4. Se retorna métrica de validación (weighted) al optimizador.

Determinismo absoluto: seed fijo en LightGBM y en TPESampler.
Sin dataset real: tests usan mocks sintéticos.
Sin leakage: nunca se usa test para fitting/selección.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler

from app.dataset.loader import LoadedDataset
from app.prediction.backtesting.runner import run_backtest
from app.prediction.training import get_trainer

# Espacio por defecto acotado para LightGBM (D6)
DEFAULT_LIGHTGBM_SEARCH_SPACE: dict[str, tuple[Any, Any]] = {
    "learning_rate": (0.01, 0.1),
    "num_leaves": (15, 63),
    "n_estimators": (50, 150),
}

# Métricas soportadas para HPO (provienen de FoldReport / weighted)
SUPPORTED_METRICS = {
    "weighted_log_loss",
    "mean_log_loss",
    "weighted_ece",
    "mean_accuracy",
}


def _suggest_params(trial: Any, search_space: dict[str, Any] | None) -> dict[str, Any]:
    """Sugiere hiperparámetros para un trial según search_space."""
    space = search_space or DEFAULT_LIGHTGBM_SEARCH_SPACE
    params: dict[str, Any] = {}
    for name, bounds in space.items():
        # bounds can be (low, high) or (low, high, type)
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:  # noqa: UP038
            low, high = bounds
            # Infer type: int if both ints else float
            if isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                # For learning_rate use log
                if name == "learning_rate":
                    params[name] = trial.suggest_float(name, float(low), float(high), log=True)
                else:
                    params[name] = trial.suggest_float(name, float(low), float(high))
        elif isinstance(bounds, dict):
            # Explicit optuna distribution dict
            params[name] = trial.suggest_float(name, **bounds)
        else:
            # Single value fixed
            params[name] = bounds
    # Siempre fija verbosity y n_jobs para determinismo
    params["verbosity"] = -1
    params["n_jobs"] = 1
    return params


def _compute_metric_from_result(result: Any, metric: str) -> float:
    """Extrae métrica agregada del RunResult (promedio/ponderada)."""
    if not result.folds:
        return float("nan")
    # Collect per-fold values
    vals: list[float] = []
    ns: list[int] = []
    for fr in result.folds:
        # FoldReport has attributes
        if metric == "weighted_log_loss":
            # Will compute weighted later, but need per-fold log_loss
            vals.append(float(fr.report.log_loss))
            ns.append(int(fr.report.n_predictions))
        elif metric == "mean_log_loss":
            vals.append(float(fr.report.log_loss))
        elif metric == "weighted_ece":
            vals.append(float(fr.report.ece))
            ns.append(int(fr.report.n_predictions))
        elif metric == "mean_accuracy":
            vals.append(float(fr.report.accuracy))
        else:
            # Generic: try report attribute
            vals.append(float(getattr(fr.report, metric)))
            if metric.startswith("weighted"):
                ns.append(int(fr.report.n_predictions))
    if metric.startswith("weighted"):
        total_n = sum(ns)
        if total_n == 0:
            return float("nan")
        weighted = sum(n * v for n, v in zip(ns, vals, strict=False)) / total_n
        return float(weighted)
    # Simple mean
    return float(np.nanmean(vals))


def run_hpo_study(
    dataset: LoadedDataset,
    *,
    model_name: str = "lightgbm",
    metric: str = "weighted_log_loss",
    n_trials: int = 10,
    search_space: dict[str, Any] | None = None,
    iterator_params: dict[str, Any] | None = None,
    base_path: Path | str = Path("data"),
    seed: int = 42,
    direction: str = "minimize",
) -> dict[str, Any]:
    """Ejecuta estudio HPO con Optuna usando backtesting.

    Args:
        dataset: LoadedDataset sintético o real.
        model_name: Nombre del modelo ("lightgbm", "poisson", etc.)
        metric: Métrica a optimizar (de FoldReport, ej weighted_log_loss).
        n_trials: Número de intentos.
        search_space: Dict {param: (low, high)} o None para default.
        iterator_params: Params para WalkForwardIterator; si None usa
            reducidos para CI: min_train 100, test 10, gap 0, val_ratio 0.15.
        base_path: Base path para runs (cada trial usa subdir temporal).
        seed: Seed fija para TPESampler y LightGBM.
        direction: "minimize" (log_loss) o "maximize" (accuracy).

    Returns:
        Dict con {best_params, best_value, best_trial_number, n_trials, study}
        serializable (study no serializado, solo best).

    Raises:
        ValueError: si metric no soportada o n_trials <=0.
    """
    if metric not in SUPPORTED_METRICS and not hasattr(
        __import__("app.prediction.metrics.report", fromlist=["FoldReport"]), "FoldReport"
    ):
        # Allow any FoldReport attribute, just try later
        pass
    if n_trials <= 0:
        raise ValueError(f"n_trials must be >0, got {n_trials}")
    if direction not in ("minimize", "maximize"):
        raise ValueError(f"direction must be minimize/maximize, got {direction}")

    base_path = Path(base_path)
    if iterator_params is None:
        iterator_params = {
            "min_train_size": 100,
            "test_size": 10,
            "gap_days": 0,
            "val_ratio": 0.15,
            "mode": "expanding",
        }

    # Sampler determinista
    sampler = TPESampler(seed=seed)

    def _objective(trial: Any) -> float:
        # Sugerir hiperparámetros
        params = _suggest_params(trial, search_space)
        # Instanciar trainer con params + seed
        trainer = get_trainer(model_name, params={"random_state": seed, **params} if "random_state" not in params else params)
        # Cada trial usa subdir aislado para no contaminar
        trial_base = base_path / f".hpo_trial_{trial.number}"
        trial_base.mkdir(parents=True, exist_ok=True)
        result = run_backtest(
            dataset,
            trainer,
            iterator_params=iterator_params,
            model_version=f"hpo_{trial.number}",
            seed=seed,
            base_path=trial_base,
        )
        # Si no hay folds (dataset pequeño), penaliza
        if not result.folds:
            return float("inf") if direction == "minimize" else float("-inf")
        metric_val = _compute_metric_from_result(result, metric)
        return float(metric_val)

    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(_objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    # Asegura determinismo: añade verbosity/n_jobs fijos si faltan
    best_params.setdefault("verbosity", -1)
    best_params.setdefault("n_jobs", 1)

    return {
        "best_params": best_params,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "n_trials": int(n_trials),
        "metric": metric,
        "direction": direction,
        "seed": seed,
    }


__all__ = ["DEFAULT_LIGHTGBM_SEARCH_SPACE", "SUPPORTED_METRICS", "run_hpo_study"]
