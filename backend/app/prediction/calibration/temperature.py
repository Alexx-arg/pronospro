# ruff: noqa: N803, N806
"""Temperature scaling calibrator (Sprint 5.4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.prediction.calibration.base import (
    CLIP_EPS,
    _array_to_probas,
    _clip_for_log,
    _probas_to_array,
    _softmax,
    _validate_probas_and_targets,
)
from app.prediction.contracts import CalibratorKind, MatchProbabilities

T_MIN: float = 0.1
T_MAX: float = 10.0
DEFAULT_T: float = 1.0


class TemperatureCalibrator:
    """Temperature scaling: ``p_cal = softmax(log(p)/T)``."""

    kind: CalibratorKind = CalibratorKind.TEMPERATURE

    def __init__(
        self,
        *,
        t_min: float = T_MIN,
        t_max: float = T_MAX,
        init_t: float = DEFAULT_T,
    ) -> None:
        if not (0 < t_min < t_max):
            raise ValueError(f"t_min/t_max must satisfy 0 < t_min < t_max, got {t_min}, {t_max}")
        if not (t_min <= init_t <= t_max):
            raise ValueError(f"init_t {init_t} must be in [t_min, t_max]")
        self.t_min: float = float(t_min)
        self.t_max: float = float(t_max)
        self.temperature: float = float(init_t)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    def _nll(self, T: float, log_p: Any, targets: Any) -> float:
        # logits = log(p)/T → softmax
        logits = log_p / T
        proba = _softmax(logits)
        # gather p_true
        n = targets.shape[0]
        p_true = proba[np.arange(n), targets]
        p_true = _clip_for_log(p_true, eps=1e-15)
        return float(-np.log(p_true).mean())

    def fit(
        self,
        raw_probs: Any,
        targets: Any,
    ) -> TemperatureCalibrator:
        arr, t = _validate_probas_and_targets(raw_probs, targets)
        if arr.shape[0] == 0:
            raise ValueError("cannot fit temperature with n=0")
        # Clip for log
        clipped = _clip_for_log(arr, eps=CLIP_EPS)
        log_p = np.log(clipped)

        # Try scipy if available for more precise optimisation;
        # fallback to deterministic grid + refine.
        best_T = self._fit_with_scipy_or_grid(log_p, t)
        self.temperature = float(np.clip(best_T, self.t_min, self.t_max))
        self._fitted = True
        return self

    def _fit_with_scipy_or_grid(self, log_p: Any, targets: Any) -> float:
        # Attempt scipy
        try:
            import scipy.optimize as _opt  # type: ignore[import-untyped]

            def _obj(T: float) -> float:
                return self._nll(float(T), log_p, targets)

            res = _opt.minimize_scalar(
                _obj,
                bounds=(self.t_min, self.t_max),
                method="bounded",
                options={"xatol": 1e-5},
            )
            if res.success and np.isfinite(res.x):
                return float(res.x)
        except Exception:
            pass
        # Fallback: deterministic grid search + local refinement
        return self._grid_search(log_p, targets)

    def _grid_search(self, log_p: Any, targets: Any) -> float:
        # Log-spaced grid between t_min and t_max
        grid = np.logspace(np.log10(self.t_min), np.log10(self.t_max), num=100)
        best_T = self.t_min
        best_nll = float("inf")
        for t_val in grid:  # type: ignore[attr-defined]
            nll = self._nll(float(t_val), log_p, targets)
            if nll < best_nll:
                best_nll = nll
                best_T = float(t_val)
        # Refine around best with smaller steps
        for _ in range(3):
            lo = max(self.t_min, best_T * 0.8)
            hi = min(self.t_max, best_T * 1.25)
            fine = np.linspace(lo, hi, num=50)
            improved = False
            for t_val in fine:  # type: ignore[attr-defined]
                nll = self._nll(float(t_val), log_p, targets)
                if nll < best_nll - 1e-9:
                    best_nll = nll
                    best_T = float(t_val)
                    improved = True
            if not improved:
                break
        return float(best_T)

    # ------------------------------------------------------------------
    def transform(self, probs: Any) -> list[MatchProbabilities]:
        arr = _probas_to_array(probs)
        from app.prediction.metrics.classification import validate_multiclass_probabilities

        arr = validate_multiclass_probabilities(arr)
        # If not fitted, use init value (1.0 → identity)
        T = float(self.temperature)
        if T <= 0:
            raise ValueError(f"temperature must be >0, got {T}")
        clipped = _clip_for_log(arr, eps=CLIP_EPS)
        log_p = np.log(clipped)
        logits = log_p / T
        cal = _softmax(logits)
        # Copy semantics
        return _array_to_probas(np.array(cal, copy=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "temperature",
            "temperature": float(self.temperature),
            "t_min": float(self.t_min),
            "t_max": float(self.t_max),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemperatureCalibrator:
        if data.get("kind") != "temperature":
            raise ValueError(f"expected kind temperature, got {data.get('kind')}")
        obj = cls(
            t_min=float(data.get("t_min", T_MIN)),
            t_max=float(data.get("t_max", T_MAX)),
            init_t=float(data.get("temperature", DEFAULT_T)),
        )
        obj.temperature = float(data["temperature"])
        obj._fitted = True
        return obj


__all__ = ["DEFAULT_T", "T_MAX", "T_MIN", "TemperatureCalibrator"]
