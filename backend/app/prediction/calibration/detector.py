"""Calibration detector (Sprint 5.4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.prediction.calibration.base import _probas_to_array
from app.prediction.calibration.dirichlet import DirichletCalibrator
from app.prediction.calibration.identity import IdentityCalibrator
from app.prediction.calibration.temperature import TemperatureCalibrator
from app.prediction.contracts import CalibratorKind
from app.prediction.metrics.calibration import expected_calibration_error

ECE_LOW: float = 0.02
ECE_HIGH: float = 0.05
N_DIRICHLET_MIN: int = 5000


class CalibrationDetector:
    """Decides calibrator based on validation ECE and n."""

    def __init__(
        self,
        *,
        ece_low: float = ECE_LOW,
        ece_high: float = ECE_HIGH,
        n_dirichlet_min: int = N_DIRICHLET_MIN,
        lambda_l2: float = 0.1,
    ) -> None:
        self.ece_low: float = float(ece_low)
        self.ece_high: float = float(ece_high)
        self.n_dirichlet_min: int = int(n_dirichlet_min)
        self.lambda_l2: float = float(lambda_l2)

    def detect(
        self,
        raw_probs: Any,
        targets: Any,
    ) -> Any:
        """Return a *new* calibrator instance per decision.

        The returned calibrator is **unfitted** — caller must call
        ``.fit(raw_probs, targets)``.  Decision uses validation data
        only; never test/train.
        """
        arr = _probas_to_array(raw_probs)
        # Validate via central validator (ensures simplex)
        from app.prediction.metrics.classification import validate_multiclass_probabilities

        arr = validate_multiclass_probabilities(arr)
        n = arr.shape[0]
        if n == 0:
            raise ValueError("cannot detect calibration with n=0")
        # Validate targets shape
        t = np.asarray(targets, dtype=np.int64)
        if t.shape[0] != n:
            raise ValueError(f"targets length {t.shape[0]} != n {n}")

        ece = expected_calibration_error(t, arr, n_bins=10)

        if ece < self.ece_low:
            return IdentityCalibrator()
        if ece >= self.ece_high and n >= self.n_dirichlet_min:
            return DirichletCalibrator(lambda_l2=self.lambda_l2)
        return TemperatureCalibrator()

    def decide_kind(
        self,
        raw_probs: Any,
        targets: Any,
    ) -> CalibratorKind:
        """Convenience: return kind without constructing calibrator."""
        cal = self.detect(raw_probs, targets)
        return cal.kind  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece_low": self.ece_low,
            "ece_high": self.ece_high,
            "n_dirichlet_min": self.n_dirichlet_min,
            "lambda_l2": self.lambda_l2,
        }


__all__ = [
    "ECE_HIGH",
    "ECE_LOW",
    "N_DIRICHLET_MIN",
    "CalibrationDetector",
]
