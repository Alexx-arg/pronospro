"""Identity calibrator (Sprint 5.4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.prediction.calibration.base import (
    _array_to_probas,
    _probas_to_array,
    _validate_probas_and_targets,
)
from app.prediction.contracts import CalibratorKind, MatchProbabilities


class IdentityCalibrator:
    """No-op calibrator — returns probabilities unchanged."""

    kind: CalibratorKind = CalibratorKind.IDENTITY

    def fit(
        self,
        raw_probs: Any,
        targets: Any,
    ) -> IdentityCalibrator:
        # Validate inputs even though we learn nothing — ensures caller
        # mistakes are caught at fit time, not silently at transform.
        _validate_probas_and_targets(raw_probs, targets)
        return self

    def transform(self, probs: Any) -> list[MatchProbabilities]:
        arr = _probas_to_array(probs)
        # Validate using same central helper (ensures simplex even for identity)
        from app.prediction.metrics.classification import validate_multiclass_probabilities

        arr = validate_multiclass_probabilities(arr)
        # Return copy — never mutate input
        copy = np.array(arr, copy=True)
        return _array_to_probas(copy)

    # Alias for sklearn-style API
    def __call__(self, probs: Any) -> list[MatchProbabilities]:
        return self.transform(probs)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "identity"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentityCalibrator:
        if data.get("kind") != "identity":
            raise ValueError(f"expected kind identity, got {data.get('kind')}")
        return cls()


__all__ = ["IdentityCalibrator"]
