"""Structured metrics report (Sprint 5.3).

A :class:`FoldReport` captures every metric for one fold's test block.
It is immutable (frozen dataclass) and serialisable to JSON via
:func:`FoldReport.to_dict`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class FoldReport:
    """Immutable per-fold metrics report.

    Attributes:
        model_name: ``ModelName`` string (e.g. ``"poisson"``).
        model_version: Artifact version (e.g. ``"v001"``).
        dataset_version: Dataset version (e.g. ``"v001"``).
        fold_id: Fold index.
        n_predictions: ``len(test_block)``.
        accuracy: 1X2 accuracy.
        log_loss: Multiclass cross-entropy.
        brier_home, brier_draw, brier_away, brier_multiclass: Brier.
        ece, mce: Calibration.
        n_bins: ECE/MCE bins count.
        confusion_matrix: 3×3 list of lists (row=true, col=pred).
        confidence_buckets: List of per-bucket dicts
            (see :func:`confidence_bucket_metrics`).
        reliability_bins: List of per-bin dicts for ECE.
        mae_home_goals, mae_away_goals, rmse_home_goals,
        rmse_away_goals, rmse_total_goals, poisson_loglik:
            Optional goal metrics (``None`` when model does not
            produce goal expectations, e.g. GB multiclass).
        calibration_bins: Duplicate of ``n_bins`` for JSON compatibility.
    """

    model_name: str
    model_version: str
    dataset_version: str
    fold_id: int
    n_predictions: int
    accuracy: float
    log_loss: float
    brier_home: float
    brier_draw: float
    brier_away: float
    brier_multiclass: float
    ece: float
    mce: float
    n_bins: int
    confusion_matrix: list[list[int]]
    confidence_buckets: list[dict[str, Any]] = field(default_factory=list)
    reliability_bins: list[dict[str, Any]] = field(default_factory=list)
    # Goal metrics — optional
    mae_home_goals: float | None = None
    mae_away_goals: float | None = None
    rmse_home_goals: float | None = None
    rmse_away_goals: float | None = None
    rmse_total_goals: float | None = None
    poisson_loglik: float | None = None
    # Alias for compatibility
    calibration_bins: int | None = None

    def __post_init__(self) -> None:
        if self.n_predictions <= 0:
            raise ValueError(f"n_predictions must be >0, got {self.n_predictions}")
        if self.fold_id < 0:
            raise ValueError(f"fold_id must be >=0, got {self.fold_id}")
        # Confusion matrix must be 3x3
        if len(self.confusion_matrix) != 3 or any(len(row) != 3 for row in self.confusion_matrix):
            raise ValueError(f"confusion_matrix must be 3x3, got {self.confusion_matrix}")
        total = sum(sum(row) for row in self.confusion_matrix)
        if total != self.n_predictions:
            raise ValueError(
                f"confusion_matrix sum {total} != n_predictions {self.n_predictions}"
            )
        # Fill alias if not provided
        if self.calibration_bins is None:
            object.__setattr__(self, "calibration_bins", self.n_bins)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        d = asdict(self)
        # asdict already converted nested structures; ensure plain types
        return d

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FoldReport:
        """Inverse of :meth:`to_dict`."""
        return cls(**data)


__all__ = ["FoldReport"]
