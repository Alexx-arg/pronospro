"""Explicit missing-data policy (Sprint 5.2).

Source of truth for the translation ``None → nan`` and for the
no-imputation rule in the generic feature layer.

Phase 4 already distinguishes:
    * ``None`` = missing (insufficient history, no snapshot, etc.)
    * ``0``    = mathematically zero (e.g. home_wins_last_3 == 0)

Phase 5.2 preserves that distinction:
    * ``None → np.nan``  (never 0)
    * ``0    → 0.0``      (never nan)

Model-specific imputations (Poisson: test-block mean-impute,
Gradient Boosting: pass-through nan) live in their own trainers /
calibrators, NOT here. This module only exposes *observational*
helpers — counting / masking — never mutating.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def has_missing(vector: Any) -> bool:
    """Return ``True`` if *any* entry in ``vector`` is ``nan``.

    ``vector`` is expected to be the ``feature_vector`` from
    :class:`FixtureFeatures` (float32), but any floating ndarray is
    accepted. Integer arrays never have missing in this codebase
    (integers are promoted to float32 by the vectoriser).
    """
    return bool(np.isnan(vector).any())


def missing_mask(vector: Any) -> Any:
    """Return a boolean mask where ``True`` marks ``nan`` positions."""
    return np.isnan(vector)


def count_missing(vector: Any) -> int:
    """Return the number of ``nan`` entries in ``vector``."""
    return int(np.isnan(vector).sum())


def missing_fraction(vector: Any) -> float:
    """Return ``count_missing / len(vector)`` in ``[0, 1]``.

    For an empty vector (``len == 0``) the fraction is defined as
    ``0.0`` — callers never pass empty vectors in production, but
    returning 0 is more useful than raising in edge-case tests.
    """
    n = int(vector.size)
    if n == 0:
        return 0.0
    return float(count_missing(vector) / n)


def count_missing_batch(matrix: Any) -> Any:
    """Per-row ``count_missing`` for a 2-D matrix.

    Args:
        matrix: shape ``(n_rows, n_features)``.

    Returns:
        1-D array of shape ``(n_rows,)`` with per-row NaN counts.
    """
    if int(matrix.ndim) != 2:
        raise ValueError(f"batch matrix must be 2-D, got shape {matrix.shape}")
    return np.isnan(matrix).sum(axis=1)


__all__ = [
    "count_missing",
    "count_missing_batch",
    "has_missing",
    "missing_fraction",
    "missing_mask",
]
