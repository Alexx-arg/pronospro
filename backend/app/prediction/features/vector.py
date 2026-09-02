"""Feature vectorisation: LoadedExample → FixtureFeatures (Sprint 5.2).

Single responsibility: translate one :class:`LoadedExample` (Phase 4
output, dict-based) into the canonical :class:`FixtureFeatures`
(NamedTuple carrying a dense ``np.ndarray[float32]`` + feature name
tuple). No DB, no recomputation, no imputation beyond ``None → nan``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.dataset.loader import LoadedExample
from app.features.example import FEATURE_NAMES
from app.prediction.contracts import FixtureFeatures


def loaded_example_to_features(example: LoadedExample) -> FixtureFeatures:
    """Convert one :class:`LoadedExample` into a :class:`FixtureFeatures`.

    Contract (validated at runtime):

    * ``feature_names`` is exactly ``tuple(FEATURE_NAMES)`` — same order,
      same length, same strings.
    * ``feature_vector`` has ``len == len(FEATURE_NAMES)`` and
      ``dtype == np.float32``.
    * ``None`` in the dict becomes ``np.nan`` (missing preserved).
    * ``0`` (or ``0.0``) becomes ``0.0`` (math-zero preserved, not
      conflated with missing).
    * Negative and floating values are preserved verbatim as ``float``.
    * ``fixture_id`` / ``kickoff`` are copied unchanged.
    * Never mutates ``example``.

    Raises:
        ValueError: if ``FEATURE_NAMES`` is empty or the produced vector
            length mismatches ``FEATURE_NAMES`` (sanity guard; would
            indicate a registry bug, not a data bug).
        TypeError: if a feature value is not ``None``/``int``/``float``
            (e.g. a stray string from a future schema change).

    Notes:
        The dict ``example.features`` is allowed to be sparse (missing
        keys are treated as ``None → nan``). The loader-produced
        ``LoadedExample`` is always dense (66 keys), but synthetic
        examples in tests may be sparse.
    """
    if len(FEATURE_NAMES) == 0:
        raise ValueError("FEATURE_NAMES is empty — registry is broken")

    n = len(FEATURE_NAMES)
    # Validate that dict does not contain unknown feature names.
    # Missing keys are allowed (treated as None → nan) so synthetic
    # examples in tests can be sparse; unknown keys are always a bug.
    unknown = set(example.features.keys()) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature names present: {sorted(unknown)!r}")

    # Build python list of floats / nans in canonical order.
    values: list[float] = []
    for name in FEATURE_NAMES:
        raw = example.features.get(name)
        if raw is None:
            values.append(float("nan"))
        elif isinstance(raw, (int, float, np.integer, np.floating)):  # noqa: UP038  # type: ignore[arg-type]
            # Explicit float conversion preserves 0 → 0.0, negatives, etc.
            # Using float() then np.float32 ensures no loss beyond fp32.
            values.append(float(raw))
        else:
            raise TypeError(
                f"feature {name!r} has unsupported type {type(raw).__name__}: {raw!r}"
            )

    if len(values) != n:
        raise ValueError(
            f"feature vector length {len(values)} != len(FEATURE_NAMES) {n}"
        )

    arr: Any = np.array(values, dtype=np.float32)

    # Defensive: shape must be (n,) and not (n,1) etc.
    if arr.shape != (n,):
        raise ValueError(f"unexpected vector shape {arr.shape}, expected ({n},)")

    return FixtureFeatures(
        fixture_id=example.fixture_id,
        kickoff=example.kickoff,
        feature_vector=arr,
        feature_names=tuple(FEATURE_NAMES),
    )


def examples_to_matrix(
    examples: list[LoadedExample],
) -> tuple[Any, list[FixtureFeatures]]:
    """Convenience helper: vectorise many examples.

    Returns a 2-D array of shape ``(n_examples, n_features)`` and the
    list of per-row :class:`FixtureFeatures` (kept for kickoff/fixture_id
    auditing). The matrix is ``dtype float32`` and stacked row-major.

    This helper is intentionally tiny — callers that need memory control
    can loop ``loaded_example_to_features`` themselves.
    """
    if len(examples) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), []
    feats = [loaded_example_to_features(ex) for ex in examples]
    mat = np.stack([f.feature_vector for f in feats], axis=0)
    return mat, feats


__all__ = ["examples_to_matrix", "loaded_example_to_features"]
