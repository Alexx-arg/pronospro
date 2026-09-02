"""Sprint 5.2 — Missing policy observational helpers.

Tests for :mod:`app.prediction.features.missing`:

* None → nan preserved (no 0-fill)
* count_missing / missing_mask / has_missing / missing_fraction
* no imputation beyond None→nan at this layer
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest
from app.dataset.loader import LoadedExample
from app.features.example import FEATURE_NAMES
from app.prediction.features.missing import (
    count_missing,
    count_missing_batch,
    has_missing,
    missing_fraction,
    missing_mask,
)
from app.prediction.features.vector import loaded_example_to_features

T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


def _example(features: dict[str, float | int | None]) -> LoadedExample:
    return LoadedExample(
        fixture_id=1,
        kickoff=T0,
        competition_id=39,
        season_id=2024,
        home_team_id=10,
        away_team_id=20,
        features=features,
        targets={},
    )


# ------------------------------------------------------------------
# count_missing
# ------------------------------------------------------------------

def test_count_missing_all_nan() -> None:
    ex = _example({})
    vec = loaded_example_to_features(ex).feature_vector
    assert count_missing(vec) == len(FEATURE_NAMES)


def test_count_missing_cero_si_sin_nan() -> None:
    feats = {name: 1.0 for name in FEATURE_NAMES}
    ex = _example(feats)
    vec = loaded_example_to_features(ex).feature_vector
    assert count_missing(vec) == 0


def test_count_missing_mixto() -> None:
    feats = {FEATURE_NAMES[0]: None, FEATURE_NAMES[1]: 0, FEATURE_NAMES[2]: 5.0}
    vec = loaded_example_to_features(_example(feats)).feature_vector
    # FEATURE_NAMES[0] is NaN, rest of 65: 2 real + 63 NaN → total NaN = 64
    # But we explicitly check count_missing == 64? Let's compute directly:
    # Only indices 1 and 2 are non-nan, rest nan → 66 -2 =64
    assert count_missing(vec) == 64


# ------------------------------------------------------------------
# missing_mask
# ------------------------------------------------------------------

def test_missing_mask_marca_nan_correctamente() -> None:
    feats = {FEATURE_NAMES[0]: None, FEATURE_NAMES[1]: 0.0, FEATURE_NAMES[2]: 3.14}
    vec = loaded_example_to_features(_example(feats)).feature_vector
    mask = missing_mask(vec)
    assert mask.dtype == bool
    assert mask.shape == vec.shape
    assert mask[0]  # NaN
    assert not mask[1]  # 0.0 is not missing
    assert not mask[2]  # 3.14 is not missing


def test_missing_mask_no_falsos_positivos_para_cero() -> None:
    feats = {FEATURE_NAMES[0]: 0}
    vec = loaded_example_to_features(_example(feats)).feature_vector
    mask = missing_mask(vec)
    idx = FEATURE_NAMES.index(FEATURE_NAMES[0])
    assert not mask[idx]


# ------------------------------------------------------------------
# has_missing
# ------------------------------------------------------------------

def test_has_missing_true_si_hay_nan() -> None:
    ex = _example({FEATURE_NAMES[0]: None})
    vec = loaded_example_to_features(ex).feature_vector
    assert has_missing(vec) is True


def test_has_missing_false_si_no_hay_nan() -> None:
    feats = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    vec = loaded_example_to_features(_example(feats)).feature_vector
    assert has_missing(vec) is False


# ------------------------------------------------------------------
# missing_fraction
# ------------------------------------------------------------------

def test_missing_fraction_todo_nan_es_uno() -> None:
    ex = _example({})
    vec = loaded_example_to_features(ex).feature_vector
    assert missing_fraction(vec) == pytest.approx(1.0)


def test_missing_fraction_nada_nan_es_cero() -> None:
    feats = {name: 1.0 for name in FEATURE_NAMES}
    vec = loaded_example_to_features(_example(feats)).feature_vector
    assert missing_fraction(vec) == pytest.approx(0.0)


def test_missing_fraction_mitad() -> None:
    # Set half to real, half to NaN
    half = len(FEATURE_NAMES) // 2
    feats: dict[str, float | int | None] = {}
    for i, name in enumerate(FEATURE_NAMES):
        feats[name] = float(i) if i < half else None
    vec = loaded_example_to_features(_example(feats)).feature_vector
    assert missing_fraction(vec) == pytest.approx(0.5 if len(FEATURE_NAMES) % 2 == 0 else (len(FEATURE_NAMES) - half) / len(FEATURE_NAMES))


def test_missing_fraction_empty_vector() -> None:
    empty = np.array([], dtype=np.float32)
    assert missing_fraction(empty) == 0.0


# ------------------------------------------------------------------
# count_missing_batch
# ------------------------------------------------------------------

def test_count_missing_batch_shape() -> None:
    feats1 = {FEATURE_NAMES[0]: None}
    feats2 = {name: 1.0 for name in FEATURE_NAMES}
    from app.prediction.features.vector import examples_to_matrix

    ex1 = _example(feats1)
    ex2 = _example(feats2)
    mat, _ = examples_to_matrix([ex1, ex2])
    counts = count_missing_batch(mat)
    assert counts.shape == (2,)
    assert counts[1] == 0
    assert counts[0] > 0


def test_count_missing_batch_requiere_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        count_missing_batch(np.array([1.0, float("nan")], dtype=np.float32))


# ------------------------------------------------------------------
# Invariante central: None → nan, no → 0
# ------------------------------------------------------------------

def test_none_es_nan_no_cero_explicito() -> None:
    """Spec §5: must demonstrate feature=None → nan and NOT None→0."""
    ex_none = _example({FEATURE_NAMES[5]: None})
    ex_zero = _example({FEATURE_NAMES[5]: 0})
    vec_none = loaded_example_to_features(ex_none).feature_vector
    vec_zero = loaded_example_to_features(ex_zero).feature_vector
    idx = FEATURE_NAMES.index(FEATURE_NAMES[5])
    assert np.isnan(vec_none[idx])
    assert vec_zero[idx] == 0.0
    assert not math.isnan(float(vec_zero[idx]))
    # Ensure they are distinguishable
    assert not np.array_equal(
        np.array([vec_none[idx]]), np.array([vec_zero[idx]]), equal_nan=False
    )
