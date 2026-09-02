"""Sprint 5.2 — Feature vector invariants.

Verifies ``loaded_example_to_features``:

* size == len(FEATURE_NAMES)
* order == FEATURE_NAMES
* None → nan, 0 → 0.0, negatives / floats preserved
* fixture_id / kickoff preserved
* feature_names == tuple(FEATURE_NAMES)
* error on unknown feature name
* deterministic
* no imputation beyond None→nan
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest
from app.dataset.loader import LoadedExample
from app.features.example import FEATURE_NAMES
from app.prediction.features.vector import loaded_example_to_features

UTC_TZ = UTC
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC_TZ)


def _example(
    *,
    fixture_id: int = 1,
    kickoff: datetime = T0,
    features: dict[str, float | int | None] | None = None,
) -> LoadedExample:
    return LoadedExample(
        fixture_id=fixture_id,
        kickoff=kickoff,
        competition_id=39,
        season_id=2024,
        home_team_id=10,
        away_team_id=20,
        features=features if features is not None else {},
        targets={},
    )


# ------------------------------------------------------------------
# 1. vector tiene tamaño correcto
# ------------------------------------------------------------------

def test_vector_tiene_tamano_correcto() -> None:
    ex = _example()
    f = loaded_example_to_features(ex)
    assert f.feature_vector.shape == (len(FEATURE_NAMES),)
    assert len(f.feature_vector) == len(FEATURE_NAMES)


# ------------------------------------------------------------------
# 2. orden coincide con FEATURE_NAMES
# ------------------------------------------------------------------

def test_orden_coincide_con_feature_names() -> None:
    feats: dict[str, float | int | None] = {}
    for i, name in enumerate(FEATURE_NAMES):
        feats[name] = float(i * 10 + 1)
    ex = _example(features=feats)
    f = loaded_example_to_features(ex)
    for i, name in enumerate(FEATURE_NAMES):
        assert f.feature_vector[i] == pytest.approx(float(i * 10 + 1))
        assert f.feature_names[i] == name


# ------------------------------------------------------------------
# 3. None se convierte en np.nan
# ------------------------------------------------------------------

def test_none_se_convierte_en_nan() -> None:
    name = FEATURE_NAMES[0]
    ex = _example(features={name: None})
    f = loaded_example_to_features(ex)
    idx = FEATURE_NAMES.index(name)
    assert math.isnan(float(f.feature_vector[idx]))
    assert np.isnan(f.feature_vector[idx])


def test_none_no_se_convierte_en_cero() -> None:
    """Spec §5.16: None → nan y NO None → 0."""
    name = FEATURE_NAMES[1]
    ex = _example(features={name: None})
    f = loaded_example_to_features(ex)
    idx = FEATURE_NAMES.index(name)
    assert f.feature_vector[idx] != 0.0
    assert np.isnan(f.feature_vector[idx])


# ------------------------------------------------------------------
# 4. cero permanece cero
# ------------------------------------------------------------------

def test_cero_permanece_cero() -> None:
    name = FEATURE_NAMES[2]
    for zero in (0, 0.0):
        ex = _example(features={name: zero})  # type: ignore[dict-item]
        f = loaded_example_to_features(ex)
        idx = FEATURE_NAMES.index(name)
        assert f.feature_vector[idx] == 0.0
        assert not np.isnan(f.feature_vector[idx])


# ------------------------------------------------------------------
# 5. valores negativos válidos se preservan
# ------------------------------------------------------------------

def test_valores_negativos_se_preservan() -> None:
    # ELO difference can be negative
    feats = {FEATURE_NAMES[0]: -42, FEATURE_NAMES[1]: -0.5}
    ex = _example(features=feats)
    f = loaded_example_to_features(ex)
    assert f.feature_vector[0] == pytest.approx(-42.0)
    assert f.feature_vector[1] == pytest.approx(-0.5)


# ------------------------------------------------------------------
# 6. floats válidos se preservan
# ------------------------------------------------------------------

def test_floats_validos_se_preservan() -> None:
    feats = {FEATURE_NAMES[0]: 1.2345, FEATURE_NAMES[3]: 99.9}
    ex = _example(features=feats)
    f = loaded_example_to_features(ex)
    assert f.feature_vector[0] == pytest.approx(1.2345, rel=1e-5)
    assert f.feature_vector[3] == pytest.approx(99.9, rel=1e-5)


# ------------------------------------------------------------------
# 7. fixture_id se preserva
# ------------------------------------------------------------------

def test_fixture_id_se_preserva() -> None:
    ex = _example(fixture_id=9999)
    f = loaded_example_to_features(ex)
    assert f.fixture_id == 9999


# ------------------------------------------------------------------
# 8. kickoff se preserva
# ------------------------------------------------------------------

def test_kickoff_se_preserva() -> None:
    kick = datetime(2026, 3, 10, 15, 30, tzinfo=UTC_TZ)
    ex = _example(kickoff=kick)
    f = loaded_example_to_features(ex)
    assert f.kickoff == kick


# ------------------------------------------------------------------
# 9. feature_names coincide exactamente
# ------------------------------------------------------------------

def test_feature_names_coincide_exactamente() -> None:
    ex = _example()
    f = loaded_example_to_features(ex)
    assert f.feature_names == tuple(FEATURE_NAMES)
    assert len(f.feature_names) == len(FEATURE_NAMES)


# ------------------------------------------------------------------
# 10. número incorrecto / nombres desconocidos → error
# ------------------------------------------------------------------

def test_unknown_feature_name_raise_error() -> None:
    ex = _example(features={"feature_inexistente_xyz": 1})
    with pytest.raises(ValueError, match="unknown feature"):
        loaded_example_to_features(ex)


def test_dtype_es_float32() -> None:
    ex = _example(features={FEATURE_NAMES[0]: 1})
    f = loaded_example_to_features(ex)
    assert f.feature_vector.dtype == np.float32


# ------------------------------------------------------------------
# 14. no se hace ninguna imputación
# ------------------------------------------------------------------

def test_no_se_hace_imputacion_media_o_cero() -> None:
    # Only NaN should appear for missing; no zero-fill
    name_a = FEATURE_NAMES[0]
    name_b = FEATURE_NAMES[1]
    ex = _example(features={name_a: None, name_b: 5.0})
    f = loaded_example_to_features(ex)
    idx_a = FEATURE_NAMES.index(name_a)
    idx_b = FEATURE_NAMES.index(name_b)
    assert np.isnan(f.feature_vector[idx_a])
    assert f.feature_vector[idx_b] == 5.0
    # Ensure no other position was silently zero-imputed to hide NaN:
    # All positions not set in dict should be NaN (sparse dict)
    nan_count = int(np.isnan(f.feature_vector).sum())
    # We set 1 real value, so 65 should be NaN
    assert nan_count == len(FEATURE_NAMES) - 1


# ------------------------------------------------------------------
# 15. vectorización determinista
# ------------------------------------------------------------------

def test_vectorizacion_determinista() -> None:
    feats = {FEATURE_NAMES[0]: 10, FEATURE_NAMES[5]: None, FEATURE_NAMES[10]: -3.5}
    ex = _example(features=feats)
    f1 = loaded_example_to_features(ex)
    f2 = loaded_example_to_features(ex)
    assert np.array_equal(f1.feature_vector, f2.feature_vector, equal_nan=True)
    assert f1.feature_names == f2.feature_names
    assert f1.fixture_id == f2.fixture_id


# ------------------------------------------------------------------
# 16. dos llamadas al mismo ejemplo producen el mismo resultado
# ------------------------------------------------------------------

def test_dos_llamadas_mismo_ejemplo_mismo_resultado() -> None:
    feats = {FEATURE_NAMES[2]: 0, FEATURE_NAMES[3]: None}
    ex = _example(fixture_id=42, features=feats)
    a = loaded_example_to_features(ex)
    b = loaded_example_to_features(ex)
    assert a.fixture_id == b.fixture_id
    assert a.kickoff == b.kickoff
    assert a.feature_names == b.feature_names
    assert np.array_equal(a.feature_vector, b.feature_vector, equal_nan=True)


# ------------------------------------------------------------------
# Extra: no muta LoadedExample
# ------------------------------------------------------------------

def test_no_muta_loaded_example() -> None:
    feats = {FEATURE_NAMES[0]: 5}
    ex = _example(features=feats)
    before = dict(ex.features)
    loaded_example_to_features(ex)
    assert ex.features == before


def test_valor_no_numerico_raise_type_error() -> None:
    ex = _example(features={FEATURE_NAMES[0]: "no-numerico"})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        loaded_example_to_features(ex)
