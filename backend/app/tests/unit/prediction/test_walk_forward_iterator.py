"""Sprint 5.1 — WalkForwardIterator core behaviour.

Covers all mandatory scenarios listed in the Sprint 5.1 brief
(§"TESTS OBLIGATORIOS").

The dataset is synthesised in-memory as :class:`LoadedDataset`; no
PostgreSQL or CSV I/O is used — the iterator consumes only the typed
``rows`` list and validates ``(kickoff, fixture_id)`` ordering.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from app.dataset.loader import LoadedDataset, LoadedExample
from app.dataset.manifest import DatasetManifest
from app.prediction.backtesting.iterator import WalkForwardIterator
from app.prediction.contracts import WalkForwardMode

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(fixture_id: int, kickoff: datetime) -> LoadedExample:
    return LoadedExample(
        fixture_id=fixture_id,
        kickoff=kickoff,
        competition_id=39,
        season_id=2024,
        home_team_id=fixture_id,
        away_team_id=fixture_id + 1000,
        features={},
        targets={},
    )


def _dataset(rows: list[LoadedExample]) -> LoadedDataset:
    manifest = DatasetManifest(
        dataset_version="v001",
        generated_at=T0,
        feature_definition_version="fd_v1",
        data_cutoff=rows[-1].kickoff if rows else T0,
        source_schema_version="schema_v1",
        row_count=len(rows),
        feature_names=(),
        target_names=(),
        start_date=rows[0].kickoff if rows else T0,
        end_date=rows[-1].kickoff if rows else T0,
        competitions=(39,),
        seasons=(2024,),
        csv_sha256="x" * 64,
        extras={},
    )
    return LoadedDataset(manifest=manifest, rows=rows)


def _linear_rows(n: int, delta_days: int = 1) -> list[LoadedExample]:
    return [
        _row(i + 1, T0 + timedelta(days=i * delta_days))
        for i in range(n)
    ]


def _irregular_rows() -> list[LoadedExample]:
    kicks = [
        T0,
        T0 + timedelta(days=2),
        T0 + timedelta(days=3),
        T0 + timedelta(days=7),
        T0 + timedelta(days=8),
        T0 + timedelta(days=15),
        T0 + timedelta(days=16),
        T0 + timedelta(days=17),
        T0 + timedelta(days=20),
        T0 + timedelta(days=25),
        T0 + timedelta(days=26),
        T0 + timedelta(days=30),
    ]
    return [_row(i + 1, k) for i, k in enumerate(kicks)]


# ---------------------------------------------------------------------------
# 1. Dataset desordenado → error explícito
# ---------------------------------------------------------------------------

def test_dataset_desordenado_raises_value_error() -> None:
    rows = _linear_rows(6)
    # Break ordering by swapping two rows: row[2] gets a *later* kickoff
    # than row[3] would violate (kickoff, fixture_id) ordering.
    rows[2], rows[3] = rows[3], rows[2]
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=2,
        test_size=1,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=1,
    )
    with pytest.raises(ValueError, match="not ordered"):
        list(it)


def test_dataset_desordenado_por_fixture_id_tambien_falla() -> None:
    k = T0
    rows = [
        _row(2, k),
        _row(1, k),  # same kickoff but fixture_id 1 < 2 violates order
        _row(3, T0 + timedelta(days=1)),
        _row(4, T0 + timedelta(days=2)),
        _row(5, T0 + timedelta(days=3)),
        _row(6, T0 + timedelta(days=4)),
    ]
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=2,
        test_size=1,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=1,
    )
    with pytest.raises(ValueError, match="not ordered"):
        list(it)


# ---------------------------------------------------------------------------
# 2. Dataset insuficiente → cero folds (sin folds inválidos)
# ---------------------------------------------------------------------------

def test_dataset_insuficiente_produce_cero_folds() -> None:
    rows = _linear_rows(3)
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=3,
        test_size=1,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=1,
    )
    folds = list(it)
    # 3 rows train + need 1 val + 1 test = at least 5 rows needed
    assert folds == []


def test_dataset_insuficiente_train_menor_que_min_produce_cero_folds() -> None:
    rows = _linear_rows(1)
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=2,
        test_size=1,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=1,
    )
    assert list(it) == []


# ---------------------------------------------------------------------------
# 3. Dataset mínimo exacto → exactamente un fold
# ---------------------------------------------------------------------------

def test_dataset_minimo_exacto_produce_un_fold() -> None:
    # min_train=2, val=1, test=1, gap=0 → need exactly 4 rows
    rows = _linear_rows(4)
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=2,
        test_size=1,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=1,
    )
    folds = list(it)
    assert len(folds) == 1
    f = folds[0]
    assert len(f.train_indices) == 2
    assert len(f.val_indices) == 1
    assert len(f.test_indices) == 1


# ---------------------------------------------------------------------------
# 4. gap=0 → train→val y val→test pueden ser consecutivos
# ---------------------------------------------------------------------------

def test_gap_cero_permite_bloques_consecutivos() -> None:
    rows = _linear_rows(10, delta_days=1)
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=3,
        test_size=2,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=2,
    )
    folds = list(it)
    assert len(folds) >= 1
    for f in folds:
        # gap=0: train_end still strictly < val_start because kickoffs differ
        assert f.train_end < f.val_start
        assert f.val_end < f.test_start


# ---------------------------------------------------------------------------
# 5. gap>0 → gap temporal respetado
# ---------------------------------------------------------------------------

def test_gap_positivo_respetado_temporalmente() -> None:
    rows = _linear_rows(12, delta_days=2)
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=3,
        test_size=2,
        gap_days=3,
        mode=WalkForwardMode.EXPANDING,
        val_size=2,
    )
    folds = list(it)
    assert len(folds) >= 1
    for f in folds:
        assert f.train_end + timedelta(days=3) <= f.val_start
        assert f.val_end + timedelta(days=3) <= f.test_start


def test_gap_no_es_numero_de_filas() -> None:
    # 5 rows en el mismo día (mismo kickoff salvo +1h), gap=1 día
    # debe saltar TODAS las filas que estén dentro del gap window,
    # no solo "gap filas".
    base = T0
    rows = [
        _row(1, base),
        _row(2, base + timedelta(hours=1)),
        _row(3, base + timedelta(hours=2)),
        _row(4, base + timedelta(days=1)),  # exactly 1 day later
        _row(5, base + timedelta(days=1, hours=1)),
        _row(6, base + timedelta(days=2)),
        _row(7, base + timedelta(days=3)),
        _row(8, base + timedelta(days=4)),
    ]
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=2,
        test_size=1,
        gap_days=1,
        mode=WalkForwardMode.EXPANDING,
        val_size=1,
    )
    folds = list(it)
    for f in folds:
        assert f.train_end + timedelta(days=1) <= f.val_start


# ---------------------------------------------------------------------------
# 6. Timestamps repetidos — nunca dividir un mismo kickoff
# ---------------------------------------------------------------------------

def test_timestamps_repetidos_no_dividen_grupo() -> None:
    # 3 rows con mismo kickoff T0, luego 2 con T1, etc.
    rows = [
        _row(1, T0),
        _row(2, T0),
        _row(3, T0),
        _row(4, T0 + timedelta(days=1)),
        _row(5, T0 + timedelta(days=1)),
        _row(6, T0 + timedelta(days=2)),
        _row(7, T0 + timedelta(days=3)),
        _row(8, T0 + timedelta(days=4)),
    ]
    ds = _dataset(rows)
    # min_train=4 caería en medio del grupo T1 → shrink deja train=3 (<4)
    # → ningún fold debe emitirse (hard floor)
    it = WalkForwardIterator(
        ds,
        min_train_size=4,
        test_size=1,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=1,
    )
    assert list(it) == []

    # Con min=3 sí puede emitir y nunca divide grupo
    it2 = WalkForwardIterator(
        ds,
        min_train_size=3,
        test_size=1,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=1,
    )
    folds = list(it2)
    assert len(folds) >= 1
    for f in folds:
        train_kicks = {ds.rows[i].kickoff for i in f.train_indices}
        val_kicks = {ds.rows[i].kickoff for i in f.val_indices}
        test_kicks = {ds.rows[i].kickoff for i in f.test_indices}
        assert train_kicks.isdisjoint(val_kicks)
        assert val_kicks.isdisjoint(test_kicks)
        assert train_kicks.isdisjoint(test_kicks)


def test_train_por_debajo_del_min_por_kickoff_no_genera_fold() -> None:
    """min_train_size es un mínimo real (hard floor).

    Si el agrupamiento por kickoff idéntico deja len(train) <
    min_train_size, ese fold no se genera — verificar explícitamente
    el caso pedido en la revisión de Sprint 5.1.
    """
    # Grupo T0 con 3 filas, T1 con 2 filas → min=4 cae dentro de T1
    rows = [
        _row(10, T0),
        _row(11, T0),
        _row(12, T0),
        _row(20, T0 + timedelta(days=1)),
        _row(21, T0 + timedelta(days=1)),
        _row(30, T0 + timedelta(days=2)),
        _row(31, T0 + timedelta(days=3)),
        _row(32, T0 + timedelta(days=4)),
    ]
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=4,
        test_size=1,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=1,
    )
    folds = list(it)
    # Debe ser vacío porque el único primer fold posible violaría min
    assert folds == []
    # Con min=3 sí hay folds y el train nunca es <3
    it2 = WalkForwardIterator(
        ds,
        min_train_size=3,
        test_size=1,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=1,
    )
    folds2 = list(it2)
    assert len(folds2) >= 1
    for f in folds2:
        assert len(f.train_indices) >= 3


# ---------------------------------------------------------------------------
# 7. Timestamps irregulares
# ---------------------------------------------------------------------------

def test_timestamps_irregulares_gap_temporal_aun_vale() -> None:
    rows = _irregular_rows()
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=3,
        test_size=2,
        gap_days=3,
        mode=WalkForwardMode.EXPANDING,
        val_size=1,
    )
    folds = list(it)
    assert len(folds) >= 1
    for f in folds:
        assert f.train_end + timedelta(days=3) <= f.val_start
        assert f.val_end + timedelta(days=3) <= f.test_start


# ---------------------------------------------------------------------------
# 8. Reproducibilidad con mismos parámetros
# ---------------------------------------------------------------------------

def test_reproducibilidad_con_mismos_parametros() -> None:
    rows = _linear_rows(20)
    ds = _dataset(rows)
    kwargs = {
        "min_train_size": 5,
        "test_size": 2,
        "gap_days": 0,
        "mode": WalkForwardMode.EXPANDING,
        "val_size": 2,
    }
    folds_a = list(WalkForwardIterator(ds, **kwargs))
    folds_b = list(WalkForwardIterator(ds, **kwargs))
    assert len(folds_a) == len(folds_b)
    for a, b in zip(folds_a, folds_b, strict=True):
        assert a.train_indices == b.train_indices
        assert a.val_indices == b.val_indices
        assert a.test_indices == b.test_indices
        assert a.train_start == b.train_start
        assert a.val_start == b.val_start
        assert a.test_start == b.test_start


# ---------------------------------------------------------------------------
# 9. Expanding aumenta train
# ---------------------------------------------------------------------------

def test_expanding_aumenta_train_monotonicamente() -> None:
    rows = _linear_rows(20)
    ds = _dataset(rows)
    folds = list(
        WalkForwardIterator(
            ds,
            min_train_size=4,
            test_size=2,
            gap_days=0,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    )
    assert len(folds) >= 2
    train_lens = [len(f.train_indices) for f in folds]
    assert train_lens == sorted(train_lens)
    # Estrictamente creciente en expanding
    for a, b in zip(train_lens, train_lens[1:], strict=False):
        assert b > a
    # Siempre empieza en 0
    for f in folds:
        assert f.train_indices[0] == 0


# ---------------------------------------------------------------------------
# 10. Sliding mantiene tamaño esperado
# ---------------------------------------------------------------------------

def test_sliding_mantiene_tamano_aproximado() -> None:
    rows = _linear_rows(20)
    ds = _dataset(rows)
    folds = list(
        WalkForwardIterator(
            ds,
            min_train_size=4,
            test_size=2,
            gap_days=0,
            mode=WalkForwardMode.SLIDING,
            val_size=2,
        )
    )
    assert len(folds) >= 2
    train_lens = [len(f.train_indices) for f in folds]
    # Todos ~4 (puede variar por equal-kickoff, pero nunca <4 salvo que
    # el fold no se emita)
    for ln in train_lens:
        assert ln >= 4
        assert ln <= 6  # tolerancia por clumping (no exacta)
    # Ventanas se desplazan: cada fold empieza más adelante
    starts = [f.train_indices[0] for f in folds]
    assert starts == sorted(starts)
    for a, b in zip(starts, starts[1:], strict=False):
        assert b > a


# ---------------------------------------------------------------------------
# 11. val_ratio vs val_size mutuamente excluyentes
# ---------------------------------------------------------------------------

def test_val_ratio_y_val_size_mutuamente_excluyentes() -> None:
    rows = _linear_rows(10)
    ds = _dataset(rows)
    with pytest.raises(ValueError, match="mutually exclusive"):
        WalkForwardIterator(
            ds,
            min_train_size=3,
            test_size=1,
            gap_days=0,
            mode=WalkForwardMode.EXPANDING,
            val_ratio=0.2,
            val_size=2,
        )


def test_val_ratio_proporcional_al_train() -> None:
    rows = _linear_rows(20)
    ds = _dataset(rows)
    # train=10 → val should be round(10*0.2)=2
    it = WalkForwardIterator(
        ds,
        min_train_size=10,
        test_size=2,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_ratio=0.2,
    )
    folds = list(it)
    assert len(folds) >= 1
    assert len(folds[0].val_indices) == 2


# ---------------------------------------------------------------------------
# 12. Datos futuros no modifican folds anteriores
# ---------------------------------------------------------------------------

def test_datos_futuros_no_modifican_folds_anteriores() -> None:
    rows_short = _linear_rows(12)
    rows_long = _linear_rows(20)
    ds_short = _dataset(rows_short)
    ds_long = _dataset(rows_long)
    kwargs = {
        "min_train_size": 4,
        "test_size": 2,
        "gap_days": 0,
        "mode": WalkForwardMode.EXPANDING,
        "val_size": 2,
    }
    folds_short = list(WalkForwardIterator(ds_short, **kwargs))
    folds_long = list(WalkForwardIterator(ds_long, **kwargs))
    # Los folds de la versión corta deben ser prefijo exacto de la larga
    assert len(folds_long) >= len(folds_short)
    for a, b in zip(folds_short, folds_long, strict=False):
        assert a.train_indices == b.train_indices
        assert a.val_indices == b.val_indices
        assert a.test_indices == b.test_indices


# ---------------------------------------------------------------------------
# 13. __len__ y to_config
# ---------------------------------------------------------------------------

def test_len_consistente_con_iter() -> None:
    rows = _linear_rows(12)
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=3,
        test_size=2,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=2,
    )
    assert len(it) == len(list(it))


def test_to_config_serialisable() -> None:
    rows = _linear_rows(8)
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=3,
        test_size=1,
        gap_days=1,
        mode=WalkForwardMode.SLIDING,
        val_ratio=0.2,
    )
    cfg = it.to_config()
    assert cfg["min_train_size"] == 3
    assert cfg["gap_days"] == 1
    assert cfg["val_ratio"] == 0.2
