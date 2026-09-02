"""Sprint 5.1 — Fold chronology invariants.

Every fold yielded by :class:`WalkForwardIterator` must satisfy:

* train strictly before val (``train_end < val_start``)
* val strictly before test (``val_end < test_start``)
* gap temporal respected (both frontiers)
* non-empty blocks, no overlap, monotonic fold_id
* cutoffs and gap_end semantics
* Fold immutability (frozen dataclass)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from app.dataset.loader import LoadedDataset, LoadedExample
from app.dataset.manifest import DatasetManifest
from app.prediction.backtesting.fold import Fold
from app.prediction.backtesting.iterator import WalkForwardIterator
from app.prediction.contracts import WalkForwardMode

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _row(fid: int, kickoff: datetime) -> LoadedExample:
    return LoadedExample(
        fixture_id=fid,
        kickoff=kickoff,
        competition_id=39,
        season_id=2024,
        home_team_id=fid,
        away_team_id=fid + 1000,
        features={},
        targets={},
    )


def _dataset(rows: list[LoadedExample]) -> LoadedDataset:
    m = DatasetManifest(
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
    return LoadedDataset(manifest=m, rows=rows)


def _linear(n: int) -> list[LoadedExample]:
    return [_row(i + 1, T0 + timedelta(days=i)) for i in range(n)]


# ---------------------------------------------------------------------------
# Train strictly before val
# ---------------------------------------------------------------------------

def test_train_estrictamente_antes_que_val() -> None:
    ds = _dataset(_linear(15))
    folds = list(
        WalkForwardIterator(
            ds,
            min_train_size=4,
            test_size=2,
            gap_days=1,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    )
    assert len(folds) >= 1
    for f in folds:
        assert f.train_end < f.val_start
        assert f.train_start <= f.train_end
        assert f.val_start <= f.val_end
        assert ds.rows[f.train_indices[-1]].kickoff < ds.rows[f.val_indices[0]].kickoff


def test_val_estrictamente_antes_que_test() -> None:
    ds = _dataset(_linear(15))
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
    for f in folds:
        assert f.val_end < f.test_start
        assert ds.rows[f.val_indices[-1]].kickoff < ds.rows[f.test_indices[0]].kickoff


# ---------------------------------------------------------------------------
# Gap respected (temporal)
# ---------------------------------------------------------------------------

def test_gap_temporal_respetado_train_val() -> None:
    ds = _dataset(_linear(15))
    gap = 2
    folds = list(
        WalkForwardIterator(
            ds,
            min_train_size=4,
            test_size=2,
            gap_days=gap,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    )
    for f in folds:
        assert f.train_end + timedelta(days=gap) <= f.val_start


def test_gap_temporal_respetado_val_test() -> None:
    ds = _dataset(_linear(15))
    gap = 2
    folds = list(
        WalkForwardIterator(
            ds,
            min_train_size=4,
            test_size=2,
            gap_days=gap,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    )
    for f in folds:
        assert f.val_end + timedelta(days=gap) <= f.test_start


# ---------------------------------------------------------------------------
# Cutoffs / gap_end semantics
# ---------------------------------------------------------------------------

def test_cutoffs_son_max_kickoff_de_cada_bloque() -> None:
    ds = _dataset(_linear(12))
    folds = list(
        WalkForwardIterator(
            ds,
            min_train_size=3,
            test_size=2,
            gap_days=0,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    )
    for f in folds:
        assert f.training_cutoff == f.train_end
        assert f.validation_cutoff == f.val_end
        assert f.test_cutoff == f.test_end
        # Must equal max kickoff of corresponding block
        assert f.training_cutoff == max(ds.rows[i].kickoff for i in f.train_indices)
        assert f.validation_cutoff == max(ds.rows[i].kickoff for i in f.val_indices)
        assert f.test_cutoff == max(ds.rows[i].kickoff for i in f.test_indices)


def test_gap_end_es_val_start() -> None:
    ds = _dataset(_linear(12))
    folds = list(
        WalkForwardIterator(
            ds,
            min_train_size=3,
            test_size=2,
            gap_days=0,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    )
    for f in folds:
        assert f.gap_end == f.val_start


def test_gap_end_con_gap_positivo_sigue_siendo_val_start() -> None:
    ds = _dataset(_linear(14))
    folds = list(
        WalkForwardIterator(
            ds,
            min_train_size=3,
            test_size=2,
            gap_days=2,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    )
    for f in folds:
        assert f.gap_end == f.val_start
        # Temporal width is still verifiable via train_end vs gap_end
        assert f.train_end + timedelta(days=2) <= f.gap_end


# ---------------------------------------------------------------------------
# Fold invariants: non-empty, no overlap, monotonic fold_id, to_summary
# ---------------------------------------------------------------------------

def test_fold_nunca_vacio() -> None:
    ds = _dataset(_linear(12))
    for f in list(
        WalkForwardIterator(
            ds,
            min_train_size=3,
            test_size=2,
            gap_days=0,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    ):
        assert len(f.train_indices) > 0
        assert len(f.val_indices) > 0
        assert len(f.test_indices) > 0


def test_fold_sin_overlap_entre_bloques() -> None:
    ds = _dataset(_linear(14))
    for f in list(
        WalkForwardIterator(
            ds,
            min_train_size=3,
            test_size=2,
            gap_days=1,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    ):
        assert set(f.train_indices).isdisjoint(set(f.val_indices))
        assert set(f.train_indices).isdisjoint(set(f.test_indices))
        assert set(f.val_indices).isdisjoint(set(f.test_indices))


def test_fold_ids_monotonicos_desde_cero() -> None:
    ds = _dataset(_linear(14))
    folds = list(
        WalkForwardIterator(
            ds,
            min_train_size=3,
            test_size=2,
            gap_days=0,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    )
    assert [f.fold_id for f in folds] == list(range(len(folds)))


def test_fold_es_inmutable() -> None:
    ds = _dataset(_linear(8))
    f = list(
        WalkForwardIterator(
            ds,
            min_train_size=3,
            test_size=1,
            gap_days=0,
            mode=WalkForwardMode.EXPANDING,
            val_size=1,
        )
    )[0]
    with pytest.raises(AttributeError):
        f.fold_id = 99  # type: ignore[misc]
    with pytest.raises(AttributeError):
        f.train_indices = (0,)  # type: ignore[misc]


def test_fold_rechaza_overlap_en_construccion_directa() -> None:
    dt = T0
    with pytest.raises(ValueError, match="overlap"):
        Fold(
            fold_id=0,
            train_indices=(0, 1),
            val_indices=(1, 2),
            test_indices=(3,),
            train_start=dt,
            train_end=dt,
            val_start=dt,
            val_end=dt,
            test_start=dt,
            test_end=dt,
            gap_end=dt,
            training_cutoff=dt,
            validation_cutoff=dt,
            test_cutoff=dt,
        )


def test_fold_to_summary_json_serialisable() -> None:
    import json

    ds = _dataset(_linear(8))
    f = list(
        WalkForwardIterator(
            ds,
            min_train_size=3,
            test_size=1,
            gap_days=0,
            mode=WalkForwardMode.EXPANDING,
            val_size=1,
        )
    )[0]
    summary = f.to_summary()
    # Must survive json.dumps
    json.dumps(summary)
    assert summary["fold_id"] == 0
    assert summary["n_train"] == len(f.train_indices)
