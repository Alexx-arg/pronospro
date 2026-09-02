"""Sprint 5.1 — No fixture overlap between train/val/test.

Anti-leakage guarantee: no ``fixture_id`` that appears in ``test``
(or ``val``) may appear in ``train``; no index may be shared between
any two blocks; adding future fixtures must not alter earlier folds.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.dataset.loader import LoadedDataset, LoadedExample
from app.dataset.manifest import DatasetManifest
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
# No index appears in two blocks of the same fold
# ---------------------------------------------------------------------------

def test_no_overlap_indices_train_val_test() -> None:
    ds = _dataset(_linear(18))
    for f in list(
        WalkForwardIterator(
            ds,
            min_train_size=4,
            test_size=2,
            gap_days=1,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    ):
        s_tr = set(f.train_indices)
        s_va = set(f.val_indices)
        s_te = set(f.test_indices)
        assert s_tr.isdisjoint(s_va), f"train/val overlap in fold {f.fold_id}"
        assert s_tr.isdisjoint(s_te), f"train/test overlap in fold {f.fold_id}"
        assert s_va.isdisjoint(s_te), f"val/test overlap in fold {f.fold_id}"


# ---------------------------------------------------------------------------
# No fixture_id appears in train that also appears in val/test
# ---------------------------------------------------------------------------

def test_test_fixture_nunca_aparece_en_train() -> None:
    ds = _dataset(_linear(18))
    for f in list(
        WalkForwardIterator(
            ds,
            min_train_size=4,
            test_size=2,
            gap_days=0,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    ):
        train_fids = {ds.rows[i].fixture_id for i in f.train_indices}
        test_fids = {ds.rows[i].fixture_id for i in f.test_indices}
        assert train_fids.isdisjoint(test_fids), f"leak train→test fold {f.fold_id}"


def test_val_fixture_nunca_aparece_en_train() -> None:
    ds = _dataset(_linear(18))
    for f in list(
        WalkForwardIterator(
            ds,
            min_train_size=4,
            test_size=2,
            gap_days=0,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    ):
        train_fids = {ds.rows[i].fixture_id for i in f.train_indices}
        val_fids = {ds.rows[i].fixture_id for i in f.val_indices}
        assert train_fids.isdisjoint(val_fids), f"leak train→val fold {f.fold_id}"


def test_test_fixture_nunca_aparece_en_val() -> None:
    ds = _dataset(_linear(18))
    for f in list(
        WalkForwardIterator(
            ds,
            min_train_size=4,
            test_size=2,
            gap_days=0,
            mode=WalkForwardMode.EXPANDING,
            val_size=2,
        )
    ):
        val_fids = {ds.rows[i].fixture_id for i in f.val_indices}
        test_fids = {ds.rows[i].fixture_id for i in f.test_indices}
        assert val_fids.isdisjoint(test_fids), f"leak val→test fold {f.fold_id}"


# ---------------------------------------------------------------------------
# Future data does not retroactively alter earlier folds
# ---------------------------------------------------------------------------

def test_futuros_no_alteran_folds_previos_expanding() -> None:
    rows_short = _linear(14)
    rows_long = _linear(20)
    ds_short = _dataset(rows_short)
    ds_long = _dataset(rows_long)
    kwargs = {
        "min_train_size": 4,
        "test_size": 2,
        "gap_days": 0,
        "mode": WalkForwardMode.EXPANDING,
        "val_size": 2,
    }
    short_folds = list(WalkForwardIterator(ds_short, **kwargs))
    long_folds = list(WalkForwardIterator(ds_long, **kwargs))
    # First N folds must be identical; future rows only add folds
    assert len(long_folds) >= len(short_folds)
    for s, lg in zip(short_folds, long_folds, strict=False):
        assert s.train_indices == lg.train_indices
        assert s.val_indices == lg.val_indices
        assert s.test_indices == lg.test_indices


def test_futuros_no_alteran_folds_previos_sliding() -> None:
    rows_short = _linear(14)
    rows_long = _linear(20)
    ds_short = _dataset(rows_short)
    ds_long = _dataset(rows_long)
    kwargs = {
        "min_train_size": 4,
        "test_size": 2,
        "gap_days": 0,
        "mode": WalkForwardMode.SLIDING,
        "val_size": 2,
    }
    short_folds = list(WalkForwardIterator(ds_short, **kwargs))
    long_folds = list(WalkForwardIterator(ds_long, **kwargs))
    assert len(long_folds) >= len(short_folds)
    for s, lg in zip(short_folds, long_folds, strict=False):
        assert s.train_indices == lg.train_indices


# ---------------------------------------------------------------------------
# No index reused across folds in expanding mode (train grows but test
# indices are always new).
# ---------------------------------------------------------------------------

def test_test_indices_across_folds_son_distintos() -> None:
    ds = _dataset(_linear(24))
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
    seen_test: set[int] = set()
    for f in folds:
        assert seen_test.isdisjoint(set(f.test_indices))
        seen_test.update(f.test_indices)


# ---------------------------------------------------------------------------
# Timestamps idénticos: ninguna partición divide un mismo kickoff
# ---------------------------------------------------------------------------

def test_no_split_mismo_kickoff_entre_train_val() -> None:
    # Tres fixtures al mismo kickoff T0, luego tres en T1
    rows = [
        _row(1, T0),
        _row(2, T0),
        _row(3, T0),
        _row(4, T0 + timedelta(days=1)),
        _row(5, T0 + timedelta(days=1)),
        _row(6, T0 + timedelta(days=1)),
        _row(7, T0 + timedelta(days=2)),
        _row(8, T0 + timedelta(days=3)),
        _row(9, T0 + timedelta(days=4)),
    ]
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=3,
        test_size=1,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=1,
    )
    for f in it:
        train_kicks = {ds.rows[i].kickoff for i in f.train_indices}
        val_kicks = {ds.rows[i].kickoff for i in f.val_indices}
        test_kicks = {ds.rows[i].kickoff for i in f.test_indices}
        assert train_kicks.isdisjoint(val_kicks)
        assert val_kicks.isdisjoint(test_kicks)


def test_no_split_mismo_kickoff_entre_val_test() -> None:
    rows = [
        _row(1, T0),
        _row(2, T0 + timedelta(days=1)),
        _row(3, T0 + timedelta(days=2)),
        _row(4, T0 + timedelta(days=3)),
        _row(5, T0 + timedelta(days=3)),
        _row(6, T0 + timedelta(days=3)),
        _row(7, T0 + timedelta(days=4)),
        _row(8, T0 + timedelta(days=5)),
    ]
    ds = _dataset(rows)
    it = WalkForwardIterator(
        ds,
        min_train_size=3,
        test_size=1,
        gap_days=0,
        mode=WalkForwardMode.EXPANDING,
        val_size=2,
    )
    for f in it:
        val_kicks = {ds.rows[i].kickoff for i in f.val_indices}
        test_kicks = {ds.rows[i].kickoff for i in f.test_indices}
        assert val_kicks.isdisjoint(test_kicks)
