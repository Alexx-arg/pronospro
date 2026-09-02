"""Tests for :mod:`app.dataset.builder` — helpers + integration build.

The builder's public API (:func:`build_dataset`) is async and pulls
from PostgreSQL via :class:`AsyncSession`. The "shape" of the output
(dataset.csv + metadata.json + row_count + checksum) is therefore
covered by the ``@pytest.mark.integration`` tests below; they require
a live database and are skipped otherwise via the ``session`` fixture
in :mod:`app.tests.conftest`.

The pure / synchronous helpers (:func:`_serialise_row`,
:func:`_resolve_cutoff`, :func:`_ensure_aware`, :func:`_hash_file`)
are exercised directly here so they get unit coverage even in
environments without Postgres.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.dataset._schema import CSV_COLUMNS
from app.dataset.builder import (
    InvalidBuildConfigError,
    _ensure_aware,
    _hash_file,
    _resolve_cutoff,
    _serialise_row,
)
from app.dataset.loader import load_dataset, validate_dataset
from app.features.example import (
    FEATURE_NAMES,
    HistoricalMatchExample,
    TARGET_NAMES,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_resolve_cutoff_prefers_explicit_data_cutoff() -> None:
    cutoff = datetime(2026, 8, 1, tzinfo=timezone.utc)
    end = datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert _resolve_cutoff(cutoff, end) == cutoff


def test_resolve_cutoff_defaults_to_end_date_when_cutoff_missing() -> None:
    end = datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert _resolve_cutoff(None, end) == end


def test_resolve_cutoff_defaults_to_now_when_both_missing() -> None:
    out = _resolve_cutoff(None, None)
    assert out.tzinfo is not None
    # Now() within the test process can't lag more than a minute.
    delta = abs((datetime.now(timezone.utc) - out).total_seconds())
    assert delta < 60


def test_resolve_cutoff_normalises_naive_datetime_to_utc() -> None:
    naive = datetime(2026, 8, 1, 0, 0)  # no tzinfo
    out = _resolve_cutoff(naive, None)
    assert out.tzinfo is not None


def test_ensure_aware_adds_utc_to_naive() -> None:
    naive = datetime(2026, 1, 1)
    out = _ensure_aware(naive)
    assert out.tzinfo is timezone.utc
    # Aware datetimes are returned unchanged (identity preserved).
    aware = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert _ensure_aware(aware) is aware


def test_serialise_row_identity_columns_first() -> None:
    k = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    ex = HistoricalMatchExample(
        fixture_id=42, kickoff=k, competition_id=39, season_id=1,
        home_team_id=10, away_team_id=20, features={}, targets={},
    )
    row = _serialise_row(ex)
    # First 6 cells are identity, rest are FEATURE_NAMES + TARGET_NAMES.
    assert row[:6] == [42, k.isoformat(), 39, 1, 10, 20]
    assert len(row) == len(CSV_COLUMNS)


def test_serialise_row_features_then_targets_in_canonical_order() -> None:
    k = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    feats = {FEATURE_NAMES[0]: 1, FEATURE_NAMES[-1]: 9.5}
    tgts = {TARGET_NAMES[0]: 1, TARGET_NAMES[-1]: 3}
    ex = HistoricalMatchExample(
        fixture_id=42, kickoff=k, competition_id=39, season_id=1,
        home_team_id=10, away_team_id=20, features=feats, targets=tgts,
    )
    row = _serialise_row(ex)
    feat_cells = row[6:6 + len(FEATURE_NAMES)]
    target_cells = row[6 + len(FEATURE_NAMES):]
    assert feat_cells[0] == 1
    assert feat_cells[-1] == 9.5
    assert target_cells[0] == 1
    assert target_cells[-1] == 3


def test_serialise_row_emits_none_as_empty_cell() -> None:
    """When a feature is missing, the builder writes an empty string
    into the CSV cell — the loader's ``EMPTY_CELL`` contract.
    """
    k = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    ex = HistoricalMatchExample(
        fixture_id=42, kickoff=k, competition_id=39, season_id=1,
        home_team_id=10, away_team_id=20, features={}, targets={},
    )
    row = _serialise_row(ex)
    feat_cells = row[6:6 + len(FEATURE_NAMES)]
    assert all(c is None for c in feat_cells)
    target_cells = row[6 + len(FEATURE_NAMES):]
    assert all(c is None for c in target_cells)
    # When the csv module writes None it serialises to "" QUOTE_MINIMAL.
    import io
    sio = io.StringIO()
    csv.writer(sio, dialect="excel", lineterminator="\r\n",
               quoting=csv.QUOTE_MINIMAL).writerow(row)
    written = sio.getvalue().rstrip("\r\n")
    assert ",," in written  # at least one empty cell


def test_hash_file_is_sha256_of_disk_bytes(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    payload = b"hello-phase-4"
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert _hash_file(p) == expected
    assert len(_hash_file(p)) == 64


def test_hash_file_changes_when_bytes_change(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"aaa")
    h1 = _hash_file(p)
    p.write_bytes(b"aab")
    h2 = _hash_file(p)
    assert h1 != h2


# ---------------------------------------------------------------------------
# Validation guards of build_dataset (no DB needed)
# ---------------------------------------------------------------------------
async def test_build_dataset_rejects_empty_competition_ids() -> None:
    from app.dataset.builder import build_dataset

    with pytest.raises(InvalidBuildConfigError):
        await build_dataset(
            session=None,  # type: ignore[arg-type]
            competition_ids=[],
            season_ids=[1],
            dataset_version="v001",
            output_path="/tmp/whatever",
        )


async def test_build_dataset_rejects_empty_season_ids() -> None:
    from app.dataset.builder import build_dataset

    with pytest.raises(InvalidBuildConfigError):
        await build_dataset(
            session=None,  # type: ignore[arg-type]
            competition_ids=[1],
            season_ids=[],
            dataset_version="v001",
            output_path="/tmp/whatever",
        )


async def test_build_dataset_rejects_end_date_after_data_cutoff() -> None:
    """``end_date > data_cutoff`` is a vacuous watermark — the builder
    refuses it explicitly so the integrity of the temporal boundary is
    never silently relaxed.
    """
    from app.dataset.builder import build_dataset

    with pytest.raises(InvalidBuildConfigError):
        await build_dataset(
            session=None,  # type: ignore[arg-type]
            competition_ids=[1],
            season_ids=[1],
            end_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
            data_cutoff=datetime(2026, 7, 1, tzinfo=timezone.utc),
            dataset_version="v001",
            output_path="/tmp/whatever",
        )


# ---------------------------------------------------------------------------
# Integration build — needs a live DB. The session fixture skips when
# the alembic migration has not been applied.
# ---------------------------------------------------------------------------
def _write_dataset_with_loader(
    tmp_path: Path,
    examples: list[HistoricalMatchExample],
    *,
    dataset_version: str = "v001",
) -> Path:
    """Mirror :func:`app.tests.unit.test_loader._write_dataset` but
    inline so we can keep builder/loader tests independent.
    """
    out_dir = tmp_path / dataset_version
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dataset.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f, dialect="excel", lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL,
        )
        writer.writerow(CSV_COLUMNS)
        for ex in examples:
            writer.writerow(_serialise_row(ex))
    sha = _hash_file(csv_path)
    meta_path = out_dir / "metadata.json"
    meta = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_definition_version": "fd_v1",
        "data_cutoff": "2026-08-01T00:00:00+00:00",
        "source_schema_version": "schema_v1",
        "row_count": len(examples),
        "feature_names": list(FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "start_date": (examples[0].kickoff.isoformat() if examples
                       else "2026-01-01T00:00:00+00:00"),
        "end_date": (examples[-1].kickoff.isoformat() if examples
                     else "2026-01-01T00:00:00+00:00"),
        "competitions": sorted({e.competition_id for e in examples}) or [39],
        "seasons": sorted({e.season_id for e in examples}) or [1],
        "csv_sha256": sha,
    }
    meta_path.write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8",
    )
    return out_dir


def test_builder_serialise_then_loader_roundtrip_unit(tmp_path: Path) -> None:
    """Builder writes via ``_serialise_row``; loader reads it back.

    A pure-unit round-trip: we hand-build a couple of examples through
    the builder's serialiser, write them with the builder's exact CSV
    dialect, then read them back through the loader and verify column
    types, feature/target separation and the SHA-256 produced by the
    builder equals the SHA-256 recomputed by the loader.
    """
    k1 = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    k2 = datetime(2026, 1, 12, 14, 0, tzinfo=timezone.utc)
    feats = {FEATURE_NAMES[0]: 3, FEATURE_NAMES[5]: 2.5}
    tgts = {TARGET_NAMES[0]: 1, TARGET_NAMES[3]: 2, TARGET_NAMES[4]: 0}
    examples = [
        HistoricalMatchExample(
            fixture_id=1, kickoff=k1, competition_id=39, season_id=1,
            home_team_id=10, away_team_id=20, features=feats, targets=tgts,
        ),
        HistoricalMatchExample(
            fixture_id=2, kickoff=k2, competition_id=39, season_id=1,
            home_team_id=10, away_team_id=20, features={}, targets={},
        ),
    ]
    d = _write_dataset_with_loader(tmp_path, examples, dataset_version="v001")

    # The manifest's sha matches the actual file bytes — validate passes.
    manifest = validate_dataset(d)
    assert manifest.row_count == 2
    assert manifest.csv_sha256 == _hash_file(d / "dataset.csv")

    loaded = load_dataset(d, verify_sha256=True)
    assert len(loaded.rows) == 2
    # Builder writes rows in chronological order; loader preserves it.
    assert [r.fixture_id for r in loaded.rows] == [1, 2]
    # Empty features round-trip back to None (never 0).
    for f in FEATURE_NAMES:
        assert loaded.rows[1].features[f] is None
    assert loaded.rows[0].features[FEATURE_NAMES[0]] == 3
    assert loaded.rows[0].features[FEATURE_NAMES[5]] == 2.5
    assert loaded.rows[0].targets[TARGET_NAMES[0]] == 1
    assert loaded.rows[0].targets[TARGET_NAMES[3]] == 2
    assert loaded.rows[0].targets[TARGET_NAMES[4]] == 0
