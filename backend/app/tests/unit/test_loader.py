"""Tests for :mod:`app.dataset.loader`.

The loader is read-only and synchronous; tests build a tiny dataset on
disk in a ``tmp_path`` fixture and exercise the public API:

* :func:`load_dataset` happy path — types, feature/target separation,
  row_count, expected_version match.
* Empty-string cells parse to ``None``.
* Strong typing per column.
* :func:`validate_dataset` performs the SHA-256 check + header shape
  check and returns the manifest.
* Failure modes: missing CSV, missing metadata, header mismatch,
  manifest row_count mismatch, expected_version mismatch, sha256
  mismatch (the loader never silently coerces anything; every parse
  failure raises a :class:`DatasetLoadError` subclass).

The dataset bytes are produced by the builder's own serialiser
(:func:`app.dataset.builder._serialise_row`) so the round-trip
contract is exercised against the exact bytes the builder writes —
no parallel serialiser exists here.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.dataset._schema import CSV_COLUMNS, EMPTY_CELL
from app.dataset.builder import _serialise_row
from app.dataset.loader import (
    DatasetIntegrityError,
    DatasetLoadError,
    DatasetSchemaMismatch,
    DatasetVersionMismatch,
    LoadedDataset,
    load_dataset,
    load_manifest,
    validate_dataset,
)
from app.features.example import FEATURE_NAMES, HistoricalMatchExample, TARGET_NAMES


# ---------------------------------------------------------------------------
# Helpers — write a tiny dataset on disk using the builder's exact
# serialiser so loader↔builder round-trip is the contract under test.
# ---------------------------------------------------------------------------
def _example(
    *,
    fixture_id: int,
    kickoff: datetime,
    features: dict[str, float | int | None] | None = None,
    targets: dict[str, int | None] | None = None,
    competition_id: int = 39,
    season_id: int = 1,
    home_team_id: int = 10,
    away_team_id: int = 20,
) -> HistoricalMatchExample:
    return HistoricalMatchExample(
        fixture_id=fixture_id,
        kickoff=kickoff,
        competition_id=competition_id,
        season_id=season_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        features=features if features is not None else {},
        targets=targets if targets is not None else {},
    )


def _write_dataset(
    dir_path: Path,
    *,
    examples: list[HistoricalMatchExample],
    dataset_version: str = "v001",
    csv_sha256_override: str | None = None,
    extra_csv_lines: list[list[str]] | None = None,
    corrupt_sha256_in_manifest: bool = False,
) -> Path:
    """Write ``dataset.csv`` + ``metadata.json`` at ``dir_path``.

    Returns the dataset directory. Used by both the loader tests and
    the validate test below. The SHA-256 in the manifest is computed
    over the on-disk CSV unless ``csv_sha256_override`` / corruption
    flags are given.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    csv_path = dir_path / "dataset.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f, dialect="excel", lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL,
        )
        writer.writerow(CSV_COLUMNS)
        for ex in examples:
            writer.writerow(_serialise_row(ex))
        if extra_csv_lines:
            for line in extra_csv_lines:
                writer.writerow(line)

    actual_sha = _sha256_file(csv_path)

    meta_path = dir_path / "metadata.json"
    meta_obj = {
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
        "csv_sha256": (
            ("deadbeef" * 8)[:64] if corrupt_sha256_in_manifest
            else (csv_sha256_override or actual_sha)
        ),
    }
    meta_path.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")
    return dir_path


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_load_dataset_happy_path(tmp_path: Path) -> None:
    k1 = datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc)
    k2 = datetime(2026, 1, 17, 18, 0, tzinfo=timezone.utc)
    feats = {FEATURE_NAMES[0]: 1, FEATURE_NAMES[1]: 2.5}
    tgts = {TARGET_NAMES[0]: 1, TARGET_NAMES[1]: 0, TARGET_NAMES[2]: 0,
            TARGET_NAMES[3]: 2, TARGET_NAMES[4]: 1}
    examples = [
        _example(fixture_id=1, kickoff=k1, features=feats, targets=tgts),
        _example(fixture_id=2, kickoff=k2),
    ]
    d = _write_dataset(tmp_path / "v001", examples=examples, dataset_version="v001")

    loaded = load_dataset(d)

    assert isinstance(loaded, LoadedDataset)
    assert loaded.manifest.dataset_version == "v001"
    assert loaded.manifest.row_count == 2
    assert len(loaded.rows) == 2

    # Deterministic chronological order: sorted by kickoff then fixture_id.
    assert [r.fixture_id for r in loaded.rows] == [1, 2]
    assert loaded.rows[0].kickoff == k1
    assert loaded.rows[1].kickoff == k2


def test_loader_returns_strongly_typed_columns(tmp_path: Path) -> None:
    k1 = datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc)
    # Pick one feature that is float-typed and one int-typed so we can
    # assert the loader preserves the type distinction.
    feats: dict[str, float | int | None] = {
        FEATURE_NAMES[0]: 5,             # int feature
        "h2h_total_goals_mean": 2.5,     # float feature (if present)
    }
    tgts: dict[str, int | None] = {
        TARGET_NAMES[0]: 1, TARGET_NAMES[1]: 0, TARGET_NAMES[2]: 0,
        TARGET_NAMES[3]: 3, TARGET_NAMES[4]: 2,
    }
    examples = [_example(fixture_id=1, kickoff=k1, features=feats, targets=tgts)]
    d = _write_dataset(tmp_path / "v001", examples=examples)

    loaded = load_dataset(d)
    row = loaded.rows[0]

    # Identity columns — strict ints + tz-aware datetime.
    assert isinstance(row.fixture_id, int)
    assert isinstance(row.competition_id, int)
    assert isinstance(row.season_id, int)
    assert isinstance(row.home_team_id, int)
    assert isinstance(row.away_team_id, int)
    assert isinstance(row.kickoff, datetime)
    assert row.kickoff.tzinfo is not None

    # The feature dict only contains FEATURE_NAMES (no targets).
    assert set(row.features.keys()) == set(FEATURE_NAMES)
    assert set(row.targets.keys()) == set(TARGET_NAMES)
    assert set(row.features).isdisjoint(set(row.targets))

    # Stored int feature comes back as int (not float 0.0).
    assert isinstance(row.features[FEATURE_NAMES[0]], int)
    assert row.features[FEATURE_NAMES[0]] == 5


def test_loader_separates_features_and_targets(tmp_path: Path) -> None:
    k1 = datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc)
    feats = {FEATURE_NAMES[0]: 7}
    tgts = {TARGET_NAMES[3]: 3, TARGET_NAMES[4]: 1}
    examples = [_example(fixture_id=1, kickoff=k1, features=feats, targets=tgts)]
    d = _write_dataset(tmp_path / "v001", examples=examples)

    loaded = load_dataset(d)
    row = loaded.rows[0]

    assert row.features[FEATURE_NAMES[0]] == 7
    assert row.targets[TARGET_NAMES[3]] == 3
    assert row.targets[TARGET_NAMES[4]] == 1
    # Targets are NEVER reachable via the features dict.
    for tname in TARGET_NAMES:
        assert tname not in row.features


def test_loader_empty_string_parses_to_none(tmp_path: Path) -> None:
    """A cell of ``""`` becomes Python ``None`` for feature and target
    cells — never ``0`` — preserving the "math-zero is real" contract.
    """
    k1 = datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc)
    # No feature values supplied -> every feature is None on disk.
    examples = [_example(fixture_id=1, kickoff=k1)]
    d = _write_dataset(tmp_path / "v001", examples=examples)

    loaded = load_dataset(d)
    row = loaded.rows[0]
    for f in FEATURE_NAMES:
        assert row.features[f] is None, f"{f} should be None for empty CSV cell"
    for t in TARGET_NAMES:
        assert row.targets[t] is None


# ---------------------------------------------------------------------------
# expected_version
# ---------------------------------------------------------------------------
def test_expected_version_match_passes(tmp_path: Path) -> None:
    examples = [_example(fixture_id=1,
                         kickoff=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc))]
    d = _write_dataset(tmp_path / "v001", examples=examples, dataset_version="v042")
    loaded = load_dataset(d, expected_version="v042")
    assert loaded.manifest.dataset_version == "v042"


def test_expected_version_mismatch_raises(tmp_path: Path) -> None:
    examples = [_example(fixture_id=1,
                         kickoff=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc))]
    d = _write_dataset(tmp_path / "v001", examples=examples, dataset_version="v042")
    with pytest.raises(DatasetVersionMismatch):
        load_dataset(d, expected_version="v999")


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------
def test_load_dataset_missing_csv_raises(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    # Only metadata.json — no dataset.csv.
    (d / "metadata.json").write_text(
        json.dumps({
            "dataset_version": "v001",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_definition_version": "fd_v1",
            "data_cutoff": "2026-08-01T00:00:00+00:00",
            "source_schema_version": "schema_v1",
            "row_count": 0,
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "start_date": "2026-01-01T00:00:00+00:00",
            "end_date": "2026-01-01T00:00:00+00:00",
            "competitions": [39],
            "seasons": [1],
            "csv_sha256": "0" * 64,
        }),
        encoding="utf-8",
    )
    with pytest.raises(DatasetLoadError):
        load_dataset(d)


def test_load_dataset_missing_metadata_raises(tmp_path: Path) -> None:
    d = tmp_path / "no_meta"
    d.mkdir()
    (d / "dataset.csv").write_text(",".join(CSV_COLUMNS) + "\r\n", encoding="utf-8")
    with pytest.raises(DatasetLoadError):
        load_dataset(d)


def test_load_dataset_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoadError):
        load_dataset(tmp_path / "nope")


def test_load_dataset_header_mismatch_raises(tmp_path: Path) -> None:
    d = tmp_path / "v001"
    d.mkdir()
    # Write a CSV with an unexpected header.
    with (d / "dataset.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\r\n").writerow(["unexpected", "header"])
    (d / "metadata.json").write_text(
        json.dumps({
            "dataset_version": "v001",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_definition_version": "fd_v1",
            "data_cutoff": "2026-08-01T00:00:00+00:00",
            "source_schema_version": "schema_v1",
            "row_count": 0,
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "start_date": "2026-01-01T00:00:00+00:00",
            "end_date": "2026-01-01T00:00:00+00:00",
            "competitions": [39],
            "seasons": [1],
            "csv_sha256": "0" * 64,
        }),
        encoding="utf-8",
    )
    with pytest.raises(DatasetSchemaMismatch):
        load_dataset(d)


def test_load_dataset_row_count_mismatch_raises(tmp_path: Path) -> None:
    examples = [
        _example(fixture_id=i,
                 kickoff=datetime(2026, 1, i, 15, 0, tzinfo=timezone.utc))
        for i in range(1, 4)
    ]
    # Manifest claims 1 row; CSV has 3.
    d = tmp_path / "v001"
    d.mkdir()
    csv_path = d / "dataset.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f, dialect="excel", lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL,
        )
        writer.writerow(CSV_COLUMNS)
        for ex in examples:
            writer.writerow(_serialise_row(ex))
    sha = _sha256_file(csv_path)
    (d / "metadata.json").write_text(
        json.dumps({
            "dataset_version": "v001",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_definition_version": "fd_v1",
            "data_cutoff": "2026-08-01T00:00:00+00:00",
            "source_schema_version": "schema_v1",
            "row_count": 1,  # WRONG
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "start_date": "2026-01-01T00:00:00+00:00",
            "end_date": "2026-01-01T00:00:00+00:00",
            "competitions": [39],
            "seasons": [1],
            "csv_sha256": sha,
        }),
        encoding="utf-8",
    )
    with pytest.raises(DatasetLoadError):
        load_dataset(d)


def test_load_dataset_verify_sha256_match_passes(tmp_path: Path) -> None:
    examples = [_example(
        fixture_id=1, kickoff=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc),
        features={FEATURE_NAMES[0]: 3})]
    d = _write_dataset(tmp_path / "v001", examples=examples)
    # Should not raise — manifest hashes match the actual on-disk bytes.
    loaded = load_dataset(d, verify_sha256=True)
    assert loaded.manifest.row_count == 1


def test_load_dataset_verify_sha256_mismatch_raises(tmp_path: Path) -> None:
    examples = [_example(
        fixture_id=1, kickoff=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc))]
    d = _write_dataset(tmp_path / "v001", examples=examples,
                       corrupt_sha256_in_manifest=True)
    with pytest.raises(DatasetIntegrityError):
        load_dataset(d, verify_sha256=True)


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------
def test_validate_dataset_returns_manifest_on_success(tmp_path: Path) -> None:
    examples = [_example(
        fixture_id=1, kickoff=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc))]
    d = _write_dataset(tmp_path / "v001", examples=examples)
    m = validate_dataset(d)
    assert m.dataset_version == "v001"
    assert m.row_count == 1


def test_validate_dataset_missing_csv_raises(tmp_path: Path) -> None:
    d = tmp_path / "v001"
    d.mkdir()
    (d / "metadata.json").write_text(
        json.dumps({
            "dataset_version": "v001",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_definition_version": "fd_v1",
            "data_cutoff": "2026-08-01T00:00:00+00:00",
            "source_schema_version": "schema_v1",
            "row_count": 0,
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "start_date": "2026-01-01T00:00:00+00:00",
            "end_date": "2026-01-01T00:00:00+00:00",
            "competitions": [39],
            "seasons": [1],
            "csv_sha256": "0" * 64,
        }),
        encoding="utf-8",
    )
    with pytest.raises(DatasetLoadError):
        validate_dataset(d)


def test_validate_dataset_sha256_mismatch_raises(tmp_path: Path) -> None:
    examples = [_example(
        fixture_id=1, kickoff=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc))]
    d = _write_dataset(tmp_path / "v001", examples=examples,
                       corrupt_sha256_in_manifest=True)
    # validate_dataset ALWAYS recomputes the SHA-256 — this is the
    # explicit verification step (the loader never does it unless asked).
    with pytest.raises(DatasetIntegrityError):
        validate_dataset(d)


def test_validate_dataset_header_mismatch_raises(tmp_path: Path) -> None:
    d = tmp_path / "v001"
    d.mkdir()
    with (d / "dataset.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\r\n").writerow(["wrong", "columns"])
    (d / "metadata.json").write_text(
        json.dumps({
            "dataset_version": "v001",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_definition_version": "fd_v1",
            "data_cutoff": "2026-08-01T00:00:00+00:00",
            "source_schema_version": "schema_v1",
            "row_count": 0,
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "start_date": "2026-01-01T00:00:00+00:00",
            "end_date": "2026-01-01T00:00:00+00:00",
            "competitions": [39],
            "seasons": [1],
            "csv_sha256": "0" * 64,
        }),
        encoding="utf-8",
    )
    with pytest.raises(DatasetSchemaMismatch):
        validate_dataset(d)


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------
def test_load_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoadError):
        load_manifest(tmp_path)


def test_load_manifest_invalid_json_raises(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text("not valid json {", encoding="utf-8")
    with pytest.raises(DatasetLoadError):
        load_manifest(tmp_path)
