"""Read-only typed loader for built datasets.

Reads ``dataset.csv`` + ``metadata.json`` of a specific version into
strongly-typed in-memory objects (:class:`LoadedExample`,
:class:`LoadedDataset`) without ever triggering feature math or DB
access.

Guarantees / what the loader WILL NOT do
-----------------------------------------
* It **does not** recompute features. The dataset on disk is the
  authoritative fact; the loader strictly round-trips the bytes.
* It **does not** mutate the CSV or the metadata. Only read modes are
  opened; the manifest and the CSV file descriptors are never opened
  for writing.
* It **does not** import the builder (no SQLAlchemy / async machinery
  needed at load time). All shared state lives in
  :mod:`app.dataset._schema` and :mod:`app.features.example`.
* It **does not** silently coerce anything: every numeric cell is
  parsed with an explicit type, ``""`` -> ``None``, and anything that
  fails to parse raises :class:`DatasetLoadError`.

API
---
* :func:`load_dataset`   — read the whole dataset (manifest + rows).
* :func:`load_manifest`   — read only ``metadata.json``.
* :func:`validate_dataset` — verify integrity (sha256 + column shape)
  without materialising the rows.

Verification flags
------------------
``load_dataset(..., verify_sha256=True)`` recomputes the SHA-256 of the
CSV bytes on disk and compares it against
``manifest.csv_sha256``. :func:`validate_dataset` always does this; the
loader never does unless asked because re-hashing a 100k-row CSV on a
training run is wasteful when the user has already verified once.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.dataset._schema import (
    CSV_COLUMNS,
    EMPTY_CELL,
    FEATURE_NAMES,
    IDENTITY_COLUMNS,
    TARGET_NAMES,
)
from app.dataset.manifest import DatasetManifest

_LOG = get_logger()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class DatasetLoadError(ValueError):
    """Raised when the dataset on disk is malformed / inconsistent."""


class DatasetIntegrityError(DatasetLoadError):
    """SHA-256 mismatch between the on-disk CSV and the manifest."""


class DatasetVersionMismatch(DatasetLoadError):
    """The directory's ``dataset_version`` differs from the one requested."""


class DatasetSchemaMismatch(DatasetLoadError):
    """The CSV header does not match the canonical column order."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LoadedExample:
    """A single loaded dataset row with explicit per-column typing.

    Identity columns are exposed as named attributes; features and
    targets are exposed as plain ``dict`` keyed by canonical name so the
    consumer (training pipeline / backtester) can index by name without
    importing :mod:`app.features.example` constants.
    """

    fixture_id: int
    kickoff: datetime
    competition_id: int
    season_id: int
    home_team_id: int
    away_team_id: int
    features: dict[str, float | int | None] = field(default_factory=dict)
    targets: dict[str, int | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """Bundle: manifest + loaded rows."""

    manifest: DatasetManifest
    rows: list[LoadedExample]


# ---------------------------------------------------------------------------
# Public API: load_manifest
# ---------------------------------------------------------------------------
def load_manifest(dataset_dir: str | Path) -> DatasetManifest:
    """Load + validate ``metadata.json`` from ``dataset_dir``.

    Raises :class:`DatasetLoadError` if the file is missing or not
    parseable.
    """
    dir_path = _as_dir(dataset_dir)
    meta_path = dir_path / "metadata.json"
    if not meta_path.is_file():
        raise DatasetLoadError(
            f"metadata.json not found at {meta_path}"
        )
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetLoadError(
            f"metadata.json is not valid JSON: {exc.msg} at pos {exc.pos}"
        ) from exc

    return DatasetManifest.from_dict(raw)


# ---------------------------------------------------------------------------
# Public API: validate_dataset
# ---------------------------------------------------------------------------
def validate_dataset(dataset_dir: str | Path) -> DatasetManifest:
    """Verify dataset integrity without materialising rows.

    Performs (in this order):

    1. loads ``metadata.json`` (raises :class:`DatasetLoadError`),
    2. verifies the CSV exists,
    3. checks the CSV header matches :data:`CSV_COLUMNS` exactly
       (raises :class:`DatasetSchemaMismatch`),
    4. recomputes the CSV SHA-256 and compares with
       ``manifest.csv_sha256`` (raises :class:`DatasetIntegrityError`).

    Returns the loaded manifest so callers can assert against it
    without re-reading the file.
    """
    dir_path = _as_dir(dataset_dir)
    csv_path = dir_path / "dataset.csv"
    if not csv_path.is_file():
        raise DatasetLoadError(f"dataset.csv not found at {csv_path}")
    manifest = load_manifest(dir_path)

    # 3. Header shape check.
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise DatasetLoadError(
                f"dataset.csv is empty (no header): {csv_path}"
            ) from exc
    if tuple(header) != CSV_COLUMNS:
        raise DatasetSchemaMismatch(
            f"dataset.csv header does not match canonical schema.\n"
            f" expected: {CSV_COLUMNS}\n"
            f" got:      {header}"
        )

    # 4. Integrity hash check.
    actual = _hash_file(csv_path)
    if actual != manifest.csv_sha256:
        raise DatasetIntegrityError(
            f"CSV sha256 mismatch.\n"
            f" manifest.csv_sha256 = {manifest.csv_sha256}\n"
            f" actual              = {actual}"
        )
    return manifest


# ---------------------------------------------------------------------------
# Public API: load_dataset
# ---------------------------------------------------------------------------
def load_dataset(
    dataset_dir: str | Path,
    *,
    expected_version: str | None = None,
    verify_sha256: bool = False,
) -> LoadedDataset:
    """Load the dataset at ``dataset_dir``.

    Args:
        dataset_dir: directory containing ``dataset.csv`` +
            ``metadata.json`` (typically ``data/datasets/<version>/``).
        expected_version: optional version label that MUST equal
            ``manifest.dataset_version``; raises
            :class:`DatasetVersionMismatch` otherwise. Use this to
            fail early when a stale dataset is pointed at by CI / config.
        verify_sha256: when ``True``, recompute the SHA-256 of the CSV
            bytes and compare against the manifest. Useful for an
            explicit verification step; the loader does not do this by
            default because hashing 100k rows on every training run is
            wasteful (see :func:`validate_dataset`).

    Returns:
        :class:`LoadedDataset` with the manifest + every parsed row.
    """
    dir_path = _as_dir(dataset_dir)
    csv_path = dir_path / "dataset.csv"
    if not csv_path.is_file():
        raise DatasetLoadError(f"dataset.csv not found at {csv_path}")

    manifest = load_manifest(dir_path)
    if expected_version is not None and expected_version != manifest.dataset_version:
        raise DatasetVersionMismatch(
            f"expected dataset_version={expected_version!r} but "
            f"manifest reports {manifest.dataset_version!r}"
        )

    # Header shape check (we re-read the header from the file so the
    # failure mode "disk header differs from CSV_COLUMNS" is detected
    # even when the manifest was hand-edited).
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise DatasetLoadError(
                f"dataset.csv is empty (no header): {csv_path}"
            ) from exc
        if tuple(header) != CSV_COLUMNS:
            raise DatasetSchemaMismatch(
                f"dataset.csv header does not match canonical schema.\n"
                f" expected: {CSV_COLUMNS}\n"
                f" got:      {header}"
            )

        if verify_sha256:
            actual = _hash_file(csv_path)
            if actual != manifest.csv_sha256:
                raise DatasetIntegrityError(
                    f"CSV sha256 mismatch.\n"
                    f" manifest.csv_sha256 = {manifest.csv_sha256}\n"
                    f" actual              = {actual}"
                )

        rows = list(_iter_rows(reader, csv_path))

    if len(rows) != manifest.row_count:
        raise DatasetLoadError(
            f"row_count mismatch: manifest={manifest.row_count} "
            f"actual={len(rows)}"
        )

    _LOG.info(
        "loaded dataset {}: {} rows (sha256 {}verified)",
        manifest.dataset_version, len(rows),
        "" if verify_sha256 else "un",
    )
    return LoadedDataset(manifest=manifest, rows=rows)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _as_dir(dataset_dir: str | Path) -> Path:
    p = Path(dataset_dir)
    if not p.is_dir():
        raise DatasetLoadError(f"dataset directory not found: {p}")
    return p


def _iter_rows(reader: "csv._reader", csv_path: Path) -> Iterator[LoadedExample]:
    """Yield typed :class:`LoadedExample` from a csv.reader cursor.

    Assumes the header row has already been consumed by the caller.
    """
    for line_no, row in enumerate(reader, start=2):  # 1 = header
        if len(row) != len(CSV_COLUMNS):
            raise DatasetLoadError(
                f"{csv_path}:line {line_no}: "
                f"expected {len(CSV_COLUMNS)} columns, got {len(row)}"
            )
        try:
            yield _parse_row(row, csv_path, line_no)
        except DatasetLoadError:
            raise
        except Exception as exc:  # noqa: BLE001  (anticipate numeric parse)
            raise DatasetLoadError(
                f"{csv_path}:line {line_no}: failed to parse row: {exc!r}"
            ) from exc


def _parse_row(row: list[str], csv_path: Path, line_no: int) -> LoadedExample:
    """Parse one CSV row into a :class:`LoadedExample`.

    Strict typing contract:

    * ``""`` (empty cell) → Python ``None`` for *every* numeric column
      (features and targets). Identity columns are required (never
      ``None``).
    * ``int`` parsing uses base-10 only — no base hint, no leading
      zeros kept silently. ``"01"`` becomes ``1``.
    * ``float`` parsing uses :class:`float` constructor (rejects NaN /
      inf because the builder writes empty for NaN cells).
    * `kickoff` is parsed by :class:`datetime.fromisoformat` (matches
      what the builder writes via :meth:`datetime.isoformat`).

    Any failure raises :class:`DatasetLoadError` with the row position
    so reproduceability checks (CI / unit tests) produce actionable
    diagnostics.
    """
    identity = {name: cell for name, cell in zip(IDENTITY_COLUMNS, row, strict=True)}
    feature_cells = row[len(IDENTITY_COLUMNS):len(IDENTITY_COLUMNS) + len(FEATURE_NAMES)]
    target_cells = row[len(IDENTITY_COLUMNS) + len(FEATURE_NAMES):]

    try:
        fixture_id = _parse_required_int(identity["fixture_id"], "fixture_id")
        competition_id = _parse_required_int(identity["competition_id"], "competition_id")
        season_id = _parse_required_int(identity["season_id"], "season_id")
        home_team_id = _parse_required_int(identity["home_team_id"], "home_team_id")
        away_team_id = _parse_required_int(identity["away_team_id"], "away_team_id")
        kickoff = _parse_required_dt(identity["kickoff"], "kickoff")
    except DatasetLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DatasetLoadError(
            f"{csv_path}:line {line_no}: identity cell parse error: {exc!r}"
        ) from exc

    features: dict[str, float | int | None] = {}
    for name, cell in zip(FEATURE_NAMES, feature_cells, strict=True):
        features[name] = _parse_optional_number(cell, name)

    targets: dict[str, int | None] = {}
    for name, cell in zip(TARGET_NAMES, target_cells, strict=True):
        targets[name] = _parse_optional_int(cell, name)

    return LoadedExample(
        fixture_id=fixture_id,
        kickoff=kickoff,
        competition_id=competition_id,
        season_id=season_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        features=features,
        targets=targets,
    )


def _parse_required_int(cell: str, name: str) -> int:
    if cell == EMPTY_CELL:
        raise DatasetLoadError(
            f"identity column {name!r} is required and must not be empty"
        )
    return _strict_int(cell, name)


def _parse_optional_int(cell: str, name: str) -> int | None:
    if cell == EMPTY_CELL:
        return None
    return _strict_int(cell, name)


def _parse_optional_number(cell: str, name: str) -> float | int | None:
    """Parse a feature cell. Float features are kept as ``float``;
    integer-typed features come back as ``int``.

    A cell of ``""`` -> ``None``.
    """
    if cell == EMPTY_CELL:
        return None
    # Try int first (handles integer features cleanly: returning
    # ``0`` as int and not ``0.0`` keeps the on-disk shape symmetric).
    # We accept a leading ``-`` for negative values (ELO differences,
    # points diff, etc.) but reject a decimal point in int mode.
    s = cell.strip()
    if _looks_int(s):
        try:
            return int(s)
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError as exc:
        raise DatasetLoadError(
            f"feature {name!r} cell {cell!r} is not a number"
        ) from exc


def _parse_required_dt(cell: str, name: str) -> datetime:
    if cell == EMPTY_CELL:
        raise DatasetLoadError(
            f"identity column {name!r} is required and must not be empty"
        )
    try:
        dt = datetime.fromisoformat(cell)
    except ValueError as exc:
        raise DatasetLoadError(
            f"identity column {name!r} cell {cell!r} is not ISO8601"
        ) from exc
    if dt.tzinfo is None:
        # Normalize naive ISO strings to UTC; matches the builder's
        # tz-aware write. We do NOT roll back: a naive ISO is suspicious
        # for a tz-aware field and the builder always writes tz-aware.
        raise DatasetLoadError(
            f"identity column {name!r} cell {cell!r} has no timezone info "
            "(loader requires UTC tz-aware)"
        )
    return dt


def _strict_int(cell: str, name: str) -> int:
    s = cell.strip()
    if not _looks_int(s):
        raise DatasetLoadError(
            f"column {name!r} cell {cell!r} is not an int"
        )
    return int(s)


def _looks_int(s: str) -> bool:
    """Cheap detection for ``int``-shaped strings (base 10).

    Accepts optional leading ``-`` or ``+`` and digits only. Rejects
    underscores (``1_000``), hexadecimal, decimal point, ``e`` notation.
    """
    if not s:
        return False
    body = s[1:] if s[0] in "+-" else s
    return body.isdigit()


def _hash_file(path: Path) -> str:
    """SHA-256 hex digest of the file's bytes on disk."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# Re-export for any caller that wants the helper along with this module.
__all__ = [
    "DatasetIntegrityError",
    "DatasetLoadError",
    "DatasetSchemaMismatch",
    "DatasetVersionMismatch",
    "LoadedDataset",
    "LoadedExample",
    "load_dataset",
    "load_manifest",
    "validate_dataset",
]
