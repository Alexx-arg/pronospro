"""Dataset builder: PostgreSQL → CSV + metadata.json (immutable, SHA-256).

Builds a versioned historical dataset by:

1. Loading the (finished) fixture universe for each
   ``(competition, season)`` once.
2. For each fixture whose ``kickoff_time`` lies inside ``[start_date,
   end_date]`` (and ``< data_cutoff`` strict), running the
   :mod:`app.features.assembler` against that fixture's team statistics
   and the season's fixture universe.
3. Materialising every example to a deterministic-ordered CSV file, then
   computing its SHA-256 and emitting ``metadata.json``.

Determinism
-----------
The CSV row order is strictly chronological by ``(kickoff_time, fixture_id)``;
the manifest's ``csv_sha256`` is therefore a pure function of the input
subset AND the feature-definition version. Two independent runs over
the same DB state MUST produce byte-identical CSVs + matching
manifests. (Caveat: ``generated_at`` legitimately differs between runs
but lives in the manifest only, not in the CSV.) See ANTI_LEAKAGE.md
§reproducibility.

Anti-leakage guarantees pulled out from the assembler (see that module
for the per-feature story) and guarded additionally here:

* The season fixtures universe is sliced to ``kickoff_time < data_cutoff``
  — the dataset cannot see matches that haven't kicked off yet at
  ``data_cutoff`` time. This is paramount when a fixture's ``status``
  may have been updated to ``finished`` *after* ``data_cutoff``: the
  builder must NOT depend on race timing.
* The target fixture itself is omitted from every feature math via the
  per-feature ``exclude_fixture_id`` argument.
* ``end_date <= data_cutoff`` is enforced (otherwise ``data_cutoff``
  would be vacuous).
* H2H features use cross-season history: the builder pre-loads every
  finished fixture between the home/away team pair once per pair and
  forwards it to :mod:`app.features.assembler` via the
  ``h2h_fixtures`` argument. The per-fixture ``kickoff < T`` boundary
  is enforced inside :mod:`app.features.h2h` via
  :func:`app.features.asof.fixtures_before_unordered`.

Performance profile
--------------------
Per (competition, season):

* O(1) fixture SELECT.
* O(1) team_statistics SELECT.
* O(N) Python iteration where N = fixtures in window; the assembler
  slices in-memory.

H2H adds O(P) SELECTs where P = unique team pairs across the build
(loaded lazily the first time a pair appears; reused on subsequent
fixtures of the same pair). Total SQL queries ≈
``2 * len(competitions * seasons) + P``.

Schema: ``metadata.json`` shape is defined in :mod:`app.dataset.manifest`.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.dataset._schema import CSV_COLUMNS, IDENTITY_COLUMNS
from app.dataset.manifest import (
    DatasetManifest,
    FEATURE_DEFINITION_VERSION,
    SOURCE_SCHEMA_VERSION,
)
from app.features.asof import (
    load_finished_fixtures_as_of,
    load_h2h_fixtures_as_of,
    load_team_stats_as_of,
)
from app.features.assembler import build_example
from app.features.example import FEATURE_NAMES, HistoricalMatchExample, TARGET_NAMES
from app.models import Competition, Season

_LOG = get_logger()

# ``CSV_COLUMNS`` / ``IDENTITY_COLUMNS`` are re-exported for callers who
# import them from the builder (legacy path) — they live in
# :mod:`app.dataset._schema` now.


class InvalidBuildConfigError(ValueError):
    """Raised when the builder receives contradictory parameters."""


async def build_dataset(
    session: AsyncSession,
    *,
    competition_ids: list[int] | None = None,
    season_ids: list[int] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    data_cutoff: datetime | None = None,
    dataset_version: str,
    output_path: str | Path,
) -> tuple[Path, DatasetManifest]:
    """Build a dataset and materialise it on disk.

    Args:
        session: SQLAlchemy async session.
        competition_ids: competitions to include (empty dict means "all"
            but here we require explicit).
        season_ids: seasons to include.
        start_date: lower bound on kickoff_time (inclusive) — ``None``
            means "no lower bound".
        end_date: upper bound on kickoff_time (inclusive) — ``None``
            means "no upper bound".
        data_cutoff: hard temporal watermark. Must be >= ``end_date``.
            When ``None`` defaults to ``end_date`` (or ``now`` if
            ``end_date`` is also None). The builder refuses to emit any
            fixture with ``kickoff_time >= data_cutoff``.
        dataset_version: human label (e.g. ``v001``); used as the row
            signature in the manifest.
        output_path: dataset directory; the builder writes
            ``dataset.csv`` + ``metadata.json`` underneath. The
            directory is created if missing; existing files are
            overwritten (callers are responsible for not reusing a
            ``dataset_version`` once published — see DATASET.md).

    Returns:
        ``(csv_path, manifest)``. The manifest has ``csv_sha256`` filled
        in and matches the bytes on disk exactly.
    """
    cutoff = _resolve_cutoff(data_cutoff, end_date)
    if end_date is not None and cutoff < _ensure_aware(end_date):
        raise InvalidBuildConfigError(
            f"data_cutoff ({cutoff.isoformat()}) must be >= end_date "
            f"({end_date.isoformat()})"
        )

    comp_ids = list(competition_ids or [])
    sz_ids = list(season_ids or [])
    if not comp_ids:
        raise InvalidBuildConfigError("competition_ids must not be empty")
    if not sz_ids:
        raise InvalidBuildConfigError("season_ids must not be empty")

    target_comps, target_seasons = await _resolve_targets(
        session=session, competition_ids=comp_ids, season_ids=sz_ids,
    )

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dataset.csv"
    meta_path = out_dir / "metadata.json"

    examples: list[HistoricalMatchExample] = []
    all_comps: set[int] = set()
    all_seasons: set[int] = set()
    # Cross-season H2H universe cache, keyed by (low_team_id, high_team_id)
    # so pairs are interchange addressable (home/away assignment in one
    # direction or the other collapses to the same key). Loaded lazily
    # the first time a pair appears, reused by every fixture of that
    # pair across the whole build so we don't pay P SELECTs per fixture.
    h2h_cache: dict[tuple[int, int], list] = {}

    def _h2h_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    for comp_id in target_comps:
        for sz_id in target_seasons:
            season_fixture_rows = await load_finished_fixtures_as_of(
                session,
                competition_id=comp_id,
                season_id=sz_id,
            )
            # Apply data_cutoff: future-by-cutoff rows are dropped so
            # the assembler sees nothing that was still pending at
            # the watermark.
            season_fixture_rows = [
                r for r in season_fixture_rows if r.kickoff_time < cutoff
            ]
            if not season_fixture_rows:
                continue
            stats_rows = await load_team_stats_as_of(
                session, competition_id=comp_id, season_id=sz_id,
            )
            for fx in season_fixture_rows:
                if start_date is not None and fx.kickoff_time < _ensure_aware(start_date):
                    continue
                if end_date is not None and fx.kickoff_time > _ensure_aware(end_date):
                    continue
                h2h_key = _h2h_key(fx.home_team_id, fx.away_team_id)
                if h2h_key not in h2h_cache:
                    h2h_cache[h2h_key] = await load_h2h_fixtures_as_of(
                        session,
                        home_team_id=fx.home_team_id,
                        away_team_id=fx.away_team_id,
                    )
                example = build_example(
                    fixture=fx,
                    season_fixtures=season_fixture_rows,
                    stats_rows=stats_rows,
                    h2h_fixtures=h2h_cache[h2h_key],
                )
                examples.append(example)
                all_comps.add(comp_id)
                all_seasons.add(sz_id)

    # Sort by kickoff then by fixture_id for deterministic CSV ordering.
    examples.sort(key=lambda e: (e.kickoff, e.fixture_id))

    # Write CSV with deterministic column order + quote style.
    # The SHA-256 is computed over the file bytes afterward to avoid a
    # double-source-of-truth between the in-memory buffer and the on-disk
    # bytes. Empty (CSV "") means ``None``; the loader restores it.
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = _make_csv_writer(f)
        writer.writerow(CSV_COLUMNS)
        for example in examples:
            writer.writerow(_serialise_row(example))
    csv_sha = _hash_file(csv_path)

    # Build the manifest from the produced examples.
    manifest = _manifest_from_examples(
        examples=examples,
        dataset_version=dataset_version,
        data_cutoff=cutoff,
        competitions=tuple(sorted(all_comps)),
        seasons=tuple(sorted(all_seasons)),
        csv_sha256=csv_sha,
    )
    meta_path.write_text(manifest.to_json(), encoding="utf-8")

    _LOG.info(
        "build_dataset: {} examples -> {} (sha256={})",
        manifest.row_count, csv_path, csv_sha[:12],
    )
    return csv_path, manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_cutoff(
    data_cutoff: datetime | None,
    end_date: datetime | None,
) -> datetime:
    if data_cutoff is not None:
        return _ensure_aware(data_cutoff)
    if end_date is not None:
        return _ensure_aware(end_date)
    return datetime.now(timezone.utc)


def _ensure_aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _make_csv_writer(f: Any) -> Any:
    """Build a csv writer with deterministic quoting + CRLF newlines."""
    return csv.writer(
        f,
        dialect="excel",
        lineterminator="\r\n",
        quoting=csv.QUOTE_MINIMAL,
    )


def _hash_file(path: Path) -> str:
    """SHA-256 hex digest of the file's bytes on disk."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _serialise_row(example: HistoricalMatchExample) -> list[Any]:
    """Row in canonical CSV-column order (identity + features + targets)."""
    out: list[Any] = [
        example.fixture_id,
        example.kickoff.isoformat(),
        example.competition_id,
        example.season_id,
        example.home_team_id,
        example.away_team_id,
    ]
    feat = example.features
    for name in FEATURE_NAMES:
        out.append(feat.get(name))
    tgt = example.targets
    for name in TARGET_NAMES:
        out.append(tgt.get(name))
    return out


def _manifest_from_examples(
    examples: list[HistoricalMatchExample],
    *,
    dataset_version: str,
    data_cutoff: datetime,
    competitions: tuple[int, ...],
    seasons: tuple[int, ...],
    csv_sha256: str,
) -> DatasetManifest:
    if not examples:
        now = datetime.now(timezone.utc)
        return DatasetManifest(
            dataset_version=dataset_version,
            generated_at=now,
            feature_definition_version=FEATURE_DEFINITION_VERSION,
            data_cutoff=_ensure_aware(data_cutoff),
            source_schema_version=SOURCE_SCHEMA_VERSION,
            row_count=0,
            feature_names=FEATURE_NAMES,
            target_names=TARGET_NAMES,
            start_date=now,
            end_date=datetime.fromtimestamp(0, tz=timezone.utc),
            competitions=competitions,
            seasons=seasons,
            csv_sha256=csv_sha256,
        )
    start = examples[0].kickoff
    end = examples[-1].kickoff
    return DatasetManifest(
        dataset_version=dataset_version,
        generated_at=datetime.now(timezone.utc),
        feature_definition_version=FEATURE_DEFINITION_VERSION,
        data_cutoff=_ensure_aware(data_cutoff),
        source_schema_version=SOURCE_SCHEMA_VERSION,
        row_count=len(examples),
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
        start_date=_ensure_aware(start),
        end_date=_ensure_aware(end),
        competitions=competitions,
        seasons=seasons,
        csv_sha256=csv_sha256,
    )


async def _resolve_targets(
    *,
    session: AsyncSession,
    competition_ids: list[int],
    season_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Validate that the requested competitions / seasons exist in DB.

    Returns the deduplicated, sorted list of valid ids; raises if any
    is missing so the dataset build fails loudly rather than silently
    recording zero rows.
    """
    comps = (await session.execute(
        select(Competition.id).where(Competition.id.in_(competition_ids))
    )).scalars().all()
    if len(set(comps)) != len(set(competition_ids)):
        missing = set(competition_ids) - set(comps)
        raise InvalidBuildConfigError(
            f"competitions not in DB: {sorted(missing)}"
        )
    zens = (await session.execute(
        select(Season.id).where(Season.id.in_(season_ids))
    )).scalars().all()
    if len(set(zens)) != len(set(season_ids)):
        missing = set(season_ids) - set(zens)
        raise InvalidBuildConfigError(
            f"seasons not in DB: {sorted(missing)}"
        )
    return sorted(set(comps)), sorted(set(zens))


__all__ = [
    "CSV_COLUMNS",
    "IDENTITY_COLUMNS",
    "InvalidBuildConfigError",
    "build_dataset",
]
