"""Shared structural constants for the dataset CSV.

Kept separate from :mod:`app.dataset.builder` to avoid pulling the
builder (which imports heavy DB/SQLAlchemy machinery) into the loader.
The loader is read-only and should not require the async session /
SQLAlchemy / feature math at import time — only the canonical column
order and the feature/target name lists already exposed by
:mod:`app.features.example`.

Public constants
----------------
* :data:`IDENTITY_COLUMNS` — non-feature, non-target columns prefixing
  every CSV row.
* :data:`FEATURE_NAMES` — re-exported from ``app.features.example`` for
  single-source-of-truth.
* :data:`TARGET_NAMES` — same.
* :data:`CSV_COLUMNS` — full ordered column list (identity + features +
  targets). Header order of every dataset.csv.
* :data:`EMPTY_CELL` — the canonical on-disk representation of ``None``
  (empty string, ""). The loader interprets it as Python ``None`` and
  never as ``0`` or ``""``.
"""

from __future__ import annotations

from app.features.example import FEATURE_NAMES, TARGET_NAMES

#: Identity columns prepended to every CSV row (in this exact order).
IDENTITY_COLUMNS: tuple[str, ...] = (
    "fixture_id",
    "kickoff",
    "competition_id",
    "season_id",
    "home_team_id",
    "away_team_id",
)

#: All column names in canonical CSV order (matches the builder's header).
CSV_COLUMNS: tuple[str, ...] = IDENTITY_COLUMNS + FEATURE_NAMES + TARGET_NAMES

#: Canonical on-disk representation of a missing value (``None`` in
#: Python). Kept here so the loader and tests share one truth.
EMPTY_CELL: str = ""


__all__ = [
    "CSV_COLUMNS",
    "EMPTY_CELL",
    "FEATURE_NAMES",
    "IDENTITY_COLUMNS",
    "TARGET_NAMES",
]
