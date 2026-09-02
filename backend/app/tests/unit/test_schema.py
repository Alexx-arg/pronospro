"""Tests for :mod:`app.dataset._schema` (canonical CSV structural shape).

The schema module is the shared structural contract between builder and
loader. Tests here verify that:

* :data:`IDENTITY_COLUMNS` carries the 6 fixed identity columns in the
  exact order the loader parses them in.
* :data:`FEATURE_NAMES` / :data:`TARGET_NAMES` are re-exported from
  :mod:`app.features.example` (single source of truth).
* :data:`CSV_COLUMNS` is ``IDENTITY_COLUMNS + FEATURE_NAMES +
  TARGET_NAMES`` and contains no duplicates.
* :data:`EMPTY_CELL` is the canonical on-disk NaN representation.
* An unexpected column name is rejected — the policy here is strict
  ordering; the loader raises :class:`DatasetSchemaMismatch` on any
  header that is not exactly :data:`CSV_COLUMNS` (see ``test_loader``).
"""

from __future__ import annotations

from app.dataset._schema import (
    CSV_COLUMNS,
    EMPTY_CELL,
    FEATURE_NAMES,
    IDENTITY_COLUMNS,
    TARGET_NAMES,
)
from app.features.example import (
    FEATURE_NAMES as EXAMPLE_FEATURE_NAMES,
    TARGET_NAMES as EXAMPLE_TARGET_NAMES,
)


def test_identity_columns_order_and_count() -> None:
    assert IDENTITY_COLUMNS == (
        "fixture_id",
        "kickoff",
        "competition_id",
        "season_id",
        "home_team_id",
        "away_team_id",
    )
    assert len(IDENTITY_COLUMNS) == 6


def test_feature_names_are_re_exported_from_example() -> None:
    # Single source of truth — the schema module MUST NOT redefine the
    # feature list. ``is`` comparison because tuples are interned by the
    # import system when aliased.
    assert FEATURE_NAMES is EXAMPLE_FEATURE_NAMES


def test_target_names_are_re_exported_from_example() -> None:
    assert TARGET_NAMES is EXAMPLE_TARGET_NAMES


def test_csv_columns_are_identity_plus_features_plus_targets() -> None:
    assert CSV_COLUMNS == IDENTITY_COLUMNS + FEATURE_NAMES + TARGET_NAMES


def test_csv_columns_have_no_duplicates() -> None:
    # A duplicate column name would silently break CSV round-tripping
    # (the loader would parse it twice / misalign the rest of the row).
    assert len(CSV_COLUMNS) == len(set(CSV_COLUMNS))


def test_csv_columns_identity_first() -> None:
    # The loader relies on identity columns being the first len(IDENTITY)
    # cells of every row so it can parse them by position before the
    # feature/target split.
    assert CSV_COLUMNS[: len(IDENTITY_COLUMNS)] == IDENTITY_COLUMNS


def test_empty_cell_is_canonical_empty_string() -> None:
    # The loader treats ``""`` as Python ``None`` for every numeric
    # column; this single constant keeps the builder and loader in
    # sync. If this changes the loader will silently misinterpret 0
    # features as missing data.
    assert EMPTY_CELL == ""


def test_feature_and_target_names_are_disjoint() -> None:
    # A name appearing in both lists would let a feature math path read
    # the target column via the shared registry; physically impossible
    # today because they live in separate tuple constants, but the test
    # guards against refactor accidents.
    assert set(FEATURE_NAMES).isdisjoint(set(TARGET_NAMES))
