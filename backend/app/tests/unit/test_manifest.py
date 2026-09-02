"""Tests for :mod:`app.dataset.manifest` (``metadata.json`` shape).

Verifies:

* Round-trip ``DatasetManifest.to_dict`` -> ``from_dict`` is lossless
  for every contractual field.
* Datetime fields serialise to ISO8601 strings (UTC).
* Datetime fields parse back to tz-aware ``datetime`` objects.
* All 12 contractual fields are present in the serialised dict.
* Tuples (``feature_names``, ``competitions``, ``seasons``) survive the
  JSON round-trip as lists on the wire and tuples in Python.
* Missing ``extras`` defaults to an empty dict (not ``None``) so the
  type signature stays uniform.
* Naive ISO datetime inputs are normalised to UTC by ``_parse_dt``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.dataset.manifest import (
    DatasetManifest,
    FEATURE_DEFINITION_VERSION,
    SOURCE_SCHEMA_VERSION,
)


def _sample_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_version="v001",
        generated_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        feature_definition_version=FEATURE_DEFINITION_VERSION,
        data_cutoff=datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
        source_schema_version=SOURCE_SCHEMA_VERSION,
        row_count=42,
        feature_names=("a", "b", "c"),
        target_names=("t1", "t2"),
        start_date=datetime(2025, 8, 1, 0, 0, 0, tzinfo=timezone.utc),
        end_date=datetime(2026, 1, 31, 23, 0, 0, tzinfo=timezone.utc),
        competitions=(140, 39),
        seasons=(1, 2, 3),
        csv_sha256="0" * 64,
    )


def test_to_dict_then_from_dict_roundtrip_is_lossless() -> None:
    original = _sample_manifest()
    restored = DatasetManifest.from_dict(original.to_dict())
    assert restored == original


def test_to_json_then_from_dict_roundtrip_is_lossless() -> None:
    original = _sample_manifest()
    restored = DatasetManifest.from_dict(json.loads(original.to_json()))
    assert restored == original


def test_to_dict_datetimes_serialize_to_iso8601_strings() -> None:
    d = _sample_manifest().to_dict()
    for key in ("generated_at", "data_cutoff", "start_date", "end_date"):
        v = d[key]
        assert isinstance(v, str), f"{key} should be an ISO8601 str"
        # must round-trip via fromisoformat
        datetime.fromisoformat(v)


def test_from_dict_datetimes_are_tz_aware() -> None:
    m = DatasetManifest.from_dict(_sample_manifest().to_dict())
    for key in (m.generated_at, m.data_cutoff, m.start_date, m.end_date):
        assert key.tzinfo is not None


def test_to_dict_contains_all_contractual_fields() -> None:
    keys = set(_sample_manifest().to_dict().keys())
    expected = {
        "dataset_version",
        "generated_at",
        "feature_definition_version",
        "data_cutoff",
        "source_schema_version",
        "row_count",
        "feature_names",
        "target_names",
        "start_date",
        "end_date",
        "competitions",
        "seasons",
        "csv_sha256",
        "extras",
    }
    assert keys == expected


def test_tuples_round_trip_as_lists_on_wire_as_tuples_in_python() -> None:
    d = _sample_manifest().to_dict()
    assert d["feature_names"] == ["a", "b", "c"]
    assert d["competitions"] == [140, 39]
    assert d["seasons"] == [1, 2, 3]
    m = DatasetManifest.from_dict(d)
    assert m.feature_names == ("a", "b", "c")
    assert m.competitions == (140, 39)
    assert m.seasons == (1, 2, 3)


def test_extras_defaults_to_empty_dict_when_absent() -> None:
    d = _sample_manifest().to_dict()
    d.pop("extras")
    m = DatasetManifest.from_dict(d)
    assert m.extras == {}


def test_naive_iso_datetime_normalised_to_utc() -> None:
    d = _sample_manifest().to_dict()
    # strip the tz suffix of generated_at to simulate a naive ISO string
    naive = "2026-01-02T03:04:05"
    d["generated_at"] = naive
    m = DatasetManifest.from_dict(d)
    assert m.generated_at.tzinfo is timezone.utc


def test_manifest_is_frozen_dataclass() -> None:
    # Mutation must be forbidden — once produced the manifest is the
    # immutable certificate of how the dataset was built.
    m = _sample_manifest()
    try:
        m.row_count = 99  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("DatasetManifest is not frozen")


def test_feature_definition_version_constant() -> None:
    # The loader/builder import this constant as the canonical version
    # tag; bumping it MUST be a deliberate act. Test pins the value so
    # an accidental edit is caught.
    assert isinstance(FEATURE_DEFINITION_VERSION, str)
    assert FEATURE_DEFINITION_VERSION == "fd_v1"


def test_source_schema_version_constant() -> None:
    assert isinstance(SOURCE_SCHEMA_VERSION, str)
    assert SOURCE_SCHEMA_VERSION == "schema_v1"
