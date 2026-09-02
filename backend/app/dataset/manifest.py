"""Manifest dataclass + the canonical metadata.json shape.

Per the Phase 4 contract, ``metadata.json`` alongside every
``dataset.csv`` is the immutable certificate of how the dataset was
produced. Schema:

```
{
  "dataset_version": "v001",
  "generated_at":       <ISO8601 UTC>,
  "feature_definition_version": "fd_v1",
  "data_cutoff":        <ISO8601 UTC>   # upper bound on kickoff_time
  "source_schema_version":          "schema_v1",
  "row_count":          <int>,
  "feature_names":      [<str>, ...],
  "target_names":       [<str>, ...],
  "start_date":         <ISO8601 UTC>   # earliest kickoff in the dataset
  "end_date":           <ISO8601 UTC>   # latest kickoff in the dataset
  "competitions":       [<int>, ...],
  "seasons":            [<int>, ...],
  "csv_sha256":         "<64 hex>"
}
```

The class is a frozen dataclass so it cannot be mutated after the
builder has emitted it; downstream code reads it via attribute access
or via :meth:`to_dict` for JSON serialisation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

#: The current feature-definition version. Bump this (and ``dataset_version``)
#: every time a Feature math function or constant changes shape. See
#: ANTI_LEAKAGE.md for the rationale.
FEATURE_DEFINITION_VERSION: str = "fd_v1"

#: Database schema version tracked by the migration under
#: ``backend/alembic/versions/0001_initial.py``. Update only when a
#: migration modifies tables the feature math reads.
SOURCE_SCHEMA_VERSION: str = "schema_v1"


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable metadata describing one produced dataset."""

    dataset_version: str
    generated_at: datetime
    feature_definition_version: str
    data_cutoff: datetime
    source_schema_version: str
    row_count: int
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    start_date: datetime
    end_date: datetime
    competitions: tuple[int, ...]
    seasons: tuple[int, ...]
    csv_sha256: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict; datetimes -> ISO8601 (UTC)."""
        d = asdict(self)
        # asdict deep-copies datetimes as themselves; isoformat them.
        for key in ("generated_at", "data_cutoff", "start_date", "end_date"):
            v = d.get(key)
            if isinstance(v, datetime):
                d[key] = v.astimezone(timezone.utc).isoformat()
        # extras may contain datetimes too (defence).
        if isinstance(d.get("extras"), dict):
            for k, v in list(d["extras"].items()):
                if isinstance(v, datetime):
                    d["extras"][k] = v.astimezone(timezone.utc).isoformat()
        # tuple -> list for JSON (asdict already converted tuples to lists)
        return d

    def to_json(self) -> str:
        """Serialise to a pretty, deterministic JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=False,
                          ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        """Inverse of :meth:`to_dict` — parses back the manifest."""
        return cls(
            dataset_version=str(data["dataset_version"]),
            generated_at=_parse_dt(data["generated_at"]),
            feature_definition_version=str(data["feature_definition_version"]),
            data_cutoff=_parse_dt(data["data_cutoff"]),
            source_schema_version=str(data["source_schema_version"]),
            row_count=int(data["row_count"]),
            feature_names=tuple(str(x) for x in data["feature_names"]),
            target_names=tuple(str(x) for x in data["target_names"]),
            start_date=_parse_dt(data["start_date"]),
            end_date=_parse_dt(data["end_date"]),
            competitions=tuple(int(x) for x in data["competitions"]),
            seasons=tuple(int(x) for x in data["seasons"]),
            csv_sha256=str(data["csv_sha256"]),
            extras=dict(data.get("extras") or {}),
        )


def _parse_dt(value: Any) -> datetime:
    """Parse an ISO8601 string back to a tz-aware datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


__all__ = [
    "DatasetManifest",
    "FEATURE_DEFINITION_VERSION",
    "SOURCE_SCHEMA_VERSION",
]
