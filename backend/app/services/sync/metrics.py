"""Sync metrics container.

Every sync service returns a :class:`SyncMetrics` describing the outcome
of one integration pass. The scheduler logs it as structured data and the
Phase 4 admin endpoints forward it to the client (without secrets).

The counters are:

* ``requested`` — number of items the service intended to fetch
  (e.g. leagues scheduled, fixtures in the window).
* ``received``  — number of DTOs the provider actually returned.
* ``inserted``  — rows newly inserted in PostgreSQL.
* ``updated``   — rows updated in PostgreSQL (idempotent upserts).
* ``skipped``   — items intentionally skipped (e.g. finished fixture that
  had no score yet, or partial player data we couldn't validate).
* ``failed``    — items that raised an exception (logged individually).

Invariant: ``inserted + updated + skipped + failed`` approximates
``received`` (it may differ when a single DTO spans multiple rows, like a
lineup that creates both a Lineup row and N LineupPlayer rows).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SyncMetrics:
    """Counters describing one sync pass."""

    job: str
    requested: int = 0
    received: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Append an error fragment (truncated to keep memory bounded)."""
        self.errors.append(msg[:256])

    def as_dict(self) -> dict[str, object]:
        """Plain dict representation for logging / JSON serialisation."""
        return {
            "job": self.job,
            "requested": self.requested,
            "received": self.received,
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
            "failed": self.failed,
            "errors": list(self.errors),
        }
