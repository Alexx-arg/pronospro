"""Standalone sync jobs.

Each job is a thin orchestration wrapper around the corresponding sync
service in :mod:`app.services.sync`. The wrapper is responsible for:

* Resolving an async :class:`Settings` instance (env-driven).
* Obtaining a :class:`DataProvider` instance from the registry.
* Owning a SQLAlchemy transaction via :func:`app.db.session.session_scope`.
* Honouring the per-job ``SYNC_<NAME>_ENABLED`` flag — disabled jobs are
  no-ops when invoked by the scheduler but stay invocable on demand (the
  admin endpoint / CLI passes ``force=True`` to bypass the flag).
* Logging the resulting :class:`~app.services.sync.metrics.SyncMetrics`.

The jobs are intentionally side-effect-pure regarding provider/session
lifecycle: each call resolves its own provider + session, runs the sync,
commits (or rolls back inside ``session_scope``) and returns the metrics.
This keeps them trivially schedulable by APScheduler (one call per tick)
and trivially testable (dependencies injected via keyword arguments).

None of these jobs touch the prediction engine, GLM or feature store —
those belong to later phases.
"""

from __future__ import annotations

from app.tasks.sync_finished import run_sync_finished
from app.tasks.sync_injuries import run_sync_injuries
from app.tasks.sync_lineups import run_sync_lineups
from app.tasks.sync_player_statistics import run_sync_player_statistics
from app.tasks.sync_team_statistics import run_sync_team_statistics
from app.tasks.sync_teams import run_sync_teams
from app.tasks.sync_upcoming import run_sync_upcoming

__all__ = [
    "run_sync_finished",
    "run_sync_injuries",
    "run_sync_lineups",
    "run_sync_player_statistics",
    "run_sync_team_statistics",
    "run_sync_teams",
    "run_sync_upcoming",
]
