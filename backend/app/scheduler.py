"""APScheduler-based orchestrator for the sync jobs.

Phase 3 final piece: a lightweight scheduler that runs each standalone
sync job (:mod:`app.tasks`) on its own interval. The intervals and ENABLED
flags come from :class:`app.config.Settings`, so operators can tune the
cadence without redeploying code.

Design constraints:

* **Single worker is enough for Phase 3.** All jobs are I/O bound against
  the same provider (one HTTP rate limiter shared across all jobs); there
  is no benefit in parallel schedulers. APScheduler runs the jobs on the
  same asyncio event loop as the FastAPI app.
* **ENABLED flags are honoured at two layers.** The scheduler only
  registers a job whose flag is on at registration time (avoids waking
  the loop just to no-op). The job itself re-checks the flag on every
  tick so toggling it via env at runtime (relaunch) takes effect without
  code changes. Manual invocations (admin endpoint / CLI) bypass the flag
  via ``force=True``.
* **``SCHEDULER_ENABLED`` is the master switch.** When ``False`` the
  scheduler never starts: jobs stay invocable manually but no tick is
  ever scheduled.
* **Idempotent start/stop.** The instance tracks its own ``running``
  state to survive FastAPI startup retries and test re-entrancy.

APScheduler version: 3.10.x (APScheduler 4.x has a different API; pinning
in ``requirements.txt``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.tasks import (
    run_sync_finished,
    run_sync_injuries,
    run_sync_lineups,
    run_sync_player_statistics,
    run_sync_team_statistics,
    run_sync_teams,
    run_sync_upcoming,
)
from app.tasks._runner import JobResult

_LOG = get_logger()

# Type alias for the coroutine-returning job entry points.
JobCoroutine = Callable[..., Coroutine[Any, Any, JobResult]]


@dataclass(slots=True)
class JobSpec:
    """One job registration entry."""

    name: str
    func: JobCoroutine
    enabled_attr: str
    interval_minutes_attr: str


# Order matters only for logging readability: faster / more sensitive
# jobs are listed first. The metadata/competitions sync is intentionally
# NOT here: it's a one-shot bootstrap, run by the init command (Phase 4)
# rather than on a timer. See docs/ARCHITECTURE.md.
JOB_SPECS: tuple[JobSpec, ...] = (
    JobSpec("sync_upcoming", run_sync_upcoming,
            "sync_upcoming_enabled", "sync_upcoming_interval_minutes"),
    JobSpec("sync_finished", run_sync_finished,
            "sync_finished_enabled", "sync_finished_interval_minutes"),
    JobSpec("sync_lineups", run_sync_lineups,
            "sync_lineups_enabled", "sync_lineups_interval_minutes"),
    JobSpec("sync_injuries", run_sync_injuries,
            "sync_injuries_enabled", "sync_injuries_interval_minutes"),
    JobSpec("sync_teams", run_sync_teams,
            "sync_teams_enabled", "sync_teams_interval_minutes"),
    JobSpec("sync_team_statistics", run_sync_team_statistics,
            "sync_team_statistics_enabled",
            "sync_team_statistics_interval_minutes"),
    JobSpec("sync_player_statistics", run_sync_player_statistics,
            "sync_player_statistics_enabled",
            "sync_player_statistics_interval_minutes"),
)


class SyncScheduler:
    """Owns the lifecycle of the APScheduler asyncio scheduler.

    A single instance is created at app startup (or by tests). It is safe
    to call :meth:`start` / :meth:`shutdown` multiple times; only the
    first ``start`` actually boots the loop.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        scheduler: AsyncIOScheduler | None = None,
    ) -> None:
        self._settings: Settings = settings if settings is not None else get_settings()
        self._scheduler: AsyncIOScheduler = (
            scheduler if scheduler is not None else AsyncIOScheduler()
        )
        self._running: bool = False
        self._registered: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        """Return ``True`` when the scheduler is currently active."""
        return self._running

    def register_jobs(self) -> int:
        """Register every ENABLED job on the scheduler.

        Returns the number of jobs actually registered (disabled jobs are
        skipped at registration time so they never fire). Safe to call
        once per instance: a second call is a no-op.
        """
        if self._registered:
            return len(self._scheduler.get_jobs())
        self._registered = True

        n_registered = 0
        for spec in JOB_SPECS:
            enabled = bool(getattr(self._settings, spec.enabled_attr, False))
            interval_minutes = int(
                getattr(self._settings, spec.interval_minutes_attr, 0)
            )
            if not enabled:
                _LOG.info(
                    "scheduler: job {} not registered (ENABLED flag is False)",
                    spec.name,
                )
                continue
            if interval_minutes <= 0:
                _LOG.warning(
                    "scheduler: job {} has non-positive interval "
                    "({}={}); skipping registration",
                    spec.name, spec.interval_minutes_attr, interval_minutes,
                )
                continue

            self._scheduler.add_job(
                # AsyncIOScheduler can call sync or async callables; we
                # wrap so the JobResult is logged and exceptions don't
                # surf the scheduler (APScheduler logs them, but a wrapper
                # is clearer).
                _make_tick(spec.name, spec.func, self._settings),
                trigger=IntervalTrigger(minutes=interval_minutes),
                id=spec.name,
                name=spec.name,
                # Avoid pile-ups if a tick overlaps the next: APScheduler
                # refuses to start a new instance while the previous one
                # is still running when ``max_instances=1``.
                max_instances=1,
                # No coalescing: we want each scheduled tick to fire.
                coalesce=False,
                replace_existing=True,
            )
            n_registered += 1
            _LOG.info(
                "scheduler: registered {} every {} minutes",
                spec.name, interval_minutes,
            )
        return n_registered

    async def start(self) -> bool:
        """Start the scheduler if it isn't running and the master switch
        is on.

        Returns ``True`` when the scheduler actually started, ``False``
        when it was skipped (already running, or ``SCHEDULER_ENABLED``
        was ``False``).
        """
        if self._running:
            return False
        if not self._settings.scheduler_enabled:
            _LOG.info(
                "scheduler: SCHEDULER_ENABLED is False — start() is a no-op. "
                "Jobs remain invocable manually via the admin endpoint / CLI."
            )
            return False

        if not self._registered:
            self.register_jobs()

        if not self._scheduler.get_jobs():
            _LOG.warning(
                "scheduler: SCHEDULER_ENABLED is True but no job is "
                "registered (all ENABLED flags are False). start() will "
                "still run the loop, but nothing will fire."
            )

        self._scheduler.start()
        self._running = True
        _LOG.info("scheduler: started ({} jobs registered)",
                  len(self._scheduler.get_jobs()))
        return True

    async def shutdown(self, wait: bool = True) -> None:
        """Stop the scheduler if it's running. Idempotent."""
        if not self._running:
            return
        # APScheduler's shutdown is sync for AsyncIOScheduler when called
        # from inside the loop thread; the ``wait`` flag asks it to give
        # in-flight jobs a chance to finish.
        self._scheduler.shutdown(wait=wait)
        self._running = False
        self._registered = False
        _LOG.info("scheduler: shutdown complete")

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------
    def list_registered_jobs(self) -> list[str]:
        """Return the ids of the jobs currently registered.

        Used by tests to assert the registration filtered out disabled
        jobs and applied the configured intervals.
        """
        return [job.id for job in self._scheduler.get_jobs()]


def _make_tick(
    name: str,
    func: JobCoroutine,
    settings: Settings,
) -> Callable[[], Coroutine[Any, Any, None]]:
    """Wrap a job entry point so its ``JobResult`` is logged and any
    exception bubbles no further than loguru.
    """

    async def _tick() -> None:
        try:
            result: JobResult = await func(settings=settings)
            if result.skipped:
                _LOG.info("tick {}: skipped (disabled)", name)
            elif result.metrics is not None and result.metrics.failed:
                _LOG.warning(
                    "tick {}: ran with {} failures",
                    name, result.metrics.failed,
                )
            else:
                _LOG.info("tick {}: ok", name)
        except Exception as exc:  # noqa: BLE001  (never crash the loop)
            _LOG.error("tick {}: crashed {}", name, str(exc)[:512])

    return _tick


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# A single scheduler instance per worker process. The FastAPI lifespan
# handler (Phase 4) calls :func:`get_scheduler` at startup and
# :func:`shutdown_scheduler` at teardown.
_scheduler_instance: SyncScheduler | None = None


def get_scheduler(settings: Settings | None = None) -> SyncScheduler:
    """Return the lazily-initialised scheduler singleton."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = SyncScheduler(settings=settings)
    return _scheduler_instance


async def shutdown_scheduler(wait: bool = True) -> None:
    """Tear down the scheduler singleton (used by app lifespan teardown)."""
    global _scheduler_instance
    if _scheduler_instance is not None:
        await _scheduler_instance.shutdown(wait=wait)
        _scheduler_instance = None


__all__ = [
    "JOB_SPECS",
    "JobSpec",
    "SyncScheduler",
    "get_scheduler",
    "shutdown_scheduler",
]
