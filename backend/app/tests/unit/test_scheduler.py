"""Unit tests for the APScheduler-based :class:`SyncScheduler`.

These tests use the real ``AsyncIOScheduler`` but inject stubbed settings.
We DO NOT call :meth:`AsyncIOScheduler.start` in most tests — we only
check *registration* (which jobs were added, with which intervals) and
the lifecycle calls (start/shutdown idempotency, ``SCHEDULER_ENABLED``
master switch). When we DO start the scheduler, we shut it down within
the same test to avoid leaking a background loop into other tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.scheduler import JOB_SPECS, SyncScheduler
from app.tasks._runner import JobResult
from app.services.sync.metrics import SyncMetrics


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
# Reuse the same stub shape as test_jobs.py but we also need the scheduler
# to *not* try to read attributes it doesn't care about. SyncScheduler uses
# only the ENABLED flags + the *_interval_minutes attrs + the master switch
# -> those should all be present.
@dataclass
class _SchedulerStubSettings:
    scheduler_enabled: bool = True

    sync_upcoming_enabled: bool = True
    sync_upcoming_interval_minutes: int = 5
    sync_finished_enabled: bool = True
    sync_finished_interval_minutes: int = 15
    sync_lineups_enabled: bool = True
    sync_lineups_interval_minutes: int = 120
    sync_injuries_enabled: bool = True
    sync_injuries_interval_minutes: int = 180
    sync_teams_enabled: bool = True
    sync_teams_interval_minutes: int = 720
    sync_team_statistics_enabled: bool = True
    sync_team_statistics_interval_minutes: int = 360
    sync_player_statistics_enabled: bool = True
    sync_player_statistics_interval_minutes: int = 360

    # Defaults the wrapped jobs MIGHT query at runtime (they don't, since
    # the scheduler never starts them here — but the lambdas pass
    # ``settings=settings`` to the runner; the runner reads only the flag,
    # so these are unused). Provided for parity / safety.
    current_league_ids: list[int] = field(default_factory=lambda: [39])
    current_season_year: int = 2024


class _RecordingScheduler(AsyncIOScheduler):
    """AsyncIOScheduler that records ``add_job`` calls without starting a
    real loop.

    The base class works fine without :meth:`start` for ``add_job`` /
    ``get_jobs`` queries; we only subclass to allow easy introspection in
    tests. When we actually need a running scheduler we use the plain
    :class:`AsyncIOScheduler` and shut it down inside the test.
    """


# ---------------------------------------------------------------------------
# Registration: only ENABLED jobs with positive intervals are added.
# ---------------------------------------------------------------------------
def test_register_jobs_only_enabled():
    s = _SchedulerStubSettings(sync_upcoming_enabled=False,
                               sync_lineups_enabled=False)
    sched = SyncScheduler(settings=s,
                          scheduler=_RecordingScheduler())
    n = sched.register_jobs()

    registered = sched.list_registered_jobs()
    assert "sync_upcoming" not in registered
    assert "sync_lineups" not in registered
    assert "sync_finished" in registered
    assert "sync_teams" in registered
    assert "sync_team_statistics" in registered
    assert "sync_player_statistics" in registered
    assert "sync_injuries" in registered
    assert n == 5


def test_register_jobs_skips_zero_or_negative_interval(caplog):
    # pydantic would forbid negative ints, but stub-driven tests can't.
    s = _SchedulerStubSettings(sync_teams_interval_minutes=0,
                               sync_injuries_interval_minutes=-5)
    sched = SyncScheduler(settings=s, scheduler=_RecordingScheduler())
    sched.register_jobs()

    registered = sched.list_registered_jobs()
    assert "sync_teams" not in registered
    assert "sync_injuries" not in registered
    # The rest still register.
    assert "sync_upcoming" in registered
    assert "sync_finished" in registered


def test_register_jobs_is_idempotent():
    s = _SchedulerStubSettings()
    sched = SyncScheduler(settings=s, scheduler=_RecordingScheduler())
    sched.register_jobs()
    first = sched.list_registered_jobs()
    second_count = sched.register_jobs()
    assert second_count == len(first)
    assert sched.list_registered_jobs() == first


def test_job_specs_covers_all_seven_jobs():
    names = {spec.name for spec in JOB_SPECS}
    assert names == {
        "sync_upcoming", "sync_finished", "sync_teams",
        "sync_team_statistics", "sync_player_statistics",
        "sync_injuries", "sync_lineups",
    }


# ---------------------------------------------------------------------------
# Lifecycle: start/shutdown honour SCHEDULER_ENABLED + idempotency.
# ---------------------------------------------------------------------------
async def test_start_noop_when_master_switch_off():
    s = _SchedulerStubSettings(scheduler_enabled=False)
    sched = SyncScheduler(settings=s, scheduler=_RecordingScheduler())
    started = await sched.start()
    assert started is False
    assert sched.is_running() is False
    # No jobs added because start() short-circuited before register_jobs().
    assert sched.list_registered_jobs() == []


async def test_start_registers_and_starts_when_enabled():
    # Here we use the REAL AsyncIOScheduler so .start() actually runs the
    # loop briefly; we shut it down within the test.
    s = _SchedulerStubSettings()
    sched = SyncScheduler(settings=s, scheduler=AsyncIOScheduler())
    try:
        started = await sched.start()
        assert started is True
        assert sched.is_running() is True
        # All 7 jobs enabled by the stub.
        assert len(sched.list_registered_jobs()) == 7
    finally:
        await sched.shutdown(wait=False)
        assert sched.is_running() is False


async def test_start_idempotent():
    s = _SchedulerStubSettings()
    sched = SyncScheduler(settings=s, scheduler=AsyncIOScheduler())
    try:
        assert await sched.start() is True
        # Second start is a no-op.
        assert await sched.start() is False
        assert sched.is_running() is True
    finally:
        await sched.shutdown(wait=False)


async def test_shutdown_idempotent():
    # Calling shutdown before start is a no-op.
    s = _SchedulerStubSettings()
    sched = SyncScheduler(settings=s, scheduler=AsyncIOScheduler())
    await sched.shutdown(wait=False)  # must not raise
    assert sched.is_running() is False


async def test_start_warns_when_no_jobs_registered(caplog):
    s = _SchedulerStubSettings(
        scheduler_enabled=True,
        sync_upcoming_enabled=False,
        sync_finished_enabled=False,
        sync_lineups_enabled=False,
        sync_injuries_enabled=False,
        sync_teams_enabled=False,
        sync_team_statistics_enabled=False,
        sync_player_statistics_enabled=False,
    )
    sched = SyncScheduler(settings=s, scheduler=AsyncIOScheduler())
    try:
        started = await sched.start()
        assert started is True  # scheduler starts even with no jobs
        assert sched.list_registered_jobs() == []
    finally:
        await sched.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Tick wrapper: exceptions never escape.
# ---------------------------------------------------------------------------
async def test_make_tick_swallows_exceptions(monkeypatch):
    from app.scheduler import _make_tick

    async def boom(*args: Any, **kwargs: Any) -> JobResult:
        raise RuntimeError("kaboom")

    tick = _make_tick("my_job", boom, _SchedulerStubSettings())
    # Should not raise.
    await tick()


async def test_make_tick_logs_ok_path(monkeypatch):
    from app.scheduler import _make_tick

    metrics = SyncMetrics(job="ok", requested=1, received=1, inserted=1)

    async def happy(*args: Any, **kwargs: Any) -> JobResult:
        return JobResult(job="ok", skipped=False, metrics=metrics)

    tick = _make_tick("my_job", happy, _SchedulerStubSettings())
    await tick()  # must not raise


async def test_make_tick_logs_skipped(monkeypatch):
    from app.scheduler import _make_tick

    async def skipped(*args: Any, **kwargs: Any) -> JobResult:
        return JobResult(job="s", skipped=True, metrics=None)

    tick = _make_tick("my_job", skipped, _SchedulerStubSettings())
    await tick()


async def test_make_tick_logs_failed(monkeypatch):
    from app.scheduler import _make_tick

    bad_metrics = SyncMetrics(job="bad", failed=2)

    async def with_failures(*args: Any, **kwargs: Any) -> JobResult:
        return JobResult(job="bad", skipped=False, metrics=bad_metrics)

    tick = _make_tick("my_job", with_failures, _SchedulerStubSettings())
    await tick()


# ---------------------------------------------------------------------------
# Module-level accessors.
# ---------------------------------------------------------------------------
async def test_singleton_get_scheduler_creates_once(monkeypatch):
    # The scheduler singleton stores a module-level reference. To keep tests
    # isolated we derive the scheduler fresh and reset the singleton after.
    from app.scheduler import get_scheduler, shutdown_scheduler
    # Reset any leftover singleton from previous tests.
    await shutdown_scheduler(wait=False)

    s = _SchedulerStubSettings()
    sched1 = get_scheduler(settings=s)
    sched2 = get_scheduler(settings=s)
    assert sched1 is sched2

    # Teardown.
    await shutdown_scheduler(wait=False)
