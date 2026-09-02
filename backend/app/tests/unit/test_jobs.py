"""Unit tests for the standalone sync jobs (no DB / no HTTP).

These tests cover the contract of the jobs layer:

* The Job ENABLED flag is honoured: ``force=False`` + flag off -> ``skipped``.
* ``force=True`` bypasses the flag.
* The job forwards the right kwargs to the underlying sync service.
* The shared ``JobResult`` shape is correct (``ok`` reflects ``failed``).
* When the underlying sync service raises, the job does NOT re-raise —
  it fabricates a ``SyncMetrics(job=..., failed=1)`` and returns it.

Strategy: we monkeypatch the real sync service with a spy that records
its calls and returns a controllable ``SyncMetrics``. We pass an injected
``session`` (any non-None object) so the runner never opens its own
``session_scope`` (which would require a live PostgreSQL). The injected
``provider`` is likewise any sentinel — the spy never uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.services.sync.metrics import SyncMetrics
from app.tasks._runner import JobResult
from app.tasks import (
    sync_finished as sync_finished_mod,
    sync_injuries as sync_injuries_mod,
    sync_lineups as sync_lineups_mod,
    sync_player_statistics as sync_player_statistics_mod,
    sync_team_statistics as sync_team_statistics_mod,
    sync_teams as sync_teams_mod,
    sync_upcoming as sync_upcoming_mod,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
# The sync services are duck-typed against Settings; the lambda flag
# readers access named attrs. We expose ALL the *_enabled / *_interval
# attrs the scheduler / jobs look at so the same stub works for every job.
@dataclass
class _StubSettings:
    """Duck-typed Settings exposing the ENABLED flags + minor attrs."""

    # Master switch (used by SyncScheduler tests, not the jobs directly).
    scheduler_enabled: bool = True

    sync_upcoming_enabled: bool = True
    sync_upcoming_interval_minutes: int = 5
    sync_upcoming_days: int = 7

    sync_finished_enabled: bool = True
    sync_finished_interval_minutes: int = 15
    sync_finished_window_hours: int = 6

    sync_teams_enabled: bool = True
    sync_teams_interval_minutes: int = 720

    sync_team_statistics_enabled: bool = True
    sync_team_statistics_interval_minutes: int = 360

    sync_player_statistics_enabled: bool = True
    sync_player_statistics_interval_minutes: int = 360

    sync_injuries_enabled: bool = True
    sync_injuries_interval_minutes: int = 180

    sync_lineups_enabled: bool = True
    sync_lineups_interval_minutes: int = 120

    # Some sync services read these attributes too — provide defaults so
    # the stub is "settings-shaped" enough for any job that mistakenly
    # forwards it to the real service (the spy ignores them).
    current_league_ids: list[int] = field(default_factory=lambda: [39])
    current_season_year: int = 2024


class _Spy:
    """Records the kwargs of the last call and returns a controllable
    ``SyncMetrics``."""

    def __init__(self, job: str, *, fail: bool = False) -> None:
        self.job = job
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> SyncMetrics:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError(f"forced failure in {self.job}")
        metrics = SyncMetrics(job=self.job, requested=1, received=1, inserted=1)
        return metrics


_SENTINEL_PROVIDER = object()  # any non-None value; the spy ignores it
_SENTINEL_SESSION = object()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _patch_spy(monkeypatch: pytest.MonkeyPatch, module: Any, name: str,
               fail: bool = False) -> _Spy:
    """Replace ``module.<name>`` (the imported sync service) with a spy."""
    spy = _Spy(job=name, fail=fail)
    monkeypatch.setattr(module, name, spy)
    return spy


# ---------------------------------------------------------------------------
# sync_upcoming
# ---------------------------------------------------------------------------
async def test_run_sync_upcoming_invokes_service_and_returns_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _patch_spy(monkeypatch, sync_upcoming_mod, "sync_upcoming_fixtures")
    result = await sync_upcoming_mod.run_sync_upcoming(
        settings=_StubSettings(),
        provider=_SENTINEL_PROVIDER,
        session=_SENTINEL_SESSION,
        league_ids=[39, 140],
        days=14,
    )
    assert isinstance(result, JobResult)
    assert result.skipped is False
    assert result.ok is True
    assert result.metrics is not None
    assert result.metrics.job == "sync_upcoming"
    # The spy got provider/session/settings + the extra overrides.
    assert spy.calls, "sync_upcoming_fixtures was never called"
    call = spy.calls[0]
    assert call["provider"] is _SENTINEL_PROVIDER
    assert call["session"] is _SENTINEL_SESSION
    assert call["league_ids"] == [39, 140]
    assert call["days"] == 14


async def test_run_sync_upcoming_skipped_when_disabled_and_not_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _patch_spy(monkeypatch, sync_upcoming_mod, "sync_upcoming_fixtures")
    s = _StubSettings(sync_upcoming_enabled=False)
    result = await sync_upcoming_mod.run_sync_upcoming(
        settings=s, provider=_SENTINEL_PROVIDER, session=_SENTINEL_SESSION,
    )
    assert result.skipped is True
    assert result.metrics is None
    assert spy.calls == []  # not invoked


async def test_run_sync_upcoming_force_bypasses_disabled_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _patch_spy(monkeypatch, sync_upcoming_mod, "sync_upcoming_fixtures")
    s = _StubSettings(sync_upcoming_enabled=False)
    result = await sync_upcoming_mod.run_sync_upcoming(
        settings=s,
        provider=_SENTINEL_PROVIDER,
        session=_SENTINEL_SESSION,
        force=True,
    )
    assert result.skipped is False
    assert spy.calls != []
    assert result.metrics is not None


async def test_run_sync_upcoming_propagates_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_spy(monkeypatch, sync_upcoming_mod, "sync_upcoming_fixtures",
               fail=True)
    result = await sync_upcoming_mod.run_sync_upcoming(
        settings=_StubSettings(),
        provider=_SENTINEL_PROVIDER,
        session=_SENTINEL_SESSION,
    )
    # The job MUST NOT raise — it fabricates a failed SyncMetrics.
    assert result.skipped is False
    assert result.metrics is not None
    assert result.metrics.failed == 1
    assert result.ok is False
    assert result.metrics.errors  # at least one error logged


# ---------------------------------------------------------------------------
# sync_finished: extra kwarg forwarded
# ---------------------------------------------------------------------------
async def test_run_sync_finished_forwards_window_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _patch_spy(monkeypatch, sync_finished_mod, "sync_finished_fixtures")
    await sync_finished_mod.run_sync_finished(
        settings=_StubSettings(),
        provider=_SENTINEL_PROVIDER,
        session=_SENTINEL_SESSION,
        window_hours=24,
        league_ids=[61],
    )
    assert spy.calls
    assert spy.calls[0]["window_hours"] == 24
    assert spy.calls[0]["league_ids"] == [61]


# ---------------------------------------------------------------------------
# sync_lineups: extra kwarg forwarded
# ---------------------------------------------------------------------------
async def test_run_sync_lineups_forwards_fixture_external_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _patch_spy(monkeypatch, sync_lineups_mod, "sync_lineups")
    await sync_lineups_mod.run_sync_lineups(
        settings=_StubSettings(),
        provider=_SENTINEL_PROVIDER,
        session=_SENTINEL_SESSION,
        fixture_external_ids=[1, 2, 3],
    )
    assert spy.calls
    assert spy.calls[0]["fixture_external_ids"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Sanity: every job skips when its own flag is False
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("module", "func_name", "flag_attr"),
    [
        (sync_teams_mod, "run_sync_teams", "sync_teams_enabled"),
        (sync_team_statistics_mod, "run_sync_team_statistics",
         "sync_team_statistics_enabled"),
        (sync_player_statistics_mod, "run_sync_player_statistics",
         "sync_player_statistics_enabled"),
        (sync_injuries_mod, "run_sync_injuries", "sync_injuries_enabled"),
        (sync_finished_mod, "run_sync_finished", "sync_finished_enabled"),
        (sync_lineups_mod, "run_sync_lineups", "sync_lineups_enabled"),
        (sync_upcoming_mod, "run_sync_upcoming", "sync_upcoming_enabled"),
    ],
)
async def test_each_job_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    func_name: str,
    flag_attr: str,
) -> None:
    # Find the sync service name the module imports (it's the attr the
    # run_sync_* helper passes to run_job). All of our modules name the
    # sync service as ``sync_*`` lowercased — we discover it by listing
    # the module's __all__ (or known attrs).
    sync_service_name = _module_sync_service_name(module)
    spy = _patch_spy(monkeypatch, module, sync_service_name)

    s = _StubSettings(**{flag_attr: False})
    runner = getattr(module, func_name)
    result = await runner(
        settings=s, provider=_SENTINEL_PROVIDER, session=_SENTINEL_SESSION,
    )
    assert result.skipped is True, (
        f"{func_name} should skip when {flag_attr}=False"
    )
    assert result.metrics is None
    assert spy.calls == []


def _module_sync_service_name(module: Any) -> str:
    """Return the name of the sync service each job module imports."""
    known = {
        sync_upcoming_mod: "sync_upcoming_fixtures",
        sync_finished_mod: "sync_finished_fixtures",
        sync_teams_mod: "sync_teams",
        sync_team_statistics_mod: "sync_team_statistics",
        sync_player_statistics_mod: "sync_player_statistics",
        sync_injuries_mod: "sync_injuries",
        sync_lineups_mod: "sync_lineups",
    }
    return known[module]
