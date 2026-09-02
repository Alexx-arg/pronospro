"""Shared scaffolding for the standalone sync jobs.

The per-job modules are one-liners that delegate here. Centralising the
boilerplate keeps the actual job files readable and ensures every job
obeys the same contract:

    settings -> provider -> session_scope -> sync service -> log -> return

The helper is **private** to this package (leading underscore) and is not
part of the public ``app.tasks`` API — callers should invoke the
``run_sync_*`` coroutines directly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.session import session_scope
from app.providers.base import DataProvider
from app.providers.registry import get_provider
from app.services.sync.metrics import SyncMetrics

_LOG = get_logger()


@dataclass(slots=True)
class JobResult:
    """Outcome of one job invocation.

    ``metrics`` is ``None`` when the job was skipped because its ENABLED
    flag was ``False`` (and ``force`` was not requested).
    """

    job: str
    skipped: bool
    metrics: SyncMetrics | None

    @property
    def ok(self) -> bool:
        """``True`` when the job ran without failures (or was skipped)."""
        if self.skipped or self.metrics is None:
            return True
        return self.metrics.failed == 0


def _resolve_provider(
    provider: DataProvider | None,
    settings: Settings,
) -> DataProvider:
    return provider if provider is not None else get_provider(settings)


async def run_job(
    *,
    job_name: str,
    enabled_flag_fn: Callable[[Settings], bool],
    force: bool,
    sync_fn: Callable[..., Awaitable[SyncMetrics]],
    settings: Settings | None,
    provider: DataProvider | None,
    session: AsyncSession | None,
    sync_kwargs: dict[str, Any] | None,
) -> JobResult:
    """Run one sync job, honouring the ENABLED flag.

    Args:
        job_name: human-readable name used for logging.
        enabled_flag_fn: reads the ENABLED boolean from the resolved
            settings — e.g. ``lambda s: s.sync_upcoming_enabled``. Keeps
            the per-job flag knowledge in the per-job module.
        force: when ``True`` the ENABLED flag is bypassed. Used by the
            admin endpoint / CLI for manual invocations.
        sync_fn: the underlying sync service coroutine factory.
        settings: explicit settings override (tests). ``None`` reads the
            env-driven singleton.
        provider: explicit provider override (tests). ``None`` resolves
            via the registry.
        session: explicit async session override (tests). When ``None``
            the helper opens its own :func:`session_scope`.
        sync_kwargs: extra keyword arguments forwarded to ``sync_fn``
            (e.g. ``window_hours`` for the finished-fixtures job).
    """
    s = settings if settings is not None else get_settings()
    enabled = enabled_flag_fn(s)

    if not enabled and not force:
        _LOG.info("job {} skipped (disabled, force=False)", job_name)
        return JobResult(job=job_name, skipped=True, metrics=None)

    p = _resolve_provider(provider, s)
    kw = dict(sync_kwargs or {})

    async def _execute(sess: AsyncSession) -> SyncMetrics:
        return await sync_fn(
            provider=p,
            session=sess,
            settings=s,
            **kw,
        )

    try:
        if session is not None:
            # Caller owns the transaction (e.g. an integration test). We
            # must NOT commit/rollback here; just run the sync.
            metrics = await _execute(session)
        else:
            async with session_scope() as sess:
                metrics = await _execute(sess)
    except Exception as exc:  # noqa: BLE001  (top-level job guard)
        _LOG.error("job {} crashed: {}", job_name, str(exc)[:512])
        metrics = SyncMetrics(job=job_name)
        metrics.failed = 1
        metrics.add_error(f"job crashed: {exc!r}")
        # session_scope already rolled back when an exception propagated.

    _LOG.info("job {} done: {}", job_name, metrics.as_dict())
    return JobResult(job=job_name, skipped=False, metrics=metrics)


__all__ = ["JobResult", "run_job"]
