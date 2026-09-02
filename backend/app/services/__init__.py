"""Sub-package with sync services.

Each ``sync_*`` service orchestrates one round of data integration:

    Provider → DTO normalization → Sync service → Repository → PostgreSQL

Services are async, take a session (the caller owns the transaction) and
the :class:`DataProvider` to use, and return a :class:`SyncMetrics` object
that the scheduler logs and exposes via the admin endpoints (Phase 4).
"""
