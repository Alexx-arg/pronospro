"""Prediction engine (Phase 5).

Consumes **exclusively** the immutable ``LoadedDataset`` produced by
Phase 4 (see :mod:`app.dataset.loader`). The engine never reads
PostgreSQL and never recomputes features — those guarantees are
enforced by the contracts in :mod:`app.prediction.contracts` and
validated by structural tests in ``app/tests/unit/prediction``.

This package is offline / CLI-only in Phase 5: no FastAPI or Flutter
integration is provided. FastAPI endpoints and Flutter consumers are
explicitly deferred to later phases.
"""

from __future__ import annotations

__all__: list[str] = []
