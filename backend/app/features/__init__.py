"""Feature engineering layer (Phase 4).

Builds historical match examples from the normalised PostgreSQL schema
(:mod:`app.models`) by computing features **strictly from data available
before each fixture's kickoff**. Targets (post-match result) are kept
in a separate namespace so feature code can never consume target code by
accident.

The layer is split as:

* :mod:`app.features.rolling`  — window aggregations that always exclude
  the fixture under analysis.
* :mod:`app.features.asof`     — loaders that pull candidates (fixtures,
  team_statistics, ...) already filtered by ``kickoff < T`` /
  ``as_of_date < T``.
* :mod:`app.features.form`, :mod:`app.features.goals`,
  :mod:`app.features.homeaway`, :mod:`app.features.h2h`,
  :mod:`app.features.standings`, :mod:`app.features.rest`,
  :mod:`app.features.elo`, :mod:`app.features.xg` — concrete feature
  families built on top of the two previous layers.
* :mod:`app.features.assembler` — combines everything into a
  :class:`~app.features.example.HistoricalMatchExample`, separating
  ``features`` (pre-match) from ``targets`` (post-match).
* :mod:`app.features.example`  — the immutable DTO +
  :class:`FeatureNames` registry.

Nothing here imports from the prediction engine, GLM, or the providers.
Dependency direction:

    models / repositories → features → dataset
"""

from __future__ import annotations
