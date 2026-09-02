"""Tests for the Sprint 5.0 prediction-engine foundation.

Sprint 5.0 only ships contracts, artifact dataclasses, settings,
seeds, and storage layout helpers. Tests here are STRICTLY
structural — they ensure:
  1. imports resolve,
  2. ``ModelName`` is exactly the three approved models,
  3. ``MatchProbabilities`` admits a valid 1X2 distribution,
  4. ``ModelArtifact`` is genuinely frozen (mutation raises),
  5. ``PredictionSettings`` validates its bounds,
  6. storage layout paths are deterministic (same input → same output),
  7. seed derivation is reproducible + per-fold distinct.

No model logic, no Poisson formula, no walk-forward iteration, no store
I/O is tested here — those live in Sprints 5.1 / 5.3 / 5.4 / 5.5+.
"""

from __future__ import annotations
