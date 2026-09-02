"""Historical dataset package (Phase 4).

Builds versioned historical datasets from the normalised PostgreSQL
schema, materialising them as **immutable** CSV + metadata.json on
disk so the dataset can be:

* regenerated deterministically from the same source state (data +
  code revision),
* verified by SHA-256 checksum,
* loaded later for training / walk-forward backtesting / model
  comparison (Phase 5+).

This package does NOT touch the model training / prediction engine.
"""

from __future__ import annotations
