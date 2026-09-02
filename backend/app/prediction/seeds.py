"""Deterministic seeding utilities for the prediction engine.

A single source of truth for seed handling across all Sprint 5.x
trainers. Two responsibilities:

1. **Per-run base seed**: the value of
   :attr:`PredictionSettings.random_seed` is forked into a per-fold
   derived seed so that two fold trainers don't accidentally share
   the same RNG state (which would make fold-to-fold leakiness more
   subtle to detect).
2. **Pure / hash-able API**: every helper here is deterministic and
   pure — the output of :func:`derive_seed` is a pure function of its
   inputs, no global RNG state. The actual RNG construction is left
   to the trainer (numpy's ``default_rng`` is the recommended one but
   not pinned at this layer — keeps the contract library-agnostic).

Reproducibility contract (see ``docs/PHASE_5.md`` §9.4) tests:
* ``test_reproducible_seed_derivation`` — same inputs same outputs.
* ``test_seed_per_fold_differs`` — folds inside a run get distinct seeds.
"""

from __future__ import annotations

import hashlib


def derive_seed(*, base_seed: int, fold_index: int) -> int:
    """Derive a 32-bit fold seed from ``base_seed`` + ``fold_index``.

    Deterministic, pure, hash-based: same ``(base_seed, fold_index)``
    always produces the same integer.

    The hash is SHA-256 for collision-resistance; we truncate to the
    first 32 bits (int4) so the result fits numpy's seed slot
    (``np.random.default_rng`` accepts any non-negative int).

    Raises:
        ValueError: if ``base_seed < 0`` or ``fold_index < 0`` (a
            negative seed is a Python ``random`` oddity, not a
            contract we want leaking through).
    """
    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}")
    if fold_index < 0:
        raise ValueError(f"fold_index must be non-negative, got {fold_index}")
    payload = f"{base_seed}:{fold_index}".encode()
    digest = hashlib.sha256(payload).digest()
    # Take the first 4 bytes as a big-endian unsigned int.
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


__all__ = ["derive_seed"]
