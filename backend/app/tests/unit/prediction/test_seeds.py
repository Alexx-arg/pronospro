"""Sprint 5.0 — seed derivation is deterministic, pure, per-fold distinct
and bounded to a 32-bit unsigned int (numpy-compatible)."""

from __future__ import annotations

import pytest
from app.prediction.seeds import derive_seed


def test_derive_seed_is_deterministic() -> None:
    assert derive_seed(base_seed=42, fold_index=0) == derive_seed(base_seed=42, fold_index=0)


def test_derive_seed_distinct_for_distinct_fold_indices() -> None:
    base = 42
    seeds = {derive_seed(base_seed=base, fold_index=i) for i in range(10)}
    assert len(seeds) == 10  # all distinct


def test_derive_seed_distinct_for_distinct_base_seeds() -> None:
    bases = {derive_seed(base_seed=b, fold_index=0) for b in range(10)}
    assert len(bases) == 10


def test_derive_seed_returns_non_negative_32bit_int() -> None:
    s = derive_seed(base_seed=42, fold_index=3)
    assert isinstance(s, int)
    assert 0 <= s <= 2**32 - 1


def test_derive_seed_rejects_negative_base_seed() -> None:
    with pytest.raises(ValueError):
        derive_seed(base_seed=-1, fold_index=0)


def test_derive_seed_rejects_negative_fold_index() -> None:
    with pytest.raises(ValueError):
        derive_seed(base_seed=42, fold_index=-1)
