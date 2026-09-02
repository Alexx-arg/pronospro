"""Unit tests for :mod:`app.features.rolling` (window helpers).

Covers the core invariant of this phase: a window of size ``N`` sent
the first ``N`` items of the (already-filtered) history does NOT
silently include items beyond ``N`` and treats missing values per the
documented contract (``None`` for partial window, ``0`` reserved for
math-zero).

These tests do not touch the DB; they verify the math directly.
"""

from __future__ import annotations

from app.features.rolling import (
    rolling_count,
    rolling_mean,
    rolling_rate,
    rolling_sum,
    window_size,
)


def test_rolling_sum_excludes_items_past_n() -> None:
    """A 10-item history summed over N=5 must ignore items #6..#10."""
    history = [1.0, 2.0, 3.0, 5.0, 7.0, 100.0, 1000.0, 10000.0]
    # First 5: 1+2+3+5+7 = 18. The next 3 (100, 1000, 10000) MUST be
    # dropped — they are not in the window.
    assert rolling_sum(history, 5, lambda x: x) == 18.0


def test_rolling_sum_returns_none_for_partial_window() -> None:
    """If fewer than N items are available the sum is None, not 0."""
    history = [1.0, 2.0]  # only 2 items
    assert rolling_sum(history, 5, lambda x: x) is None
    # But the math-zero of a full 5-window IS a real zero.
    assert rolling_sum([0.0, 0.0, 0.0, 0.0, 0.0], 5, lambda x: x) == 0.0


def test_rolling_mean_skips_per_sample_none() -> None:
    """``None`` cells are skipped in the mean, not treated as 0."""
    history: list[float | None] = [None, 4.0, None, 8.0, None]
    # Only 4 and 8 are usable -> mean = 6. None does not pull DOWN.
    assert rolling_mean(history, 5, lambda x: x) == 6.0


def test_rolling_mean_returns_none_when_all_cells_none() -> None:
    """Even a full window of missing cells yields None."""
    history: list[float | None] = [None, None, None, None, None]
    assert rolling_mean(history, 5, lambda x: x) is None


def test_rolling_count_returns_zero_for_empty_window_not_none() -> None:
    """Counts have a natural zero baseline."""
    assert rolling_count([], 5, lambda x: True) == 0
    assert rolling_count([1, 2, 3], 5, lambda x: False) == 0
    # Window present: count the predicate matches among the first N.
    assert rolling_count([True, False, True, True, False], 5, lambda x: x) == 3


def test_rolling_rate_returns_none_for_partial_window() -> None:
    """The rate is undefined for a partial window."""
    assert rolling_rate([True, True], 5, lambda x: x) is None
    # Full window with all True -> 1.0.
    assert rolling_rate([True, True, True, True, True], 5, lambda x: x) == 1.0
    # Full window with all False -> 0.0 (real zero, not None).
    assert rolling_rate([False, False, False, False, False], 5, lambda x: x) == 0.0


def test_window_size_is_min_of_n_and_len() -> None:
    assert window_size([1, 2, 3], 5) == 3
    assert window_size([1, 2, 3, 4, 5, 6, 7], 5) == 5
    assert window_size([], 5) == 0
    assert window_size([1, 2, 3], 0) == 0
