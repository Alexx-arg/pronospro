"""Rolling-window aggregations over an already-filtered history stream.

Every function in this module takes the **finished** matches of a single
team (or pair of teams, for H2H) **already ordered by ``kickoff_time``
descending** and sliced to ``kickoff_time < T`` by the caller.

The functions NEVER re-filter by ``kickoff_time`` themselves: the loader
(:mod:`app.features.asof`) is the single point of truth for the as-of
boundary. That keeps these helpers pure, trivially testable and reusable
both for the dataset builder and for the anti-leakage test suite.

Windows used across the project (see docs/FEATURES.md):

* ``3``  — short-term form
* ``5``  — typical recent sample
* ``10`` — medium-term window

Missing-data semantics
----------------------
Every ``*_last_N`` feature returns ``None`` (NOT ``0``) when fewer than
``N`` qualifying fixtures exist. ``0`` is returned only when N fixtures
exist and the aggregated value is **mathematically** zero (e.g. 0 goals
scored across 5 completed matches).

Count-style helpers (``rolling_count``) return ``0`` when no fixture
matches the predicate — that IS the correct zero semantics for a count
and is documented per-feature in docs/FEATURES.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def _take_n(history: Iterable[T], n: int) -> list[T]:
    """Return the first ``n`` items of ``history`` as a fresh list.

    ``history`` is assumed to already be filtered (kickoff < T) and
    ordered newest-first by the caller. ``n <= 0`` returns ``[]``.
    """
    if n <= 0:
        return []
    out: list[T] = []
    for item in history:
        out.append(item)
        if len(out) >= n:
            break
    return out


def window_size(history: list[T], n: int) -> int:
    """Effective window size: ``min(len(history), n)`` (0 if ``n <= 0``)."""
    if n <= 0:
        return 0
    return min(len(history), n)


def rolling_sum(
    history: list[T],
    n: int,
    value: Callable[[T], float | None],
) -> float | None:
    """Sum of ``value(item)`` over the first ``n`` items.

    Returns ``None`` if fewer than ``n`` items qualify (window not full)
    so downstream features can keep the explicit "missing" semantics.
    Returns a non-``None`` value when the window is full (including 0.0
    when every qualifying item evaluates to 0 / None).

    A ``None`` from ``value(item)`` is treated as a missing data point
    inside the window — the window item is still COUNTED toward ``n``
    but contributes 0 to the sum. Use :func:`rolling_mean` if you want
    missing values to bias the average downward instead.
    """
    window = _take_n(history, n)
    if len(window) < n:
        return None
    total = 0.0
    for item in window:
        v = value(item)
        if v is not None:
            total += v
    return total


def rolling_mean(
    history: list[T],
    n: int,
    value: Callable[[T], float | None],
) -> float | None:
    """Arithmetic mean of ``value(item)`` over the first ``n`` items.

    Returns ``None`` when the window is not full or when every item in
    the window was missing (``value(item) is None``). Returns a
    non-``None`` value otherwise — including ``0.0`` when the non-missing
    items all evaluated to zero.
    """
    window = _take_n(history, n)
    if len(window) < n:
        return None
    total = 0.0
    have = 0
    for item in window:
        v = value(item)
        if v is not None:
            total += v
            have += 1
    if have == 0:
        return None
    return total / have


def rolling_count(
    history: list[T],
    n: int,
    predicate: Callable[[T], bool],
) -> int:
    """Count items among the first ``n`` satisfying ``predicate``.

    Unlike ``rolling_sum``/``rolling_mean``, this returns ``0`` (NOT
    ``None``) when the window is empty — counts have a natural zero
    baseline. When the window is partial (fewer than ``n`` items) we
    still count what's available; callers that require a full sample
    should wrap this with :func:`window_size` checks.
    """
    window = _take_n(history, n)
    return sum(1 for item in window if predicate(item))


def rolling_rate(
    history: list[T],
    n: int,
    predicate: Callable[[T], bool],
) -> float | None:
    """Fraction of the first ``n`` items satisfying ``predicate``.

    Returns ``None`` when fewer than ``n`` items are available (the
    rate is undefined for a partial window). Returns ``0.0`` when N
    items exist and none satisfy the predicate, and ``1.0`` when all
    of them do.
    """
    window = _take_n(history, n)
    if len(window) < n:
        return None
    if n == 0:
        return None
    matches = sum(1 for item in window if predicate(item))
    return matches / n


__all__ = [
    "rolling_count",
    "rolling_mean",
    "rolling_rate",
    "rolling_sum",
    "window_size",
]
