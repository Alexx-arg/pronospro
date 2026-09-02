"""Walk-forward fold (Sprint 5.1).

A :class:`Fold` is an immutable, fully-described train/val/test slice
produced by :class:`app.prediction.backtesting.iterator.WalkForwardIterator`.

Design contract (see ``docs/PHASE_5.md`` §6 and Sprint 5.1 brief):

* The fold carries **row indices** of the underlying
  :class:`app.dataset.loader.LoadedDataset.rows` list — never raw rows
  or stolen references. Indices are positions in a list already ordered
  by ``(kickoff, fixture_id)`` (Phase 4 builder guarantee), so spatial
  adjacency in the index tuple mirrors temporal adjacency.
* Every chronology invariant (train strictly before val, val strictly
  before test, no overlap, gap respected) is enforced by the iterator
  at construction time; :class:`Fold` itself is structural only.
* Cutoffs use real kickoff timestamps, not row counts.

Timestamps semantics
--------------------

>>> train_start   = rows[train_indices[0]].kickoff
>>> train_end     = rows[train_indices[-1]].kickoff
>>> val_start     = rows[val_indices[0]].kickoff
>>> val_end       = rows[val_indices[-1]].kickoff
>>> test_start    = rows[test_indices[0]].kickoff
>>> test_end      = rows[test_indices[-1]].kickoff
>>> training_cutoff   = train_end
>>> validation_cutoff = val_end
>>> test_cutoff       = test_end

``gap_end`` carries an unambiguous meaning: the **earliest kickoff a
row may legally have to still belong to val_start's block** — i.e.
``val_start`` itself when ``gap_days == 0``. It is the lower inclusive
frontier of the val block **after** the train block has been closed:

    gap_end := val_start

In formulas:

    train_end + timedelta(days=gap_days) <= val_start
    val_start is the first kickoff included in the val block
    ⇒ gap_end (the legal-lower-bound of val) is val_start

This is *not* a separate absolute instant distinct from ``val_start`` —
it's a documented synonym so callers / tests can refer to "the point
just after the train+gap boundary" without reinventing it. When gap =
0 we still expose it so downstream code (artifact metadata) has a
single name to read regardless of gap configuration.

Indirect equality ``gap_end == val_start`` holds **always**. The
*temporal* width of the gap is captured by comparing ``train_end`` and
``gap_end`` (the train→val front-gap), and ``val_end`` vs ``test_start``
(the val→test front-gap).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

#: Type alias for the index tuples carried by a fold. Frozen folds must
#: be hashable, so we use tuples (not lists). Indices are non-negative
#: ints into ``LoadedDataset.rows``.
IndexTuple = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward fold: ``(train, val, test)`` slices + cutoffs.

    Invariants (enforced by the iterator when building the instance):

    * ``len(train_indices) > 0`` and ``len(val_indices) > 0`` and
      ``len(test_indices) > 0``.
    * The three index sets are pairwise disjoint.
    * Temporal ordering: ``train_end < val_start < val_end < test_start``
      (strict ``<`` on real kickoff timestamps at the boundaries).

    The dataclass itself performs no defensive validation: it trusts the
    iterator's construction contract. A ``Fold`` is a value object — the
    iterator chooses which indices/timestamps land in it, then yields it
    fully-formed. Tests re-check the invariants from the outside.
    """

    fold_id: int
    train_indices: IndexTuple
    val_indices: IndexTuple
    test_indices: IndexTuple
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime
    gap_end: datetime
    training_cutoff: datetime
    validation_cutoff: datetime
    test_cutoff: datetime

    def __post_init__(self) -> None:
        """Lightweight structural checks.

        These are cheap and serve as defence-in-depth in case a future
        caller constructs a :class:`Fold` outside the iterator. They
        intentionally do NOT recompute the temporal invariants on raw
        dataset rows (that requires the dataset reference, which the
        fold deliberately does not keep). The iterator is authoritative
        for temporal invariants.
        """
        if self.fold_id < 0:
            raise ValueError(
                f"fold_id must be non-negative, got {self.fold_id}"
            )
        if len(self.train_indices) == 0:
            raise ValueError("Fold.train_indices must be non-empty")
        if len(self.val_indices) == 0:
            raise ValueError("Fold.val_indices must be non-empty")
        if len(self.test_indices) == 0:
            raise ValueError("Fold.test_indices must be non-empty")

        # Set disjointness — O(n) via sorted scan, since indices are
        # positions in an ordered list and thus *tend* to ranges. We
        # don't rely on that tendency for correctness; we just use sets.
        s_train = set(self.train_indices)
        s_val = set(self.val_indices)
        s_test = set(self.test_indices)
        if s_train & s_val:
            raise ValueError(
                "Fold train_indices and val_indices overlap: "
                f"shared={sorted(s_train & s_val)[:5]}..."
            )
        if s_train & s_test:
            raise ValueError(
                "Fold train_indices and test_indices overlap: "
                f"shared={sorted(s_train & s_test)[:5]}..."
            )
        if s_val & s_test:
            raise ValueError(
                "Fold val_indices and test_indices overlap: "
                f"shared={sorted(s_val & s_test)[:5]}..."
            )

        # No index may be negative.
        for idx in (*self.train_indices, *self.val_indices, *self.test_indices):
            if idx < 0:
                raise ValueError(f"Fold index must be non-negative, got {idx}")

    # ------------------------------------------------------------------
    # Hashing / equality
    # ------------------------------------------------------------------
    # ``@dataclass(frozen=True, slots=True)`` already generates a
    # sensible ``__hash__`` and ``__eq__`` from all fields. Datetimes
    # are hashable, tuples are hashable, ints are hashable ⇒ Fold is
    # hashable out of the box. No override needed.

    def to_summary(self) -> dict[str, Any]:
        """JSON-serialisable summary dict-of-primitives view of the fold.

        Useful when persisting run metadata (Sprint 5.8). Datetimes
        are ISO-encoded so the dict survives ``json.dumps``.
        """
        return {
            "fold_id": self.fold_id,
            "n_train": len(self.train_indices),
            "n_val": len(self.val_indices),
            "n_test": len(self.test_indices),
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "val_start": self.val_start.isoformat(),
            "val_end": self.val_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "gap_end": self.gap_end.isoformat(),
            "training_cutoff": self.training_cutoff.isoformat(),
            "validation_cutoff": self.validation_cutoff.isoformat(),
            "test_cutoff": self.test_cutoff.isoformat(),
        }


__all__ = ["Fold", "IndexTuple"]
