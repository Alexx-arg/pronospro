"""Walk-forward iterator (Sprint 5.1).

Produced folds are temporal non-overlapping slices of the
:class:`app.dataset.loader.LoadedDataset` produced by Phase 4. Every
fold is consumed by trainers (Sprint 5.5–5.7) and evaluated against the
test block via Sprint 5.3 metrics.

Canonical contract — see ``docs/PHASE_5.md`` §6 and the Sprint 5.1
brief in the project notes.

Algorithm (high level)
----------------------

Each iteration is defined by a pair of cursors ``(train_start_idx,
train_end_idx)`` describing the inclusive row range of the train
block. From that pair, the iterator computes:

1. Val block start: the first row index ``ix`` such that
   ``rows[ix].kickoff >= rows[train_end_idx].kickoff + gap``,
   expanded forward through equal-kickoff rows so val gets the whole
   kickoff group the boundary opens on.
2. Val block end: ``val_start + val_target - 1``, where ``val_target``
   is ``val_size`` (fixed) or ``round(train_len * val_ratio)``
   (proportional). End may grow to swallow a trailing equal-kickoff
   tail, but never past the point where test becomes impossible.
3. Test block start: first row index ``ix`` such that
   ``rows[ix].kickoff >= rows[val_end_idx].kickoff + gap``.
4. Test block end: ``test_start + test_size - 1``, then grown to
   swallow a trailing equal-kickoff tail.

The train cursors for the next iteration are computed at the *end* of
the current iteration:

* ``EXPANDING``: ``train_start_idx = 0`` always; ``train_end_idx`` of
  the next iteration = ``test_end_idx`` of the current (the entire
  seen-so-far history becomes train).
* ``SLIDING``: ``train_end_idx`` of next = ``test_end_idx`` of current;
  ``train_start_idx`` of next is chosen so the train window has
  approximately ``min_train_size`` rows (i.e. ``test_end_idx -
  min_train_size + 1``), nudged backward through equal-kickoff rows so
  we never split a kickoff group with the previous (now-vanished)
  train.

Equal-kickoff rule (canonical for this package)
-----------------------------------------------

Real-world fixtures share exact kickoff timestamps (same kick-off
minute across leagues). Index-order ties via ``fixture_id`` are an
implementation artefact — a higher ``fixture_id`` does NOT mean a
later fixture. We must preserve *temporal* integrity, never
``fixture_id`` integrity.

**Rule:** a temporal frontier that would split a group of equal-kickoff
rows is forbidden. The frontier is pushed deterministically so the
entire equal-kickoff group joins the **right-hand** block (i.e. the
block the boundary was about to open).

* train→val boundary drawn at row ``ix``: all rows ``j >= ix`` with
  ``rows[j].kickoff == rows[ix].kickoff`` stay together on the val
  side. In practice, this means the train tail is shrunk backward
  until ``rows[train_end].kickoff < rows[train_end + 1].kickoff``.
* val→test boundary: same rule, applied to the val tail.

Consequences: val/test sizes may deviate from their nominal targets
by a few rows. Anti-leak guarantee ``max(train.kickoff) <
min(val.kickoff)`` (strict ``<``) is preserved by construction.

Failure-mode policy
-------------------

* **Insufficient data**: the iterator stops cleanly. Yielding zero
  folds is a valid outcome and is easily testable by callers.
* **Broken invariant in a fold that did satisfy the size params**: a
  :class:`WalkForwardError` is raised — a hard construction error,
  not silent silent INFO log. This only fires when equal-kickoff
  clumping truly can't separate the blocks (e.g. all rows share one
  kickoff).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from typing import Any

from app.dataset.loader import LoadedDataset
from app.prediction.backtesting.fold import Fold, IndexTuple
from app.prediction.contracts import WalkForwardMode


class WalkForwardError(ValueError):
    """Raised when a fold that nominally satisfies the size parameters
    still violates a chrono-temporal invariant.

    Distinct from "insufficient data" (which is a clean stop, no
    exception).
    """


# ---------------------------------------------------------------------------
# Iterator
# ---------------------------------------------------------------------------
class WalkForwardIterator:
    """Yield :class:`Fold` objects from a :class:`LoadedDataset`.

    Construction does no work — iteration is lazy. Each iteration
    determines a ``(train, val, test)`` triple, validates temporality
    strictly, yields a :class:`Fold`, then advances cursors per
    ``mode``.

    Reproducibility: the iterator is a pure function of
    ``(dataset, params)``. Calling it twice on the same inputs yields
    identical folds (validated by ``test_reproducibility_with_same_params``).
    """

    def __init__(
        self,
        dataset: LoadedDataset,
        *,
        min_train_size: int,
        test_size: int,
        gap_days: int,
        mode: WalkForwardMode,
        val_ratio: float | None = None,
        val_size: int | None = None,
    ) -> None:
        """Construct an iterator.

        Args:
            dataset: Phase 4 ``LoadedDataset``. Must be ordered by
                ``(kickoff, fixture_id)`` — verified at iteration start.
            min_train_size: minimum size of the first train block
                (also the sliding-block target size when
                ``mode == SLIDING``). Must be ``>= 1``.
            test_size: number of rows in each test block. Must be
                ``>= 1``.
            gap_days: temporal buffer (calendar days) between
                train→val and val→test. ``0`` is valid.
            mode: :data:`WalkForwardMode.EXPANDING` or ``SLIDING``.
            val_ratio: proportion of *current* train size to use as
                the val block size. Mutually exclusive with
                ``val_size``. Must be in ``(0, 1)``.
            val_size: fixed row count for the val block. Mutually
                exclusive with ``val_ratio``. Must be ``>= 1``.

        Raises:
            ValueError: if the size parameters are inconsistent.
        """
        if min_train_size < 1:
            raise ValueError(
                f"min_train_size must be >= 1, got {min_train_size}"
            )
        if test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")
        if gap_days < 0:
            raise ValueError(f"gap_days must be >= 0, got {gap_days}")
        if val_ratio is not None and val_size is not None:
            raise ValueError(
                "val_ratio and val_size are mutually exclusive — "
                "provide exactly one."
            )
        if val_ratio is not None and not (0.0 < val_ratio < 1.0):
            raise ValueError(
                f"val_ratio must satisfy 0.0 < r < 1.0, got {val_ratio}"
            )
        if val_size is not None and val_size < 1:
            raise ValueError(f"val_size must be >= 1, got {val_size}")
        if not isinstance(mode, WalkForwardMode):
            raise ValueError(
                f"mode must be a WalkForwardMode, got {type(mode).__name__}"
            )

        self._dataset = dataset
        self._min_train_size = min_train_size
        self._test_size = test_size
        self._gap = timedelta(days=gap_days)
        self._gap_days = gap_days
        self._mode = mode
        self._val_ratio = val_ratio
        self._val_size = val_size

    # ------------------------------------------------------------------
    # Public iteration
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Fold]:
        # Validate dataset ordering once, up-front, on iteration start.
        # Failure here is a hard error.
        self._validate_ordering_or_raise()

        rows = self._dataset.rows
        n = len(rows)
        if n < self._min_train_size:
            # Not even enough for a single train block — clean stop,
            # zero folds. (Caller of `list(WalkForwardIterator(...))`
            # gets `[]`.)
            return

        # First iteration: train = rows[0 .. min_train_size - 1],
        # then shrunk backward if row just past the tail shares its
        # kickoff.
        train_start_idx = 0
        train_end_idx = self._min_train_size - 1
        train_end_idx = self._shrink_train_tail(train_end_idx, n)
        # min_train_size is a REAL floor (PHASE_5.md §6.5 — "min_train_size
        # is a hard floor, not a nominal target").
        #
        # If the equal-kickoff shrink brings train below ``min_train_size``,
        # the first fold cannot honour the parameter and is therefore
        # invalid. We do NOT silently emit a short fold. We stop iteration
        # without yielding — yielding zero folds is the valid, caller-
        # observable outcome (``list(WalkForwardIterator(...)) == []``).
        #
        # Why not "skip the first fold and continue" — in EXPANDING the
        # train block starts at row 0 forever; there is no way to start
        # later because train would no longer be the earliest contiguous
        # slice. In SLIDING, the first fold's train is the dataset's
        # first ``min_train_size`` rows; skipping would silently shift
        # the train window ahead, breaking reproducibility / comparable-
        # across-models semantics (§6.4). Stop is the only honest answer.
        if self._train_len(train_start_idx, train_end_idx) < self._min_train_size:
            return

        fold_id = 0
        while True:
            #### 1. Determine val block ####
            train_len = train_end_idx - train_start_idx + 1
            val_target = self._val_target(train_len)
            val_start_idx = self._scan_past_gap(
                train_end_idx, n
            )
            if val_start_idx is None:
                # Nothing past the gap → insufficient data, stop.
                return

            # Move val_start backward isn't allowed (would split a
            # kickoff group with train). The shrink of train tail
            # above already ensured rows[train_end].kickoff <
            # rows[val_start].kickoff if they're different rows; if
            # they share kickoff it's because train was shrunk and
            # val_start inherits the next kickoff after train's tail.
            # But _scan_past_gap returns the first index whose kickoff
            # >= train_tail_kickoff + gap, so when gap=0 we may get
            # val_start_idx = train_end_idx + 1 with same kickoff —
            # that would be a leak. Enforce: val_start must have
            # kickoff strictly greater than train_end's, OR (only when
            # gap=0) at least a different kickoff value.
            if rows[val_start_idx].kickoff == rows[train_end_idx].kickoff:
                # Equal kickoff after the gap scan: clumping is
                # impossible to break (gap=0 and tail of train shares
                # kickoff with the candidate val_start). This is a
                # hard error: the parameters were honoured but the
                # dataset doesn't allow chrono-separation.
                raise WalkForwardError(
                    f"train→val boundary cannot be temporally separated: "
                    f"rows[{train_end_idx}].kickoff == "
                    f"rows[{val_start_idx}].kickoff == "
                    f"{rows[val_start_idx].kickoff.isoformat()}, "
                    "and gap_days=0 doesn't help. "
                    "Equal-kickoff rule can't split this group."
                )

            # Compute val_end (inclusive). Nominal = val_start + val_target - 1.
            val_end_idx = val_start_idx + val_target - 1
            # If the nominal val block runs off the end of the dataset,
            # stop cleanly — we don't emit short val blocks. The brief
            # explicitly forbids partial folds; "stop when data
            # insufficient" is a valid, no-error outcome.
            if val_end_idx >= n:
                return
            # Now expand val_end forward to swallow equal-kickoff tail,
            # but only across rows that share val_end's kickoff. Note
            # equal-kickoff swallowing may push val_end further into
            # the dataset; that's fine, the test block builder below
            # will scan past the (possibly enlarged) val_end + gap.
            v_e = val_end_idx
            while (
                v_e + 1 < n
                and rows[v_e + 1].kickoff == rows[v_e].kickoff
            ):
                v_e += 1
            val_end_idx = v_e

            #### 2. Determine test block ####
            test_start_idx = self._scan_past_gap(val_end_idx, n)
            if test_start_idx is None:
                return
            if rows[test_start_idx].kickoff == rows[val_end_idx].kickoff:
                raise WalkForwardError(
                    f"val→test boundary cannot be temporally separated: "
                    f"rows[{val_end_idx}].kickoff == "
                    f"rows[{test_start_idx}].kickoff == "
                    f"{rows[test_start_idx].kickoff.isoformat()}, "
                    "and gap_days=0 doesn't help."
                )
            test_end_idx = test_start_idx + self._test_size - 1
            if test_end_idx >= n:
                # Not enough rows for a full test block of test_size.
                # Per Sprint 5.1 brief: stop cleanly, do not emit a
                # partial fold.
                return
            # Swallow equal-kickoff tail (test tail can grow freely
            # to the end of the dataset; no downstream constraint).
            t_e = test_end_idx
            while (
                t_e + 1 < n
                and rows[t_e + 1].kickoff == rows[t_e].kickoff
            ):
                t_e += 1
            test_end_idx = t_e

            #### 3. Build the index tuples ####
            train_indices: IndexTuple = tuple(
                range(train_start_idx, train_end_idx + 1)
            )
            val_indices: IndexTuple = tuple(
                range(val_start_idx, val_end_idx + 1)
            )
            test_indices: IndexTuple = tuple(
                range(test_start_idx, test_end_idx + 1)
            )

            #### 4. Build + yield the Fold (with strict chrono check) ####
            fold = self._build_fold(
                fold_id,
                train_indices,
                val_indices,
                test_indices,
            )
            yield fold
            fold_id += 1

            #### 5. Advance cursors for the next iteration ####
            # Common to both modes: the new train_end_idx is the
            # current test_end_idx (inclusive). All rows up to the end
            # of test are "past" → available for the next train.
            next_train_end_idx = test_end_idx
            # If next_train_end_idx is the last row, the next iteration
            # has no val block space — stop.
            if next_train_end_idx >= n - 1:
                return

            if self._mode == WalkForwardMode.EXPANDING:
                next_train_start_idx = 0
            else:
                # SLIDING: keep train window ~min_train_size long.
                # New train_start (target) = next_train_end - min_train_size + 1
                tgt = next_train_end_idx - self._min_train_size + 1
                if tgt < 0:
                    tgt = 0
                # Nudge tgt backward so we don't leave a split kickoff
                # group behind: if rows[tgt] shares kickoff with
                # rows[tgt - 1], include tgt - 1 in the new train
                # (i.e. decrement tgt). This keeps the kickoff group
                # intact on one side of the train boundary.
                while (
                    tgt > 0
                    and rows[tgt].kickoff == rows[tgt - 1].kickoff
                ):
                    tgt -= 1
                next_train_start_idx = tgt

            # Apply the new cursors. The train tail shrink (for the
            # train→val boundary of the next iteration) is applied at
            # the *top* of the next loop iteration.
            train_start_idx = next_train_start_idx
            train_end_idx = next_train_end_idx
            # Apply equal-kickoff shrink to the train tail BEFORE the
            # next iteration uses it. This is the symmetric operation
            # to the initial shrink above.
            train_end_idx = self._shrink_train_tail(train_end_idx, n)
            # Same min_train_size floor as the initial shrink: if the
            # shrink collapses train below the requested minimum, stop.
            if self._train_len(train_start_idx, train_end_idx) < self._min_train_size:
                return
            # Loop continues; next iter computes a fresh val/test.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_ordering_or_raise(self) -> None:
        """Verify ``rows`` is ordered by ``(kickoff, fixture_id)``.

        On failure: ``ValueError`` citing the offending indices. We
        NEVER reorder silently — Phase 4 owns the order.
        """
        rows = self._dataset.rows
        for i in range(len(rows) - 1):
            a_k = rows[i].kickoff
            a_f = rows[i].fixture_id
            b_k = rows[i + 1].kickoff
            b_f = rows[i + 1].fixture_id
            if (a_k, a_f) > (b_k, b_f):
                raise ValueError(
                    "dataset.rows is not ordered by "
                    "(kickoff, fixture_id): "
                    f"row[{i}]=(kickoff={a_k.isoformat()}, "
                    f"fixture_id={a_f}) > "
                    f"row[{i + 1}]=(kickoff={b_k.isoformat()}, "
                    f"fixture_id={b_f}). "
                    "Reordering silently is forbidden; rebuild the "
                    "dataset via Phase 4 builder or sort it explicitly "
                    "before invoking the iterator."
                )

    def _train_len(self, start_idx: int, end_idx: int) -> int:
        """Inclusive row count of the train block ``[start_idx, end_idx]``.

        Returns ``0`` (or a negative-effective value) when ``end_idx <
        start_idx`` (collapsed block). The caller treats any value
        ``< min_train_size`` as "fold cannot honour the minimum train
        size; stop cleanly without emitting this fold".
        """
        return end_idx - start_idx + 1

    def _shrink_train_tail(self, train_end_idx: int, n: int) -> int:
        """Equal-kickoff rule applied to the train tail.

        Walk ``train_end_idx`` backward while the *next* row
        (``train_end_idx + 1``) shares the same kickoff as
        ``train_end_idx``. The equal-kickoff tail therefore joins the
        right-hand block (val) once the next iteration starts scanning
        from ``train_end_idx + 1``.

        Edge case: if this shrink collapses train below the
        :attr:`_min_train_size` floor (or even to nothing, when many
        rows share one kickoff), the caller is responsible for
        detecting the shortfall via :meth:`_train_len` and stopping
        cleanly without emitting an under-sized fold.
        """
        rows = self._dataset.rows
        while (
            train_end_idx >= 0
            and train_end_idx + 1 < n
            and rows[train_end_idx + 1].kickoff
            == rows[train_end_idx].kickoff
        ):
            train_end_idx -= 1
        return train_end_idx

    def _scan_past_gap(
        self, prev_end_idx: int, n: int
    ) -> int | None:
        """Return the first index ``i`` such that
        ``rows[i].kickoff >= rows[prev_end_idx].kickoff + gap``, or
        ``None`` if no row satisfies it.

        ``prev_end_idx`` is the inclusive end of the previous block.
        """
        rows = self._dataset.rows
        prev_kickoff = rows[prev_end_idx].kickoff
        idx = prev_end_idx + 1
        while idx < n:
            if rows[idx].kickoff - prev_kickoff >= self._gap:
                return idx
            idx += 1
        return None

    def _val_target(self, train_len: int) -> int:
        if self._val_size is not None:
            return self._val_size
        if self._val_ratio is not None:
            return max(1, int(round(train_len * self._val_ratio)))
        # Unreachable: __init__ requires one of the two.
        raise WalkForwardError(
            "internal: neither val_size nor val_ratio set after "
            "post-init validation"
        )

    def _build_fold(
        self,
        fold_id: int,
        train_indices: IndexTuple,
        val_indices: IndexTuple,
        test_indices: IndexTuple,
    ) -> Fold:
        rows = self._dataset.rows
        train_start_dt = rows[train_indices[0]].kickoff
        train_end_dt = rows[train_indices[-1]].kickoff
        val_start_dt = rows[val_indices[0]].kickoff
        val_end_dt = rows[val_indices[-1]].kickoff
        test_start_dt = rows[test_indices[0]].kickoff
        test_end_dt = rows[test_indices[-1]].kickoff

        # Strict chronology defences. Hard errors because the size
        # params were honoured; arriving here means equal-kickoff
        # clumping truly couldn't separate the blocks. Should not fire
        # in practice because the helpers above raise earlier, but
        # kept for defence-in-depth.
        if not (train_end_dt < val_start_dt):
            raise WalkForwardError(
                f"fold {fold_id}: train_end ({train_end_dt.isoformat()}) "
                f"not strictly before val_start "
                f"({val_start_dt.isoformat()})."
            )
        if not (val_end_dt < test_start_dt):
            raise WalkForwardError(
                f"fold {fold_id}: val_end ({val_end_dt.isoformat()}) "
                f"not strictly before test_start "
                f"({test_start_dt.isoformat()})."
            )
        if not (train_end_dt + self._gap <= val_start_dt):
            raise WalkForwardError(
                f"fold {fold_id}: temporal gap not respected between "
                f"train and val: train_end ({train_end_dt.isoformat()}) "
                f"+ gap_days={self._gap_days} > val_start "
                f"({val_start_dt.isoformat()})."
            )
        if not (val_end_dt + self._gap <= test_start_dt):
            raise WalkForwardError(
                f"fold {fold_id}: temporal gap not respected between "
                f"val and test: val_end ({val_end_dt.isoformat()}) "
                f"+ gap_days={self._gap_days} > test_start "
                f"({test_start_dt.isoformat()})."
            )

        # gap_end: by module convention, gap_end == val_start_dt
        # (the lower inclusive frontier of val after the train+gap
        # boundary closes). See the Fold module docstring for full
        # rationale.
        gap_end = val_start_dt

        return Fold(
            fold_id=fold_id,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_start=train_start_dt,
            train_end=train_end_dt,
            val_start=val_start_dt,
            val_end=val_end_dt,
            test_start=test_start_dt,
            test_end=test_end_dt,
            gap_end=gap_end,
            training_cutoff=train_end_dt,
            validation_cutoff=val_end_dt,
            test_cutoff=test_end_dt,
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of folds yielded. Materialises a twin iterator."""
        twin = WalkForwardIterator(
            self._dataset,
            min_train_size=self._min_train_size,
            test_size=self._test_size,
            gap_days=self._gap_days,
            mode=self._mode,
            val_ratio=self._val_ratio,
            val_size=self._val_size,
        )
        return sum(1 for _ in twin)

    def to_config(self) -> dict[str, Any]:
        """Return the iterator configuration as a JSON-serialisable
        dict. Useful for run-id hashing (see ``docs/PHASE_5.md §6.4``).
        """
        return {
            "min_train_size": self._min_train_size,
            "test_size": self._test_size,
            "gap_days": self._gap_days,
            "mode": str(self._mode.value),
            "val_ratio": self._val_ratio,
            "val_size": self._val_size,
        }


__all__ = ["WalkForwardError", "WalkForwardIterator"]
