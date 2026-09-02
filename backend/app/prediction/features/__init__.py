"""Feature vectorisation + explicit missing-data policy.

Sprint 5.0 reserve: package skeleton only. ``vector.py`` and
``missing.py`` are scheduled for Sprint 5.2
(see ``docs/PHASE_5.md`` §16).

Policy (already locked by PHASE_5.md §5 / §13):
* ``None`` from the CSV → ``np.nan`` (preserved; never auto-imputed to ``0``).
* ``0`` is the math-zero (kept as ``0``).
"""

from __future__ import annotations

__all__: list[str] = []
