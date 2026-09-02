"""Unit tests (no DB connection).

Phase 2 introduces only persistence. Pure unit tests cover the immutability
contract at the Python level: the ``PredictionRepository`` /
``PredictionOutcomeRepository`` methods ``update``/``delete``/``merge`` raise
:class:`PredictionImmutableError` even before any DB interaction occurs.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import PredictionImmutableError
from app.repositories.prediction import PredictionRepository
from app.repositories.prediction_outcome import PredictionOutcomeRepository


def test_prediction_repository_update_is_forbidden() -> None:
    repo = PredictionRepository.__new__(PredictionRepository)
    repo.session = None  # not used; the guard must short-circuit before DB
    with pytest.raises(PredictionImmutableError):
        repo.update()


def test_prediction_repository_delete_is_forbidden() -> None:
    repo = PredictionRepository.__new__(PredictionRepository)
    repo.session = None
    with pytest.raises(PredictionImmutableError):
        repo.delete()


def test_prediction_repository_merge_is_forbidden() -> None:
    repo = PredictionRepository.__new__(PredictionRepository)
    repo.session = None
    with pytest.raises(PredictionImmutableError):
        repo.merge()


def test_outcome_repository_update_is_forbidden() -> None:
    repo = PredictionOutcomeRepository.__new__(PredictionOutcomeRepository)
    repo.session = None
    with pytest.raises(PredictionImmutableError):
        repo.update()


def test_outcome_repository_delete_is_forbidden() -> None:
    repo = PredictionOutcomeRepository.__new__(PredictionOutcomeRepository)
    repo.session = None
    with pytest.raises(PredictionImmutableError):
        repo.delete()
