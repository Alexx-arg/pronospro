"""Custom runtime exceptions for the application.

Phase 2 only defines the persistence-related error hierarchy. Service and
provider errors will be added in later phases. The error strings here are
publicly documented and must remain stable.
"""

from __future__ import annotations


class AppError(Exception):
    """Root class for all application-raised exceptions."""


class PersistenceError(AppError):
    """Generic persistence-layer failure."""


class PredictionImmutableError(PersistenceError):
    """Raised when an operation attempts to mutate an immutable prediction.

    The Python layer never tries to UPDATE or DELETE a prediction: this
    exception exists as a defense-in-depth signal so that any accidental
    mutation through SQLAlchemy (e.g. via session dirty tracking) is
    surfaced explicitly before reaching the database trigger.
    """


class EntityNotFoundError(PersistenceError):
    """Raised when an entity lookup returns no row."""


class ConstraintViolationError(PersistenceError):
    """Raised when a database CHECK / UNIQUE / FK constraint is violated."""


class RollbackError(PersistenceError):
    """Raised when an explicit transaction rollback fails."""
