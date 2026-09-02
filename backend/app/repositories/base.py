"""Generic async repository base class."""

from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base


TModel = TypeVar("TModel", bound=Base)


class BaseRepository(Generic[TModel]):
    """Repository base providing read primitives.

    Specialized repositories subclass this and add their own custom queries
    (e.g. ``upsert`` for operational entities, ``insert``-only for
    immutable ones). The base class deliberately exposes only ``get`` and
    ``list``: write semantics differ enough between entities to warrant
    explicit method declarations on each subclass.
    """

    model: type[TModel]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, entity_id: int) -> TModel | None:
        """Fetch a row by primary key.

        Returns ``None`` when not found. The caller decides whether to raise
        :class:`EntityNotFoundError`.
        """
        return await self.session.get(self.model, entity_id)

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TModel]:
        """Return a slice of rows ordered by primary key (stable pagination)."""
        stmt = (
            select(self.model)
            .order_by(self.model.id)  # type: ignore[attr-defined]
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def refresh(self, instance: TModel) -> TModel:
        """Expire and reload ``instance`` from the database."""
        await self.session.refresh(instance)
        return instance
