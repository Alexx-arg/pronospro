"""``ModelVersion`` repository."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, update

from app.models import ModelVersion
from app.repositories.base import BaseRepository


class ModelVersionRepository(BaseRepository[ModelVersion]):
    """Repository for :class:`ModelVersion`.

    ``model_versions`` is mutable so creating new versions and toggling
    ``is_active`` is permitted. The partial unique index
    ``uq_model_active_per_name`` guarantees only one active version per
    model name.
    """

    model = ModelVersion

    async def create(
        self,
        *,
        name: str,
        version: str,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        is_active: bool = False,
    ) -> ModelVersion:
        """Insert a new model version."""
        mv = ModelVersion(
            name=name,
            version=version,
            description=description,
            parameters=parameters or {},
            is_active=is_active,
        )
        self.session.add(mv)
        await self.session.flush()
        return mv

    async def activate(self, *, model_version_id: int) -> ModelVersion:
        """Activate a model version (deactivating siblings of the same name).

        The DB-level partial unique index is the real guard; this method
        issues an UPDATE first to flip siblings to inactive and avoids a
        transient two-rows-active state.
        """
        target = await self.get(model_version_id)
        if target is None:
            raise LookupError(f"ModelVersion {model_version_id} not found")
        await self.session.execute(
            update(ModelVersion)
            .where(ModelVersion.name == target.name)
            .where(ModelVersion.id != model_version_id)
            .where(ModelVersion.is_active.is_(True))
            .values(is_active=False)
        )
        target.is_active = True
        await self.session.flush()
        return target

    async def get_active(self, *, name: str) -> ModelVersion | None:
        """Return the currently-active version of ``name`` (or None)."""
        stmt = select(ModelVersion).where(
            ModelVersion.name == name,
            ModelVersion.is_active.is_(True),
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()
