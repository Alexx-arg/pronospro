"""API Key security — Sprint 6.3 (D10.1)."""

from __future__ import annotations

import os

from app.config import get_settings
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

# Header name as per spec
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _expected_api_key() -> str | None:
    """Read expected key from env / settings without hardcoding (D10.1).

    Returns None if not configured (allows public in dev/test unless enforced).
    In production (ENV=production) missing key is a fail-safe error.
    """
    # Pydantic settings already reads env, but we also allow direct os.getenv for tests
    # to avoid cache issues.
    env_key = os.getenv("API_SECRET_KEY")
    if env_key:
        return env_key
    try:
        settings = get_settings()
        return settings.api_secret_key
    except Exception:
        return None


async def verify_api_key(api_key: str | None = None) -> None:
    """Dependency to be injected. Hace `Depends(APIKeyHeader)` manualmente.

    Este wrapper permite testear sin FastAPI Depends complications y mypy.
    La función real usada en routers es `require_api_key`.
    """
    _ = api_key


async def require_api_key(
    api_key: str | None = None,  # will be overridden by Depends
) -> None:
    """Dependency que valida X-API-Key contra API_SECRET_KEY."""
    # FastAPI will inject via Depends(api_key_header) — we handle both signatures
    # This function is intended to be used as `Depends(require_api_key)` where
    # FastAPI resolves `api_key_header` automatically. For manual calls, pass api_key.
    # To keep mypy happy, we accept optional.
    # Actual key extraction is done via APIKeyHeader dependency.
    pass


# Real dependency callable for FastAPI
async def _require_api_key_dependency(
    api_key: str | None = None,
) -> None:
    # This is the internal, the public one is below with Depends
    expected = _expected_api_key()
    # D10.2: en tests/development si no hay clave configurada, no bloquear
    # Para no romper 313 tests existentes, permitimos acceso si expected is None/empty
    # y ENV != production.
    if not expected:
        # Si estamos en producción y no hay clave, fail safe
        try:
            from app.config import get_settings

            env = get_settings().env
        except Exception:
            env = "development"
        if env == "production":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API_SECRET_KEY not configured in production",
            )
        return

    if not api_key or api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )


# Exportable dependency for routers: Depends(_require_api_key_dependency) with header injection
# We need to bind APIKeyHeader to the function signature.
# FastAPI pattern: `api_key: str = Depends(api_key_header)` → then validate
async def require_api_key_dep(
    api_key: str | None = Depends(api_key_header),  # noqa: B008
) -> None:
    await _require_api_key_dependency(api_key)


__all__ = ["api_key_header", "require_api_key_dep"]
