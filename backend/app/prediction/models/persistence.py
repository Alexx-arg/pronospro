"""Persistencia genérica con joblib — Sprint 5.13.

Funciones save_model / load_model para ModelArtifact o cualquier
objeto Predictor (sklearn/LightGBM) que use numpy internamente.
Garantiza manejo correcto de Paths y directorios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]


def save_model(obj: Any, path: Path | str) -> Path:
    """Serializa ``obj`` con joblib en ``path``.

    Crea directorios padres si no existen. Sobrescribe si ya existe
    (el artefacto final es idempotente por seed).

    Returns:
        Path absoluto del archivo guardado.
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)
    return p


def load_model(path: Path | str) -> Any:
    """Carga objeto serializado con joblib.

    Args:
        path: Ruta al archivo .joblib

    Returns:
        Objeto deserializado (ModelArtifact o Predictor).

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el archivo no es un joblib válido.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"model file not found: {p}")
    try:
        obj: Any = joblib.load(p)
    except Exception as exc:
        raise ValueError(f"failed to load joblib artifact {p}: {exc}") from exc
    return obj


__all__ = ["load_model", "save_model"]
