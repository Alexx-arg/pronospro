# ruff: noqa: N803, N806
"""Entrenamiento final con 100% del dataset — Sprint 5.13.

Único lugar legítimo para usar el dataset completo sin splits (D7.1).
No usa WalkForwardIterator. Persiste artefacto vía persistence (joblib).

Tipado estricto: artefactos cumplen Predictor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.dataset.loader import LoadedDataset
from app.prediction.models.persistence import save_model
from app.prediction.training import get_trainer


def train_final_model(
    dataset: LoadedDataset,
    model_name: str,
    best_params: dict[str, Any],
    output_dir: Path | str,
    *,
    seed: int = 42,
    model_version: str = "production",
) -> Path:
    """Entrena modelo final con 100% de los datos y persiste artefacto.

    No usa iterador temporal — todo el dataset es train (D7.1).
    Aislado de hpo.py y compare.py (D7.2).

    Args:
        dataset: LoadedDataset completo.
        model_name: Nombre del modelo ("lightgbm", "poisson", etc.)
        best_params: Hiperparámetros ganadores del HPO.
        output_dir: Directorio donde guardar ``{model_name}_production.joblib``.
        seed: Seed fija para determinismo.
        model_version: Versión para ModelArtifact.

    Returns:
        Path absoluto del archivo guardado.

    Raises:
        ValueError: si dataset vacío o model_name desconocido.
    """
    if not dataset.rows:
        raise ValueError("dataset is empty")

    from app.prediction.features.vector import loaded_example_to_features

    # Convertir todo el dataset (100% sin split)
    features = [loaded_example_to_features(ex) for ex in dataset.rows]
    targets = [ex.targets for ex in dataset.rows]

    # Instanciar trainer vía factory con best_params + seed fijo
    trainer = get_trainer(model_name, params=best_params)

    # Entrenar con 100% (sin val/test) — training_cutoff = max kickoff
    training_cutoff = max(ex.kickoff for ex in dataset.rows)
    artifact = trainer.train(
        features,
        targets,
        hyperparameters=best_params,
        seed=seed,
        model_version=model_version,
        training_data_version=dataset.manifest.dataset_version,
        feature_definition_version=dataset.manifest.feature_definition_version,
        training_cutoff=training_cutoff,
    )

    # Recuperar modelo fitted para persistir el predictor real
    # Los trainers guardan _last_model; si no, persistimos el artifact
    model: Any = None
    if hasattr(trainer, "get_model"):
        try:
            model = trainer.get_model(artifact)
        except Exception:
            try:
                model = trainer.get_model()
            except Exception:
                model = None
    if model is None:
        model = getattr(trainer, "_last_model", artifact)

    # Validación tipada D7.3: debe cumplir Predictor (predict)
    if not hasattr(model, "predict"):
        # Fallback al artifact si el modelo no es Predictor (ej. sklearn puro)
        model = artifact

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{model_name}_production.joblib"
    out_path = output_dir / filename
    return save_model(model, out_path)


__all__ = ["train_final_model"]
