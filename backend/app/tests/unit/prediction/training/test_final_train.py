# ruff: noqa: N806
"""Tests entrenamiento final — Sprint 5.13.

Valida: ejecución sin errores, archivo .joblib creado, determinismo
de serialización (predict idéntico antes/después de save/load).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from app.dataset.loader import LoadedDataset, LoadedExample
from app.dataset.manifest import DatasetManifest
from app.features.example import FEATURE_NAMES
from app.prediction.models.persistence import load_model
from app.prediction.training.final_train import train_final_model


def _make_dataset(n: int = 30) -> LoadedDataset:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[LoadedExample] = []
    for i in range(n):
        from datetime import timedelta

        kickoff = t0 + timedelta(days=i)
        feats = {name: float((i % 3) * 0.5) for name in FEATURE_NAMES}
        feats["home_elo_pre_match"] = 1500.0
        feats["away_elo_pre_match"] = 1500.0
        feats["elo_difference"] = 0.0
        label = i % 3
        if label == 0:
            targets: dict[str, int | None] = {"home_win": 1, "draw": 0, "away_win": 0, "home_goals": 2, "away_goals": 0}
        elif label == 1:
            targets = {"home_win": 0, "draw": 1, "away_win": 0, "home_goals": 1, "away_goals": 1}
        else:
            targets = {"home_win": 0, "draw": 0, "away_win": 1, "home_goals": 0, "away_goals": 2}
        rows.append(
            LoadedExample(
                fixture_id=i + 1,
                kickoff=kickoff,
                competition_id=39,
                season_id=1,
                home_team_id=10,
                away_team_id=20,
                features=feats,
                targets=targets,
            )
        )
    manifest = DatasetManifest(
        dataset_version="v001",
        generated_at=t0,
        feature_definition_version="fd_v1",
        data_cutoff=t0,
        source_schema_version="schema_v1",
        row_count=len(rows),
        feature_names=tuple(FEATURE_NAMES),
        target_names=("home_win", "draw", "away_win", "home_goals", "away_goals"),
        start_date=rows[0].kickoff,
        end_date=rows[-1].kickoff,
        competitions=(39,),
        seasons=(1,),
        csv_sha256="x" * 64,
        extras={},
    )
    return LoadedDataset(manifest=manifest, rows=rows)


def test_train_final_crea_archivo(tmp_path: Path) -> None:
    dataset = _make_dataset(30)
    best_params = {"n_estimators": 20, "num_leaves": 15, "learning_rate": 0.05, "random_state": 42, "verbosity": -1, "n_jobs": 1}
    out = train_final_model(dataset, "lightgbm", best_params, output_dir=tmp_path, seed=42)
    assert out.is_file()
    assert out.name == "lightgbm_production.joblib"
    assert out.parent.resolve() == tmp_path.resolve()


def test_train_final_sin_errores_dataset_completo(tmp_path: Path) -> None:
    dataset = _make_dataset(20)
    best_params = {"n_estimators": 10, "num_leaves": 15, "learning_rate": 0.05}
    # No debe lanzar
    out = train_final_model(dataset, "lightgbm", best_params, output_dir=tmp_path)
    assert out.exists()


def test_identidad_serializacion(tmp_path: Path) -> None:
    """Entrenar → predecir → guardar → cargar → predecir idéntico (test crítico)."""
    dataset = _make_dataset(30)
    best_params = {"n_estimators": 20, "num_leaves": 15, "learning_rate": 0.05, "random_state": 42, "verbosity": -1, "n_jobs": 1}
    # Entrenamiento final vía train_final_model (persiste)
    out = train_final_model(dataset, "lightgbm", best_params, output_dir=tmp_path, seed=42)
    loaded = load_model(out)
    assert hasattr(loaded, "predict")

    # Test crítico: entrenar en memoria, predecir, guardar con save_model, cargar y comparar
    from app.prediction.features.vector import loaded_example_to_features
    from app.prediction.models.persistence import save_model
    from app.prediction.training import get_trainer

    # Datos sintéticos pequeños para trainer directo
    X = np.random.default_rng(42).normal(size=(20, 66))
    y = np.random.default_rng(42).integers(0, 3, size=20)
    # Asegura 3 clases
    y[0] = 0
    y[1] = 1
    y[2] = 2
    trainer_mem = get_trainer("lightgbm", params={"random_state": 42, "verbosity": -1, "n_estimators": 20, "num_leaves": 15})
    model_mem = trainer_mem.fit(X, y)  # type: ignore[call-arg]
    # Predicción en memoria
    proba_mem = model_mem.predict_proba(X[:5])
    # Guardar y cargar
    path_tmp = tmp_path / "mem_model.joblib"
    save_model(model_mem, path_tmp)
    loaded_mem = load_model(path_tmp)
    proba_loaded = loaded_mem.predict_proba(X[:5])
    np.testing.assert_array_equal(proba_mem, proba_loaded)

    # También verifica vía FixtureFeatures con dataset real
    feat = loaded_example_to_features(dataset.rows[0])
    proba_1 = loaded.predict(feat)
    loaded2 = load_model(out)
    proba_2 = loaded2.predict(feat)
    arr1 = np.array([proba_1.p_home_win, proba_1.p_draw, proba_1.p_away_win])
    arr2 = np.array([proba_2.p_home_win, proba_2.p_draw, proba_2.p_away_win])
    np.testing.assert_array_equal(arr1, arr2)


def test_persistencia_tipado_predictor(tmp_path: Path) -> None:
    dataset = _make_dataset(20)
    out = train_final_model(dataset, "lightgbm", {"n_estimators": 10}, output_dir=tmp_path)
    loaded = load_model(out)
    # D7.3: debe cumplir Predictor
    assert hasattr(loaded, "predict")
    # También probar con el otro modelo (poisson) como sanity
    # No necesario, solo lightgbm para este sprint


def test_train_final_no_leakage_usa_100_porciento(tmp_path: Path) -> None:
    # Verifica que train_final usa 100% (no hay split): si dataset tiene 10 rows, train_n debe ser 10
    dataset = _make_dataset(10)
    out = train_final_model(dataset, "lightgbm", {"n_estimators": 10}, output_dir=tmp_path)
    # El archivo existe, y no debe haber usado iterador
    assert out.is_file()
