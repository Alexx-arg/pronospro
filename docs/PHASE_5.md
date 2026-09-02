# PHASE_5.md — Prediction Engine (Diseño aprobado)

> Estado: **DISEÑO APROBADO**. Implementación en curso por sprints.
> Sprint actual: **5.0 — Fundamentos** (sin entrenar nada todavía).
>
> Documento consolidado. Toda decisión aquí tomada es canónica para la
> implementación de Fase 5; cambios requieren reaprobación explícita y
> bump de este documento.

---

## 0. Principios rectores

1. **Fuente única**: los modelos consumen **únicamente** `LoadedDataset`
   de Fase 4. Nada de leer PostgreSQL ni recalcular features.
2. **Pasado → futuro**: toda evaluación es walk-forward. Nunca random
   split.
3. **Inmutabilidad de predicciones**: una predicción histórica guardada
   no se sobrescribe aunque se entrene un modelo nuevo.
4. **Reproducibilidad bit-exacta**: la tupla
   `(dataset_version, feature_definition_version, model_name,
   model_version, hyperparameters, seed)` reproduce artifact y
   métricas.
5. **Comparación justa**: todos los modelos son evaluados sobre el
   mismo walk-forward iterador y los mismos fixtures de test.
6. **Distinción Elo-feature vs Elo-baseline** (ver §11): el
   `elo_difference` del CSV es input del baseline Elo; el baseline Elo
   produce la probabilidad. Dos roles, mismo math, distinto punto del
   pipeline.

---

## 1. Arquitectura del prediction engine

```
                  LoadedDataset  (Fase 4)
                        │
                        ▼
        ┌────────────────────────────────────┐
        │   BacktestIterator (walk-forward)   │
        └────────────────────────────────────┘
              │                          │
              ▼                          ▼
        train_block                  test_block
              │                          │
              ▼                          │
        ┌──────────┐                     │
        │ Trainers │                     │
        ├──────────┤                     │
        │ EloTrainer (baseline)          │
        │ PoissonTrainer                 │
        │ GradientBoostingTrainer        │
        └──────────┘                     │
              │                          │
              ▼                          ▼
        Model artifact ──fit Calibrator ──► CalibratedModel
                                            │
                                            ▼
                                     Predictor.predict(fixture_rows)
                                            │
                                            ▼
                                MatchPrediction (P(home/draw/away)
                                                 + P(goals))
                                            │
                            ┌───────────────┴───────────────┐
                            ▼                               ▼
                PredictionStore (persistente)         MetricsEvaluator
                            │                               │
                            ▼                               ▼
                predictions / model_runs          metrics.json por run
```

| Capa | Rol | Vive en |
|---|---|---|
| dataset loader | Ya existe (Fase 4) | `app/dataset/loader.py` |
| walk-forward iterator | Corta el dataset en ventanas train+test respetando `kickoff` | `app/prediction/backtesting/iterator.py` |
| trainers | Producen `ModelArtifact` inmutable | `app/prediction/training/*.py` |
| calibrators | Ajustan probs sobre holdout temporal | `app/prediction/calibration/*.py` |
| predictors | Aplican el artifact a un bloque de test | `app/prediction/models/*.py` |
| metrics | Computan métricas sobre las predicciones vs targets | `app/prediction/metrics/*.py` |
| stores | Persisten artifacts, predicciones y runs | `app/prediction/storage/*.py` |
| runner | Orquesta backtesting end-to-end | `app/prediction/backtesting/runner.py` |

Nada de esto se conecta a FastAPI ni a Flutter en Fase 5: es CLI /
script / jobs offline.

---

## 2. Estructura de carpetas

```
backend/app/prediction/
    __init__.py
    contracts.py              # interfaces (Model, Predictor, Calibrator, ...)
    artifacts.py              # ModelArtifact, ModelManifest, PredictionRecord
    seeds.py                 # utilidades de seeding determinista

    features/
        vector.py             # LoadedExample → numpy vector (ghost de missing)
        missing.py            # política de imputación EXPLÍCITA
                              # (None→np.nan, sin imputar 0)

    models/
        elo_baseline.py      # Elo → P(home/draw/away) vía λ + mat Poisson
        poisson.py           # λ_home, λ_away → matriz de goles → 1X2
        gradient_boosting.py # clasificador 1X2 (multiclass) sobre 66 features

    training/
        base.py              # Trainer interface
        elo_trainer.py
        poisson_trainer.py
        gb_trainer.py
        splitter.py          # TrainValSplit temporal (NO random)

    calibration/
        base.py              # Calibrator interface
        identity.py          # no-op (cuando no conviene calibrar)
        temperature.py       # temperature scaling (1-param)
        dirichlet.py         # Dirichlet calibration (K²+K-1 params)
        detector.py          # decide calibrador según modelo + n + ECE

    backtesting/
        iterator.py          # WalkForwardIterator (folding temporal)
        runner.py            # orquesta un model_run completo
        fold.py              # Fold (train_idx, val_idx, test_idx)

    metrics/
        classification.py    # accuracy, log_loss, brier, confusion, ECE/MCE
        calibration.py       # reliability diagrams, ECE
        goals.py             # MAE goles, RMSE, Poisson likelihood
        report.py            # CompilationReport (dict por model_run)

    storage/
        artifacts.py         # ModelArtifactStore (bin + manifest)
        runs.py              # RunsStore (cada backtesting produce un run_row)
        predictions.py       # PredictionStore (cada predicción por fixture)
        layout.py            # rutas canónicas bajo data/models/<m>/<v>/

    config.py                # PredictionSettings (pydantic-settings)

backend/app/tests/
    unit/prediction/...
    integration/prediction/...

data/
    datasets/v001/                # Fase 4 (ya existe)
    models/<model_name>/<model_version>/
        artifact.bin
        manifest.json
        calibrator.json
    models/runs/<run_id>/
        config.json
        folds/fold_001/
            train_meta.json
            test_meta.json
            predictions.csv
            metrics.json
        summary.json
    predictions/<model_name>/<model_version>/<fixture_id>.json
```

Política de almacenamiento: **sin Parquet**, coherente con Fase 4
(sin `pyarrow`). Predicciones por fixture se guardan como archivos
JSON individuales inmutables. Tablas por fold como CSV.

---

## 3. Contratos (Sprint 5.0)

Definidos en `backend/app/prediction/contracts.py`:

- `ModelName` (StrEnum): `ELO_BASELINE`, `POISSON`, `GRADIENT_BOOSTING`.
- `MatchProbabilities(NamedTuple)`: `p_home_win`, `p_draw`,
  `p_away_win` (floats, `sum == 1`, `0 ≤ k ≤ 1`),
  `p_home_goals: dict[int, float] | None`,
  `p_away_goals: dict[int, float] | None`.
- `FixtureFeatures(NamedTuple)`: `fixture_id`, `kickoff`,
  `feature_vector: np.ndarray`, `feature_names: tuple[str, ...]`.
- `Predictor(Protocol)`: `predict(self, x: FixtureFeatures) -> MatchProbabilities`.
- `Trainer(Protocol)`: `name: ModelName`, `train(...)`.
- `Calibrator(Protocol)`: `fit`, `transform`, `kind`.
- `Fold` (immuable dataclass): rangos de índices con cronología
  garantizada por el iterator.

`ModelArtifact` (frozen) en `artifacts.py`:

```json
{
  "model_name": "gradient_boosting",
  "model_version": "v001",
  "training_data_version": "v001",
  "feature_definition_version": "fd_v1",
  "inputs": {
    "feature_names": ["home_elo_pre_match", ...],
    "feature_definition_version": "fd_v1",
    "head_features": null
  },
  "hyperparameters": { ... },
  "training_cutoff": "2024-12-31T00:00:00+00:00",
  "created_at": "2026-08-15T12:00:00+00:00",
  "metrics": { "train_log_loss": ..., "val_log_loss": ..., ... },
  "fitted_seed": 42,
  "payload_sha256": "<64hex>",
  "payload_ref": "artifact.bin"
}
```

---

## 4. Flujo de entrenamiento

Por cada fold `(train, val, test)` y modelo `M`:

1. **Cargar** `LoadedDataset` por `training_data_version`.
2. **Anti-leakage** garantizado por el iterador: índices ordenados por
   `(kickoff, fixture_id)` (Fase 4 garantiza el orden en el CSV).
3. **Convertir a matriz**: `LoadedExample → np.ndarray[float32]`,
   `None → np.nan` (explícito; preserva política missing/zero de
   Fase 4). `targets` → `int8` 1X2 y `int8` goles.
4. **Entrenar** `Trainer.train(train_block, hp, seed) → ModelArtifact`
   (aún no calibrado).
5. **Calibrar** sobre `val_block` (NO sobre `train`, NO sobre `test`):
   `calibrator.fit(raw_probs_val, targets_val)`.
6. **Evaluar** sobre `test_block` con el `CalibratedPredictor`.
7. **Persistir** artifacts + runs + predictions.
8. **Verificación runtime** (tests dedicados): en cada fold, ningún
   `fixture_id` del `test_block` puede estar en `train` ni en `val`.

---

## 5. Flujo de inferencia

1. Cargar artifact `ModelArtifactStore.load(name, version)`.
2. Construir `FixtureFeatures`:
   - En backtest: desde `LoadedDataset.rows` del fold test.
   - En runtime Fase 5+ (futuro): desde `build_example` con
     `data_cutoff = fixture.kickoff_time` (misma capa as-of Fase 4).
3. `predict(features) → MatchProbabilities`.
4. Persistir `PredictionRecord` en
   `data/predictions/<name>/<version>/<fixture_id>.json`.
   - Si ya existe: **no sobrescribir**; log WARNING + devolver la
     existente. Garantiza inmutabilidad histórica.

Diferencia: en runtime el fixture no está terminado (no targets); en
backtest sí, pero el modelo no los ve (siguiente fold del walk-forward).

---

## 6. Walk-forward

### 6.1 Parámetros del iterador

| Param | Default | Significado |
|---|---|---|
| `min_train_size` | 200 | tamaño **mínimo real** del bloque `train` (hard floor, ver §6.5) |
| `val_ratio` | 0.15 | bloque de validación temporal posterior a `train` (alternativa a `val_size`) |
| `val_size` | — | tamaño fijo de `val` (override del caller, mutuamente excluyente con `val_ratio`) |
| `test_size` | 50 | bloque a predecir |
| `gap_days` | 1 | holgura **temporal** (días de calendario, no nº de filas) entre `train→val` y `val→test` |
| `mode` | `expanding` | default; alternativa `sliding` |

`val_ratio` y `val_size` son mutuamente excluyentes en `WalkForwardIterator`.
`PredictionSettings` expone solo `val_ratio`; `val_size` es un override
explícito del caller (tests / experimentos).

### 6.2 Skew

```
kickoff timeline ----------------------------------------------------------------->
             [-------- train --------][-- val --]gap[-- test --]
fold 0       ^min_train_size                              ^
fold 1              [----------- train -----------][val][g][t]
fold 2                       [----------------- train ------------][v][g][t]
...
```

- `train.kickoff < val.kickoff < test.kickoff` — siempre (timestamps reales).
- `gap_days` es temporal: `min(val.kickoff) >= max(train.kickoff) + gap_days`
  y `min(test.kickoff) >= max(val.kickoff) + gap_days`.
- Si los datos son insuficientes para formar un triple `(train, val, test)`
  completo, el iterador termina limpiamente sin producir ese fold
  (cero folds es un resultado válido — Sprint 5.1, decisión D).

### 6.3 Anti-leakage garantizado por construction

`WalkForwardIterator` consume el dataset ya ordenado (Fase 4 ordena
por `(kickoff, fixture_id)`). Los folds son rangos de índices no
solapados.

Tests (`test_walk_forward_iterator.py`):

- `test_train_idx_max_kickoff < val_idx_min_kickoff`
- `test_val_idx_max_kickoff < test_idx_min_kickoff`
- `test_no_fixture_in_test_appears_in_train`
- `test_no_fixture_in_val_appears_in_train`
- `test_gap_respected`
- `test_first_fold_skipped_when_insufficient_train`

### 6.5 Reglas de frontera temporal (Sprint 5.1)

#### 6.5.1 Integridad por kickoff idéntico

Puede haber múltiples partidos con exactamente el mismo `kickoff`.
`fixture_id` mayor **no** implica posterioridad temporal.

**Regla conservadora — no dividir un timestamp:**

Si una frontera temporal (`train→val` o `val→test`) cae dentro de un
grupo de filas con idéntico `kickoff == T`, la frontera se desplaza
de forma **determinista hacia atrás** (el train se achica) para que el
grupo completo de `T` quede en el bloque de la **derecha** (val o test).
Con ello `max(train.kickoff) < min(val.kickoff)` (estricto) se preserva
por construcción.

Consecuencia: los tamaños reales de `val`/`test` pueden desviarse del
target nominal en unas pocas filas (se expanden para tragar la cola de
kickoff idéntico). Esta variación es documentada e inevitable.

#### 6.5.2 `min_train_size` es un mínimo real (hard floor)

`min_train_size` **no** es un target aproximado. El iterador nunca
emite un fold cuyo `train` efectivo sea menor que `min_train_size`.

Si el ajuste por kickoff idéntico (§6.5.1) deja `len(train) <
min_train_size`, **ese fold no se genera** y la iteración termina
limpiamente (cero folds o tantos como ya se emitieron — nunca un fold
parcial que viole el parámetro). Ejemplo:

```
min_train_size = 4
rows[0].kickoff == rows[1].kickoff == rows[2].kickoff == T0
rows[3].kickoff == rows[4].kickoff == T1   # misma T1
train nominal = rows[0..3]  (4 filas) pero rows[3] y rows[4] comparten T1
→ train se achica a rows[0..2] (3 filas) para no dividir T1
→ 3 < 4 ⇒ fold inválido ⇒ no se emite (list(iterator) == [])
```

Este caso se valida en `test_train_below_min_due_to_equal_kickoff_produces_no_fold`.

#### 6.5.3 Gap temporal real

`walk_forward_gap_days` se interpreta como `timedelta(days=gap_days)` sobre
timestamps `kickoff` reales. `gap_end` se define como `val_start` (la
frontera inferior inclusiva de val) y el ancho temporal del gap se verifica
comparando `train_end + gap <= val_start` y `val_end + gap <= test_start`.

### 6.6 Feature Vector & Missing Policy (Sprint 5.2)

**Origen (Fase 4):**

* `None` en `HistoricalMatchExample.features` / `LoadedExample.features`
  significa *missing* — no hay historia suficiente, snapshot ausente, etc.
* `0` significa *cero matemático* — p.ej. `home_wins_last_3 == 0` porque
  el equipo perdió los 3.

Los dos valores **no** son intercambiables y el builder/loader preserva
esa distinción (`""` en CSV → `None`, `0` en CSV → `0`).

**Transformación Sprint 5.2 (`features/vector.py`):**

```
LoadedExample  →  FixtureFeatures
  fixture_id      fixture_id          (copiado)
  kickoff         kickoff             (copiado)
  features dict   feature_vector      np.ndarray[float32] len==66
                  feature_names       tuple(FEATURE_NAMES) canónico
```

Reglas:

* Se itera exactamente `FEATURE_NAMES` (66, orden canónico) — no se
  reordena, no se recalcula, no se consulta PostgreSQL/API-Football.
* `None → np.nan`  (missing preservado; nunca 0)
* `0 / 0.0 → 0.0`  (cero preservado; nunca nan)
* Negativos y floats preservados verbatim (`float(val)` → `float32`).
* `feature_names == tuple(FEATURE_NAMES)` siempre; si la longitud no
  coincide se lanza `ValueError`; si aparecen nombres desconocidos se
  lanza `ValueError`; tipos no-numérico → `TypeError`.
* Determinista: dos llamadas con el mismo `LoadedExample` producen
  vectores `equal_nan`-idénticos.

**Política de missing en capa genérica (`features/missing.py`):**

La capa genérica **no imputa**. Solo observa:

* `has_missing(vec) → bool`
* `missing_mask(vec) → NDArray[bool]`
* `count_missing(vec) → int`
* `missing_fraction(vec) → float in [0,1]`
* `count_missing_batch(mat) → NDArray[int]`

La imputación, si corresponde, es **específica del modelo** y vive en
su trainer (no aquí):

* Poisson: `train → drop_row`, `test → impute_train_mean` (§12.7)
* Gradient Boosting: `nan` passthrough — LightGBM lo maneja nativo (§13)
* Elo baseline: no consume features con missing relevante

Esto garantiza que `None → nan` vs `0 → 0.0` jamás se confundan antes de
que el modelo decida.

### 6.7 Reproducibilidad

El orden de folds depende solo de `(dataset_version, iterator_params)`.
Cualquier modelo entrenado sobre los mismos `(dataset_version, params)`
recorre los mismos folds.

`run_id = sha256(dataset_version || iterator_params || model_name ||
model_version)[..16]`. Comparación multi-modelo: leer todos los
`run_id` con el mismo `(dataset_version, iterator_params)` y producir
`comparison_<date>.csv`.

---

## 7. Calibración multiclass

### 7.1 Marco — simplex y no "Platt + renorm ingenuo"

Una distribución `p ∈ R^3` válida (suma 1, no-negativa) es un punto
del 2-simplex. Cualquier calibración debe ser un mapeo **del simplex
al simplex**. Métodos aprobados:

| Método | Transformación | Params | Requiere |
|---|---|---|---|
| `identity` | `p_cal = p_raw` | 0 | nada |
| `temperature scaling` | `p_cal = softmax(log p_raw / T)` | 1 (T>0) | salida softmax-calibrable multiclass |
| `Dirichlet calibration` | `p_cal = softmax(W·log p + b)`, `W = I + A` con `A` L2-regularizado | 12 | `n_val ≥ 5000` |

Isotonic queda descartado como default: alto riesgo de overfit en 3
clases + no garantiza simplex por construction.

### 7.2 Decisión por modelo

| Modelo | Método default | Razón |
|---|---|---|
| Elo baseline | `temperature` (1-param T) | salida ya válida (Poisson → simplex); corrige slope sin mover ranking |
| Poisson | `temperature` (1-param T) | GLM Poisson tiende a over/under-confident; T corrige slope |
| GB multiclass | Decisión sobre val: medir `ECE_raw` y `n_val` |
|   | `ECE < 0.02` | `identity` |
|   | `0.02 ≤ ECE < 0.05` | `temperature` |
|   | `ECE ≥ 0.05` y `n_val ≥ 5000` | `Dirichlet` (regularizado) |
|   | `ECE ≥ 0.05` y `n_val < 5000` | `temperature` (no overfit) |

### 7.3 Transformaciones

#### Temperature scaling

```
z_k = log p_raw_k
p_cal_k = softmax(z_k / T) = exp(z_k / T) / Σ_l exp(z_l / T)
```

- `T ∈ [0.1, 10.0]` acotado por optimizer.
- `T = 1` → identity. `T > 1` → calienta (más uniforme). `T < 1` → más
  confidente.
- Suma = 1 por softmax (no renorm post-calibración).

#### Dirichlet calibration

```
log p_cal_k = W · log(p_raw_clipped) + b     (clip_min = 1e-12)
p_cal_k    = softmax(log p_cal_k)

W = I + A, A regularizado con L2 fuerte (λ = 0.1 default)
```

- `n_params = 12` (9 + 3).
- Optimizado minimizando NLL sobre val.
- Garantiza estar en simplex por softmax final.

### 7.4 Caso extremo — clase con prob minúscula

`p_raw_k < 1e-12`:

- Temperature: `z_k = log p_raw_k → muy negativo`, softmax →
  `p_cal_k ≈ 0.0` (no NaN; `exp(muy_negativo) = 0` es válido en
  simplex). Masa redistribuye en las otras dos.
- Dirichlet: usa `log(p_clipped)` con `clip_min = 1e-12`, documentado.

### 7.5 Suma = 1 — por construction, no por post-renorm

```
softmax(z) = exp(z) / Σ_l exp(z_l)
Σ_k softmax(z)_k = 1.0    (idempotente)
```

Test: `test_temperature_scaling_preserves_simplex`.

### 7.6 `CalibrationDetector`

```python
class CalibrationDetector:
    def detect(self, model_name, raw_probs_val, targets_val) -> CalibrationChoice:
        ece = compute_ece(raw_probs_val, targets_val)
        n = len(targets_val)
        if ece < 0.02:
            return Identity()
        if n >= 5000 and ece >= 0.05:
            return Dirichlet(reg_l2=0.1)
        return Temperature(min=0.1, max=10.0)
```

La elección se persiste en `ModelArtifact.calibration_method` y en
`calibrator.json`.

Tests:

- `test_detector_identity_when_ece_low`
- `test_detector_temperature_when_few_val`
- `test_detector_dirichlet_when_many_val_high_ece`
- `test_temperature_preserves_simplex`
- `test_dirichlet_preserves_simplex`
- `test_calibration_handles_neary_zero_prob`
- `test_calibration_T1_equals_identity`
- `test_reproducible_calibrator_from_seed`

### 7.7 Implementación Sprint 5.4

**Entrada/salida:** todos los calibradores reciben `Sequence[MatchProbabilities]`
o `ndarray (n,3)` y `targets (n,)`; salida siempre `(n,3)` finita,
`0≤p≤1`, `sum≈1` (`atol=1e-6`), misma shape, sin mutar el input.

**Identity:** `fit` valida y no aprende; `transform` copia y valida simplex.

**Temperature:** `p_cal=softmax(log(clip(p,1e-12))/T)`, `T∈[0.1,10]`,
optimiza `NLL` sobre *validation* con búsqueda acotada (scipy `bounded`
o grid log-space determinista fallback). `T=1` ≡ identity, `T>1`
suaviza, `T<1` hace más confidente. Maneja `p≈0` vía `clip` antes de
`log` sin alterar política de missing; nunca produce `NaN/inf`.

**Dirichlet:** `p_cal=softmax((I+A) log(clip(p))+b)`, `A` regularizado
`λ=0.1` (`W=I+A`, 12 params). Optimiza `NLL+λ||A||²` sobre validation;
`softmax` estable, `clip 1e-12`. Entrenable con cualquier `n` en tests,
pero el detector solo lo elige con `n≥5000` y `ECE≥0.05`.

**Detector:** usa *solo* `val` (nunca `test`/`train`):
`ECE<0.02 → Identity`, `ECE≥0.05 y n≥5000 → Dirichlet`, resto → Temperature.
Determinista; decisión serializable.

**Simplex:** garantizado por `softmax` (no clipping final); validación
centralizada `validate_multiclass_probabilities`.

**Datos vacíos/inválidos:** `n=0`, `NaN/inf`, `shape≠(n,3)`, `targets∉{0,1,2}`,
`p∉[0,1]` o `sum≠1` → `ValueError` explícito.

**Serialización:**
* Identity → `{"kind":"identity"}`  
* Temperature → `{"kind":"temperature","temperature":T,"t_min":0.1,"t_max":10}`  
* Dirichlet → `{"kind":"dirichlet","W":[[..]],"b":[..],"lambda_l2":0.1}`  
(No `CalibratorStore` aún.)

**Prohibición:** calibrar con `test` o `train` está prohibido por
diseño; el runner (Sprint 5.8) lo impone estructuralmente.

---

## 8. Métricas

### 8.1 Mínimas (todos los modelos)

| Métrica | Tipo | Definición |
|---|---|---|
| `accuracy` | clasif | `argmax(p) == resultado real` |
| `log_loss` | probabilística | `-mean log p_true` (multiclass cross-entropy) |
| `brier_home`, `brier_draw`, `brier_away` | probabilística | `(p_k - y_k)^2` por clase |
| `ece` (Expected Calibration Error) | calibración | bins × |confianza - accuracy| ponderado |
| `mce` (Max Calibration Error) | calibración | max sobre bins |
| `confusion_matrix` | clasif | 3×3 |
| `n_predictions` | tamaño | `len(test)` |
| `per_confidence` | buckets | accuracy / log loss por bucket `[0.4-0.5)...[0.9-1.0]` |

### 8.2 Adicionales según modelo

| Modelo | Extra |
|---|---|
| Elo baseline | `brier_ranking` (diagnóstico: Brier sobre `p_home - p_away` vs signo de diff-goals) |
| Poisson | `poisson_loglik`, `mae_home_goals`, `mae_away_goals`, `rmse_goals` |
| GB | `feature_importance_top10`, OOF log loss en train+val |

### 8.3 Cálculo

- Bins ECE/MCE: 10 (default), configurable, reportado en `metrics.json`.
- Buckets de confianza: 6 buckets `[0.4,0.5)...[0.9,1.0]` — default
  `DEFAULT_CONFIDENCE_BUCKETS = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]`. Confidence
  `<0.4` **no se asigna a ningún bucket** (queda excluido del reporte
  por bucket pero contado en accuracy/log_loss global). `sum(n_bucket) =
  n_{conf>=0.4}`; el reporte documenta explícitamente el conteo excluido.
- Toda métrica persiste en
  `data/models/runs/<run_id>/folds/<fold_id>/metrics.json`.

### 8.4 Convenciones de targets y validación (Sprint 5.3)

**Targets:** `0 = home_win`, `1 = draw`, `2 = away_win` (orden fijo).

**Probabilidades:** `y_proba` shape `(n, 3)` columnas `[home, draw, away]`.

**Validación centralizada** `validate_multiclass_probabilities` (→ `ValueError`):
* `0 <= p <= 1` elemento-a-elemento, finite (no NaN/inf), `n>0`,
  `shape[1]==3`, `sum(p) ≈ 1` con `atol=1e-6`. Nunca clipping/normalización
  silenciosa.

### 8.5 Fórmulas Sprint 5.3

**Accuracy:** `mean(argmax(p) == y)`.

**Log loss:** `-mean(log(clip(p_true, eps=1e-15)))`. Solo estabilización
numérica; inputs inválidos ya fueron rechazados.

**Brier por clase:** `mean((p_k - y_onehot_k)^2)` → `brier_home/draw/away`.
**Brier multiclass:** `mean(sum_k (p_k - y_k)^2) = sum(briers)` (agregado).

**Confusión:** `matrix[true, pred] += 1`, `shape (3,3)`, `sum == n`.

**ECE:** bins uniformes `[0,1]` (`n_bins=10`): confidence `= max(p)`,
pred `= argmax(p)`. Por bin: `acc_b = mean(correct)`, `conf_b = mean(conf)`,
`ECE = Σ (n_b/n) * |acc_b - conf_b|`. Bins vacíos peso 0. **MCE** = `max |acc_b-conf_b|`.

**Reliability bins:** lista serializable por bin `{bin_id, bin_lower, bin_upper,
count, accuracy, mean_confidence, abs_gap, weight}`; siempre `len==n_bins`.

**Confidence buckets:** intervalos `[edges[i], edges[i+1])` excepto último
inclusivo; por bucket `accuracy, mean_confidence, log_loss` (con mismo `eps`).

**Goals:**
* `mae_home = mean|y_true_home - y_pred_home|`, idem `mae_away`.
* `rmse_home = sqrt(mean((y_true - y_pred)^2))`, idem away.
* `rmse_total = rmse(y_true_home+y_true_away, y_pred_home+y_pred_away)`.
* `poisson_loglik` — dos formas:
  * **Rate:** con `λ>0`: `mean(-λ + k*log λ - lgamma(k+1))` por home+away.
  * **Dict:** con `{k: prob}` (conversor Poisson): `mean(log p_home[k] + log p_away[k])`,
    `k` ausente → `eps`.

### 8.6 Report (Sprint 5.3)

`FoldReport` (frozen, `to_dict()/to_json()`) con: `model_name, model_version,
dataset_version, fold_id, n_predictions, accuracy, log_loss, brier_home/draw/away/multiclass,
ece, mce, n_bins, confusion_matrix (3×3 list), confidence_buckets, reliability_bins,
mae_home_goals, mae_away_goals, rmse_home/away/total, poisson_loglik`. Validaciones:
`n_predictions>0`, matriz 3×3 con suma `== n`, `fold_id>=0`; `calibration_bins` alias de `n_bins`.

---

## 9. Model versioning

### 9.1 Manifest por artifact

Ver `§3`. La tupla reproducible:

```
(dataset_version,
 feature_definition_version,
 model_name,
 model_version,
 hyperparameters,        # incluye K, HFA, β0, β1, max_goals, λ_l2, nan_policy, ...
 seed)
```

### 9.2 Inmutabilidad

- Un `model_version` publicado nunca se reescribe. Re-entrenar con otra
  seed/data → otro `model_version`.
- `PredictionRecord` y `metrics.json` son append-only.
- `RunsStore.save(run_id, ...)` con `overwrite=False` detecta `run_id`
  existente y falla (no pisa).
- Re-correr backtest con nota distinta: bump del `run_id` con un
  `--note "..."` que entra en el hash.

### 9.3 Predicciones históricas no se modifican

Cuando entrene `v002`, `v001` sigue disponible. En runtime, el
consumidor especifica `(model_name, model_version)`. No hay "último
modelo" automático; esa decisión es operacional (CLI / Fase 6).

### 9.4 Reproducibilidad — tests

- `test_reproducibility_train_twice_same_artifact_sha256`
- `test_reproducibility_seed_change_diff_artifact`

---

## 10. Comparación entre modelos

Todos los modelos evaluados sobre el **mismo** `dataset_version` y el
**mismo** `WalkForwardIterator` (mismos params → mismos folds → mismos
fixtures). Garantía:

- `test_compare_runs_only_same_dataset_and_iterator_params` falla si se
  intenta comparar modelos entrenados sobre `(dataset_version, params)`
  distintos.

Un script `compare_runs.py` (CLI offline, no FastAPI) lee todos los
`summary.json` con el mismo `(dataset_version, iterator_params)` y
produce `data/models/runs/comparison_<date>.csv` con las métricas
agregadas (media y std por fold, log loss acumulado ponderado, ECE
acumulado ponderado, etc.).

No se elige ganador en Fase 5. Se **presenta** la tabla. La decisión
es post-Fase 5.

---

## 11. Elo-feature vs Elo-baseline

| Aspecto | `home_elo_pre_match`, `away_elo_pre_match`, `elo_difference` (Fase 4) | `Elo baseline` (Fase 5) |
|---|---|---|
| Qué es | Feature del dataset (input) | Modelo (output) |
| Dónde vive | `app/features/elo.py` | `app/prediction/models/elo_baseline.py` |
| Consume el CSV | No — el builder lo computa en Fase 4 | Sí — lee los 3 campos del `LoadedDataset` |
| Produce | 3 columnas del CSV | `MatchProbability` (p_home/draw/away) |
| Cómo pasa a probs | No pasa — es una feature | mapea ratings → λ → matriz Poisson → 1X2 (ver §11.1) |

### 11.1 Modelo Elo baseline aprobado — Poisson-Elo con cierre analítico

El baseline Elo **no** traduce directamente el rating a `1X2` (eso
rompería los invariantes del simplex). Usa la diferencia de ratings
para fijar las dos λ y luego el conversor Poisson (compartido con el
modelo Poisson) produce la distribución 1X2 válida.

#### Constantes

- `BASE_RATING = 1500` (compartido con Fase 4, no reaprendido).
- `K` configurable (grid search sobre val: `K ∈ {15, 20, 25, 30}`).
- `HFA` configurable (grid search sobre val: `HFA ∈ {50, 65, 80}`).

#### Inputs (del CSV)

- `home_elo_pre_match`
- `away_elo_pre_match`
- `elo_difference = home_elo_pre_match − away_elo_pre_match`

#### Paso 1 — ratings → λ

Dado `D = (R_home + HFA) − R_away`:

```
log λ_home = β0_home + β1 · D
log λ_away = β0_away − β1 · D      # simétrico
```

- `β0_home, β0_away`: interceptos de liga (goles promedio por campo),
  aprendidos sobre `train_block` como media de goles por equipo/campo
  (agregación global, no por fixture).
- `β1`: sensibilidad Elo → goles. Único parámetro aprendido por grid
  search (`β1 ∈ {0.0015, 0.0020, 0.0025}`), elegido por **log loss
  sobre val**.
- `λ > 0` siempre (exponencial positiva); `|β1·D| < 5` típico ⇒ sin
  overflow.
- Determinista dados `(D, β0_home, β0_away, β1)`.

#### Paso 2 — λ → 1X2

Conversor **único** `poisson_to_1x2(λ_home, λ_away, max_goals)`
compartido con el modelo Poisson (ver §12.5).

#### Paso 3 — calibración

La salida cruda del paso 2 es ya una distribución 1X2 válida. La
calibración (§7) se aplica encima; nunca puede romper el simplex
(transformaciones válidas; testeadas por construction).

#### Invariantes garantizados (tests)

- `0 ≤ p_home, p_draw, p_away ≤ 1`
- `p_home + p_draw + p_away == 1.0` (`rtol=1e-9`)
- `R_home == R_away` (sin HFA) ⇒ `p_home ≈ p_away`, `p_draw > 0`
- `R_home >> R_away` ⇒ `p_home → 1`, `p_draw, p_away → 0`

#### Hiperparámetros del artifact

```json
{
  "K": 20, "HFA": 65,
  "beta_0_home": 0.32,
  "beta_0_away": 0.12,
  "beta_1": 0.0020,
  "max_goals": 10
}
```

#### Por qué no es "trampa" respecto a Poisson

- Poisson come las 66 features; Elo baseline come 3.
- Poisson entrena regresión Poisson completa por cabeza con muchas
  features; Elo baseline ajusta 3 parámetros globales + 1 grid search.
- Elo baseline sigue siendo el lower bound: si Poisson/GB no le ganan,
  no aportan valor. Verdaderamente diferentes en grado de libertad.

---

## 12. Poisson

### 12.1 Modelo

```
log λ_home = β_home · x_home_features + α_home
log λ_away = β_away · x_away_features + α_away
```

- `λ_home, λ_away > 0` por exponencial (overflow controlado en §12.5).
- Distribución de goles asumida **independiente** (limitación
  documentada; no Dixon-Coles en la iteración inicial).
- `P(home=i, away=j) = Poisson(i; λ_home) · Poisson(j; λ_away)`.

### 12.2 Partición de features (estricta)

**Regla**: las 66 features se particionan en dos subconjuntos
asignados a cada cabeza. Una feature **no** puede aparecer
en las dos cabezas (evita redundancia y colinealidad inducida) salvo
las 8 que describen el **par** y no a un equipo: `h2h_*` (5) +
`elo_difference` (1) + `table_points_difference` (1) +
`table_goal_difference_difference` (1) — Opción A aprobada Sprint 5.6.
Con ello `λ_home` **37** features y `λ_away` **37** features,
overlap **8**, cobertura `37+37-8 = 66` exacta sin modificar `FEATURE_NAMES`.

#### Cabeza `λ_home` (37)

| Categoría | Features |
|---|---|
| Forma local | `home_wins_last_3/5/10`, `home_draws_last_3/5/10`, `home_losses_last_3/5/10`, `home_points_last_5/10` |
| Goles local | `home_goals_for_last_5/10`, `home_goals_against_last_5/10`, `home_goals_for_mean_last_5/10`, `home_goals_against_mean_last_5/10` |
| Home/away split | `home_home_points_last_5`, `home_home_goals_for_last_5`, `home_home_goals_against_last_5` |
| Standings local | `home_table_position`, `home_points_table`, `home_goal_difference_table` |
| Rest local | `home_rest_days` |
| Elo | `home_elo_pre_match`, `elo_difference` |
| xG local | `home_xg_season_asof`, `home_xga_season_asof` |
| H2H | `h2h_home_wins`, `h2h_away_wins`, `h2h_draws`, `h2h_total_goals_mean`, `h2h_sample_size` |
| Tabla diffs (par) | `table_points_difference`, `table_goal_difference_difference` |

#### Cabeza `λ_away` (37)

| Categoría | Features |
|---|---|
| Forma visitante | `away_wins_last_3/5/10`, `away_draws_last_3/5/10`, `away_losses_last_3/5/10`, `away_points_last_5/10` |
| Goles visitante | `away_goals_for_last_5/10`, `away_goals_against_last_5/10`, `away_goals_for_mean_last_5/10`, `away_goals_against_mean_last_5/10` |
| Home/away split | `away_away_points_last_5`, `away_away_goals_for_last_5`, `away_away_goals_against_last_5` |
| Standings visitante | `away_table_position`, `away_points_table`, `away_goal_difference_table` |
| Rest visitante | `away_rest_days` |
| Elo | `away_elo_pre_match`, `elo_difference` |
| xG visitante | `away_xg_season_asof`, `away_xga_season_asof` |
| H2H | (las mismas 5 features H2H que `λ_home` — dictadas por el par, no por equipo; coeficiente esperado de signo opuesto) |
| Tabla diffs (par) | `table_points_difference`, `table_goal_difference_difference` |

#### Verificación anti-redundancia (tests en Sprint 5.6)

- `test_features_partition_is_disjoint_except_h2h_elo_table_diffs` (8 permitidos)
- `test_features_partition_covers_all_66` (`37+37-8=66`, orden no importa, sin duplicados fuera de 8)
- `test_no_target_features_in_either_head`

### 12.3 Conversión λ → 1X2 (matemática)

Para `λ_home, λ_away > 0` y `M = max_goals`:

```
P(i, j) = Poisson(i; λ_home) · Poisson(j; λ_away)     para 0 ≤ i, j ≤ M

P(home win) = Σ_{i=0..M} Σ_{j=0..i-1} P(i, j)
P(draw)      = Σ_{k=0..M} P(k, k)
P(away win)  = 1 - P(home win) - P(draw)               # cierre por resta
```

`Poisson(k; λ) = exp(-λ) λ^k / k!`.

### 12.4 `max_goals`

- Default `max_goals = 10`.
- Cola Poisson `P(k > 10; λ)` para `λ ≤ 6` es `< 1e-4`.
- El cierre final se hace **por resta** (más estable que sumar tres
  términos).

### 12.5 Overflow y normalización

Tres pasos en este orden, sin ramas condicionales:

1. **Cálculo de λ estable**:
   ```
   log λ = β · x + α
   λ    = exp(log λ)
   ```
   Si `log λ > 30` ⇒ acotar a `log λ = 30` (decision documentada:
   prevenir `inf` acotando **el argumento de la tasa**, no haciendo
   clipping de probs).

2. **`Poisson(k; λ)` en log-space**:
   ```
   log_pmf(k) = -λ + k·log(λ) - gammaln(k+1)
   pmf(k)    = exp(log_pmf(k))
   ```
   `scipy.special.gammaln` y `scipy.special.logsumexp` para acumular.
   No se calculan factoriales directos.

3. **Cierre por resta**: `P(away) = 1 - P(home) - P(draw)`.
   `|1 - suma| < 1e-9` después del cierre.

El conversor `poisson_to_1x2(λ_home, λ_away, max_goals)` es **único**
para Elo baseline (§11.1) y Poisson. Implementado una sola vez.

### 12.6 Tests

- `test_poisson_to_1x2_sums_to_one` (sobre malla `(λ_home, λ_away) ∈
  {0.1, 0.5, 1.0, 2.0, 5.0}²`, todas las tripletas suman 1 con
  `rtol=1e-9`)
- `test_poisson_to_1x2_all_nonneg`
- `test_poisson_to_1x2_extreme_lambda` (`λ_home=30, λ_away=0.1` ⇒
  `p_home ≈ 1`, sin `inf` ni `NaN`)
- `test_poisson_to_1x2_equal_lambdas` ⇒ `p_home == p_away`, `p_draw > 0`
- `test_poisson_total_mass_above_max_goals` (`Σ P(i,j) ≥ 0.9999` para
  λ razonables)
- `test_poisson_features_partition_disjoint_except_h2h`
- `test_poisson_loglambda_overflow_capped`

### 12.7 Missing data en Poisson

- `None → np.nan`.
- Regresión Poisson **no maneja NaN nativamente** (a diferencia de
  LightGBM). Política:
  - En `train_block`: **descartar** el fixture si cualquier feature
    asignada a esa cabeza es NaN. Cuenta y reporta en `metrics.json`
    como `train_block_dropouts`.
  - En `test_block`: si una feature es NaN, **imputar a la media del
    train_block por feature** (no a 0). Documentado y testeado.
- Tests: `test_poisson_handles_nan_in_train_by_row_drop`,
  `test_poisson_imputes_test_nan_with_train_mean_not_zero`.

### 12.8 Hiperparámetros

```json
{
  "model_name": "poisson",
  "hyperparameters": {
    "alpha_home": <fit>, "alpha_away": <fit>,
    "beta_home": {...}, "beta_away": {...},
    "regularization_l2": 0.01,
    "max_goals": 10,
    "nan_policy_train": "drop_row",
    "nan_policy_test": "impute_train_mean"
  }
}
```

---

## 13. Gradient Boosting

- Clasificación **multiclass 1X2** (3 clases, 0=home,1=draw,2=away) sobre las **66 FEATURE_NAMES** completas.
- Cabezas de goles no en la iteración inicial (`p_home_goals=None`).
- Manejo de missing: `None→np.nan` y **passthrough nativo** al estimador (**no imputar 0 ni mediana**). `HistGradientBoostingClassifier` maneja `NaN` nativo igual que LightGBM.
- Librería GB Sprint 5.7: **decisión D1 aprobada — `sklearn.ensemble.HistGradientBoostingClassifier`** (sin nueva dependencia; `lightgbm`/`xgboost` descartados en 5.7). Hiperparámetros v1 fijos: `learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, l2_regularization=0.0, loss=log_loss, early_stopping=False, random_state=seed` (serializables, deterministas, `training_cutoff=max(train.kickoff)`).
- Tests: `test_gb_imputes_missing_explicit` (asevera que no se imputa
  0), `test_gb_multiclass_probs_sum_to_one`,
  `test_gb_artifact_manifest.py`.

---

## 14. Anti-leakage walk-forward — resumen de garantías

1. Iterator divide índices del CSV ordenado; los rangos no se
   solapan.
2. `Trainer.train` recibe solo `train_block`; el artifact no guarda
   referencias al dataset entero.
3. `Calibrator.fit` recibe solo `val_block`.
4. `Predictor.predict` recibe un solo `FixtureFeatures`; no puede leer
   el dataset.
5. El artifact persiste `training_cutoff` y `val_cutoff`.
6. Tests de integración inyectan fixtures fantasma futuros y
   verifican que las predicciones del fold de test no cambian
   (patrón de los `test_anti_leakage_*` de Fase 4).
7. Random seed por fold (determinismo reproducible a nivel run).

---

## 15. Limitaciones conocidas

1. **Poisson independence assumption**: λ_home y λ_away se ajustan
   por separado. No se modela correlación (bivariado Poisson /
   Dixon-Coles) en la primera iteración. Limitación documentada;
   segunda iteración candidata.
2. **Walk-forward no responsable de drift**: cambios de reglas o
   estilos no se detectan hasta que la data drifting entra en train.
   Documentado.
3. **Calibración por fold**: se calibra por separado en cada fold.
   Más honesto que "calibrar online" pero introduce varianza entre
   folds. Documentado.
4. **Baseline Elo no reproduce recatestería**: `p_draw` se obtiene
   por conversión Poisson válida (no residuo).
5. **No se tunea iterating hyperparams**: cada `model_version` fija
   sus hiperparámetros. Auto-tuning → feature opcional Fase 5.1.
6. **Dirichlet calibration requiere `n_val ≥ 5000`**: si n baja,
   fallback a temperature. Documentado.

---

## 16. Orden de implementación

### Sprint 5.0 — Fundamentos (sin entrenar nada) ← **en curso**

1. `docs/PHASE_5.md` ← este documento.
2. Estructura de carpetas vacía.
3. `contracts.py`, `artifacts.py`, `seeds.py`, `prediction/config.py`,
   `storage/layout.py`.
4. Tests estructurales.

### Sprint 5.1 — Walk-forward iterator

8. `backtesting/fold.py`.
9. `backtesting/iterator.py`.
10. Tests.

### Sprint 5.2 — Features vector + missing policy

11. `features/vector.py`.
12. `features/missing.py`.
13. Tests.

### Sprint 5.3 — Métricas

14. `metrics/classification.py`, `calibration.py`, `goals.py`,
    `report.py`.
15. Tests.

### Sprint 5.4 — Calibradores

16. `calibration/base.py` + `identity.py`.
17. `calibration/temperature.py`.
18. `calibration/dirichlet.py`.
19. `calibration/detector.py`.
20. Tests.

### Sprint 5.5 — Elo baseline

21. `models/elo_baseline.py`.
22. `training/elo_trainer.py` (grid search `(K, HFA)` + `β1` sobre
    val).
23. Tests (invariantes del simplex, math).

### Sprint 5.6 — Poisson

24. Decisión Poisson lib D1 aprobada Sprint 5.6: `sklearn.linear_model.PoissonRegressor` (`alpha=regularization_l2=0.01`, D2 fijo, sin grid) — `statsmodels` descartado; `poisson_to_1x2` único en `models/_poisson.py`.
25. `models/poisson.py` (incluye `poisson_to_1x2` compartido con Elo
    baseline).
26. `training/poisson_trainer.py`.
27. Tests.

### Sprint 5.7 — Gradient Boosting

28. Decisión GB lib D1 aprobada Sprint 5.7: `sklearn.ensemble.HistGradientBoostingClassifier` (D1, sin `lightgbm`/`xgboost`/`catboost` en 5.7) — `66 FEATURE_NAMES`, `None→NaN` passthrough nativo, `loss=log_loss`, `learning_rate=0.1, max_iter=100, max_leaf_nodes=31` fijos.
29. `models/gradient_boosting.py`.
30. `training/gb_trainer.py`.
31. Tests.

### Sprint 5.8 — Runner + stores

32. `storage/artifacts.py`, `runs.py`, `predictions.py`.
33. `backtesting/runner.py`.
34. Tests.

### Sprint 5.9 — End-to-end con dataset real (criterios D1-D5 aprobados)

**Alcance D1:** solo `tests/integration/prediction/test_e2e_v001.py`, sin CLI ni scripts adicionales.

**Walk-forward D2 (solo tests 5.9, no modifica defaults `PredictionSettings`):**
`min_train_size=100, test_size=10, gap_days=0, val_ratio=0.15, mode=expanding` — reducido para CI determinista.

**Modelos D3:** los tres modelos (Elo, Poisson, Gradient Boosting) sobre **exactamente los mismos folds** para validar pipeline común y comparación justa.

**Dataset D4:** verificar `data/datasets/v001/dataset.csv + metadata.json`; si no existe, tests hacen `pytest.skip` explícito (no dataset falso).

**Documentación D5:** esta sección documenta criterios y corrige discrepancias 5.6/5.9.

35. `load_dataset` + `validate_dataset` reales sobre v001 (orden `(kickoff,fixture_id)`).
36. `WalkForwardIterator` real con params D2.
37. E2E `run_backtest` por modelo (Elo/Poisson/GB) con métricas finitas, simplex, `training_cutoff`, persistencia `FoldReport/metrics.json`, `artifacts/manifest.json`, `predictions.csv` y `calibrator.json` por fold val-only.

### Sprint 5.10 — Comparación

38. `backtesting/compare.py` (CLI).
39. Producción de `comparison_<date>.csv`.

### Sprint 5.11 — Cierre documental

40-43. Actualizar `docs/ARCHITECTURE.md`, `DATA_FLOW.md`,
`DEPENDENCIES.md`, crear `docs/MODELS.md`.

---

## 17. Dependencias

**Ya presentes** en `backend/requirements.txt` y respetadas:

- `numpy==2.1.1`
- `scikit-learn==1.5.2`
- `statsmodels==0.14.2`
- `pandas==2.2.3`

**A agregar en sprints futuros** (no en 5.0):

- `lightgbm` (candidato Sprint 5.7 — pendiente decisión).
- `scipy` (ya viene con pandas/sklearn; necesario para `gammaln` /
  `logsumexp` en Poisson — Sprint 5.6).

_No se agregan dependencias en Sprint 5.0 que el código no necesite
realmente._

---

## 18. Configuración (PredictionSettings)

`backend/app/prediction/config.py` — `pydantic-settings`
independiente (no mergueada en `app/config.py` para mantener el
namespace limpio y porque el prediction engine es totalmente offline):

- `dataset_version: str = "v001"`
- `walk_forward_min_train_size: int = 200`
- `walk_forward_val_ratio: float = 0.15`
- `walk_forward_test_size: int = 50`
- `walk_forward_gap_days: int = 1`
- `walk_forward_mode: Literal["expanding", "sliding"] = "expanding"`
- `calibration_bins: int = 10`
- `confidence_buckets: str = "0.4,0.5,0.6,0.7,0.8,0.9,1.0"`
  (mismo patrón que `CONFIDENCE_BUCKETS` en `app/config.py`)
- `random_seed: int = 42`
- `max_goals: int = 10`
- `nan_policy: Literal["drop_row", "impute_train_mean"] = "drop_row"`

Variables de entorno (prefijo `PREDICTION_` en `.env.example`):
ver `Appendix A`.

---

## Appendix A — `.env.example` (variables Sprint 5.0)

```ini
# ---------- Prediction Engine (Fase 5) ----------
PREDICTION_DATASET_VERSION=v001
PREDICTION_WALK_FORWARD_MIN_TRAIN_SIZE=200
PREDICTION_WALK_FORWARD_VAL_RATIO=0.15
PREDICTION_WALK_FORWARD_TEST_SIZE=50
PREDICTION_WALK_FORWARD_GAP_DAYS=1
PREDICTION_WALK_FORWARD_MODE=expanding
PREDICTION_CALIBRATION_BINS=10
PREDICTION_CONFIDENCE_BUCKETS=0.4,0.5,0.6,0.7,0.8,0.9,1.0
PREDICTION_RANDOM_SEED=42
PREDICTION_MAX_GOALS=10
PREDICTION_NAN_POLICY=drop_row
```
