# DATASET.md — Dataset histórico reproducible

> Fase 4 — Sistema de feature engineering anti-data-leakage.
> Este documento describe **el dataset** almacenado en disco.
> Las features en sí están documentadas en [`FEATURES.md`](FEATURES.md).
> Las garantías anti-leakage en [`ANTI_LEAKAGE.md`](ANTI_LEAKAGE.md).

## 1. Fuente de verdad

La **fuente de verdad** es y sigue siendo **PostgreSQL**. El esquema
normalizado (`fixtures`, `teams`, `team_statistics`, `competitions`,
`seasons`, etc.) es la única entrada al pipeline de feature
engineering.

El CSV en disco es una **materialización inmutable y versionada** de
una consulta determinística sobre PostgreSQL, no una fuente
alternativa de datos. Si dos builds ejecutados sobre el mismo estado
de DB + la misma revisión de código producen bytes distintos, eso es
un bug.

El pipeline es:

```
PostgreSQL (esquema normalizado)
        │
        ▼
Historical Feature Builder  (app/dataset/builder.py)
        │   • cargadores as-of (app/features/asof.py)
        │   • familias de features  (app/features/*)
        │   • assembler            (app/features/assembler.py)
        ▼
data/datasets/<version>/
        ├── dataset.csv           (immutable)
        └── metadata.json         (immutable)
        │
        ▼
Loader  (app/dataset/loader.py)   (read-only)
        │
        ▼
LoadedDataset  →  training / backtesting / model comparison  (Fase 5+)
```

## 2. Formato en disco

```
data/datasets/
  v001/
    dataset.csv
    metadata.json
  v002/
    dataset.csv
    metadata.json
```

- **No se usa Parquet.** Formato CSV (por compatibilidad, diff y
  auditabilidad humana) + `metadata.json`.
- **Cada versión es un directorio** cuyo nombre **es** el
  `dataset_version` (`v001`, `v002`, …).
- Cada directorio contiene **exactamente dos archivos**:
  `dataset.csv` y `metadata.json`. No se escriben auxiliares, no se
  cachean partials.
- Un `dataset_version` publicado **no se reescribe**. Quien quiera
  regenerar bajo un cambio de features o un nuevo data_cutoff debe
  publicar un **nuevo** `dataset_version`. Esta es la garantía de
  reproducibilidad: el consumidor downstream puede citar
  `dataset_version=v001` y saber que sus bytes no mutaron.

## 3. Estructura del CSV — `dataset.csv`

El header es el orden canónico definido en
`app/dataset/_schema.py::CSV_COLUMNS`:

```
fixture_id, kickoff, competition_id, season_id, home_team_id, away_team_id,
<FEATURE_NAMES…>, <TARGET_NAMES…>
```

- Las 6 primeras columnas son **identidad** (no son features ni
  targets).
- Le siguen las features (orden de `app/features/example.py::FEATURE_NAMES`).
- Al final los targets (orden de `TARGET_NAMES`).
- Las filas están **ordenadas asc por `(kickoff, fixture_id)`** para
  que el CSV sea determinista.
- El dialecto es `excel` con `lineterminator="\r\n"` y
  `QUOTE_MINIMAL`. La nueva-línea fija a CRLF asegura igualdad de
  bytes entre Windows / Linux / CI.
- Una celda numérica faltante se serializa como **string vacío `""`**
  (constante `EMPTY_CELL` en `app/dataset/_schema.py`). El loader la
  vuelve a interpretar como Python `None` — nunca como `0`. El valor
  `0` se reserva para el caso en que el cálculo es genuinamente cero.

## 4. Estructura del manifest — `metadata.json`

Campos obligatorios (todos presentes, sin defaults client-side):

| campo                        | tipo                  | descripción                                              |
|------------------------------|-----------------------|---------------------------------------------------------|
| `dataset_version`            | `str`                 | etiqueta del dataset (`v001`). Coincide con el directorio. |
| `generated_at`               | ISO8601 UTC           | timestamp de generación. **Único** campo que legitimately cambia entre runs idénticas. |
| `feature_definition_version` | `str`                 | versión de las fórmulas (`fd_v1`). Se bump antes de publicar un `dataset_version` cuando cambia cualquier feature math. |
| `data_cutoff`                | ISO8601 UTC           | watermark temporal estricto. El dataset nunca contiene fixtures con `kickoff_time >= data_cutoff`. Debe ser `>= end_date`. |
| `source_schema_version`      | `str`                 | versión del esquema PostgreSQL leído (`schema_v1`). Se bump cuando una migración modifica tablas que el feature math consume. |
| `row_count`                  | `int`                 | número de filas de `dataset.csv`. El loader lo valida contra las filas reales. |
| `feature_names`             | `list[str]`           | lista **completa y ordenada** de FEATURES que aparecen en el CSV. Equivale a `FEATURE_NAMES`. |
| `target_names`              | `list[str]`           | ídem, para TARGETS. |
| `start_date`                 | ISO8601 UTC           | `kickoff` más temprano en el dataset. `end_date` = más tardío. |
| `competitions`               | `list[int]`           | IDs internos (`competitions.id`) incluidos. |
| `seasons`                    | `list[int]`           | IDs internos de `seasons.id` incluidos. |
| `csv_sha256`                 | `str` (64 hex)        | SHA-256 sobre los bytes del `dataset.csv` en disco. |
| `extras`                     | `dict`                | defaults a `{}`. Para campos futuros opcionales. |

`feature_names` aparece en el manifest para que un consumidor (por
ejemplo, un notebook de entrenamiento) pueda **assert** la forma de
la fila en vez de asumirla. Si `feature_names` cambia entre
versiones, el consumidor debe negarse a cargar (o migrar
explícitamente) en vez de silentemente desalinear.

## 5. SHA-256 y verificación de integridad

El manifest provee `csv_sha256`, válido contra los bytes **exactos**
del archivo en disco. La verificación es un flujo de dos pasos:

```python
from app.dataset.loader import validate_dataset, load_dataset

# 1) Verifica metadata.json + header shape + recomputa SHA-256 y
#    compara con el manifest. Lanza DatasetIntegrityError si hay mismatch.
manifest = validate_dataset("data/datasets/v001")

# 2) Carga el dataset (ya verificado). El flag verify_sha256=True
#    repite el hash por si el archivo en disco difiere del verificado.
loaded = load_dataset("data/datasets/v001", verify_sha256=True)
```

- `validate_dataset` **siempre** recomputa el SHA-256.
- `load_dataset` **no** lo recomputa por defecto (hashar 100k filas
  en cada training run es caro cuando la verificación ya se hizo
  una vez en CI). El flag `verify_sha256=True` lo activa.

Errores diferenciados del loader:

- `DatasetLoadError` — problema genérico (carpeta inexistente,
  metadata faltante, JSON inválido, fila mal tipada).
- `DatasetIntegrityError` — mismatch SHA-256 (subclase de `DatasetLoadError`).
- `DatasetSchemaMismatch` — el header del CSV no coincide con
  `CSV_COLUMNS` (subclase).
- `DatasetVersionMismatch` — el `dataset_version` que el consumidor
  esperaba no coincide con el del manifest (subclase).

## 6. Reproducibilidad

Una build `(estado de DB, revisión de código)` es **determinística**:

- Las fixtures se ordenan asc por `(kickoff_time, fixture_id)` antes
  de materializar el CSV.
- El CSV se escribe con dialect `excel` + `lineterminator="\r\n"` +
  `QUOTE_MINIMAL` en todos los SO.
- El `csv_sha256` se calcula **sobre los bytes escritos en disco**
  (no sobre una copia en memoria), via `hashlib.sha256` con hash
  streaming de 64K chunks.
- La única variación legítima entre dos runs idénticos es
  `generated_at` (timestamp del momento de la build). Vive en el
  manifest, no en el CSV — por lo que el `csv.sha256` es **idéntico**
  run a run.

Reproducibilidad rota si:

- se muta `FEATURE_NAMES` / `TARGET_NAMES` sin bump
  `feature_definition_version` y `dataset_version`,
- se introduce aleatoriedad en cualquier feature math,
- se cambia el dialect CSV sin bump,
- se exporta desde DB con ordenamiento no determinista.

`ANT_LEAKAGE.md §reproducibility` contiene más detalles.

## 7. Cómo generar un dataset

```python
import asyncio
from datetime import datetime, timezone
from app.db.session import get_session_factory
from app.dataset.builder import build_dataset


async def main() -> None:
    session_factory = get_session_factory()
    async with session_factory() as session:
        csv_path, manifest = await build_dataset(
            session=session,
            competition_ids=[39],
            season_ids=[1, 2],
            start_date=datetime(2023, 8, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
            # cutoff estricto: el dataset NUNCA incluirá fixtures con
            # kickoff_time >= data_cutoff. Debe ser >= end_date.
            data_cutoff=datetime(2026, 8, 1, tzinfo=timezone.utc),
            dataset_version="v042",
            output_path="data/datasets/v042",
        )
    print(csv_path, manifest.csv_sha256)


asyncio.run(main())
```

Errores explícitos del builder:

- `InvalidBuildConfigError("competition_ids must not be empty")`
- `InvalidBuildConfigError("season_ids must not be empty")`
- `InvalidBuildConfigError("data_cutoff … must be >= end_date")`
- `InvalidBuildConfigError("competitions not in DB: …")`
- `InvalidBuildConfigError("seasons not in DB: …")`

El builder escribe en disco **aún** si `row_count == 0` (header + un
`metadata.json` con `row_count=0` y `start_date=now`). Esto permite
verificar la mecánica del pipeline sin tener fixtures cargados.

## 8. Cómo cargar un dataset

```python
from app.dataset.loader import load_dataset, validate_dataset

# A) Verificación única en CI:
manifest = validate_dataset("data/datasets/v042")

# B) Carga en un pipeline de training:
loaded = load_dataset("data/datasets/v042",
                      expected_version="v042",
                      verify_sha256=False)  # ya validado en CI

print(loaded.manifest.row_count)
for row in loaded.rows:
    print(row.fixture_id, row.kickoff, row.features, row.targets)
```

El loader **no importa** al builder (no arrastra SQLAlchemy ni
async). El loader es síncrono y read-only, fijo a
`app.dataset._schema` y `app.dataset.manifest` para los contratos
estructurales.

Contract de tipos del loader:

- Las columnas de identidad son **int** y `kickoff` es
  `datetime` (tz-aware UTC intepretado de ISO8601).
- Cada feature es `int | float | None` según el formato del string
  (si la celda es un entero como `"5"` vuelve `int`,
  si es decimal como `"5.5"` vuelve `float`, si es `""` vuelve
  `None`).
- Los targets son `int | None`.

## 9. Versionado y migración

Reglas:

1. **Nunca reescribir** un `dataset_version` publicado.
2. Al cambiar cualquier feature math: bump
   `FEATURE_DEFINITION_VERSION` (en
   `app/dataset/manifest.py`) **y** `dataset_version`. El cambio en
   la fórmula invalida todos los datasets publicados con el mismo
   `FEATURE_DEFINITION_VERSION`.
3. Al cambiar el esquema PostgreSQL leído: bump
   `SOURCE_SCHEMA_VERSION`.
4. El consumidor downstream checkea `expected_version` y falla con
   `DatasetVersionMismatch` si el dato en disco difiere. Usar para
   que CI detecte un stale dataset en CI temprano.

Anti-feature evitado: **no** hay script "upgrade v001 → v002". Cada
`dataset_version` materializa el estado completo del feature set en
ese momento. Migrar significa publicar una nueva versión y volver a
verificar.
