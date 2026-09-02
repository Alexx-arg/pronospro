# Arquitectura

Aplicación de análisis y predicción deportiva de fútbol. Frontend Flutter que
consume un backend FastAPI. El backend es la **única** capa autorizada para
hablar con API-Football y con el servicio de explicación GLM-5.2.

> **Regla de oro.** GLM-5.2 nunca decide probabilidades. Solo recibe un JSON
> estructurado ya calculado por los modelos (Elo / Poisson / Gradient Boosting)
> y devuelve una explicación textual y listas de factores. Los números que
> ve el usuario provienen siempre de los modelos deterministas, no del LLM.

---

## 1. Diagrama de contexto

```
                 +-----------------------+
                 |  Flutter App (Mobile) |
                 |  Dart + Riverpod      |
                 |  Dio + GoRouter        |
                 +-----------+-----------+
                             | HTTPS (JSON)
                             v
+---------------------------------------------+
|  Backend FastAPI (Python 3.12)              |
|  - API layer (Pydantic)                     |
|  - Service layer (sync / prediction / eval) |
|  - Repository layer                         |
|  - Provider layer (pluggable data source)   |
+---+-----------------+--------------+--------+
    |                 |              |
    v                 v              v
+--------+    +----------------+   +-------------------+
|Postgres|    | API-Football   |   | GLM Explanation   |
| 16 DB  |    | (HTTP via httpx|   | Service (GLM-5.2  |
|        |    |  + tenacity)   |    via OpenAI-compat)|
+--------+    +----------------+   +-------------------+
```

Componentes no frontales:
- `worker` (mismo código que el backend) que ejecuta el scheduler de
  sincronización y evaluación.
- `postgres` con datos OLTP normalizados.

---

## 2. Backend — capas (clean architecture, hexagonal-lite)

```
api        -> schemas Pydantic v2 + routers FastAPI
       |
services   -> orquestación: sync, prediction (elo/poisson/gb), evaluation,
       |     explanation (GLM). Aquí vive la lógica de negocio.
       |
repositories -> abstracción de persistencia sobre SQLAlchemy async.
       |
models     -> ORM (SQLAlchemy declarative). 1:1 con esquema PostgreSQL.
       |
db         -> engine async (asyncpg), session, base declarativa.
       v
providers  -> contratos pluggables de datos externos. El primero: API-Football.
              Incluye adaptador httpx + rate limit + retries + persistencia
              "last sync watermark".
```

Reglas de capa:
- `api` no toca `models` ni SQLAlchemy. Habla con `services`.
- `services` no conoce routers. Depende de `repositories` y `providers`.
- `providers` no escriben SQL. Devuelven DTOsnormalizados (schemas internos)
  que `repositories` traducen a entidades ORM.

### Provider abstraction

```python
class DataProvider(Protocol):
    name: str
    def fetch_upcoming_fixtures(self, *, league: int, season: int, date_from, date_to) ...
    def fetch_finished_fixtures(self, ...) ...
    def fetch_team_statistics(self, *, team_id, league, season) ...
    def fetch_player_statistics(self, ...) ...
    def fetch_injuries(self, *, league, season) ...
    def fetch_lineups(self, *, fixture_id) ...
    def fetch_head_to_head(self, *, team_a, team_b, last) ...
```

`APIFootballProvider` implementa el contrato. Cambiar de proveedor solo exige
implementar otro adapter. La selección se configura via `.env`
(`DATA_PROVIDER=api_football`).

---

## 3. Frontend — capas (Riverpod + clean)

```
presentation
  screens/ (home, match_detail, predictions, performance, prediction_history)
  widgets/ (componentes reutilizables:概率 barra, form chips, etc.)
  providers/ (Riverpod NotifierProvider / FutureProvider)
        |
domain
  entities/  (tipos Dart puros, sin Dio)
  repositories/ (contratos abstractos)
  usecases/   (casos de uso ponibles por los providers)
        |
data
  datasources/remote/ (Dio client + ApiEndpoints)
  models/ (DTOs JSON serializables con freezed/json_serializable)
  repositories/ (implementaciones que consumen datasources)
        v
core (theme Material 3, routing GoRouter, Dio con interceptors, errors, utils)
```

Reglas:
- Las pantallas solo leen/escriben via Riverpod providers.
- Ningún widget llama a Dio directamente.
- Sin API keys en Flutter. El dispositivo solo conoce la URL del backend.

---

## 4. Aislamiento estricto de GLM

El servicio de explicación es **independiente** y no tiene acceso a:
- DB (no SQLAlchemy, no repos).
- API-Football.
- Cálculo de probabilidades.

Su contrato:

```
Input  (JSON estructurado por el Orquestador):
{
  "fixture": {...},
  "home_team_stats": {...},
  "away_team_stats": {...},
  "h2h": [...],
  "injuries": [...],
  "prediction": {
     "home_probability", "draw_probability", "away_probability",
     "expected_home_goals", "expected_away_goals", "confidence"
  },
  "model_version": "..."
}

Output:
{
  "summary": "...",
  "main_factors": ["...","..."],
  "risk_factors": ["...","..."],
  "confidence_explanation": "..."
}
```

Garantías:
- El LLM **no recibe** el prompt para calcular probabilidades; recibe el
  resultado ya calculado. La salida textual se valida contra el schema
  esperado y se descarta/rega si viola el contrato.
- Se prohíben en el system prompt: cambiar probabilidades, inventar
  estadísticas/lesiones/resultados, usar información fuera del JSON.
- La salida del LLM se persiste en `predictions.explanation` como JSONB.
  Los números de la predicción **no** se alteran jamás.

---

## 5. Inmutabilidad de `predictions`

- Una `prediction` insertada es **definitiva**. Aplicación y scripts solo
  tienen permisos `SELECT`/`INSERT` sobre la tabla. Un trigger `BEFORE UPDATE`
  lanza `RAISE EXCEPTION` por defensa en profundidad (ver `SCHEMA.md`).
- Para "corregir" un modelo se crea una **nueva** `model_version` y nuevas
  predicciones. Nunca se sobreescribe el historial.
- `features_snapshot` guarda el estado exacto de los features al kick-off,
  serializable y reproducible para backtesting.

---

## 6. Antifugas de datos (anti data-leakage)

- Las predicciones solo se generan a partir de features con `as_of_date < kickoff`.
- El orquestador de predicción filtra estadísticas/injuries por
  `kickoff_time`.
- El backtesting es **walk-forward**: cada fold solo entrena con fixtures
  `< t` y valida en `[t, t+h)`. No hay shuffle.
- En features de forma reciente (win streak, xG rolling), se reconstruyen
  excluyendo filas del futuro inmediato.

---

## 7. Modelos (capa `services/prediction/`)

| Modelo | Estado | Salida |
|---|---|---|
| Elo | inicial | P(gana|local), P(empate), P(gana|visitante) vía ratings diferencia |
| Poisson | inicial | lambda_x, lambda_y → distribuciones de goles → P(1X2) |
| Gradient Boosting | posterior | clasificación softmax + regresión de goles |

Cada modelo publica un `predict(features_snapshot) -> PredictionDTO`
con la misma firma. El `PredictorOrchestrator`:
1. Construye el snapshot de features (con watermark `< kickoff`).
2. Llama al(es) modelo(s) activo(s).
3. Escribe **una sola** `Prediction` por `(fixture, model_version)`.
4. Llama al `ExplanationService` pasándole el DTO ya cerrado.

---

## 8. Calidad. Mapeo a requisitos

| Requisito | Dónde se aplica |
|---|---|
| Type hints | `mypy --strict` en backend, `dart analyze --fatal-infos` en frontend |
| Validación | Pydantic v2 en DTOs + CHECK constraints en DB |
| Manejo de errores | `app/core/exceptions.py` + handlers FastAPI |
| Logging | `loguru` estructurado, context vars por request |
| Retries | `tenacity` en providers y GLM client |
| Rate limiting | `aiolimiter` en provider y en GLM |
| Tests unitarios | `pytest`, `flutter test`, módulos puros |
| Tests de integración | `pytest-asyncio` + Postgres container (testcontainers/respx) |

---

## 9. Infraestructura

- `docker-compose.yml` levanta `postgres`, `backend` (FastAPI/uvicorn), y
  `worker` (mismo image, comando `python -m app.tasks.scheduler`).
- `.env` (no commiteado) + `.env.example` (plantilla).
- `backend/Dockerfile` slim Python 3.12 + asyncpg.
- Frontend Flutter se construye fuera de compose (es una app Android); solo
  separamos su `pubspec.yaml` para fijar dependencias.
