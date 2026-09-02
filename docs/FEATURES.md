# FEATURES.md — Catálogo de features del dataset histórico (Fase 4)

> Estado: refleja **exactamente** el código de `backend/app/features/`
> en el commit actual. Toda feature listada aquí está implementada;
> toda feature implementada está listada aquí.
>
> El orden de las features en el CSV = orden de
> `app.features.example.FEATURE_NAMES`. No modificar este documento
> sin bump `feature_definition_version` (ver
> [`DATASET.md`](DATASET.md)).

## 1. Introducción

### Qué es una feature

Una **feature** es una columna pre-match del dataset. Cada fixture con
kickoff `T` produce una fila de features que describe el estado de los
dos equipos estrictamente **antes** de `T`.

### Todas las features son pre-match

Una feature jamás puede leer:

- el resultado del propio fixture objetivo,
- estadísticas posteriores al kickoff,
- partidos posteriores al kickoff,
- standings posteriores al kickoff,
- snapshots de `team_statistics` con `as_of_date >= kickoff.date()`,
- cualquier dato derivado del futuro.

### Feature vs target

El assembler (`app/features/assembler.py`) construye las features
**primero**, a partir de un `FixtureRow` que ya incluye `home_goals` /
`away_goals` (porque está terminado en DB), **pero el feature math
nunca los lee**. Las `targets` se escriben después en un dict
separado. No hay contenedor compartido ni aliasing — físicamente
imposible que una feature lea un target.

La distinción nominal:

- **Feature**: `HistoricalMatchExample.features[name]`. Vive en el
  bloque central del CSV, calculada pre-match.
- **Target**: `HistoricalMatchExample.targets[name]`. Vive al final
  del CSV, leída del resultado post-match.

### Regla as-of

Para un fixture con kickoff `T`:

> **Solo se puede utilizar información con timestamp estrictamente `data_time < T`.**

- El fixture objetivo jamás puede contribuir a sus propias features
  (se excluye por `exclude_fixture_id`).
- Los datos posteriores no pueden modificar una feature histórica ya
  construida.
- Toda función de feature recibe `kickoff` y `exclude_fixture_id` y
  filtra en `app/features/asof.py` (único punto del boundary).

### Política de missing data

- **`None`** = "missing" (no hay suficiente historia, snapshot
  ausente, ventana parcial). El builder lo serializa como `""` y el
  loader lo reinterpreta como `None`.
- **`0`** = "el cálculo es genuinamente cero" (ventana completa donde
  el agregado es `0`, por ejemplo 0 goles en 5 partidos). El builder
  lo serializa como `"0"`.
- **Nunca se imputa `0` por missing.** El consumidor puede imputar
  después si lo desea; el contrato del dataset es explícito.

---

## 2. Identidad / contexto

Las 6 primeras columnas del CSV. **No son features ni targets** —
son identificadores para que el consumidor pueda rastrear cada fila al
fixture real.

| Columna            | Tipo             | Descripción                                  |
|--------------------|------------------|----------------------------------------------|
| `fixture_id`       | `int`            | ID interno `fixtures.id` (BIGSERIAL). **NO** el `external_id` del proveedor. |
| `kickoff`          | `datetime` (UTC) | `fixtures.kickoff_time`. Frontera temporal as-of de todas las features. |
| `competition_id`   | `int`            | ID interno `competitions.id`.              |
| `season_id`        | `int`            | ID interno `seasons.id`.                    |
| `home_team_id`     | `int`            | ID interno `teams.id` del local.           |
| `away_team_id`     | `int`            | ID interno `teams.id` del visitante.        |

El `external_id` del proveedor NO se persiste en el dataset: la
separación external_id vs id interno (Fase 2) garantiza que las FK
internas siempre sean el BIGSERIAL.

---

## 3. Categorías de features

### 3.1 Forma (Form)

**Implementación**: `app/features/form.py`.
**Ventanas**: `(3, 5, 10)` — constante `_WINDOWS = (3, 5, 10)`.
**Rolling helper**: `rolling_count` (para W/D/L), `rolling_sum`
(para puntos).
**Fuente**: historial del equipo dentro de la misma temporada
(`season_fixtures`), filtrado por `team_history_before`.
**Boundary temporal**: `kickoff_time < T` (estricto, vía
`team_history_before`).
**Exclusión del target**: `exclude_fixture_id` pasado por el assembler.

Para cada ventana `N ∈ {3, 5, 10}` y para cada equipo (home/away):

#### `home_wins_last_{3,5,10}` / `away_wins_last_{3,5,10}`

**Descripción**: Cantidad de victorias del equipo en sus `N` últimos
partidos finalizados antes de `T`.

**Fórmula**:
```
rolling_count(history, N, lambda fx: fx.outcome_for(team_id) == "W")
```
donde `history` = `team_history_before(...)` ya ordenado newest-first.

**Fuente**: `FixtureRow` de la misma temporada.

**Ventana**: últimos `N` (con `N ∈ {3,5,10}`) fixtures finalizados
estrictamente antes de `T`.

**Boundary**: `kickoff_time < T`.

**Exclusión del target**: sí, por `exclude_fixture_id`.

**Missing**: si `len(history) < N` (ventana parcial) → `None` y
`missing_report[...] = "fewer than N finished matches before T"`.

**Tipo**: `int | None` cuando la ventana es completa; `None` cuando
es parcial.

**Unidad**: count (partidos).

#### `home_draws_last_{3,5,10}` / `away_draws_last_{3,5,10}`

Igual a `wins` pero con predicado `outcome_for(team_id) == "D"`.

#### `home_losses_last_{3,5,10}` / `away_losses_last_{3,5,10}`

Igual a `wins` pero con predicado `outcome_for(team_id) == "L"`.

#### `home_points_last_{5,10}` / `away_points_last_{5,10}`

**Descripción**: Puntos acumulados por el equipo en sus `N` últimos
partidos finalizados antes de `T`. Solo se publica para `N ∈ {5, 10}`
(no `3`).

**Fórmula**:
```
rolling_sum(history, N, lambda fx: fx.points_for(team_id))
```
donde `points_for` = `{W: 3, D: 1, L: 0}`.

**Missing**: si `len(history) < N` → `None`.

**Tipo**: `int | None`.

**Unidad**: puntos.

---

### 3.2 Goles (Goals)

**Implementación**: `app/features/goals.py`.
**Ventanas**: `(5, 10)` — constante `_WINDOWS = (5, 10)`.
**Rolling helpers**: `rolling_sum` (totales), `rolling_mean` (medias).
**Fuente**: `FixtureRow` (misma temporada).
**Boundary**: `kickoff_time < T` (estricto).
**Exclusión del target**: `exclude_fixture_id`.

#### `home_goals_for_last_{5,10}` / `away_goals_for_last_{5,10}`

**Descripción**: Total de goles marcados por el equipo en sus `N`
últimos partidos finalizados antes de `T`.

**Fórmula**:
```
rolling_sum(history, N, lambda fx: fx.goals_for(team_id))
```

**Missing**: si `len(history) < N` → `None`.

**Tipo**: `int | None`. `0` = ventana completa sin goles marcados.

**Unidad**: goles (acumulados).

#### `home_goals_against_last_{5,10}` / `away_goals_against_last_{5,10}`

Igual a `goals_for` pero con `fx.goals_against(team_id)`.

#### `home_goals_for_mean_last_{5,10}` / `away_goals_for_mean_last_{5,10}`

**Fórmula**:
```
rolling_mean(history, N, lambda fx: fx.goals_for(team_id))
```
`rolling_mean` **saltea** cualquier `None` dentro de la ventana (no
los cuenta como `0`), y devuelve `None` si ninguna celda es
no-`None`. Para `FixtureRow.goals_for` este caso no se activa porque
siempre es `int` en una fila finalizada.

**Missing**: si `len(history) < N` → `None`.

**Tipo**: `float | None`.

**Unidad**: goles por partido.

#### `home_goals_against_mean_last_{5,10}` / `away_goals_against_mean_last_{5,10}`

Igual a `goals_for_mean` pero con `fx.goals_against(team_id)`.

---

### 3.3 Home / Away splits

**Implementación**: `app/features/homeaway.py`.
**Ventana**: `5` — constante `_WINDOW = 5` (no hay `3` ni `10`).
**Rolling helper**: `rolling_sum`.
**Fuente**: `FixtureRow` (misma temporada), restringido a la cancha
relevant.
**Boundary**: `kickoff_time < T` (estricto).
**Exclusión del target**: `exclude_fixture_id`.

Para el equipo **local** del fixture objetivo: su historial limitado a
partidos donde fue **local**. Para el equipo **visitante**: limitado
a donde fue **visitante**.

#### `home_home_points_last_5`

**Descripción**: Puntos acumulados por el equipo local en sus 5
últimos partidos como **local**, antes de `T`.

**Fórmula**:
```
history = team_history_before(team=home, ...)
sliced  = [fx for fx in history if fx.home_team_id == home]
rolling_sum(sliced, 5, lambda fx: fx.points_for(home))
```

#### `away_away_points_last_5`

Ídem para el visitante restringido a sus partidos como visitante
(`fx.away_team_id == away`).

#### `home_home_goals_for_last_5` / `away_away_goals_for_last_5`

**Fórmula**: `rolling_sum(sliced, 5, lambda fx: fx.goals_for(team_id))`.

**Missing**: si `len(sliced) < 5` → `None`.

#### `home_home_goals_against_last_5` / `away_away_goals_against_last_5`

Igual pero con `fx.goals_against(team_id)`.

**Tipo**: `int | None`. **Unidad**: goles (acumulados) o puntos.

---

### 3.4 H2H (Head-to-Head) — **CROSS-SEASON**

**Implementación**: `app/features/h2h.py`.
**Constante**: `MIN_SAMPLE: int = 3` (umbrazo de sample mínimo).
**Fuente**: **universal** — todos los `Fixture` finalizados entre los
dos equipos del fixture objetivo, sin importar temporada. El builder
carga este universo vía `app/features/asof.py::load_h2h_fixtures_as_of`
(una consulta `OR` que acepta cualquier asignación de localía) y lo
pasa al assembler como `h2h_fixtures`.
**Boundary temporal**: `kickoff_time < T` **estricto**, aplicado por
`fixtures_before_unordered` dentro de `compute_h2h_features`.
**Exclusión del target**: `exclude_fixture_id` (el fixture objetivo se
cae por id, incluso si compartiera kickoff con otro partido del par).

> **H2H es CROSS-SEASON.** Esto significa:
> - puede utilizar enfrentamientos de temporadas anteriores;
> - solo utiliza partidos cuyo `kickoff_time` sea estrictamente
>   anterior al `kickoff` del fixture objetivo;
> - nunca utiliza partidos futuros;
> - nunca incluye el propio fixture objetivo;
> - si no existen enfrentamientos válidos, devuelve missing según la
>   política de missing data (no inventa datos, no convierte `NULL` en
>   `0`).

> El fallback `h2h_fixtures=None` still existe en el assembler para
> callers sin el bulk loader (tests especiales, predicción runtime).
> El **builder** siempre pasa el universo cross-season real.

#### `h2h_sample_size`

**Descripción**: Cantidad de partidos H2H válidos encontrados entre
los dos equipos, **estrictamente antes** de `T`. Es la **verdadera
cardinalidad** del conjunto elegible — no se topa artificialmente.

**Fórmula**:
```
eligible = [fx for fx in h2h_fixtures
            if _is_pair_match(fx, a=home, b=away)]
ordered  = fixtures_before_unordered(eligible, kickoff=T,
                                    exclude_fixture_id=fx_id)
sample   = len(ordered)
```

**Missing**: nunca. Es siempre un `int ≥ 0`. Si `0`, el resto de las
métricas H2H quedan en `None`.

**Tipo**: `int`.

**Unidad**: partidos.

#### `h2h_home_wins`

**Descripción**: Cantidad de victorias del **local del fixture
objetivo** en los H2H estrictamente anteriores a `T`.

**Fórmula**:
```
sum(1 for fx in ordered if fx.outcome_for(home_team_id) == "W")
```

**Missing**: si `sample < MIN_SAMPLE (3)` → `None`. El
`missing_report` registra `"only {sample} H2H fixtures before T
(< MIN_SAMPLE=3)"`.

**Tipo**: `int | None`.

**Unidad**: partidos.

#### `h2h_away_wins`

Igual a `home_wins` pero para el visitante del fixture objetivo:
`fx.outcome_for(away_team_id) == "W"`.

#### `h2h_draws`

Cantidad de empates en los H2H anteriores:
`fx.outcome_for(home_team_id) == "D"`.

#### `h2h_total_goals_mean`

**Descripción**: Promedio de goles totales por partido en los H2H
estrictamente anteriores a `T`.

**Fórmula**:
```
total = sum(fx.home_goals + fx.away_goals for fx in ordered)
mean  = total / sample
```

**Missing**: si `sample < MIN_SAMPLE` → `None`.

**Tipo**: `float | None`.

**Unidad**: goles por partido.

> Contract de política: cuando `sample < MIN_SAMPLE`, las cuatro
> métricas (`home_wins`, `away_wins`, `draws`, `total_goals_mean`)
> quedan en `None`, pero `sample_size` **siempre** es el número real.
> Un consumidor puede usar `h2h_sample_size` para decidir si confía
> en la métrica o la descarta.

---

### 3.5 Standings (tabla reconstruida históricamente)

**Implementación**: `app/features/standings.py`.
**Fuente**: reconstrucción pura desde `FixtureRow` de la misma
`(competition, season)` — **no se usa** `API-Football /standings` ni
ninguna tabla `standings` persistida.
**Boundary temporal**: solo fixtures con `kickoff_time < T`.
**Exclusión del target**: `exclude_fixture_id` (el target se cae
incluso si su kickoff hubiera sido < T).

**Fórmula de reconstrucción**:
- victoria = 3 puntos
- empate = 1 punto
- derrota = 0 puntos
- `goals_for` = suma de `fx.goals_for(team_id)` sobre los
  fixtures elegibles
- `goals_against` = suma de `fx.goals_against(team_id)`
- `goal_difference = goals_for - goals_against`

**Ordenamiento de la tabla** (clave determinística):
```
(-points, -goal_difference, -goals_for, team_id ASC)
```
El `team_id ASC` es el tiebreaker determinista — las ligas reales
usan H2H / fair-play que no podemos reproducir desde fixtures solos
(ver `docs/FEATURES.md §standings` ya documentado en el código).

#### `home_table_position` / `away_table_position`

**Descripción**: Posición en la tabla reconstruida estrictamente antes
de `T`.

**Fórmula**:
```
ranked = sorted(rows.values(), key=(-points, -gd, -gf, team_id))
position = idx + 1  (1-based)
```

**Missing**: si el equipo no jugó ningún fixture terminado antes de
`T` → `None` y `missing = "home team has no finished fixture before T"`.

**Tipo**: `int | None`.

**Unidad**: posición (1 = líder).

#### `home_points_table` / `away_points_table`

**Fórmula**: `3 * w + 1 * d` acumulado en los fixtures elegibles.

**Tipo**: `int | None`.

**Unidad**: puntos.

#### `home_goal_difference_table` / `away_goal_difference_table`

**Fórmula**: `goals_for - goals_against` acumulado.

**Tipo**: `int | None`.

**Unidad**: diferencia de goles.

#### `table_points_difference`

**Fórmula**: `home_points_table - away_points_table`.

**Missing**: si **alguno** de los dos equipos no aparece en la tabla
→ `None` y `missing = "at least one team has no finished fixture
before T"`.

**Tipo**: `int | None`.

#### `table_goal_difference_difference`

**Fórmula**: `home_goal_difference_table - away_goal_difference_table`.

**Missing**: igual a `table_points_difference`.

**Tipo**: `int | None`.

---

### 3.6 Descanso (Rest)

**Implementación**: `app/features/rest.py`.
**Fuente**: el fixture terminado más reciente del equipo,
estrictamente antes de `T`.
**Boundary temporal**: `kickoff_time < T` (estricto).
**Exclusión del target**: `exclude_fixture_id`.

#### `home_rest_days` / `away_rest_days`

**Descripción**: Días enteros desde el último partido finalizado del
equipo antes de `T`.

**Fórmula**:
```
history = team_history_before(team, ...)
last   = history[0]   # newest-first
delta  = T - last.kickoff_time
days   = int(delta.total_seconds() // 86400)
```

**Missing**: si `history` está vacío (sin partido anterior) → `None`
y `missing = "no previous fixture before T"`. **No** se devuelve `0`
en este caso: `0` is reserved para el caso válido de dos partidos en
el mismo día.

**Tipo**: `int | None`.

**Unidad**: días enteros (piso hacia abajo).

> El descanso se computa actualmente **solo dentro de la temporada /
> competition-id** del fixture objetivo, porque el builder pasa
> `season_fixtures` restringido a `(competition, season)`. Si a
> futuro se ampliara el historial cross-competición, el número crece
> hasta cubrir tambien el offseason (ese cambio sería un bump de
> `feature_definition_version`).

---

### 3.7 Elo temporal

**Implementación**: `app/features/elo.py`.
**Constantes del código**:

| constante          | valor   | significado                                                |
|--------------------|---------|------------------------------------------------------------|
| `DEFAULT_K`        | `20.0`  | coeficiente de desarrollo (FIFA clásico).                  |
| `DEFAULT_HFA`      | `65.0`  | home-field advantage en puntos de rating (~0.5 goles).     |
| `BASE_RATING`      | `1500.0`| rating inicial de todo equipo al ingresar al dataset.      |

**Fuente**: fixtures finalizados de la (competition, season) del
fixture objetivo, en orden cronológico.

**Fórmula de expected score (A local, B visitante)**:
```
E_A = 1 / (1 + 10 ** ((R_B - (R_A + HFA)) / 400))
```
`HFA` se suma al rating del local.

**Actualización post-match** (Elo clásico):
```
R_A_new = R_A + K * (actual_A - E_A)
R_B_new = R_B + K * (actual_B - (1 - E_A))
```
con `actual_A/B`:
- victoria = 1.0
- empate   = 0.5
- derrota  = 0.0

**Cómo se obtiene el rating PRE-MATCH**:
`elo_stream` procesa los fixtures de la temporada asc por `kickoff`
y, para cada fixture, **primero** emite su snapshot de pre-match
(`EloPreMatch`) **antes** de fold su propio resultado en los
ratings. Por lo tanto `Y[fx].home_elo` es el rating **un instante
antes** del kickoff de `fx`, sin contar el resultado de `fx`.

**Exclusión del target**: `compute_elo_features` filtra
`season_fixtures` con `fx.kickoff_time < fixture.kickoff_time` **y**
`fx.fixture_id != fixture.fixture_id`, y luego re-inyecta el target
al final del stream para emitir su snapshot pre-match **antes** de
fold su resultado. Garantía: el rating pre-match del target no se ve
afectado por su propio resultado (ej. un 10-0 no sube el rating
pre-match).

#### `home_elo_pre_match` / `away_elo_pre_match`

**Descripción**: Rating Elo del local / visitante **inmediatamente
antes** del kickoff del fixture objetivo.

**Tipo**: `float`. **No es `None`** salvo edge case defensivo (fixture
no encontrado en el stream reconstruido; en la práctica nunca ocurre
porque el assembler lo reinyecta).

**Unidad**: puntos Elo (sin escala absoluta; sólo relativa entre los
dos equipos).

#### `elo_difference`

**Fórmula**: `home_elo_pre_match - away_elo_pre_match`.

**Tipo**: `float`.

**Unidad**: puntos Elo.

> Sub-tleza simultánea: dos fixtures del mismo día comparten
> boundary. Ambos usarán pre-match ratings que excluyen el resultado
> del otro — el procesamiento es single-pass asc, así que el
> segundo fixture sí verá el resultado del primero foldado en su
> pre-match. Esta es la convención documentada en el código; para
> hacer un "matchday freeze" habría que cambiar `compute_elo_features`
> (bump `feature_definition_version`).

---

### 3.8 xG (season-as-of snapshot)

**Implementación**: `app/features/xg.py`.
**Fuente**: snapshots estacionales de `team_statistics` (cuya
`as_of_date` marca el corte acumulativo hasta esa fecha). **No** hay
xG por partido en DB (la tabla `fixtures` no lo persiste), por eso
xG-per-match rolling **no está implementado**.
**Boundary temporal**: `as_of_date < kickoff.date()` **estricto**
(no se acepta el mismo día).
**Constante interior**: ningún window; se toma el snapshot más
reciente.

> Los nombres de las features **no** usan sufijo `last_5` ni
> `last_10` — precisamente para evitar que un consumidor las confunda
> con un rolling per-match. El nombre es `*_season_asof`.

#### `home_xg_season_asof` / `away_xg_season_asof`

**Descripción**: xG acumulativo a la fecha, tomado del snapshot de
`team_statistics` más reciente con `as_of_date < kickoff.date()`.

**Fórmula**:
```
eligible = [row for row in stats_rows
            if row.team_id == team_id and row.as_of_date < kd]
if eligible:
    latest = max(eligible, key=lambda r: r.as_of_date)
    feature = latest.xg
else:
    feature = None
```

**Missing (dos niveles)**:
1. **No existe snapshot** con `as_of_date < kickoff.date()` →
   `None` y `missing = "no team_statistics snapshot strictly before
   kickoff date"`.
2. **El snapshot existe pero `xg` (o `xga`) es `NULL`** (el
   proveedor no devolvió el dato) → `None` y `missing = "snapshot
   exists but xG is NULL (provider did not return)"`.

**Tipo**: `float | None`.

**Unidad**: xG acumulado a la fecha del snapshot (no por partido).

#### `home_xga_season_asof` / `away_xga_season_asof`

Igual a `xg_season_asof` pero con `latest.xga` (expected goals
against).

---

### 3.9 Rolling / aggregations (helpers no listados en el CSV)

**Implementación**: `app/features/rolling.py`.
NO son features del dataset — son las primitivas usadas por las
familias.

| Helper           | Salida                          | Política missing                                       |
|------------------|---------------------------------|--------------------------------------------------------|
| `rolling_sum`    | `float | None`                  | `None` si `len(window) < N`; `0.0` si N items todos `0` o `None`. |
| `rolling_mean`   | `float | None`                  | `None` si `len(window) < N`; `None` si todas las celtas son `None`; `0.0` si todas son `0`. |
| `rolling_count`  | `int` (siempre)                 | `0` es válido: ninguna fila matchea el predicado. No devuelve `None` nunca. |
| `rolling_rate`   | `float | None`                  | `None` si `len(window) < N` o `N == 0`; `0.0` si ninguna matchea; `1.0` si todas matchean. |

`window_size` devuelve `min(len(history), n)` (0 si `n <= 0`).

> El fixture objetivo nunca entra a estas helpers — ya fue excluido
> aguas arriba por `team_history_before` / `fixtures_before_unordered`.

> Missing vs cero: esta es la **única** distinción matemática
> siempre. Ventana incompleta ⇒ `None`. Ventana completa con suma
> real `0` ⇒ `0.0`. No se imputan `0` por falta de datos.

---

## 4. Features comunes (anti-leakage)

Cada feature sigue las mismas garantías, aplicadas en distintos
puntos:

| Familia    | Boundary aplicado por            | Donde está `exclude_fixture_id` |
|------------|----------------------------------|---------------------------------|
| form       | `team_history_before`            | `asof.team_history_before`      |
| goals      | `team_history_before`            | `asof.team_history_before`      |
| homeaway   | `team_history_before`            | `asof.team_history_before`      |
| h2h        | `fixtures_before_unordered`      | `asof.fixtures_before_unordered`|
| standings  | inline filter (`kickoff_time < T`) | `exclude_fixture_id` en    `standings_as_of`   |
| elo        | inline filter + re-injection      | `exclude_fixture_id` en `compute_elo_features` |
| rest       | `team_history_before`            | `asof.team_history_before`      |
| xg         | `latest_team_stats_before`       | n/a (no usa fixtures)           |

Ver [`ANTI_LEAKAGE.md`](ANTI_LEAKAGE.md) para las garantías por
familia y los ejemplos de leakage que el pipeline garantiza no ocurrir.

---

## 5. Targets

**NO son features.** Los targets se construyen a partir del **resultado
posterior** del fixture objetivo y viven en un bloque completamente
separado del dict de features.

Definidos en `app/features/example.py::TARGET_NAMES`:

| Target           | Descripción                                       | Tipo      |
|------------------|---------------------------------------------------|-----------|
| `home_win`       | `1` si `home_goals > away_goals`, `0` en otro caso. Solo es `1` en victorias locales. | `int \| None` |
| `draw`           | `1` si `home_goals == away_goals`, `0` en otro caso. | `int \| None` |
| `away_win`       | `1` si `away_goals > home_goals`, `0` en otro caso. | `int \| None` |
| `home_goals`     | `fixture.home_goals`                              | `int \| None` |
| `away_goals`     | `fixture.away_goals`                               | `int \| None` |

Las flags `home_win / draw / away_win` son **mutuamente exclusivas**
pero no se asegura que sumen 1 — se computan independientemente para
permitir tres cabezas de clasificación separadas (no son
one-hot-encoded acopladas).

> `None` solo aparece si el fixture no está terminado (`home_goals` o
> `away_goals` son `None`). En el dataset histórico (que solo
> materializa fixtures `status='finished'`) los targets siempre son
> `int`.

> **Los targets nunca pueden participar en la construcción de las
> features pre-match.** El assembler los escribe al final, en un dict
> separado, sin aliasing.

---

## 6. Anti-leakage (resumen por familia)

Ver [`ANTI_LEAKAGE.md`](ANTI_LEAKAGE.md) para el detalle completo.
Acá sólo el resumen por familia:

### Form / Goals
- Ventana estrictamente `< T` (vía `team_history_before`).
- `exclude_fixture_id` cae el target.
- Un fixture futuro inyectado al universo NO cambia las features
  (`test_anti_leakage_form_goals.py`).

### Home/Away
- Misma boundary; además restringida al rol (local/visitante).
- Un partido futuro de la misma cancha no entra
  (`test_anti_leakage_homeaway.py`).

### H2H
- **Cross-season**. El builder carga el universo completo y lo pasa
  al assembler.
- Filtro temporal estricto dentro de `fixtures_before_unordered`.
- `exclude_fixture_id`.
- Partidos posteriores al kickoff no entran
  (`test_anti_leakage_h2h.py`).

### Standings
- Reconstrucción histórica pura desde `FixtureRow` de la
  `(competition, season)`.
- Filtro `kickoff_time < T`.
- Exclusión del target por id.
- Un partido futuro inyectado NO cambia la tabla reconstruida
  (`test_anti_leakage_standings.py`).

### Elo
- Single-pass asc, emite snap pre-match **antes** de fold el resultado.
- `exclude_fixture_id` filtra el target; el rating pre-match no se ve
  afectado por el propio resultado.
- Partidos posteriores al kickoff no entran
  (`test_anti_leakage_elo.py`).

### Rest
- Solo el último fixture `< T` del equipo.
- Un partido futuro no desplaza al "último" (`test_anti_leakage_rest.py`).

### xG
- Snapshot más reciente con `as_of_date < kickoff.date()` (estricto,
  también excluye el mismo día, que es el snapshot sincronizado el día
  del partido y contiene información posterior al kickoff).
- Si solo existen snapshots en/after kickoff.date, todo xG queda
  `None` (`test_anti_leakage_xg.py`).

---

## 7. Verificación final

### Conteo de features documentadas vs código

El código de `app/features/example.py` define `FEATURE_NAMES` con
exactamente:

- Form: 18 W/D/L counts (6 ventanas × 3 outcomes × 2 scopes = 18) +
  4 points (5/10 × 2 scopes) = **22**
- Goals: 8 sums (4 features × 2 ventanas × 2 scopes) + 8 means
  (4 features × 2 ventanas × 2 scopes — mismo nombre, distinta
  métrica — en realidad `*_last_5/10` for `goals_for/against` ×
  `home/away` × `sum/mean` = 2×2×2×2 = 16) → **16**
- Home/Away: 3 features × 2 scopes = **6**
- H2H: 5 features = **5**
- Standings: 8 features (positions × 2, points × 2, gd × 2, points
  diff, gd diff) = **8**
- Rest: 2 features = **2**
- Elo: 3 features = **3**
- xG: 4 features = **4**

**Total: 22 + 16 + 6 + 5 + 8 + 2 + 3 + 4 = 66 features.**

Cada una está documentada arriba en su sección. El orden en
`FEATURE_NAMES` coincide con el orden en que aparecen en el CSV y
en este documento (Identidad → Forma → Goles → Home/Away → H2H →
Standings → Rest → Elo → xG).

### Confirmaciones

1. ✅ Las features documentadas == las constantes en
   `app/features/example.py`. Sin extras, sin faltantes.
2. ✅ H2H está **explícitamente documentado como cross-season**, con
   la nota del fallback `h2h_fixtures=None` que seguir existiendo
   para callers sin el bulk loader.
3. ✅ Cada feature documenta: nombre, descripción, fórmula, fuente,
   ventana, boundary, exclusión del target, missing, tipo, unidad.
4. ✅ Targets documentados por separado con la aclaración de que no
   son features y nunca intervienen en su construcción.

### Discrepancias detectadas

- **Rest cross-competition**: el builder actualmente pasa
  `season_fixtures` restringido a `(competition, season)`, así que
  `home_rest_days` / `away_rest_days` solo consideran partidos de la
  misma temporada. Si el primer partido del equipo en la temporada es
  contra el target, `rest_days = None`. Esto está documentado en §3.6
  como un caveat — no es un bug.
- **xG season-as-of**: el nombre del feature NO tiene `last_N` ni
  `last_5`. Es `xg_season_as_of`. Esto es intencional para evitar
  confusión con rollings per-match que no existen en Phase 4.
- **H2H cross-season en el builder**: tras la corrección de [`Fase 4
  H2H`](../backend/app/dataset/builder.py), el builder carga el
  universo cross-season y lo pasa al assembler. El asm assembler
  mantienen el fallback `None` para callers externos, como se
  documenta en §3.4.
- **Elo matchday simultáneo**: si dos fixtures del mismo día
  comparten el mismo "matchday", el target que procesa segundo ya ve
  el primero foldeado en su pre-match rating. Esto es por la paseada
  single-pass asc; documentado en §3.7.
