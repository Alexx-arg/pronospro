# ANTI_LEAKAGE.md — Garantías anti-data-leakage (Fase 4)

> Estado: refleja **exactamente** el código de
> `backend/app/features/` y `backend/app/dataset/`. El catálogo
> completo de features y sus fórmulas está en
> [`FEATURES.md`](FEATURES.md); el formato del dataset, reglas de
> versioning y checksum en [`DATASET.md`](DATASET.md).

## 1. Regla temporal fundamental

### Definición formal

Para un fixture objetivo con kickoff `T`, una feature `F` solamente
puede depender de información disponible con timestamp estrictamente:

```
data_time < T
```

- El propio fixture objetivo queda excluido de las features.
- Los partidos posteriores al kickoff quedan excluidos.
- Cualquier dato posterior al kickoff (estadísticas, standings
  reconstruidas, snapshots `team_statistics`, H2H posteriores, etc.)
  no puede modificar una feature histórica ya construida.

### Diferencia `< T` vs `<= T`

| Boundary | Semántica                                | Resultado                                       |
|----------|------------------------------------------|-------------------------------------------------|
| `< T`    | estrictamente **antes** de `T`            | El fixture objetivo (cuyo kickoff == T) **nunca** entra en sus propias features. |
| `<= T`   | incluye el instante exacto `T`            | El fixture objetivo **sí** entraría: leakage directo del resultado propio. |

El sistema utiliza **boundary estricto `< T`** en todos los puntos
del feature math (`asof.py`, `standings.py`, `elo.py`, `h2h.py`,
`rest.py`, `xg.py`). El fixture objetivo adicionalmente se excluye
**por `exclude_fixture_id`** (defensa contra fixtures que comparten
`kickoff_time == T` por coincidencia).

### Imposibilidad de leakage temporal en el builder

El builder impone además `data_cutoff` (en UTC), un watermark
**independiente de `T`**: el dataset materializado jamás contiene
fixtures con `kickoff_time >= data_cutoff`. Esto protege contra
fixtures cuyo `status='finished'` puede haberse escrito **después** de
la fecha lógica del cutoff (condición de carrera del sync). El builder
verifica `end_date <= data_cutoff` y levanta
`InvalidBuildConfigError` si no se respeta.

---

## 2. As-of: la capa del boundary temporal

`app/features/asof.py` **centraliza** el boundary temporal. Es el
punto único donde la condición `data_time < T` se aplica sobre
filas ya cargadas; el feature math no debe reaplicar el filtro.

### Funciones exportadas

#### `load_finished_fixtures_as_of(session, *, competition_id, season_id, exclude_fixture_id=None)`

Carga **todos** los fixtures finalizados de una `(competition,
season)`. No aplica aquí el filtro `kickoff_time < T` — la lista
completa se devuelve asc por `kickoff_time` y el builder/feature
decide el cutoff aguas abajo (vía `fixtures_before`). El propietario
de "finished" es la propia DB: solo filas con `status='finished'` y
`home_goals IS NOT NULL` y `away_goals IS NOT NULL`.

#### `load_h2h_fixtures_as_of(session, *, home_team_id, away_team_id, competition_id=None)`

Carga todos los fixtures finalizados entre dos equipos (en cualquier
rol local/visitante), opcionalmente restringidos a una competencia.
Sin filtro temporal aquí — el cutoff lo aplica
`fixtures_before_unordered` dentro de `compute_h2h_features`.
Esto habilita **H2H cross-season** (ver §6).

#### `load_team_stats_as_of(session, *, competition_id, season_id, team_ids=None)`

Carga todos los snapshots de `team_statistics` para la temporada, asc
por `as_of_date`. El cutoff por snapshot (`as_of_date < kickoff.date()`)
se aplica en `app/features/xg.py::latest_team_stats_before`.

#### `fixtures_before(season_fixtures, *, kickoff, exclude_fixture_id=None)`

Slice **estricto `< T`** sobre una lista ascedente. Corta con `break`
en el primer `kickoff_time >= cutoff`. Devuelve newest-first (para
que los helpers de rolling tomen los "primeros N" como los "más
recientes N"). Excluye `exclude_fixture_id` por id.

#### `fixtures_before_unordered(...)`

Igual semántica que `fixtures_before` pero sobre listas no ordenadas
(usado por H2H y `load_h2h_fixtures_as_of`). Ordena internamente
desc por `kickoff_time`.

#### `team_history_before(*, team_id, season_fixtures, kickoff, exclude_fixture_id=None)`

Filtra solo los fixtures donde participa `team_id`, `kickoff_time <
T`, y excluye el target por id. Devuelve newest-first.

### Exclusión por `fixture_id`

`exclude_fixture_id` cae el fixture objetivo **por id interno**, no
por kickoff. Es la doble defensa:

- Boundary estricto `< T` ya excluye por tiempo cualquier fixture con
  `kickoff_time == T` (incluido el target).
- `exclude_fixture_id` cubre el caso en que dos fixtures compartan
  `kickoff_time < T_simultáneo` pero solo uno sea el target — por
  ejemplo, un matchday con varios partidos donde procesamos uno a la
  vez.

### Normalización de timestamps / timezone

`_ensure_aware(value)` normaliza un datetime naive a UTC
(`value.replace(tzinfo=timezone.utc)`). Los datetimes aware son
devueltos sin tocar. La comparación `kickoff_time < cutoff` solo es
segura cuando ambos lados son tz-aware — el as-of siempre normaliza el
cutoff antes de comparar.

El builder escribe `kickoff` en el CSV como ISO8601 tz-aware (UTC). El
loader rechaza fechas ISO sin timezone con `DatasetLoadError` ("loader
requires UTC tz-aware"), así que el boundary nunca se desplaza
silenciosamente por naive/UTC mismatch.

---

## 3. Forma / Goles / Rolling

### Ventanas 3 / 5 / 10

Las ventanas están en `form.py::_WINDOWS = (3, 5, 10)` (W/D/L y
points) y `goals.py::_WINDOWS = (5, 10)` (goles y medias).
`homeaway.py::_WINDOW = 5` (split por venue).

### Por qué las ventanas no incluyen el target

Conceptualmente:

```
target = fixture T
window = partidos con kickoff_time < T  (estricto)
```

No se utilizan:
- `T` (el fixture objetivo)
- `T+1`, `T+2`, … (futuros finalizados)
- ni siquiera un fixture simultáneo a `T` salvo para ELO (que tiene su
  propio manejo documentado en §5).

Cada feature family filtra con `team_history_before` (vía
`asof.fixtures_before`) **antes** de pasar la lista a las rolling
helpers. Las rolling helpers (`rolling_sum`, `rolling_mean`,
`rolling_count`, `rolling_rate`) son **puras**: no re-filtran, no
comprueban boundary. El boundary vive en un solo sitio.

### Ventanas parciales y la política `None` vs `0`

| Caso                                     | Salida                    |
|------------------------------------------|---------------------------|
| `len(history) < N` (ventana incompleta)  | `None` (missing)          |
| `len(history) == N` y todos los items son `0` o `None` y la operación suma | `0` (math-zero) |
| `len(history) == N` y todos los items son `None` en `rolling_mean` | `None` (no es posible en `FixtureRow.goals_for`) |

- `None` = **ausencia de datos**. El builder lo serializa como `""`
  (string vacío en el CSV).
- `0` = **valor matemáticamente cero**. El builder lo serializa como
  `"0"`.
- El loader **nunca** convierte `""` en `0`. La distinción se
  preserva end-to-end.
- `rolling_count` **no** devuelve `None`: los counts tienen una base
  natural `0`. La family de form nevertheless devuelve `None` si la
  ventana es parcial (en `form.py:65-72`) porque la documentación
  requiere ventana completa.

### Tests relevantes

- `test_rolling.py` — verificación directa de las primitivas
  (windows parciales, math-zero, etc.).
- `test_anti_leakage_form_goals.py` — futuros fixtures inyectados NO
  cambian form/goals; target excluido por id; assembler determinista.

---

## 4. Standings as-of

### Standings no se obtienen de una clasificación actual

Las features de standings **no** consumen la tabla `standings` del
proveedor ni la respuesta de `API-Football /standings`. Se
reconstruyen desde `FixtureRow` de la misma `(competition, season)`
estrictamente antes de `T`.

### Reconstrucción

Para cada fixture finalizado con `kickoff_time < T`:

```
points(team)        = 3 * wins  + 1 * draws
goals_for(team)     = Σ fx.goals_for(team)
goals_against(team) = Σ fx.goals_against(team)
goal_difference     = goals_for - goals_against
```

Reglas:
- victoria = 3 puntos
- empate = 1 punto
- derrota = 0 puntos

### Sorting determinístico real

La posición de cada equipo es su index (1-based) tras ordenar por la
clave:

```
(-points, -goal_difference, -goals_for, team_id ASC)
```

`team_id ASC` es el tiebreaker determinista — las ligas reales usan
H2H / fair-play que no podemos reconstruir desde fixtures solos.

### Exclusiones

- El fixture objetivo (`exclude_fixture_id`) **no** entra a la tabla
  reconstruida, incluso si su `kickoff_time < T` (no debería ser
  posible porque `kickoff_time == T` y boundary es estricto, pero la
  exclusión por id es defensa).
- Cualquier fixture posterior al kickoff queda fuera automáticamente
  por `kickoff_time < T`.

### Tests relevantes

- `test_anti_leakage_standings.py` — inyección de fixture futuro no
  cambia standings as-of; fixture objetivo excluido por id.

---

## 5. Elo temporal

### Constantes reales del código (`app/features/elo.py`)

| Constante        | Valor    | Significado                                          |
|------------------|----------|------------------------------------------------------|
| `BASE_RATING`    | `1500.0` | Rating inicial de todo equipo al entrar al stream.   |
| `DEFAULT_K`      | `20.0`   | Coeficiente de desarrollo (FIFA clásico).            |
| `DEFAULT_HFA`    | `65.0`   | Home-field advantage en puntos de rating (~0.5 goles). |

### Expected score

Con `A` local y `B` visitante, `HFA` se suma al rating del local:

```
E_A = 1 / (1 + 10 ** ((R_B - (R_A + HFA)) / 400))
```

### Actualización post-match

```
R_A_new = R_A + K * (actual_A - E_A)
R_B_new = R_B + K * (actual_B - (1 - E_A))
```
con `actual_A/B ∈ {1.0, 0.5, 0.0}` según victoria/empate/derrota.

### Orden temporal y snapshot pre-match

`elo_stream` procesa los fixtures de la temporada **asc por
`kickoff_time`**, y para cada fixture:

1. **Emite** un `EloPreMatch` con los ratings **actuales** (sin fold).
2. **Fold**: actualiza los ratings de beide equipos con el resultado
   del fixture.

Por lo tanto `Y[fx].home_elo` es el rating **un instante antes** del
kickoff de `fx`, sin contar su propio resultado.

### Exclusión del target

`compute_elo_features` (líneas 183-203 de `elo.py`):

1. Filtra `season_fixtures` con `fx.kickoff_time < fixture.kickoff_time`
   **y** `fx.fixture_id != fixture.fixture_id`.
2. Ordena asc.
3. **Reinyecta** el target al final del stream.
4. El stream emite el snapshot pre-match del target **antes** de fold
   su propio resultado.

El rating pre-match del target no se ve afectado por su propio
resultado (ej. un 10-0 no sube el rating pre-match del local).

### ⚠ Limitación: fixtures del mismo matchday

`elo_stream` es single-pass asc por `kickoff_time`. Cuando dos
fixtures A y B del mismo matchday (digamos, A.kickoff == B.kickoff
exactamente) se procesan:

- A se procesa primero: emite su pre-match sin contar B; fold de A.
- B se procesa después: emite su pre-match **con A ya foldeado**.

Este es el comportamiento actual. **No** se implementa un "matchday
freeze" que congele todos los pre-match del matchday antes del primer
fold. Cambiar esto requeriría agrupar por `kickoff_time` y diferir los
folds hasta terminar el matchday — bump obligatorio de
`feature_definition_version`.

Si los dos fixtures del mismo matchday **no** comparten equipo, el
efecto es nulo (el fold de A no toca los ratings de los equipos de
B). Si comparten equipo (un equipo juega dos veces el mismo día),
B sí vería el resultado de A foldeado — caso patológico hoy.

No ocultar: el orden de procesamiento dentro de un mismo
`kickoff_time` es determinista por el orden en `season_fixtures`
(asc por `fixture_id` begged por el builder — el builder ordena
ejemplos por `(kickoff, fixture_id)` antes de los stream).

### Tests relevantes

- `test_anti_leakage_elo.py` — inyección de resultado futuro no
  cambia Elo pre-match; el fixture objetivo no entra en su propio Elo
  por id.

---

## 6. H2H cross-season

### H2H es CROSS-SEASON

H2H puede utilizar enfrentamientos de **temporadas anteriores**. No
está limitado a la temporada actual del fixture objetivo.

Implementación:

- El builder llama `load_h2h_fixtures_as_of(session, home=…,
  away=…)` que trae **todos** los `Fixture` finalizados entre los dos
  equipos (sin importar `season_id`), en cualquier asignación de
  local/visitante (`OR` en el WHERE).
- El builder cachea este universo por par `(low_id, high_id)` para
  reutilizarlo en todos los fixtures del mismo par a lo largo del
  build.
- El assembler lo reenvía como `h2h_fixtures` a
  `compute_h2h_features`.

### Boundary temporal

El cutoff sigue siendo estricto:

```
kickoff_h2h < kickoff_target
```

`compute_h2h_features` filtra via `fixtures_before_unordered` que
aplica `kickoff_time < T`. Por lo tanto:

- **nunca** se utilizan partidos posteriores al kickoff objetivo;
- **nunca** se incluye el propio fixture objetivo
  (`exclude_fixture_id`);
- **nunca** se lee información futura.

### MIN_SAMPLE y política de missing

`MIN_SAMPLE = 3` en `h2h.py`. Si la muestra de H2H válida
(`kickoff_time < T`, equipo par, target excluido) tiene `sample <
3`:

- `h2h_sample_size` se publica tal cual (cardinalidad real, **no** se
  tops ni se imputa).
- `h2h_home_wins`, `h2h_away_wins`, `h2h_draws`,
  `h2h_total_goals_mean` se publican como `None` con un
  `missing_report` que documenta `"only {sample} H2H fixtures
  before T (< MIN_SAMPLE=3)"`.

No se inventa muestra ni se convierte `None` en `0`.

### Fallback para callers externos

`build_example(..., h2h_fixtures=None)` sigue siendo válido: en ese
caso `compute_h2h_features` recibe `season_fixtures` (mismísima
temporada) como fuente H2H. Esto es un fallback para callers sin el
bulk loader cross-season (tests especiales, uso runtime Fase 5+). La
protección anti-leakage se preserva igual (`< T` y `exclude_id`),
solo cambia el universo semántico (mismísima temporada vs
cross-season).

El **builder** de Phase 4 **nunca** usa ese fallback — siempre pasa
`h2h_fixtures` poblado. Si una feature path futura invoca
`build_example` sin `h2h_fixtures`, debe ser consciente de la
limitación semántica.

### Tests relevantes

- `test_anti_leakage_h2h.py` — los 7 escenarios requeridos:
  cross-season anterior valido, fixture posterior no utilizado,
  fixture objetivo excluido, inyección futura no cambia features,
  missing policy, determinismo, builder pasa H2H real (vía assembler).

---

## 7. Home/Away

### Partidos usados exclusivamente anteriores al kickoff

`compute_homeaway_features` (en `homeaway.py`):

1. `history = team_history_before(team_id, season_fixtures,
   kickoff, exclude_fixture_id)` — strict `< T`, target excluido.
2. Slice secundario por cancha: solo los N donde el equipo fue local
   (para features `home_home_*`) o visitante (`away_away_*`).
3. `rolling_sum(sliced, 5, ...)`.

### Ausencia de historial suficiente

- Si `len(sliced) < 5` → feature a `None` con `missing_report`.
- Si `len(sliced) >= 5` y el agregado es `0` → feature a `0` (math-zero).

### Tests relevantes

- `test_anti_leakage_homeaway.py` — inyección de fixture futuro no
  cambia las split features (local del local, visitante del
  visitante).

### ⚠ Limitación igual a form/goals

El slice secundario por cancha puede dejar pocas muestras al inicio de
la temporada (un equipo puede haber jugado 8 partidos pero solo 2
como local). En ese caso `home_home_points_last_5` queda `None` — no
hay fallback a un universo más amplio. Es la política documentada.

---

## 8. Rest

### Determinación del "partido anterior"

`compute_rest_features` (en `rest.py`):

```
history = team_history_before(team_id, season_fixtures, kickoff,
                              exclude_fixture_id)
last = history[0]   # newest-first por asof.team_history_before
delta = kickoff - last.kickoff_time
days  = int(delta.total_seconds() // 86400)
```

- `last` es el fixture **más reciente** del equipo con
  `kickoff_time < T`.
- El target se excluye por id (no puede ser su propio "partido
  anterior").
- Un fixture futuro no desplaza al "último" — boundary `< T`.

### Missing

- `history` vacío → `None` y `missing = "no previous fixture before
  T"`.
- **No** se devuelve `0`: `0` está reservado para el caso válido de
  dos partidos el mismo día (`delta == 0`).

### ⚠ Caveat detectado en FEATURES.md (§3.6)

El builder actualmente pasa a `compute_rest_features` el universo
`season_fixtures` restringido a `(competition_id, season_id)` del
fixture objetivo. Esto significa:

- Si el primer partido del equipo en esa temporada es el propio
  target → `rest_days = None`.
- La duración del offseason (descanso desde el último partido de la
  temporada anterior) **no** está capturada por la feature actual.

Ampliar a cross-competition/season requeriría que el builder pase un
universo más amplio (igual que ya hace H2H). El cambio sería un bump
obligatorio de `feature_definition_version`. No está implementado en
Fase 4.

### Tests relevantes

- `test_anti_leakage_rest.py` — fixture futuro no cambia rest_days;
  no-history ⇒ `None` (no `0`); mismo día ⇒ `0` (math-zero, no
  missing).

---

## 9. xG

### Solo lo realmente implementado

Phase 4 **no** implementa xG por partido. La tabla `fixtures` no
persiste columnas `xg_home` / `xg_away`. El único xG disponible está
en el snapshot estacional `team_statistics`, cuyos campos `xg` /
`xga` son **acumulativos** a la fecha `as_of_date`.

Por eso las features se llaman `*_season_asof`, no
`*_last_5` / `*_last_10`. El nombre mismo es parte de la garantía
documental: prevenir que un consumidor confunda esto con un rolling
per-match.

### Fuente

`team_statistics` snapshots cargados por
`load_team_stats_as_of`. Cada snapshot tiene `(team_id,
competition_id, season_id, as_of_date, xg, xga)`. El builder pasa
todos los snapshots de la temporada al assembler; éste los reenvía
a `compute_xg_features`.

### Boundary temporal

`latest_team_stats_before` (en `xg.py`):

```
kd         = kickoff.date()
eligible   = [r for r in stats_rows if r.team_id==team_id and r.as_of_date < kd]
latest     = max(eligible, key=lambda r: r.as_of_date)
```

`as_of_date < kickoff.date()` — **estricto, también excluye el mismo
día**. Esto protege contra un snapshot sincronizado el día del
partido (`as_of_date == kickoff.date()`) que contendría información
posterior al kickoff.

### Agregación

No hay rolling ni window — se toma el snapshot más reciente
estrictamente anterior al kickoff. Si solo hay snapshots en/after
`kickoff.date()`, la feature es `None`.

### Missing — dos niveles

1. **No existe snapshot** con `as_of_date < kickoff.date()` →
   `None` y `missing = "no team_statistics snapshot strictly before
   kickoff date"`.
2. **El snapshot existe pero `xg`/`xga` es `NULL`** (el proveedor no
   devolvió esos campos) → `None` y `missing = "snapshot exists but
   xG is NULL (provider did not return)"`.

### Diferencia entre `0` y `NULL`

- `xg = 0.0` es teóricamente posible si el snapshot reporta cero xG
  acumulado (un equipo queされるzó ningún disparo en N partidos).
- `xg = NULL` significa "el proveedor no devolvió el valor" — distinto
  semánticamente de "el valor es cero".

El builder escribe ambos como difieren en disco: `0` se serializa
`"0"`, `NULL` se serializa `""`. El loader distingue igual.

### Tests relevantes

- `test_anti_leakage_xg.py` — snapshot posterior no contribuye; mismo
  día excluido; missing dos niveles; determinismo; tz-aware kickoff
  usa `.date()`.

---

## 10. Targets

Targets definidos en `app/features/example.py::TARGET_NAMES`:

| Target        | Definición                                                 |
|---------------|------------------------------------------------------------|
| `home_win`    | `1` si `home_goals > away_goals`, `0` en otro caso.        |
| `draw`        | `1` si `home_goals == away_goals`, `0` en otro caso.        |
| `away_win`    | `1` si `away_goals > home_goals`, `0` en otro caso.         |
| `home_goals`  | `fixture.home_goals` (post-match).                         |
| `away_goals`  | `fixture.away_goals` (post-match).                          |

### Por qué no pueden entrar en las features

- El assembler (`assembler.py:46-160`) construye `features` primero
  llamando a cada familia con `fixture` solo como fuente de los
  cuatro team_ids, `kickoff`, y `exclude_fixture_id`. **No** pasa
  `home_goals`/`away_goals` a ninguna familia.
- Después construye `targets` en un dict **separado**
  (`_build_targets(fixture)`). No hay aliasing entre `features` y
  `targets` — son dos contenedores distintos en el `HistoricalMatchExample`.
- Las familias de features (`form`, `goals`, `h2h`, etc.) reciben
  `FixtureRow` (que sí tiene `home_goals`/`away_goals`), pero la
  mecánica siempre filtra primero por `team_history_before` /
  `fixtures_before` — `exclude_fixture_id` cae el target antes de
  que cualquier rolling lo toque.

> **Los targets nunca pueden participar en la construcción de las
> features pre-match.** Son post-match y se utilizan después para
> evaluar o entrenar modelos (Fase 5+), nunca para construir una
> observación pre-match.

### Tests relevante

- `test_anti_leakage_form_goals.py:217-239` (`test_assembler_is_deterministic`)
  explicita: re-builder el mismo ejemplo dos veces genera el mismo
  dict de features (`ex1.features == ex2.features`) y targets es
  disjunto de features (`set(targets).isdisjoint(set(features))`).
- `test_anti_leakage_form_goals.py:108-147`
  (`test_target_fixture_excluded_from_window`) demuestra que el
  target's own `7-0` **no** entra a la ventana de goles del target
  cuando se pasa `exclude_fixture_id=42`.

---

## 11. Ejemplos de leakage y mecanismos que los evitan

### Ejemplo 1 — Usar el resultado del propio fixture

**Anti-pattern**: leer `home_goals` / `away_goals` del target en una
feature (ej. "executated home_goals_for_last_5 = home_goals").

**Mecanismo que lo evita**:
- `exclude_fixture_id` cae el target por id antes del rolling.
- El target compartió el dict con `_build_targets` — no se pasa al
  feature math como tal, el assembler pasa por separado.

### Ejemplo 2 — Usar un partido de la semana siguiente para calcular forma

**Anti-pattern**: calcular `home_wins_last_5` incluyendo fixtures de
la semana siguiente al kickoff del target.

**Mecanismo que lo evita**:
- `team_history_before` aplica `kickoff_time < T` estricto.
- `fixtures_before` corta con `break` en el primer `kickoff_time >=
  cutoff`; ningún fixture posterior entra a la lista.
- `test_anti_leakage_form_goals.py::test_form_features_ignore_future_fixture`
  verifica explícitamente.

### Ejemplo 3 — Usar standings actuales para un partido histórico

**Anti-pattern**: leer `standings` de API-Football para un fixture de
2023-09-15 (la respuesta actual refleja la temporada completa ya
jugada).

**Mecanismo que lo evita**:
- `standings.py` NO importa ni usa una tabla `standings`; reconstruye
  la tabla desde `FixtureRow` con `kickoff_time < T`.
- `test_anti_leakage_standings.py::test_future_fixture_does_not_change_standings_as_of`
  inyecta un futuro finalizado y asevera que la tabla no cambia.

### Ejemplo 4 — Usar el xG del partido objetivo

**Anti-pattern**: leer `xg` del target como "xG pre-match feature"
(`xg_for_last_5` = xg del target).

**Mecanismo que lo evita**:
- La tabla `fixtures` no persiste `xg_home`/`xg_away`. No hay
  lugar de donde leer el xG del target — físicamente imposible
  hoy.
- `xg.py` consume `team_statistics` (snapshot estacional) con
  `as_of_date < kickoff.date()` estricto. Un snapshot del día del
  partido o posterior se excluye.

### Ejemplo 5 — Actualizar Elo con el resultado del target antes de extraer su rating pre-match

**Anti-pattern**: processar el target, `fold` su `10-0`, luego
reportar `home_elo_pre_match` (ahora incluye +20 rating points).

**Mecanismo que lo evita**:
- `elo_stream` emite el snapshot **antes** de fold el resultado.
- `compute_elo_features` filtra el target por `fixture_id != target.fixture_id`
  antes de pasar al stream, y luego lo reinyecta al final para
  capturar su snapshot pre-match específicamente.
- `test_anti_leakage_elo.py::test_target_fixture_excluded_from_own_elo`
  verifica con un target `10-0` que el rating pre-match no refleja ese
  resultado.

### Ejemplo 6 — Usar H2H posterior al partido objetivo

**Anti-pattern**: para un fixture del 2024-09-01, incluir en H2H un
partido del 2024-12-25 entre los mismos equipos.

**Mecanismo que lo evita**:
- `h2h.py::compute_h2h_features` filtra via
  `fixtures_before_unordered` que aplica `kickoff_time < T`.
- `exclude_fixture_id` cae el target por id.
- `test_anti_leakage_h2h.py::test_h2h_after_kickoff_not_used` y
  `.*_adding_future_fixture_does_not_change_h2h` aseveran esto.

---

## 12. Tests anti-leakage existentes

**Solo tests que existen en el repositorio**. No se inventa cobertura.

### As-of

- `backend/app/tests/unit/test_asof.py`:
  - `test_fixtures_before_is_strict` — boundary es `<`, no `<=`.
  - `test_fixtures_before_excludes_target_fixture_id` — target te
    cae por id.
  - `test_fixtures_before_returns_newest_first` — orden newest-first
    para rolling.
  - `test_fixtures_before_unordered_skips_target` — variante
    no ordenada (H2H).
  - `test_team_history_before_filters_team_and_time` — per-team +
    per-time.
  - `test_fixtures_before_tz_naive_cutoff_is_normalised` — naive→UTC.

### Rolling / form / goals

- `backend/app/tests/unit/test_rolling.py`:
  - `test_rolling_sum_excludes_items_past_n`
  - `test_rolling_sum_returns_none_for_partial_window`
  - `test_rolling_mean_skips_per_sample_none`
  - `test_rolling_mean_returns_none_when_all_cells_none`
  - `test_rolling_count_returns_zero_for_empty_window_not_none`
  - `test_rolling_rate_returns_none_for_partial_window`
  - `test_window_size_is_min_of_n_and_len`

- `backend/app/tests/unit/test_anti_leakage_form_goals.py`:
  - `test_form_features_ignore_future_fixture`
  - `test_goals_features_ignore_future_fixture`
  - `test_target_fixture_excluded_from_window`
  - `test_window_size_10_excludes_target`
  - `test_assembler_is_deterministic` — features/targets disjuntos.

### Standings

- `backend/app/tests/unit/test_anti_leakage_standings.py`:
  - `test_future_fixture_does_not_change_standings_as_of`
  - `test_target_fixture_excluded_from_pre_match_standings`

### Elo

- `backend/app/tests/unit/test_anti_leakage_elo.py`:
  - `test_future_result_does_not_change_elo_pre_match`
  - `test_target_fixture_excluded_from_own_elo`

### H2H

- `backend/app/tests/unit/test_anti_leakage_h2h.py`:
  - `test_h2h_from_previous_season_is_used` (cross-season)
  - `test_h2h_after_kickoff_not_used`
  - `test_target_fixture_excluded_from_h2h`
  - `test_adding_future_fixture_does_not_change_h2h`
  - `test_no_prior_h2h_returns_missing_per_policy`
  - `test_insufficient_h2h_returns_missing_with_real_count`
  - `test_h2h_features_are_deterministic`
  - `test_h2h_features_order_independent_under_permutation`
  - `test_assembler_uses_h2h_fixtures_argument_when_provided` —
    verifica que el builder ya no pasa `h2h_fixtures=None`.
  - `test_assembler_falls_back_to_season_fixtures_when_h2h_none` —
    documentación del fallback para callers externos.
  - `test_h2h_counts_pair_meetings_regardless_of_venue` — el OR de
    la consulta SQL se respeta en memoria (alternancia localía).

### Home/Away

- `backend/app/tests/unit/test_anti_leakage_homeaway.py`:
  - `test_future_match_does_not_change_home_split_features`
  - `test_future_match_does_not_change_away_split_features`

### Rest

- `backend/app/tests/unit/test_anti_leakage_rest.py`:
  - `test_future_match_does_not_change_rest_days`
  - `test_rest_days_returns_missing_when_no_prior_match`
  - `test_zero_rest_days_when_two_matches_same_day` — math-zero vs
    missing.

### xG

- `backend/app/tests/unit/test_anti_leakage_xg.py`:
  - `test_post_kickoff_snapshot_does_not_contribute_to_xg`
  - `test_latest_team_stats_before_strict_boundary`
  - `test_latest_team_stats_before_returns_none_when_only_post`
  - `test_no_snapshot_returns_missing_xg`
  - `test_snapshot_with_null_xg_is_missing_not_zero` — dos niveles de
    missing.
  - `test_xg_features_are_deterministic_under_permutation`
  - `test_xg_features_with_tz_aware_kickoff_use_date_part`

### Assembler / determinismo

- Cubierto por `test_anti_leakage_form_goals.py::test_assembler_is_deterministic`.
- No existe un test dedicado `test_assembler.py` aparte — no se
  inventa.

### Builder / dataset / loader / manifest (end-to-end)

- `backend/app/tests/unit/test_builder.py` — helpers + contract
  validation (rejects invalid configs, exact serialiser-then-loader
  roundtrip de H2Hales dummies).
- `backend/app/tests/unit/test_loader.py` — happy path, tipos,
  empty-cell→None, expected_version OK / mismatch, header mismatch,
  row_count mismatch, missing CSV, missing metadata, sha256 OK /
  mismatch, `validate_dataset` paths.
- `backend/app/tests/unit/test_manifest.py` — roundtrip, ISO8601,
  tz-aware, frozen dataclass, defaults `extras={}`.
- `backend/app/tests/unit/test_schema.py` — columnas canónicas,
  identidad vs features vs targets, sin duplicados, `EMPTY_CELL`.
- `backend/app/tests/integration/test_dataset_end_to_end.py` —
  PostgreSQL→builder→CSV+manifest→loader, H2H cross-season demostrado
  real, dataset vacío.

### Tests pendientes

No se inventan. Estado actual real:

- No existe test específico anti-leakage para **goals** por separado
  del suite de `test_anti_leakage_form_goals.py` (prové inyección
  futura para goals igualmente cubierto en el archivo conjunto).
- No existe test unitario anti-leakage aislado para cada individual
  form class (W, D, L, points) — cubiertos de manera conjunta.
- No existe test que verifique específicamente el **matchday freeze**
  de Elo o su ausencia (limitación documentada en §5; no se inventa
  cobertura).

---

## 13. Reglas para nuevas features

### Checklist para nuevas features

Antes de incorporar una feature al dataset, se debe responder
explícitamente:

1. **¿Cuál es su fuente?** Postgres, snapshot, API? — concretar.
2. **¿Qué timestamp determina su disponibilidad?**
   `kickoff_time`/`as_of_date`/`finished_at`/etc.
3. **¿Es estrictamente anterior al kickoff?** El timestamp debe ser
   `< T` (estricto), no `<= T`.
4. **¿Puede incluir accidentalmente el fixture objetivo?** Demostrar
   que `exclude_fixture_id` lo cae.
5. **¿Tiene una ventana temporal?** Si es rolling, ¿cuál es N y qué
   pasa si `len(history) < N`?
6. **¿Cómo trata missing data?** `None` vs `0`. Definir el contrato.
7. **¿Puede derivarse directa o indirectamente del target?**
   Demostrar aislamiento.
8. **¿Tiene un test anti-leakage?** Al menos uno que inyecte un dato
   futuro extremo (10-0, snapshot post-kickoff, fixture de la semana
   siguiente, etc.) y verifique que la feature no cambia.
9. **¿Está documentada en FEATURES.md?** Bump
   `feature_definition_version` + `dataset_version` si la fórmula
   cambia.

**Una feature que no pueda responder esto NO debe incorporating.**

### Adicionalmente

- Toda feature matemáticamente debe ser una función de la query
  Postgres + una semilla determinista. Nada de aleatoriedad.
- Si la feature se basa en una ventana rolling, debe usar las
  primitivas de `app/features/rolling.py` (no re-implementar).
- Si la feature consume `team_statistics`, debe pasar por
  `latest_team_stats_before` (`as_of_date < kickoff.date()`).
- Si la feature consume fixtures, debe consumirlos a través de
  `asof.py` (no directamente con una query SQLAlchemy dentro del
  feature math).

---

## 14. Limitaciones reales conocidas

Enumeradas, no escondidas:

1. **Elo matchday simultáneo**: dos fixtures del mismo matchday se
   process en secuencia asc; el segundo ve el resultado del primero
   ya foldeado. No hay freeze por matchday. Si dos fixtures del
   mismo `kickoff_time` no comparten equipo, el efecto es nulo. Si
   comparten equipo (patológico, un equipo juega dos veces el mismo
   día), el segundo ve el resultado del primero.
   *(ver §5)*

2. **Rest limitado al universo que carga el builder**: el builder pasa
   `season_fixtures` restringido a `(competition, season)` para
   `compute_rest_features`. La duración del offseason (descanso
   desde la temporada anterior) **no** se captura hoy. El primer
   partido del equipo en la temporada tiene `rest_days = None`.
   *(ver §8)*

3. **xG únicamente season-as-of**: no existe xG per-match. La fuente
   es el snapshot estacional `team_statistics`. Ventanas tipo
   `xg_for_last_5` no se implementan (y no pueden, porque la tabla
   `fixtures` no persiste xg). Cambiarlo requiere migrar el esquema
   PostgreSQL (out of scope Fase 4).
   *(ver §9)*

4. **H2H fallback para callers externos**: `build_example(h2h_fixtures=None)`
   sigue siendo válido y usa `season_fixtures` (mismísima temporada)
   como fuente H2H. Es un fallback para callers sin bulk loader
   (tests especializados, Fase 5+ runtime). El builder de Fase 4
   **nunca** usa ese fallback — siempre pasa `h2h_fixtures` poblado
   desde `load_h2h_fixtures_as_of`. La protección anti-leakage se
   preserva igual (boundary strict `< T` + `exclude_fixture_id`).
   *(ver §6)*

5. **xG same-day snapshot** excluido: el cutoff `as_of_date <
   kickoff.date()` rechaza el snapshot sincronizado el día del
   partido. Si el proveedor solo publica snapshots weekly, esta
   regla puede dejar `None` los primeros fixtures de la semana de la
   publicacion. Es política defensiva preferida a leakage.

6. **Standings tiebreaker**: el sorting `-points, -gd, -gf, team_id`
   no reproduce las reglas reales de competencias que usan H2H /
   fair-play como tiebreakers. La posición publicada es una
   aproximación determinista verosímil, no la posición oficial de la
   liga.

---

## 15. Verificación

- Documento comparado contra código real:
  - `app/features/asof.py`, `form.py`, `goals.py`, `h2h.py`,
    `homeaway.py`, `rest.py`, `standings.py`, `elo.py`, `xg.py`,
    `rolling.py`, `rows.py`, `assembler.py`, `example.py`.
  - `app/dataset/builder.py`, `loader.py`, `manifest.py`,
    `_schema.py`.
- Tests relevantes listados provienen de `backend/app/tests/unit/`
  (archivos verificados) y `backend/app/tests/integration/`.
- No se inventan tests.
- No se inventan garantías.
- No se modificó código.
- No se modificó FEATURES.md.
- No avanza a Fase 5.

### Discrepancias encontradas

Ninguna nueva más allá de las ya documentadas como "limitaciones
reales conocidas" en §14. El comportamiento del código y este
documento coinciden.
