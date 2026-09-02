"""End-to-end integration test of the Phase 4 dataset pipeline.

PostgreSQL
    ↓
builder
    ↓
dataset.csv
    ↓
metadata.json
    ↓
loader
    ↓
LoadedDataset

Verifies the round-trip:

* the builder produces a non-empty ``dataset.csv`` whose row order is
  deterministic (chronological by ``(kickoff, fixture_id)``),
* the builder writes ``metadata.json`` carrying the same
  ``row_count`` as the CSV,
* the manifest ``csv_sha256`` equals the SHA-256 recomputed over the
  on-disk bytes (re-validated by :func:`app.dataset.loader.validate_dataset`),
* the loader reads the CSV back into a :class:`LoadedDataset` with
  matching ``row_count``, ``features`` (per-row), ``targets``, and
  ``dataset_version``,
* the H2H universe is cross-season: a H2H fixture strictly before the
  target's kickoff from a PRIOR season proves the builder no longer
  passes ``h2h_fixtures=None``.

This test is marked ``integration`` because it requires a live
PostgreSQL instance with the Alembic ``0001_initial`` migration
applied (the ``session`` fixture skips automatically otherwise).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.dataset.builder import build_dataset
from app.dataset.loader import (
    LoadedDataset,
    load_dataset,
    validate_dataset,
)
from app.models import Fixture
from app.tests.integration.factories import Factories

pytestmark = pytest.mark.integration


async def test_end_to_end_builder_to_loader_roundtrip(session, tmp_path) -> None:
    factories = Factories(session)

    comp = await factories.competition(name="Premier League")
    season_a = await factories.season(
        competition_id=comp.id, year=2023, is_current=False,
    )
    season_b = await factories.season(
        competition_id=comp.id, year=2024, is_current=False,
    )
    home = await factories.team(name="Arsenal FC", external_id=501)
    away = await factories.team(name="Chelsea FC", external_id=502)
    other = await factories.team(name="Tottenham", external_id=503)

    # --- Season A (2023): one prior H2H + a couple of unrelated games ---
    h2h_prior = await factories.fixture(
        competition_id=comp.id, season_id=season_a.id,
        home_team_id=home.id, away_team_id=away.id,
        kickoff_time=datetime(2023, 11, 1, 15, 0, tzinfo=timezone.utc),
        status="finished",
        home_goals=2, away_goals=1,
    )
    await factories.fixture(
        competition_id=comp.id, season_id=season_a.id,
        home_team_id=home.id, away_team_id=other.id,
        kickoff_time=datetime(2023, 11, 15, 15, 0, tzinfo=timezone.utc),
        status="finished",
        home_goals=1, away_goals=0,
    )

    # --- Season B (2024): two finished fixtures in the window we'll build,
    # plus a future-finished fixture OUTSIDE the data_cutoff that must
    # NOT enter the dataset (anti-leakage at the builder level).
    fx1 = await factories.fixture(
        competition_id=comp.id, season_id=season_b.id,
        home_team_id=home.id, away_team_id=away.id,
        kickoff_time=datetime(2024, 8, 10, 15, 0, tzinfo=timezone.utc),
        status="finished", home_goals=3, away_goals=0,
    )
    # Second finished fixture in window.
    fx_out_of_window = await factories.fixture(
        competition_id=comp.id, season_id=season_b.id,
        home_team_id=away.id, away_team_id=other.id,
        kickoff_time=datetime(2024, 8, 17, 15, 0, tzinfo=timezone.utc),
        status="finished", home_goals=1, away_goals=1,
    )
    # Future-finished fixture strictly AFTER data_cutoff: must not appear.
    await factories.fixture(
        competition_id=comp.id, season_id=season_b.id,
        home_team_id=home.id, away_team_id=away.id,
        kickoff_time=datetime(2030, 1, 1, 15, 0, tzinfo=timezone.utc),
        status="finished", home_goals=10, away_goals=10,
    )

    # Flush all factories rows into the same outer transaction.
    await session.flush()

    # --- The build itself ---
    cutoff = datetime(2024, 12, 31, tzinfo=timezone.utc)
    start = datetime(2024, 8, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)

    out_dir = tmp_path / "v_end_to_end"
    csv_path, manifest = await build_dataset(
        session=session,
        competition_ids=[comp.id],
        season_ids=[season_a.id, season_b.id],
        start_date=start,
        end_date=end,
        data_cutoff=cutoff,
        dataset_version="v_end_to_end",
        output_path=out_dir,
    )

    # ----- builder output assertions -----
    # The cutoff < everything else rules out the 2030 fixture. The
    # data_cutoff also rules out any fixture strictly before 2024-08-01
    # via the window filter (start_date, end_date). Concretely:
    #   - The 2023 H2H fixture is OUT (before start_date).
    #   - The 2024-08-10 fixture IS IN.
    #   - The 2024-08-17 fixture IS IN.
    #   - The 2030 fixture is OUT (after data_cutoff).
    assert manifest.row_count == 2
    assert manifest.dataset_version == "v_end_to_end"
    # The start_date in the manifest = earliest kickoff in the dataset.
    assert manifest.start_date == datetime(2024, 8, 10, 15, 0, tzinfo=timezone.utc)
    # CSV exists + is non-empty (header + 2 rows).
    assert csv_path.is_file()
    csv_lines = csv_path.read_bytes().split(b"\r\n")
    # Filter empty trailing entry.
    csv_lines = [ln for ln in csv_lines if ln]
    assert len(csv_lines) == 3  # header + 2 data rows

    # ----- validate_dataset: SHA-256 + header shape -----
    validated = validate_dataset(out_dir)
    assert validated.row_count == 2
    assert validated.csv_sha256 == manifest.csv_sha256

    # ----- load_dataset round-trip -----
    loaded = load_dataset(
        out_dir, expected_version="v_end_to_end", verify_sha256=True,
    )
    assert isinstance(loaded, LoadedDataset)
    assert loaded.manifest.dataset_version == "v_end_to_end"
    assert loaded.manifest.row_count == 2
    assert len(loaded.rows) == 2

    # Deterministic chronological order by (kickoff, fixture_id).
    ids = [row.fixture_id for row in loaded.rows]
    fx1_db = (await session.execute(
        select(Fixture).where(Fixture.external_id == fx1.external_id)
    )).scalar_one()
    fx2_db = (await session.execute(
        select(Fixture).where(
            Fixture.external_id == fx_out_of_window.external_id
        )
    )).scalar_one()
    # The loader returns the INTERNAL fixture ids (BIGSERIAL).
    assert ids == sorted([fx1_db.id, fx2_db.id])

    # Same feature / target *names* present in every loaded row.
    from app.features.example import FEATURE_NAMES, TARGET_NAMES
    for row in loaded.rows:
        assert set(row.features.keys()) == set(FEATURE_NAMES)
        assert set(row.targets.keys()) == set(TARGET_NAMES)

    # Targets post-match round-trip. fx1 was 3-0 -> home_win=1, draw=0,
    # away_win=0, home_goals=3, away_goals=0.
    fx1_loaded = next(r for r in loaded.rows if r.fixture_id == fx1_db.id)
    assert fx1_loaded.targets["home_win"] == 1
    assert fx1_loaded.targets["draw"] == 0
    assert fx1_loaded.targets["away_win"] == 0
    assert fx1_loaded.targets["home_goals"] == 3
    assert fx1_loaded.targets["away_goals"] == 0


async def test_builder_passes_cross_season_h2h_to_assembler(session, tmp_path) -> None:
    """Demonstrates that the builder forwards the cross-season H2H
    universe (no more ``h2h_fixtures=None``). A target fixture whose
    only H2H prior lives in a DIFFERENT season must end up with a
    populated ``h2h_sample_size >= 1`` (specifically == 1 because
    only the 2023 H2H qualifies before the 2024 kickoff). Without the
    cross-season loader, sample size would have been 0 (same-season
    fallback yields nothing here).
    """
    factories = Factories(session)

    comp = await factories.competition(name="La Liga")
    season_2023 = await factories.season(
        competition_id=comp.id, year=2023, is_current=False,
    )
    season_2024 = await factories.season(
        competition_id=comp.id, year=2024, is_current=False,
    )
    home = await factories.team(name="Real Madrid", external_id=540)
    away = await factories.team(name="FC Barcelona", external_id=541)

    # --- 2023 H2H in season_2023 (eligible when season_2024 target is built) ---
    h2h_2023 = await factories.fixture(
        competition_id=comp.id, season_id=season_2023.id,
        home_team_id=home.id, away_team_id=away.id,
        kickoff_time=datetime(2023, 10, 15, 16, 0, tzinfo=timezone.utc),
        status="finished", home_goals=2, away_goals=2,
    )

    # --- 2024 target fixture (only one finished match in season_2024 ---
    # no same-season H2H available).
    target_fx = await factories.fixture(
        competition_id=comp.id, season_id=season_2024.id,
        home_team_id=home.id, away_team_id=away.id,
        kickoff_time=datetime(2024, 9, 1, 16, 0, tzinfo=timezone.utc),
        status="finished", home_goals=1, away_goals=1,
    )
    await session.flush()

    cutoff = datetime(2024, 12, 31, tzinfo=timezone.utc)

    out_dir = tmp_path / "v_h2h_cross_season"
    csv_path, manifest = await build_dataset(
        session=session,
        competition_ids=[comp.id],
        season_ids=[season_2024.id],
        start_date=datetime(2024, 8, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
        data_cutoff=cutoff,
        dataset_version="v_h2h_cross_season",
        output_path=out_dir,
    )
    assert manifest.row_count == 1

    loaded = load_dataset(out_dir, expected_version="v_h2h_cross_season")
    # The cross-season H2H from 2023 IS present in the H2H features,
    # proving that the builder passed a populated ``h2h_fixtures``
    # argument to the assembler (not ``None``).
    row = loaded.rows[0]
    assert row.features["h2h_sample_size"] == 1
    assert row.features["h2h_draws"] == 1
    assert row.features["h2h_home_wins"] is None  # below MIN_SAMPLE (3)


async def test_builder_empty_window_returns_zero_rows(session, tmp_path) -> None:
    """A window with no finished fixtures produces a 0-row dataset
    with a deterministically-shaped manifest. The builder must NOT
    raise; the loader must accept the empty CSV.
    """
    factories = Factories(session)
    comp = await factories.competition(name="Empty League")
    season = await factories.season(
        competition_id=comp.id, year=2024, is_current=False,
    )
    home = await factories.team(name="Empty FC")
    away = await factories.team(name="Void CF")
    # Only a SCHEDULED (not finished) fixture — won't be picked up.
    await factories.fixture(
        competition_id=comp.id, season_id=season.id,
        home_team_id=home.id, away_team_id=away.id,
        kickoff_time=datetime(2024, 9, 1, 16, 0, tzinfo=timezone.utc),
        status="scheduled",
    )
    await session.flush()

    cutoff = datetime(2024, 12, 31, tzinfo=timezone.utc)
    out_dir = tmp_path / "v_empty"
    csv_path, manifest = await build_dataset(
        session=session,
        competition_ids=[comp.id],
        season_ids=[season.id],
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
        data_cutoff=cutoff,
        dataset_version="v_empty",
        output_path=out_dir,
    )
    assert manifest.row_count == 0
    # Header-only CSV.
    assert csv_path.is_file()
    bytes_ = csv_path.read_bytes()
    # Header + (empty body) -> exactly one row of bytes (plus trailing CRLF).
    lines = [ln for ln in bytes_.split(b"\r\n") if ln]
    assert len(lines) == 1
    loaded = load_dataset(out_dir, expected_version="v_empty")
    assert loaded.manifest.row_count == 0
    assert len(loaded.rows) == 0
