"""Tests for NCI-ALMANAC ETL module.

Tests ComboScore classification, CSV parsing, and synergy loading.
"""

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from negbiodb_dc.dc_db import get_connection, run_dc_migrations
from negbiodb_dc.etl_almanac import (
    classify_combo_score,
    load_almanac_synergy,
    parse_almanac_csv,
)

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_dc"


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_dc.db"
    run_dc_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def conn(tmp_db):
    c = get_connection(tmp_db)
    yield c
    c.close()


# ── ComboScore classification ────────────────────────────────────────


class TestClassifyComboScore:
    def test_strongly_synergistic(self):
        assert classify_combo_score(60.0) == "strongly_synergistic"

    def test_synergistic(self):
        assert classify_combo_score(30.0) == "synergistic"

    def test_additive(self):
        assert classify_combo_score(0.0) == "additive"

    def test_antagonistic(self):
        assert classify_combo_score(-30.0) == "antagonistic"

    def test_strongly_antagonistic(self):
        assert classify_combo_score(-60.0) == "strongly_antagonistic"

    def test_none_returns_none(self):
        assert classify_combo_score(None) is None

    def test_nan_returns_none(self):
        assert classify_combo_score(float("nan")) is None

    def test_boundary_50(self):
        assert classify_combo_score(50.0) == "synergistic"

    def test_boundary_above_50(self):
        assert classify_combo_score(50.1) == "strongly_synergistic"

    def test_boundary_20(self):
        assert classify_combo_score(20.0) == "additive"

    def test_boundary_above_20(self):
        assert classify_combo_score(20.1) == "synergistic"

    def test_boundary_neg_20(self):
        assert classify_combo_score(-20.0) == "additive"

    def test_boundary_below_neg_20(self):
        assert classify_combo_score(-20.1) == "antagonistic"

    def test_boundary_neg_50(self):
        assert classify_combo_score(-50.0) == "antagonistic"

    def test_boundary_below_neg_50(self):
        assert classify_combo_score(-50.1) == "strongly_antagonistic"


# ── CSV parsing ──────────────────────────────────────────────────────


class TestParseAlmanacCsv:
    def test_basic_parsing(self, tmp_path):
        csv_content = textwrap.dedent("""\
            NSC1,NSC2,CELLNAME,PANEL,SCORE
            123,456,MCF7,Breast,25.5
            123,456,MCF7,Breast,30.0
            123,789,A549,Lung,-15.0
        """)
        csv_path = tmp_path / "almanac.csv"
        csv_path.write_text(csv_content)

        df = parse_almanac_csv(csv_path)
        assert len(df) == 2  # Grouped: (123,456,MCF7) and (123,789,A549)
        assert "DRUG_A" in df.columns
        assert "DRUG_B" in df.columns
        assert "CELLNAME" in df.columns

    def test_mean_combo_score(self, tmp_path):
        csv_content = textwrap.dedent("""\
            NSC1,NSC2,CELLNAME,SCORE
            123,456,MCF7,20.0
            123,456,MCF7,30.0
        """)
        csv_path = tmp_path / "almanac.csv"
        csv_path.write_text(csv_content)

        df = parse_almanac_csv(csv_path)
        assert len(df) == 1
        assert abs(df.iloc[0]["COMBO_SCORE"] - 25.0) < 0.01

    def test_concentration_count(self, tmp_path):
        csv_content = textwrap.dedent("""\
            NSC1,NSC2,CELLNAME,SCORE
            123,456,MCF7,10
            123,456,MCF7,20
            123,456,MCF7,30
            123,456,MCF7,40
            123,456,MCF7,50
            123,456,MCF7,60
            123,456,MCF7,70
            123,456,MCF7,80
            123,456,MCF7,90
        """)
        csv_path = tmp_path / "almanac.csv"
        csv_path.write_text(csv_content)

        df = parse_almanac_csv(csv_path)
        assert df.iloc[0]["N_CONC"] == 9


# ── Almanac synergy loading ──────────────────────────────────────────


class TestLoadAlmanacSynergy:
    def _seed_entities(self, conn):
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('123')")
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('456')")
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('789')")
        conn.execute("INSERT INTO cell_lines (cell_line_name) VALUES ('MCF7')")
        conn.execute("INSERT INTO cell_lines (cell_line_name) VALUES ('A549')")
        conn.commit()

        compound_cache = {
            row[1]: row[0]
            for row in conn.execute("SELECT compound_id, drug_name FROM compounds")
        }
        cell_line_cache = {
            row[1]: row[0]
            for row in conn.execute("SELECT cell_line_id, cell_line_name FROM cell_lines")
        }
        return compound_cache, cell_line_cache

    def test_basic_insert(self, conn):
        compound_cache, cell_line_cache = self._seed_entities(conn)
        agg_df = pd.DataFrame({
            "DRUG_A": ["123"],
            "DRUG_B": ["456"],
            "CELLNAME": ["MCF7"],
            "COMBO_SCORE": [25.5],
            "N_CONC": [9],
        })

        stats = load_almanac_synergy(conn, agg_df, compound_cache, cell_line_cache)
        assert stats["results_inserted"] == 1

    def test_self_combination_skipped(self, conn):
        compound_cache, cell_line_cache = self._seed_entities(conn)
        agg_df = pd.DataFrame({
            "DRUG_A": ["123"],
            "DRUG_B": ["123"],
            "CELLNAME": ["MCF7"],
            "COMBO_SCORE": [10.0],
            "N_CONC": [9],
        })

        stats = load_almanac_synergy(conn, agg_df, compound_cache, cell_line_cache)
        assert stats["skipped_self_combination"] == 1
        assert stats["results_inserted"] == 0

    def test_tier_assignment_bronze(self, conn):
        """9+ concentrations → bronze tier."""
        compound_cache, cell_line_cache = self._seed_entities(conn)
        agg_df = pd.DataFrame({
            "DRUG_A": ["123"],
            "DRUG_B": ["456"],
            "CELLNAME": ["MCF7"],
            "COMBO_SCORE": [10.0],
            "N_CONC": [9],
        })

        load_almanac_synergy(conn, agg_df, compound_cache, cell_line_cache)
        tier = conn.execute(
            "SELECT confidence_tier FROM dc_synergy_results"
        ).fetchone()[0]
        assert tier == "bronze"

    def test_tier_assignment_copper(self, conn):
        """< 9 concentrations → copper tier."""
        compound_cache, cell_line_cache = self._seed_entities(conn)
        agg_df = pd.DataFrame({
            "DRUG_A": ["123"],
            "DRUG_B": ["456"],
            "CELLNAME": ["MCF7"],
            "COMBO_SCORE": [10.0],
            "N_CONC": [3],
        })

        load_almanac_synergy(conn, agg_df, compound_cache, cell_line_cache)
        tier = conn.execute(
            "SELECT confidence_tier FROM dc_synergy_results"
        ).fetchone()[0]
        assert tier == "copper"

    def test_source_db_is_nci_almanac(self, conn):
        compound_cache, cell_line_cache = self._seed_entities(conn)
        agg_df = pd.DataFrame({
            "DRUG_A": ["123"],
            "DRUG_B": ["456"],
            "CELLNAME": ["MCF7"],
            "COMBO_SCORE": [10.0],
            "N_CONC": [9],
        })

        load_almanac_synergy(conn, agg_df, compound_cache, cell_line_cache)
        source = conn.execute(
            "SELECT source_db FROM dc_synergy_results"
        ).fetchone()[0]
        assert source == "nci_almanac"

    def test_auto_creates_unknown_drug(self, conn):
        """Drugs not in cache get auto-created."""
        compound_cache, cell_line_cache = self._seed_entities(conn)
        agg_df = pd.DataFrame({
            "DRUG_A": ["NEWDRUG"],
            "DRUG_B": ["456"],
            "CELLNAME": ["MCF7"],
            "COMBO_SCORE": [5.0],
            "N_CONC": [9],
        })

        stats = load_almanac_synergy(conn, agg_df, compound_cache, cell_line_cache)
        assert stats["results_inserted"] == 1
        assert "NEWDRUG" in compound_cache

    def test_pair_normalization(self, conn):
        compound_cache, cell_line_cache = self._seed_entities(conn)
        # Ensure 456 has higher compound_id than 123
        agg_df = pd.DataFrame({
            "DRUG_A": ["456"],
            "DRUG_B": ["123"],
            "CELLNAME": ["MCF7"],
            "COMBO_SCORE": [10.0],
            "N_CONC": [9],
        })

        load_almanac_synergy(conn, agg_df, compound_cache, cell_line_cache)
        row = conn.execute(
            "SELECT compound_a_id, compound_b_id FROM dc_synergy_results"
        ).fetchone()
        assert row[0] < row[1]
