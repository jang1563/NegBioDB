"""Tests for AZ-DREAM Challenge ETL module.

Tests synergy score parsing, combination CSV parsing, and loading.
"""

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from negbiodb_dc.dc_db import get_connection, run_dc_migrations
from negbiodb_dc.etl_dream import (
    load_dream_synergy,
    parse_dream_combination_csv,
    parse_dream_synergy_scores,
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


# ── Synergy scores parsing ───────────────────────────────────────────


class TestParseDreamSynergyScores:
    def test_tab_separated(self, tmp_path):
        content = textwrap.dedent("""\
            DRUG_A\tDRUG_B\tCELL_LINE\tSYNERGY_SCORE\tSYNERGY_SCORE_BLISS
            Erlotinib\tMK-2206\tMCF7\t-5.2\t-3.1
            Erlotinib\tAZD6244\tA549\t12.0\t8.5
        """)
        path = tmp_path / "scores.txt"
        path.write_text(content)

        df = parse_dream_synergy_scores(path)
        assert len(df) == 2
        assert "DRUG_A" in df.columns
        assert "LOEWE_SCORE" in df.columns
        assert "BLISS_SCORE" in df.columns

    def test_comma_separated(self, tmp_path):
        content = textwrap.dedent("""\
            DRUG_A,DRUG_B,CELL_LINE,SYNERGY_SCORE,SYNERGY_SCORE_BLISS
            DrugX,DrugY,HCT116,7.0,5.5
        """)
        path = tmp_path / "scores.csv"
        path.write_text(content)

        df = parse_dream_synergy_scores(path)
        assert len(df) == 1
        assert abs(df.iloc[0]["LOEWE_SCORE"] - 7.0) < 0.01

    def test_missing_columns_raises(self, tmp_path):
        content = "COL1,COL2\n1,2\n"
        path = tmp_path / "bad.txt"
        path.write_text(content)

        with pytest.raises(ValueError, match="Cannot find"):
            parse_dream_synergy_scores(path)


# ── Combination CSV parsing ──────────────────────────────────────────


class TestParseDreamCombinationCsv:
    def test_basic_parsing(self, tmp_path):
        content = textwrap.dedent("""\
            COMPOUND_A,COMPOUND_B,CELL_LINE_NAME,IC_TYPE,CONC_A,CONC_B,INHIBITION
            DrugA,DrugB,MCF7,combo,0.1,0.1,30
            DrugA,DrugB,MCF7,combo,1.0,0.1,40
            DrugA,DrugC,A549,combo,0.1,0.1,20
        """)
        csv_path = tmp_path / "dream_combo.csv"
        csv_path.write_text(content)

        df = parse_dream_combination_csv(csv_path)
        assert len(df) == 2  # Two unique combinations
        assert "DRUG_A" in df.columns


# ── DREAM synergy loading ────────────────────────────────────────────


class TestLoadDreamSynergy:
    def _seed_entities(self, conn):
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('Erlotinib')")
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('MK-2206')")
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('AZD6244')")
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
        df = pd.DataFrame({
            "DRUG_A": ["Erlotinib"],
            "DRUG_B": ["MK-2206"],
            "CELL_LINE": ["MCF7"],
            "LOEWE_SCORE": [-5.2],
            "BLISS_SCORE": [-3.1],
        })

        stats = load_dream_synergy(conn, df, compound_cache, cell_line_cache)
        assert stats["results_inserted"] == 1

    def test_source_db_is_az_dream(self, conn):
        compound_cache, cell_line_cache = self._seed_entities(conn)
        df = pd.DataFrame({
            "DRUG_A": ["Erlotinib"],
            "DRUG_B": ["MK-2206"],
            "CELL_LINE": ["MCF7"],
            "LOEWE_SCORE": [5.0],
            "BLISS_SCORE": [3.0],
        })

        load_dream_synergy(conn, df, compound_cache, cell_line_cache)
        source = conn.execute(
            "SELECT source_db FROM dc_synergy_results"
        ).fetchone()[0]
        assert source == "az_dream"

    def test_tier_is_bronze(self, conn):
        """DREAM provides dose-response matrices → always bronze."""
        compound_cache, cell_line_cache = self._seed_entities(conn)
        df = pd.DataFrame({
            "DRUG_A": ["Erlotinib"],
            "DRUG_B": ["MK-2206"],
            "CELL_LINE": ["MCF7"],
            "LOEWE_SCORE": [5.0],
            "BLISS_SCORE": [3.0],
        })

        load_dream_synergy(conn, df, compound_cache, cell_line_cache)
        tier = conn.execute(
            "SELECT confidence_tier FROM dc_synergy_results"
        ).fetchone()[0]
        assert tier == "bronze"

    def test_self_combination_skipped(self, conn):
        compound_cache, cell_line_cache = self._seed_entities(conn)
        df = pd.DataFrame({
            "DRUG_A": ["Erlotinib"],
            "DRUG_B": ["Erlotinib"],
            "CELL_LINE": ["MCF7"],
            "LOEWE_SCORE": [5.0],
            "BLISS_SCORE": [3.0],
        })

        stats = load_dream_synergy(conn, df, compound_cache, cell_line_cache)
        assert stats["skipped_self_combination"] == 1
        assert stats["results_inserted"] == 0

    def test_pair_normalization(self, conn):
        compound_cache, cell_line_cache = self._seed_entities(conn)
        df = pd.DataFrame({
            "DRUG_A": ["MK-2206"],
            "DRUG_B": ["Erlotinib"],
            "CELL_LINE": ["MCF7"],
            "LOEWE_SCORE": [5.0],
            "BLISS_SCORE": [3.0],
        })

        load_dream_synergy(conn, df, compound_cache, cell_line_cache)
        row = conn.execute(
            "SELECT compound_a_id, compound_b_id FROM dc_synergy_results"
        ).fetchone()
        assert row[0] < row[1]

    def test_auto_creates_unknown_drug(self, conn):
        compound_cache, cell_line_cache = self._seed_entities(conn)
        df = pd.DataFrame({
            "DRUG_A": ["NewDrug"],
            "DRUG_B": ["Erlotinib"],
            "CELL_LINE": ["MCF7"],
            "LOEWE_SCORE": [5.0],
            "BLISS_SCORE": [3.0],
        })

        stats = load_dream_synergy(conn, df, compound_cache, cell_line_cache)
        assert stats["results_inserted"] == 1
        assert "NewDrug" in compound_cache

    def test_loewe_used_for_classification(self, conn):
        """Loewe score should be used for synergy_class when available."""
        compound_cache, cell_line_cache = self._seed_entities(conn)
        df = pd.DataFrame({
            "DRUG_A": ["Erlotinib"],
            "DRUG_B": ["MK-2206"],
            "CELL_LINE": ["MCF7"],
            "LOEWE_SCORE": [-15.0],  # strongly_antagonistic by ZIP thresholds
            "BLISS_SCORE": [5.0],    # Would be additive
        })

        load_dream_synergy(conn, df, compound_cache, cell_line_cache)
        cls = conn.execute(
            "SELECT synergy_class FROM dc_synergy_results"
        ).fetchone()[0]
        assert cls == "strongly_antagonistic"

    def test_null_loewe_falls_back_to_bliss(self, conn):
        compound_cache, cell_line_cache = self._seed_entities(conn)
        df = pd.DataFrame({
            "DRUG_A": ["Erlotinib"],
            "DRUG_B": ["MK-2206"],
            "CELL_LINE": ["MCF7"],
            "LOEWE_SCORE": [None],
            "BLISS_SCORE": [7.0],  # synergistic
        })

        load_dream_synergy(conn, df, compound_cache, cell_line_cache)
        cls = conn.execute(
            "SELECT synergy_class FROM dc_synergy_results"
        ).fetchone()[0]
        assert cls == "synergistic"

    def test_multiple_cell_lines(self, conn):
        compound_cache, cell_line_cache = self._seed_entities(conn)
        df = pd.DataFrame({
            "DRUG_A": ["Erlotinib", "Erlotinib"],
            "DRUG_B": ["MK-2206", "MK-2206"],
            "CELL_LINE": ["MCF7", "A549"],
            "LOEWE_SCORE": [5.0, -8.0],
            "BLISS_SCORE": [3.0, -6.0],
        })

        stats = load_dream_synergy(conn, df, compound_cache, cell_line_cache)
        assert stats["results_inserted"] == 2
