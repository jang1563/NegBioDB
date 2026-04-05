"""Tests for DrugComb ETL module.

Tests parsing drug/cell line identifiers, tier assignment,
synergy data loading, and auto-entity creation.
"""

import textwrap
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from negbiodb_dc.dc_db import get_connection, run_dc_migrations
from negbiodb_dc.etl_drugcomb import (
    _assign_tier,
    load_cell_lines,
    load_compounds,
    load_drugcomb_synergy,
    parse_cell_line_identifiers,
    parse_drug_identifiers,
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


# ── Tier assignment ──────────────────────────────────────────────────


class TestAssignTier:
    def test_full_matrix(self):
        tier, evidence = _assign_tier(9, True)
        assert tier == "bronze"
        assert evidence == "dose_response_matrix"

    def test_small_matrix(self):
        tier, evidence = _assign_tier(4, True)
        assert tier == "bronze"
        assert evidence == "dose_response_matrix"

    def test_minimal_matrix(self):
        tier, evidence = _assign_tier(3, True)
        assert tier == "bronze"
        assert evidence == "dose_response_matrix"

    def test_single_concentration(self):
        tier, evidence = _assign_tier(1, False)
        assert tier == "copper"
        assert evidence == "single_concentration"

    def test_many_conc_but_no_matrix(self):
        tier, evidence = _assign_tier(10, False)
        assert tier == "copper"
        assert evidence == "single_concentration"

    def test_below_threshold_matrix(self):
        tier, evidence = _assign_tier(2, True)
        assert tier == "copper"
        assert evidence == "single_concentration"


# ── Drug identifier parsing ──────────────────────────────────────────


@pytest.mark.skipif(
    not pd.io.common.import_optional_dependency("openpyxl", errors="ignore"),
    reason="openpyxl not installed",
)
class TestParseDrugIdentifiers:
    def test_basic_parsing(self, tmp_path):
        xlsx_path = tmp_path / "drugs.xlsx"
        df = pd.DataFrame({
            "drug_name": ["Aspirin", "Ibuprofen"],
            "pubchem_cid": [2244, 3672],
            "inchikey": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N", None],
            "canonical_smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"],
        })
        df.to_excel(xlsx_path, index=False)

        records = parse_drug_identifiers(xlsx_path)
        assert len(records) == 2
        assert records[0]["drug_name"] == "Aspirin"
        assert records[0]["pubchem_cid"] == 2244
        assert records[1]["inchikey"] is None

    def test_empty_rows_skipped(self, tmp_path):
        xlsx_path = tmp_path / "drugs.xlsx"
        df = pd.DataFrame({
            "drug_name": ["DrugA", "", "DrugB"],
            "pubchem_cid": [100, None, 200],
        })
        df.to_excel(xlsx_path, index=False)

        records = parse_drug_identifiers(xlsx_path)
        assert len(records) == 2


# ── Cell line identifier parsing ─────────────────────────────────────


@pytest.mark.skipif(
    not pd.io.common.import_optional_dependency("openpyxl", errors="ignore"),
    reason="openpyxl not installed",
)
class TestParseCellLineIdentifiers:
    def test_basic_parsing(self, tmp_path):
        xlsx_path = tmp_path / "cell_lines.xlsx"
        df = pd.DataFrame({
            "cell_line_name": ["MCF7", "A549"],
            "cosmic_id": [905946, 905933],
            "tissue": ["Breast", "Lung"],
            "cancer_type": ["Carcinoma", "NSCLC"],
        })
        df.to_excel(xlsx_path, index=False)

        records = parse_cell_line_identifiers(xlsx_path)
        assert len(records) == 2
        assert records[0]["cell_line_name"] == "MCF7"
        assert records[0]["cosmic_id"] == 905946


# ── Compound and cell line loading ───────────────────────────────────


class TestLoadCompounds:
    def test_insert_and_cache(self, conn):
        records = [
            {"drug_name": "Aspirin", "pubchem_cid": 2244, "inchikey": "ABCD", "canonical_smiles": "CC"},
            {"drug_name": "Ibuprofen", "pubchem_cid": 3672, "inchikey": None, "canonical_smiles": None},
        ]
        cache = load_compounds(conn, records)
        assert len(cache) == 2
        assert "Aspirin" in cache
        assert "Ibuprofen" in cache

    def test_duplicate_names_ignored(self, conn):
        records = [
            {"drug_name": "Aspirin", "pubchem_cid": 2244},
            {"drug_name": "Aspirin", "pubchem_cid": 9999},
        ]
        cache = load_compounds(conn, records)
        assert len(cache) == 1


class TestLoadCellLines:
    def test_insert_and_cache(self, conn):
        records = [
            {"cell_line_name": "MCF7", "cosmic_id": 905946, "tissue": "Breast", "cancer_type": "Carcinoma"},
        ]
        cache = load_cell_lines(conn, records)
        assert len(cache) == 1
        assert "MCF7" in cache


# ── DrugComb synergy loading ─────────────────────────────────────────


class TestLoadDrugcombSynergy:
    def _make_csv(self, tmp_path, content):
        csv_path = tmp_path / "drugcomb_data.csv"
        csv_path.write_text(textwrap.dedent(content))
        return csv_path

    def _seed_entities(self, conn):
        """Seed compounds and cell lines for testing."""
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('DrugA')")
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('DrugB')")
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('DrugC')")
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

    def test_basic_loading(self, tmp_path, conn):
        csv_path = self._make_csv(tmp_path, """\
            block_id,drug_row,drug_col,cell_line_name,conc_r,conc_c,inhibition
            1,DrugA,DrugB,MCF7,0.1,0.1,10
            1,DrugA,DrugB,MCF7,0.1,1.0,20
            1,DrugA,DrugB,MCF7,1.0,0.1,15
            1,DrugA,DrugB,MCF7,1.0,1.0,30
        """)
        compound_cache, cell_line_cache = self._seed_entities(conn)

        stats = load_drugcomb_synergy(
            conn, csv_path, compound_cache, cell_line_cache,
        )
        assert stats["results_inserted"] == 1
        assert stats["chunks_processed"] == 1

    def test_unknown_drug_skipped(self, tmp_path, conn):
        csv_path = self._make_csv(tmp_path, """\
            block_id,drug_row,drug_col,cell_line_name,conc_r,conc_c,inhibition
            1,UnknownDrug,DrugB,MCF7,0.1,0.1,10
        """)
        compound_cache, cell_line_cache = self._seed_entities(conn)
        # Auto-creates UnknownDrug so won't be skipped; check it gets created
        stats = load_drugcomb_synergy(
            conn, csv_path, compound_cache, cell_line_cache,
        )
        # UnknownDrug auto-created
        assert "UnknownDrug" in compound_cache

    def test_self_combination_skipped(self, tmp_path, conn):
        csv_path = self._make_csv(tmp_path, """\
            block_id,drug_row,drug_col,cell_line_name,conc_r,conc_c,inhibition
            1,DrugA,DrugA,MCF7,0.1,0.1,10
        """)
        compound_cache, cell_line_cache = self._seed_entities(conn)
        stats = load_drugcomb_synergy(
            conn, csv_path, compound_cache, cell_line_cache,
        )
        assert stats["results_inserted"] == 0

    def test_pair_normalization(self, tmp_path, conn):
        """compound_a_id should always be < compound_b_id."""
        csv_path = self._make_csv(tmp_path, """\
            block_id,drug_row,drug_col,cell_line_name,conc_r,conc_c,inhibition
            1,DrugB,DrugA,MCF7,0.1,0.1,10
        """)
        compound_cache, cell_line_cache = self._seed_entities(conn)
        load_drugcomb_synergy(conn, csv_path, compound_cache, cell_line_cache)

        row = conn.execute(
            "SELECT compound_a_id, compound_b_id FROM dc_synergy_results"
        ).fetchone()
        assert row[0] < row[1]

    def test_matrix_detection(self, tmp_path, conn):
        """Block with 2x2 concentration grid → has_dose_matrix = True."""
        csv_path = self._make_csv(tmp_path, """\
            block_id,drug_row,drug_col,cell_line_name,conc_r,conc_c,inhibition
            1,DrugA,DrugB,MCF7,0.1,0.1,10
            1,DrugA,DrugB,MCF7,0.1,1.0,20
            1,DrugA,DrugB,MCF7,1.0,0.1,15
            1,DrugA,DrugB,MCF7,1.0,1.0,30
        """)
        compound_cache, cell_line_cache = self._seed_entities(conn)
        load_drugcomb_synergy(conn, csv_path, compound_cache, cell_line_cache)

        row = conn.execute(
            "SELECT has_dose_matrix, num_concentrations FROM dc_synergy_results"
        ).fetchone()
        assert row[0] == 1  # has_dose_matrix
        assert row[1] == 2  # max(n_conc_r, n_conc_c)

    def test_multiple_blocks(self, tmp_path, conn):
        csv_path = self._make_csv(tmp_path, """\
            block_id,drug_row,drug_col,cell_line_name,conc_r,conc_c,inhibition
            1,DrugA,DrugB,MCF7,0.1,0.1,10
            2,DrugA,DrugC,A549,0.1,0.1,20
        """)
        compound_cache, cell_line_cache = self._seed_entities(conn)
        stats = load_drugcomb_synergy(
            conn, csv_path, compound_cache, cell_line_cache,
        )
        assert stats["results_inserted"] == 2

    def test_source_db_is_drugcomb(self, tmp_path, conn):
        csv_path = self._make_csv(tmp_path, """\
            block_id,drug_row,drug_col,cell_line_name,conc_r,conc_c,inhibition
            1,DrugA,DrugB,MCF7,0.1,0.1,10
        """)
        compound_cache, cell_line_cache = self._seed_entities(conn)
        load_drugcomb_synergy(conn, csv_path, compound_cache, cell_line_cache)

        source = conn.execute(
            "SELECT source_db FROM dc_synergy_results"
        ).fetchone()[0]
        assert source == "drugcomb"
