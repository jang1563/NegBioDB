"""Tests for PRISM drug sensitivity ETL module.

Uses synthetic data to test compound loading and sensitivity data insertion.
"""

from pathlib import Path

import pandas as pd
import pytest

from negbiodb_depmap.depmap_db import get_connection, run_ge_migrations
from negbiodb_depmap.etl_prism import load_prism_compounds, load_prism_sensitivity

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_depmap"


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_ge.db"
    run_ge_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def conn(tmp_db):
    c = get_connection(tmp_db)
    yield c
    c.close()


@pytest.fixture
def synthetic_prism(tmp_path, tmp_db):
    """Create synthetic PRISM data files."""
    # Compound metadata
    compound_data = {
        "broad_id": ["BRD-K001", "BRD-K002", "BRD-K003", "BRD-K004"],
        "name": ["DrugA", "DrugB", "DrugC", None],
        "smiles": ["CCO", "c1ccccc1", None, "CC(=O)O"],
        "InChIKey": ["LFQSCWFLJHTTHZ-UHFFFAOYSA-N", None, None, "QTBSBXVTEAMEQO-UHFFFAOYSA-N"],
        "moa": ["kinase inhibitor", "DNA damage", None, None],
        "target": ["EGFR", "TOP2A", None, None],
    }
    compound_file = tmp_path / "compound_meta.csv"
    pd.DataFrame(compound_data).to_csv(compound_file, index=False)

    # Insert cell lines for sensitivity tests
    conn = get_connection(tmp_db)
    conn.execute("INSERT INTO cell_lines (model_id, ccle_name) VALUES ('ACH-000001', 'CELL1')")
    conn.execute("INSERT INTO cell_lines (model_id, ccle_name) VALUES ('ACH-000002', 'CELL2')")
    conn.commit()
    conn.close()

    # Primary screen matrix (cell lines × compounds)
    primary_data = {
        "BRD-K001": [-0.3, -0.1],
        "BRD-K002": [-0.8, -0.5],
        "BRD-K003": [-0.2, float("nan")],
    }
    primary_df = pd.DataFrame(primary_data, index=["ACH-000001", "ACH-000002"])
    primary_file = tmp_path / "primary_screen.csv"
    primary_df.to_csv(primary_file, index_label="")

    # Secondary screen (row-oriented)
    secondary_data = {
        "broad_id": ["BRD-K001", "BRD-K002"],
        "depmap_id": ["ACH-000001", "ACH-000002"],
        "auc": [0.85, 0.92],
        "ic50": [1.5, 0.3],
        "ec50": [2.0, None],
    }
    secondary_file = tmp_path / "secondary_screen.csv"
    pd.DataFrame(secondary_data).to_csv(secondary_file, index=False)

    return {
        "db_path": tmp_db,
        "compound_file": compound_file,
        "primary_file": primary_file,
        "secondary_file": secondary_file,
    }


# ── Compound loading ─────────────────────────────────────────────────────


class TestLoadCompounds:
    def test_basic_load(self, synthetic_prism):
        stats = load_prism_compounds(
            synthetic_prism["db_path"],
            synthetic_prism["compound_file"],
        )
        assert stats["compounds_inserted"] == 4
        assert stats["compounds_with_smiles"] == 3
        assert stats["compounds_with_inchikey"] == 2

    def test_compound_metadata(self, synthetic_prism):
        load_prism_compounds(
            synthetic_prism["db_path"],
            synthetic_prism["compound_file"],
        )
        conn = get_connection(synthetic_prism["db_path"])
        try:
            row = conn.execute(
                "SELECT name, smiles, mechanism_of_action, target_name FROM prism_compounds WHERE broad_id = 'BRD-K001'"
            ).fetchone()
            assert row[0] == "DrugA"
            assert row[1] == "CCO"
            assert row[2] == "kinase inhibitor"
            assert row[3] == "EGFR"
        finally:
            conn.close()

    def test_null_fields(self, synthetic_prism):
        load_prism_compounds(
            synthetic_prism["db_path"],
            synthetic_prism["compound_file"],
        )
        conn = get_connection(synthetic_prism["db_path"])
        try:
            row = conn.execute(
                "SELECT smiles, inchikey FROM prism_compounds WHERE broad_id = 'BRD-K003'"
            ).fetchone()
            assert row[0] is None
            assert row[1] is None
        finally:
            conn.close()

    def test_idempotent(self, synthetic_prism):
        load_prism_compounds(synthetic_prism["db_path"], synthetic_prism["compound_file"])
        load_prism_compounds(synthetic_prism["db_path"], synthetic_prism["compound_file"])
        conn = get_connection(synthetic_prism["db_path"])
        try:
            count = conn.execute("SELECT COUNT(*) FROM prism_compounds").fetchone()[0]
            assert count == 4
        finally:
            conn.close()


# ── Sensitivity loading ───────────────────────────────────────────────────


class TestLoadSensitivity:
    def test_primary_screen(self, synthetic_prism):
        # Load compounds first
        load_prism_compounds(
            synthetic_prism["db_path"],
            synthetic_prism["compound_file"],
        )
        stats = load_prism_sensitivity(
            synthetic_prism["db_path"],
            primary_file=synthetic_prism["primary_file"],
            depmap_release="TEST_PRISM",
        )
        # 2 cell lines × 3 compounds, minus 1 NaN = 5
        assert stats["primary_pairs"] == 5

    def test_secondary_screen(self, synthetic_prism):
        load_prism_compounds(
            synthetic_prism["db_path"],
            synthetic_prism["compound_file"],
        )
        stats = load_prism_sensitivity(
            synthetic_prism["db_path"],
            secondary_file=synthetic_prism["secondary_file"],
            depmap_release="TEST_PRISM",
        )
        assert stats["secondary_pairs"] == 2

    def test_combined_load(self, synthetic_prism):
        load_prism_compounds(
            synthetic_prism["db_path"],
            synthetic_prism["compound_file"],
        )
        stats = load_prism_sensitivity(
            synthetic_prism["db_path"],
            primary_file=synthetic_prism["primary_file"],
            secondary_file=synthetic_prism["secondary_file"],
            depmap_release="TEST_PRISM",
        )
        assert stats["total_inserted"] == stats["primary_pairs"] + stats["secondary_pairs"]

    def test_sensitivity_values(self, synthetic_prism):
        load_prism_compounds(
            synthetic_prism["db_path"],
            synthetic_prism["compound_file"],
        )
        load_prism_sensitivity(
            synthetic_prism["db_path"],
            secondary_file=synthetic_prism["secondary_file"],
            depmap_release="TEST_PRISM",
        )
        conn = get_connection(synthetic_prism["db_path"])
        try:
            row = conn.execute(
                """SELECT auc, ic50, ec50 FROM prism_sensitivity
                WHERE screen_type = 'secondary' LIMIT 1"""
            ).fetchone()
            assert row[0] is not None  # auc should be set
            assert row[1] is not None  # ic50 should be set
        finally:
            conn.close()

    def test_dataset_version(self, synthetic_prism):
        load_prism_compounds(
            synthetic_prism["db_path"],
            synthetic_prism["compound_file"],
        )
        load_prism_sensitivity(
            synthetic_prism["db_path"],
            primary_file=synthetic_prism["primary_file"],
            depmap_release="TEST_PRISM",
        )
        conn = get_connection(synthetic_prism["db_path"])
        try:
            row = conn.execute(
                "SELECT version FROM dataset_versions WHERE name = 'prism_sensitivity'"
            ).fetchone()
            assert row is not None
            assert row[0] == "TEST_PRISM"
        finally:
            conn.close()

    def test_unmapped_compounds_skipped(self, synthetic_prism):
        """Compounds not in prism_compounds table are skipped."""
        # Don't load compounds → all should be skipped
        stats = load_prism_sensitivity(
            synthetic_prism["db_path"],
            primary_file=synthetic_prism["primary_file"],
            depmap_release="TEST_PRISM",
        )
        assert stats["primary_pairs"] == 0
        assert stats["primary_skipped_no_compound"] > 0
