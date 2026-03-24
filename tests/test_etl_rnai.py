"""Tests for DEMETER2 RNAi ETL module.

Uses synthetic data to test cell line mapping, score loading,
and concordance tier upgrades.
"""

from pathlib import Path

import pandas as pd
import pytest

from negbiodb_depmap.depmap_db import get_connection, run_ge_migrations, refresh_all_ge_pairs
from negbiodb_depmap.etl_depmap import load_cell_lines, load_genes_from_header
from negbiodb_depmap.etl_rnai import (
    _build_ccle_to_clid,
    _resolve_cell_line,
    _upgrade_concordant_pairs,
    load_demeter2,
)

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
def seeded_db(tmp_db, tmp_path):
    """DB pre-loaded with cell lines and genes for RNAi testing."""
    conn = get_connection(tmp_db)

    # Insert cell lines with CCLE names
    conn.execute(
        """INSERT INTO cell_lines (model_id, ccle_name, stripped_name)
        VALUES ('ACH-000001', 'CELL1_LUNG', 'CELL1')"""
    )
    conn.execute(
        """INSERT INTO cell_lines (model_id, ccle_name, stripped_name)
        VALUES ('ACH-000002', 'CELL2_BREAST', 'CELL2')"""
    )
    conn.execute(
        """INSERT INTO cell_lines (model_id, ccle_name, stripped_name)
        VALUES ('ACH-000003', 'CELL3_SKIN', 'CELL3')"""
    )

    # Insert genes
    conn.execute(
        "INSERT INTO genes (entrez_id, gene_symbol, is_reference_nonessential) VALUES (1001, 'GeneA', 1)"
    )
    conn.execute(
        "INSERT INTO genes (entrez_id, gene_symbol) VALUES (1002, 'GeneB')"
    )
    conn.execute(
        "INSERT INTO genes (entrez_id, gene_symbol) VALUES (1003, 'GeneC')"
    )
    conn.commit()
    conn.close()

    # Create DEMETER2 format: genes in rows, cell lines (CCLE names) in columns
    rnai_data = {
        "CELL1_LUNG": [0.08, -0.45, -1.2],   # GeneA non-ess, B marginal, C essential
        "CELL2_BREAST": [0.12, -0.50, -0.90], # GeneA non-ess, B marginal, C essential
        "UNMAPPED_LINE": [0.05, -0.20, -0.60],  # Unknown cell line
    }
    gene_labels = ["GeneA (1001)", "GeneB (1002)", "GeneC (1003)"]
    rnai_df = pd.DataFrame(rnai_data, index=gene_labels)
    rnai_file = tmp_path / "D2_combined_gene_dep_scores.csv"
    rnai_df.index.name = ""
    rnai_df.to_csv(rnai_file)

    return {"db_path": tmp_db, "rnai_file": rnai_file}


# ── Cell line resolution ──────────────────────────────────────────────────


class TestCellLineResolution:
    def test_direct_ccle_match(self, conn):
        conn.execute(
            "INSERT INTO cell_lines (model_id, ccle_name) VALUES ('ACH-001', 'A549_LUNG')"
        )
        conn.commit()
        ccle_map, stripped_map = _build_ccle_to_clid(conn)
        assert _resolve_cell_line("A549_LUNG", ccle_map, stripped_map) is not None

    def test_stripped_name_match(self, conn):
        conn.execute(
            "INSERT INTO cell_lines (model_id, stripped_name) VALUES ('ACH-001', 'A549')"
        )
        conn.commit()
        ccle_map, stripped_map = _build_ccle_to_clid(conn)
        # Stripped matching normalizes to uppercase and removes separators
        assert _resolve_cell_line("A549", ccle_map, stripped_map) is not None

    def test_unmatched_returns_none(self, conn):
        ccle_map, stripped_map = _build_ccle_to_clid(conn)
        assert _resolve_cell_line("UNKNOWN_CELL", ccle_map, stripped_map) is None


# ── Full RNAi loading ─────────────────────────────────────────────────────


class TestLoadDemeter2:
    def test_basic_load(self, seeded_db):
        stats = load_demeter2(
            seeded_db["db_path"],
            seeded_db["rnai_file"],
            depmap_release="TEST_DEMETER2",
        )
        assert stats["pairs_inserted"] > 0
        assert stats["cell_lines_mapped"] == 2  # CELL1_LUNG + CELL2_BREAST
        assert stats["cell_lines_unmapped"] == 1  # UNMAPPED_LINE

    def test_essential_excluded(self, seeded_db):
        load_demeter2(
            seeded_db["db_path"],
            seeded_db["rnai_file"],
            depmap_release="TEST_DEMETER2",
        )
        conn = get_connection(seeded_db["db_path"])
        try:
            genec = conn.execute(
                "SELECT gene_id FROM genes WHERE gene_symbol = 'GeneC'"
            ).fetchone()[0]
            count = conn.execute(
                "SELECT COUNT(*) FROM ge_negative_results WHERE gene_id = ? AND source_db = 'demeter2'",
                (genec,),
            ).fetchone()[0]
            # GeneC scores: -1.2, -0.90 → both below -0.8 threshold → excluded
            assert count == 0
        finally:
            conn.close()

    def test_screen_record(self, seeded_db):
        load_demeter2(
            seeded_db["db_path"],
            seeded_db["rnai_file"],
            depmap_release="TEST_DEMETER2",
        )
        conn = get_connection(seeded_db["db_path"])
        try:
            row = conn.execute(
                "SELECT screen_type, algorithm FROM ge_screens WHERE source_db = 'demeter2'"
            ).fetchone()
            assert row[0] == "rnai"
            assert row[1] == "DEMETER2"
        finally:
            conn.close()

    def test_evidence_type(self, seeded_db):
        load_demeter2(
            seeded_db["db_path"],
            seeded_db["rnai_file"],
            depmap_release="TEST_DEMETER2",
        )
        conn = get_connection(seeded_db["db_path"])
        try:
            types = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT evidence_type FROM ge_negative_results WHERE source_db = 'demeter2'"
                ).fetchall()
            }
            assert types == {"rnai_nonessential"}
        finally:
            conn.close()

    def test_dataset_version(self, seeded_db):
        load_demeter2(
            seeded_db["db_path"],
            seeded_db["rnai_file"],
            depmap_release="TEST_DEMETER2",
        )
        conn = get_connection(seeded_db["db_path"])
        try:
            row = conn.execute(
                "SELECT name, version FROM dataset_versions WHERE name = 'demeter2_rnai'"
            ).fetchone()
            assert row is not None
            assert row[1] == "TEST_DEMETER2"
        finally:
            conn.close()


# ── Concordance upgrade ───────────────────────────────────────────────────


class TestConcordanceUpgrade:
    def test_concordance_upgrade(self, seeded_db):
        """Bronze CRISPR pair + RNAi concordant → silver upgrade."""
        conn = get_connection(seeded_db["db_path"])
        try:
            # Insert a CRISPR screen
            conn.execute(
                """INSERT INTO ge_screens (source_db, depmap_release, screen_type, algorithm)
                VALUES ('depmap', 'test', 'crispr', 'Chronos')"""
            )
            screen_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            genea = conn.execute(
                "SELECT gene_id FROM genes WHERE gene_symbol = 'GeneA'"
            ).fetchone()[0]
            cl1 = conn.execute(
                "SELECT cell_line_id FROM cell_lines WHERE model_id = 'ACH-000001'"
            ).fetchone()[0]

            # Insert bronze CRISPR result
            conn.execute(
                """INSERT INTO ge_negative_results
                (gene_id, cell_line_id, screen_id, gene_effect_score, dependency_probability,
                 evidence_type, confidence_tier, source_db, source_record_id, extraction_method)
                VALUES (?, ?, ?, -0.7, 0.45, 'crispr_nonessential', 'bronze',
                        'depmap', 'crispr_test', 'score_threshold')""",
                (genea, cl1, screen_id),
            )
            conn.commit()

            # Load RNAi (which includes concordant data for GeneA)
            load_demeter2(
                seeded_db["db_path"],
                seeded_db["rnai_file"],
                depmap_release="TEST_DEMETER2",
            )

            # Check the CRISPR result was upgraded
            row = conn.execute(
                """SELECT confidence_tier, evidence_type FROM ge_negative_results
                WHERE gene_id = ? AND cell_line_id = ? AND source_db = 'depmap'""",
                (genea, cl1),
            ).fetchone()
            assert row[0] == "silver"
            assert row[1] == "multi_screen_concordant"
        finally:
            conn.close()

    def test_no_upgrade_without_rnai(self, seeded_db):
        """Bronze CRISPR pair without RNAi data stays bronze."""
        conn = get_connection(seeded_db["db_path"])
        try:
            conn.execute(
                """INSERT INTO ge_screens (source_db, depmap_release, screen_type, algorithm)
                VALUES ('depmap', 'test', 'crispr', 'Chronos')"""
            )
            screen_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            genea = conn.execute(
                "SELECT gene_id FROM genes WHERE gene_symbol = 'GeneA'"
            ).fetchone()[0]
            cl3 = conn.execute(
                "SELECT cell_line_id FROM cell_lines WHERE model_id = 'ACH-000003'"
            ).fetchone()[0]

            # Insert bronze CRISPR result for cell line 3 (unmapped in DEMETER2)
            conn.execute(
                """INSERT INTO ge_negative_results
                (gene_id, cell_line_id, screen_id, gene_effect_score, dependency_probability,
                 evidence_type, confidence_tier, source_db, source_record_id, extraction_method)
                VALUES (?, ?, ?, -0.7, 0.45, 'crispr_nonessential', 'bronze',
                        'depmap', 'crispr_cl3', 'score_threshold')""",
                (genea, cl3, screen_id),
            )
            conn.commit()

            # Run concordance upgrade (no RNAi data for cl3)
            upgraded = _upgrade_concordant_pairs(conn)
            assert upgraded == 0

            row = conn.execute(
                """SELECT confidence_tier FROM ge_negative_results
                WHERE gene_id = ? AND cell_line_id = ? AND source_db = 'depmap'""",
                (genea, cl3),
            ).fetchone()
            assert row[0] == "bronze"
        finally:
            conn.close()
