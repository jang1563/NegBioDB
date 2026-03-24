"""Tests for DepMap CRISPR ETL module.

Uses synthetic 5-gene × 4-cell-line matrices to test the full ETL pipeline
without requiring actual DepMap downloads (~1 GB).
"""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from negbiodb_depmap.depmap_db import get_connection, run_ge_migrations, refresh_all_ge_pairs
from negbiodb_depmap.etl_depmap import (
    assign_tier,
    load_cell_lines,
    load_depmap_crispr,
    load_genes_from_header,
    load_reference_gene_sets,
    parse_gene_column,
)

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_depmap"


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary GE database with migrations applied."""
    db_path = tmp_path / "test_ge.db"
    run_ge_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def conn(tmp_db):
    c = get_connection(tmp_db)
    yield c
    c.close()


@pytest.fixture
def synthetic_data(tmp_path):
    """Create synthetic DepMap CSV files for testing.

    5 genes × 4 cell lines:
      - GeneA (1001): clearly non-essential (positive scores)
      - GeneB (1002): marginal (near threshold)
      - GeneC (1003): essential (very negative scores) — should be excluded
      - GeneD (1004): mixed (non-essential in some, essential in others)
      - GeneE (1005): NaN in all cell lines — should be skipped
    """
    genes = [
        "GeneA (1001)", "GeneB (1002)", "GeneC (1003)",
        "GeneD (1004)", "GeneE (1005)",
    ]
    cell_lines = ["ACH-000001", "ACH-000002", "ACH-000003", "ACH-000004"]

    # Gene effect scores (Chronos: 0 = no effect, -1 = essential)
    ge_data = {
        genes[0]: [0.05, 0.10, -0.15, 0.02],    # clearly non-essential
        genes[1]: [-0.45, -0.55, -0.70, -0.48],  # marginal
        genes[2]: [-1.2, -1.5, -0.95, -1.1],     # essential
        genes[3]: [0.01, -0.90, -0.30, 0.05],    # mixed
        genes[4]: [float("nan")] * 4,             # NaN
    }
    ge_df = pd.DataFrame(ge_data, index=cell_lines)
    ge_file = tmp_path / "CRISPRGeneEffect.csv"
    ge_df.to_csv(ge_file, index_label="ModelID")

    # Dependency probability (0-1)
    dep_data = {
        genes[0]: [0.05, 0.10, 0.20, 0.08],    # low prob → non-essential
        genes[1]: [0.40, 0.55, 0.70, 0.45],    # marginal
        genes[2]: [0.95, 0.98, 0.80, 0.90],    # high prob → essential
        genes[3]: [0.10, 0.85, 0.35, 0.05],    # mixed
        genes[4]: [float("nan")] * 4,           # NaN
    }
    dep_df = pd.DataFrame(dep_data, index=cell_lines)
    dep_file = tmp_path / "CRISPRGeneDependency.csv"
    dep_df.to_csv(dep_file, index_label="ModelID")

    # Model.csv (cell line metadata)
    model_data = {
        "ModelID": cell_lines,
        "CCLEName": ["CELL1_LUNG", "CELL2_BREAST", "CELL3_SKIN", "CELL4_COLON"],
        "StrippedCellLineName": ["CELL1", "CELL2", "CELL3", "CELL4"],
        "OncotreeLineage": ["Lung", "Breast", "Skin", "Colon"],
        "OncotreePrimaryDisease": [
            "Non-Small Cell Lung Cancer", "Breast Cancer",
            "Melanoma", "Colorectal Cancer",
        ],
    }
    model_df = pd.DataFrame(model_data)
    model_file = tmp_path / "Model.csv"
    model_df.to_csv(model_file, index=False)

    # Essential gene set
    ess_file = tmp_path / "AchillesCommonEssentialControls.csv"
    ess_file.write_text("GeneC (1003)\n")

    # Non-essential gene set
    ne_file = tmp_path / "AchillesNonessentialControls.csv"
    ne_file.write_text("GeneA (1001)\n")

    return {
        "gene_effect_file": ge_file,
        "dependency_file": dep_file,
        "model_file": model_file,
        "essential_file": ess_file,
        "nonessential_file": ne_file,
    }


# ── Unit tests ────────────────────────────────────────────────────────────


class TestParseGeneColumn:
    def test_standard_format(self):
        assert parse_gene_column("TP53 (7157)") == ("TP53", 7157)

    def test_gene_with_dash(self):
        assert parse_gene_column("HLA-A (3105)") == ("HLA-A", 3105)

    def test_extra_spaces(self):
        # DepMap format has no internal spaces in parens, but outer whitespace is stripped
        assert parse_gene_column("  BRAF (673)  ") == ("BRAF", 673)

    def test_invalid_format(self):
        assert parse_gene_column("ModelID") is None

    def test_no_entrez(self):
        assert parse_gene_column("UNKNOWN") is None


class TestAssignTier:
    def test_gold_tier(self):
        assert assign_tier(0.05, 0.1, is_reference_nonessential=True) == "gold"

    def test_gold_requires_ref_nonessential(self):
        # Without reference flag → silver
        assert assign_tier(0.05, 0.1, is_reference_nonessential=False) == "silver"

    def test_silver_tier(self):
        assert assign_tier(-0.4, 0.3, is_reference_nonessential=False) == "silver"

    def test_bronze_tier(self):
        assert assign_tier(-0.7, 0.45, is_reference_nonessential=False) == "bronze"

    def test_bronze_low_effect(self):
        assert assign_tier(-0.75, 0.45, is_reference_nonessential=False) == "bronze"

    def test_no_dep_prob(self):
        # dep_prob=None defaults to 1.0 (conservative)
        assert assign_tier(0.05, None, is_reference_nonessential=True) == "bronze"

    def test_silver_boundary(self):
        # Exactly at boundary: > -0.5
        assert assign_tier(-0.49, 0.49, is_reference_nonessential=False) == "silver"


# ── Cell line loading ─────────────────────────────────────────────────────


class TestCellLineLoading:
    def test_load_cell_lines(self, conn, synthetic_data):
        result = load_cell_lines(conn, synthetic_data["model_file"])
        assert len(result) == 4
        assert "ACH-000001" in result
        assert "ACH-000004" in result

    def test_cell_line_metadata(self, conn, synthetic_data):
        load_cell_lines(conn, synthetic_data["model_file"])
        row = conn.execute(
            "SELECT ccle_name, lineage, primary_disease FROM cell_lines WHERE model_id = 'ACH-000001'"
        ).fetchone()
        assert row[0] == "CELL1_LUNG"
        assert row[1] == "Lung"
        assert row[2] == "Non-Small Cell Lung Cancer"

    def test_load_idempotent(self, conn, synthetic_data):
        load_cell_lines(conn, synthetic_data["model_file"])
        result = load_cell_lines(conn, synthetic_data["model_file"])
        assert len(result) == 4  # no duplicates


# ── Gene loading ──────────────────────────────────────────────────────────


class TestGeneLoading:
    def test_load_genes_from_header(self, conn, synthetic_data):
        result = load_genes_from_header(conn, synthetic_data["gene_effect_file"])
        assert len(result) == 5  # 5 genes
        # Check genes in DB
        count = conn.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
        assert count == 5

    def test_gene_metadata(self, conn, synthetic_data):
        load_genes_from_header(conn, synthetic_data["gene_effect_file"])
        row = conn.execute(
            "SELECT gene_symbol, entrez_id FROM genes WHERE gene_symbol = 'GeneA'"
        ).fetchone()
        assert row is not None
        assert row[1] == 1001


class TestReferenceGeneSets:
    def test_mark_essential(self, conn, synthetic_data):
        load_genes_from_header(conn, synthetic_data["gene_effect_file"])
        load_reference_gene_sets(
            conn,
            essential_file=synthetic_data["essential_file"],
        )
        row = conn.execute(
            "SELECT is_common_essential FROM genes WHERE gene_symbol = 'GeneC'"
        ).fetchone()
        assert row[0] == 1

    def test_mark_nonessential(self, conn, synthetic_data):
        load_genes_from_header(conn, synthetic_data["gene_effect_file"])
        load_reference_gene_sets(
            conn,
            nonessential_file=synthetic_data["nonessential_file"],
        )
        row = conn.execute(
            "SELECT is_reference_nonessential FROM genes WHERE gene_symbol = 'GeneA'"
        ).fetchone()
        assert row[0] == 1


# ── Full pipeline ─────────────────────────────────────────────────────────


class TestFullPipeline:
    def test_load_depmap_crispr(self, tmp_db, synthetic_data):
        stats = load_depmap_crispr(
            tmp_db,
            gene_effect_file=synthetic_data["gene_effect_file"],
            dependency_file=synthetic_data["dependency_file"],
            model_file=synthetic_data["model_file"],
            essential_file=synthetic_data["essential_file"],
            nonessential_file=synthetic_data["nonessential_file"],
            depmap_release="test_release",
        )
        assert stats["cell_lines_loaded"] == 4
        assert stats["genes_loaded"] == 5
        assert stats["pairs_inserted"] > 0
        assert stats["pairs_skipped_nan"] == 4  # GeneE in all 4 cell lines

    def test_essential_genes_excluded(self, tmp_db, synthetic_data):
        stats = load_depmap_crispr(
            tmp_db,
            gene_effect_file=synthetic_data["gene_effect_file"],
            dependency_file=synthetic_data["dependency_file"],
            model_file=synthetic_data["model_file"],
            depmap_release="test_release",
        )
        # GeneC is essential (gene_effect < -0.8 AND dep_prob > 0.5)
        # ACH-000001: -1.2, 0.95 → skip (effect<-0.8 AND prob>0.5)
        # ACH-000002: -1.5, 0.98 → skip
        # ACH-000003: -0.95, 0.80 → skip (effect<-0.8 AND prob>0.5)
        # ACH-000004: -1.1, 0.90 → skip
        conn = get_connection(tmp_db)
        try:
            genec_id = conn.execute(
                "SELECT gene_id FROM genes WHERE gene_symbol = 'GeneC'"
            ).fetchone()[0]
            count = conn.execute(
                "SELECT COUNT(*) FROM ge_negative_results WHERE gene_id = ?",
                (genec_id,),
            ).fetchone()[0]
            assert count == 0
        finally:
            conn.close()

    def test_nan_values_skipped(self, tmp_db, synthetic_data):
        stats = load_depmap_crispr(
            tmp_db,
            gene_effect_file=synthetic_data["gene_effect_file"],
            dependency_file=synthetic_data["dependency_file"],
            model_file=synthetic_data["model_file"],
            depmap_release="test_release",
        )
        assert stats["pairs_skipped_nan"] == 4

    def test_tiering_distribution(self, tmp_db, synthetic_data):
        stats = load_depmap_crispr(
            tmp_db,
            gene_effect_file=synthetic_data["gene_effect_file"],
            dependency_file=synthetic_data["dependency_file"],
            model_file=synthetic_data["model_file"],
            essential_file=synthetic_data["essential_file"],
            nonessential_file=synthetic_data["nonessential_file"],
            depmap_release="test_release",
        )
        total_tiered = stats["tier_gold"] + stats["tier_silver"] + stats["tier_bronze"]
        assert total_tiered == stats["pairs_inserted"]

    def test_screen_record_created(self, tmp_db, synthetic_data):
        load_depmap_crispr(
            tmp_db,
            gene_effect_file=synthetic_data["gene_effect_file"],
            dependency_file=synthetic_data["dependency_file"],
            model_file=synthetic_data["model_file"],
            depmap_release="test_release",
        )
        conn = get_connection(tmp_db)
        try:
            row = conn.execute(
                "SELECT screen_type, algorithm FROM ge_screens WHERE source_db = 'depmap'"
            ).fetchone()
            assert row[0] == "crispr"
            assert row[1] == "Chronos"
        finally:
            conn.close()

    def test_dataset_version_recorded(self, tmp_db, synthetic_data):
        load_depmap_crispr(
            tmp_db,
            gene_effect_file=synthetic_data["gene_effect_file"],
            dependency_file=synthetic_data["dependency_file"],
            model_file=synthetic_data["model_file"],
            depmap_release="test_release",
        )
        conn = get_connection(tmp_db)
        try:
            row = conn.execute(
                "SELECT name, version FROM dataset_versions WHERE name = 'depmap_crispr'"
            ).fetchone()
            assert row is not None
            assert row[1] == "test_release"
        finally:
            conn.close()

    def test_pair_aggregation(self, tmp_db, synthetic_data):
        load_depmap_crispr(
            tmp_db,
            gene_effect_file=synthetic_data["gene_effect_file"],
            dependency_file=synthetic_data["dependency_file"],
            model_file=synthetic_data["model_file"],
            depmap_release="test_release",
        )
        conn = get_connection(tmp_db)
        try:
            count = refresh_all_ge_pairs(conn)
            conn.commit()
            assert count > 0
            # Verify gene_degree and cell_line_degree are set
            row = conn.execute(
                "SELECT gene_degree, cell_line_degree FROM gene_cell_pairs LIMIT 1"
            ).fetchone()
            assert row[0] is not None
            assert row[1] is not None
        finally:
            conn.close()

    def test_idempotent_reload(self, tmp_db, synthetic_data):
        """Running ETL twice should not duplicate records."""
        load_depmap_crispr(
            tmp_db,
            gene_effect_file=synthetic_data["gene_effect_file"],
            dependency_file=synthetic_data["dependency_file"],
            model_file=synthetic_data["model_file"],
            depmap_release="test_release",
        )
        stats1 = load_depmap_crispr(
            tmp_db,
            gene_effect_file=synthetic_data["gene_effect_file"],
            dependency_file=synthetic_data["dependency_file"],
            model_file=synthetic_data["model_file"],
            depmap_release="test_release",
        )
        conn = get_connection(tmp_db)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM ge_negative_results"
            ).fetchone()[0]
            assert count == stats1["pairs_inserted"]
        finally:
            conn.close()
