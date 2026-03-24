"""Tests for NegBioDB GE (Gene Essentiality) database layer.

Tests migration, connection, table creation, and pair aggregation.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from negbiodb_depmap.depmap_db import (
    create_ge_database,
    get_connection,
    refresh_all_ge_pairs,
    run_ge_migrations,
)

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_depmap"


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary GE database with all migrations applied."""
    db_path = tmp_path / "test_ge.db"
    run_ge_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def conn(tmp_db):
    """Get a connection to the temporary GE database."""
    c = get_connection(tmp_db)
    yield c
    c.close()


# ── Migration tests ───────────────────────────────────────────────────


class TestMigrations:
    def test_migration_creates_all_tables(self, conn):
        """All 10 expected tables should exist after migration."""
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {
            "schema_migrations",
            "dataset_versions",
            "genes",
            "cell_lines",
            "ge_screens",
            "ge_negative_results",
            "gene_cell_pairs",
            "ge_split_definitions",
            "ge_split_assignments",
            "prism_compounds",
            "prism_sensitivity",
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    def test_migration_version_recorded(self, conn):
        """Migration 001 should be recorded in schema_migrations."""
        versions = {
            row[0]
            for row in conn.execute(
                "SELECT version FROM schema_migrations"
            ).fetchall()
        }
        assert "001" in versions

    def test_migration_idempotent(self, tmp_db):
        """Running migrations twice should not fail or duplicate."""
        applied = run_ge_migrations(tmp_db, MIGRATIONS_DIR)
        assert applied == [], "No new migrations expected on second run"

    def test_create_ge_database(self, tmp_path):
        """create_ge_database convenience wrapper should work."""
        db_path = tmp_path / "convenience.db"
        result = create_ge_database(db_path, MIGRATIONS_DIR)
        assert result == db_path
        assert db_path.exists()


# ── Connection tests ──────────────────────────────────────────────────


class TestConnection:
    def test_wal_mode(self, conn):
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, conn):
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1


# ── Schema constraint tests ──────────────────────────────────────────


class TestSchemaConstraints:
    def test_gene_entrez_unique(self, conn):
        """Entrez ID should be unique."""
        conn.execute(
            "INSERT INTO genes (entrez_id, gene_symbol) VALUES (7157, 'TP53')"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO genes (entrez_id, gene_symbol) VALUES (7157, 'TP53_DUP')"
            )

    def test_cell_line_model_id_unique(self, conn):
        """Model ID should be unique."""
        conn.execute(
            "INSERT INTO cell_lines (model_id, ccle_name) VALUES ('ACH-000001', 'A549_LUNG')"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO cell_lines (model_id, ccle_name) VALUES ('ACH-000001', 'DUP')"
            )

    def test_confidence_tier_check(self, conn):
        """Invalid confidence tier should fail."""
        conn.execute("INSERT INTO genes (entrez_id, gene_symbol) VALUES (1, 'A')")
        conn.execute("INSERT INTO cell_lines (model_id) VALUES ('ACH-000001')")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO ge_negative_results
                (gene_id, cell_line_id, evidence_type, confidence_tier,
                 source_db, source_record_id, extraction_method)
                VALUES (1, 1, 'crispr_nonessential', 'platinum',
                        'depmap', 'test', 'score_threshold')"""
            )

    def test_evidence_type_check(self, conn):
        """Invalid evidence type should fail."""
        conn.execute("INSERT INTO genes (entrez_id, gene_symbol) VALUES (1, 'A')")
        conn.execute("INSERT INTO cell_lines (model_id) VALUES ('ACH-000001')")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO ge_negative_results
                (gene_id, cell_line_id, evidence_type, confidence_tier,
                 source_db, source_record_id, extraction_method)
                VALUES (1, 1, 'invalid_type', 'bronze',
                        'depmap', 'test', 'score_threshold')"""
            )

    def test_screen_type_check(self, conn):
        """Screen type must be crispr or rnai."""
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO ge_screens
                (source_db, depmap_release, screen_type)
                VALUES ('depmap', '25Q3', 'sirna')"""
            )

    def test_negative_result_dedup(self, conn):
        """Duplicate gene-cell_line-screen-source should be rejected."""
        conn.execute("INSERT INTO genes (entrez_id, gene_symbol) VALUES (1, 'A')")
        conn.execute("INSERT INTO cell_lines (model_id) VALUES ('ACH-000001')")
        conn.execute(
            """INSERT INTO ge_screens (source_db, depmap_release, screen_type)
            VALUES ('depmap', '25Q3', 'crispr')"""
        )
        conn.commit()
        conn.execute(
            """INSERT INTO ge_negative_results
            (gene_id, cell_line_id, screen_id, evidence_type, confidence_tier,
             source_db, source_record_id, extraction_method)
            VALUES (1, 1, 1, 'crispr_nonessential', 'bronze',
                    'depmap', 'r1', 'score_threshold')"""
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO ge_negative_results
                (gene_id, cell_line_id, screen_id, evidence_type, confidence_tier,
                 source_db, source_record_id, extraction_method)
                VALUES (1, 1, 1, 'crispr_nonessential', 'bronze',
                        'depmap', 'r2', 'score_threshold')"""
            )

    def test_prism_broad_id_unique(self, conn):
        """PRISM broad_id should be unique."""
        conn.execute(
            "INSERT INTO prism_compounds (broad_id, name) VALUES ('BRD-K001', 'Drug1')"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO prism_compounds (broad_id, name) VALUES ('BRD-K001', 'Drug2')"
            )

    def test_foreign_key_gene(self, conn):
        """FK from ge_negative_results to genes should be enforced."""
        conn.execute("INSERT INTO cell_lines (model_id) VALUES ('ACH-000001')")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO ge_negative_results
                (gene_id, cell_line_id, evidence_type, confidence_tier,
                 source_db, source_record_id, extraction_method)
                VALUES (999, 1, 'crispr_nonessential', 'bronze',
                        'depmap', 'test', 'score_threshold')"""
            )

    def test_foreign_key_cell_line(self, conn):
        """FK from ge_negative_results to cell_lines should be enforced."""
        conn.execute("INSERT INTO genes (entrez_id, gene_symbol) VALUES (1, 'A')")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO ge_negative_results
                (gene_id, cell_line_id, evidence_type, confidence_tier,
                 source_db, source_record_id, extraction_method)
                VALUES (1, 999, 'crispr_nonessential', 'bronze',
                        'depmap', 'test', 'score_threshold')"""
            )


# ── Pair aggregation tests ────────────────────────────────────────────


def _insert_test_data(conn):
    """Insert synthetic test data for pair aggregation tests."""
    # 3 genes, 2 cell lines
    conn.execute("INSERT INTO genes (gene_id, entrez_id, gene_symbol) VALUES (1, 7157, 'TP53')")
    conn.execute("INSERT INTO genes (gene_id, entrez_id, gene_symbol) VALUES (2, 673, 'BRAF')")
    conn.execute("INSERT INTO genes (gene_id, entrez_id, gene_symbol) VALUES (3, 2064, 'ERBB2')")
    conn.execute("INSERT INTO cell_lines (cell_line_id, model_id) VALUES (1, 'ACH-000001')")
    conn.execute("INSERT INTO cell_lines (cell_line_id, model_id) VALUES (2, 'ACH-000002')")

    # Screen
    conn.execute(
        """INSERT INTO ge_screens (screen_id, source_db, depmap_release, screen_type, algorithm)
        VALUES (1, 'depmap', '25Q3', 'crispr', 'Chronos')"""
    )
    conn.execute(
        """INSERT INTO ge_screens (screen_id, source_db, depmap_release, screen_type, algorithm)
        VALUES (2, 'demeter2', 'DEMETER2_v6', 'rnai', 'DEMETER2')"""
    )

    # Gene 1 (TP53) non-essential in both cell lines (CRISPR)
    conn.execute(
        """INSERT INTO ge_negative_results
        (gene_id, cell_line_id, screen_id, gene_effect_score, dependency_probability,
         evidence_type, confidence_tier, source_db, source_record_id, extraction_method)
        VALUES (1, 1, 1, 0.05, 0.1, 'crispr_nonessential', 'gold', 'depmap', 'r1', 'score_threshold')"""
    )
    conn.execute(
        """INSERT INTO ge_negative_results
        (gene_id, cell_line_id, screen_id, gene_effect_score, dependency_probability,
         evidence_type, confidence_tier, source_db, source_record_id, extraction_method)
        VALUES (1, 2, 1, 0.02, 0.15, 'crispr_nonessential', 'silver', 'depmap', 'r2', 'score_threshold')"""
    )

    # Gene 1 (TP53) also non-essential in cell line 1 via RNAi
    conn.execute(
        """INSERT INTO ge_negative_results
        (gene_id, cell_line_id, screen_id, gene_effect_score, dependency_probability,
         evidence_type, confidence_tier, source_db, source_record_id, extraction_method)
        VALUES (1, 1, 2, 0.08, NULL, 'rnai_nonessential', 'bronze', 'demeter2', 'r3', 'score_threshold')"""
    )

    # Gene 2 (BRAF) non-essential in cell line 1 only
    conn.execute(
        """INSERT INTO ge_negative_results
        (gene_id, cell_line_id, screen_id, gene_effect_score, dependency_probability,
         evidence_type, confidence_tier, source_db, source_record_id, extraction_method)
        VALUES (2, 1, 1, -0.3, 0.35, 'crispr_nonessential', 'bronze', 'depmap', 'r4', 'score_threshold')"""
    )

    conn.commit()


class TestPairAggregation:
    def test_refresh_pair_count(self, conn):
        """Should create correct number of aggregated pairs."""
        _insert_test_data(conn)
        count = refresh_all_ge_pairs(conn)
        conn.commit()
        # Gene 1 in cell line 1 (2 sources), Gene 1 in cell line 2 (1 source),
        # Gene 2 in cell line 1 (1 source) = 3 pairs
        assert count == 3

    def test_multi_source_pair(self, conn):
        """Pair with CRISPR + RNAi should have num_sources=2."""
        _insert_test_data(conn)
        refresh_all_ge_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT num_screens, num_sources, best_confidence, min_gene_effect, max_gene_effect
            FROM gene_cell_pairs WHERE gene_id = 1 AND cell_line_id = 1"""
        ).fetchone()
        assert row is not None
        assert row[0] == 2  # num_screens
        assert row[1] == 2  # num_sources (depmap + demeter2)
        assert row[2] == "gold"  # best_confidence
        assert row[3] == pytest.approx(0.05)  # min_gene_effect
        assert row[4] == pytest.approx(0.08)  # max_gene_effect

    def test_single_source_pair(self, conn):
        """Pair with single source should have num_sources=1."""
        _insert_test_data(conn)
        refresh_all_ge_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT num_screens, num_sources, best_confidence
            FROM gene_cell_pairs WHERE gene_id = 2 AND cell_line_id = 1"""
        ).fetchone()
        assert row is not None
        assert row[0] == 1  # num_screens
        assert row[1] == 1  # num_sources
        assert row[2] == "bronze"

    def test_gene_degree(self, conn):
        """Gene 1 should have degree 2 (non-essential in 2 cell lines)."""
        _insert_test_data(conn)
        refresh_all_ge_pairs(conn)
        conn.commit()
        row = conn.execute(
            "SELECT gene_degree FROM gene_cell_pairs WHERE gene_id = 1 LIMIT 1"
        ).fetchone()
        assert row[0] == 2

    def test_cell_line_degree(self, conn):
        """Cell line 1 should have degree 2 (2 genes non-essential)."""
        _insert_test_data(conn)
        refresh_all_ge_pairs(conn)
        conn.commit()
        row = conn.execute(
            "SELECT cell_line_degree FROM gene_cell_pairs WHERE cell_line_id = 1 LIMIT 1"
        ).fetchone()
        assert row[0] == 2

    def test_mean_gene_effect(self, conn):
        """Mean gene effect should be computed correctly."""
        _insert_test_data(conn)
        refresh_all_ge_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT mean_gene_effect FROM gene_cell_pairs
            WHERE gene_id = 1 AND cell_line_id = 1"""
        ).fetchone()
        # Average of 0.05 (CRISPR) and 0.08 (RNAi) = 0.065
        assert row[0] == pytest.approx(0.065)

    def test_refresh_clears_old_pairs(self, conn):
        """Refreshing should delete old pairs and split assignments."""
        _insert_test_data(conn)
        refresh_all_ge_pairs(conn)
        conn.commit()

        # Add a split assignment
        conn.execute(
            """INSERT INTO ge_split_definitions
            (split_name, split_strategy) VALUES ('test_split', 'random')"""
        )
        pair_id = conn.execute("SELECT pair_id FROM gene_cell_pairs LIMIT 1").fetchone()[0]
        conn.execute(
            """INSERT INTO ge_split_assignments (pair_id, split_id, fold)
            VALUES (?, 1, 'train')""",
            (pair_id,),
        )
        conn.commit()

        # Refresh again
        count = refresh_all_ge_pairs(conn)
        conn.commit()
        assert count == 3

        # Split assignments should be cleared
        sa_count = conn.execute(
            "SELECT COUNT(*) FROM ge_split_assignments"
        ).fetchone()[0]
        assert sa_count == 0

    def test_empty_results(self, conn):
        """Refreshing with no results should produce 0 pairs."""
        count = refresh_all_ge_pairs(conn)
        assert count == 0


# ── Index tests ───────────────────────────────────────────────────────


class TestIndices:
    def test_key_indices_exist(self, conn):
        """Critical indices should exist for query performance."""
        indices = {
            row[1]
            for row in conn.execute("PRAGMA index_list('ge_negative_results')").fetchall()
        }
        assert "idx_ge_nr_gene" in indices
        assert "idx_ge_nr_cell_line" in indices
        assert "idx_ge_nr_pair" in indices
        assert "idx_ge_nr_tier" in indices
        assert "idx_ge_nr_unique_source" in indices

    def test_cell_line_indices(self, conn):
        """Cell line lookup indices should exist."""
        indices = {
            row[1]
            for row in conn.execute("PRAGMA index_list('cell_lines')").fetchall()
        }
        assert "idx_cell_lines_ccle" in indices
        assert "idx_cell_lines_stripped" in indices

    def test_prism_indices(self, conn):
        """PRISM bridge indices should exist."""
        indices = {
            row[1]
            for row in conn.execute("PRAGMA index_list('prism_compounds')").fetchall()
        }
        assert "idx_prism_inchikey" in indices
        assert "idx_prism_chembl" in indices
