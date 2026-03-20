"""Tests for NegBioDB PPI domain database layer."""

import sqlite3
from pathlib import Path

import pytest

from negbiodb_ppi.ppi_db import (
    create_ppi_database,
    get_connection,
    refresh_all_ppi_pairs,
    run_ppi_migrations,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_ppi"


@pytest.fixture
def ppi_db(tmp_path):
    """Create a temporary PPI database with all migrations applied."""
    db_path = tmp_path / "test_ppi.db"
    run_ppi_migrations(db_path, MIGRATIONS_DIR)
    return db_path


class TestPPIMigrations:
    """Test PPI schema creation and migrations."""

    def test_create_ppi_database(self, tmp_path):
        db_path = tmp_path / "test.db"
        result = create_ppi_database(db_path, MIGRATIONS_DIR)
        assert result == db_path
        assert db_path.exists()

    def test_migration_001_applied(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            versions = conn.execute(
                "SELECT version FROM schema_migrations"
            ).fetchall()
            assert ("001",) in versions
        finally:
            conn.close()

    def test_idempotent_migrations(self, ppi_db):
        """Running migrations twice should not fail."""
        applied = run_ppi_migrations(ppi_db, MIGRATIONS_DIR)
        assert applied == []

    def test_all_tables_exist(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            expected = {
                "schema_migrations",
                "dataset_versions",
                "proteins",
                "ppi_experiments",
                "ppi_negative_results",
                "protein_protein_pairs",
                "ppi_split_definitions",
                "ppi_split_assignments",
            }
            assert expected.issubset(tables), f"Missing: {expected - tables}"
        finally:
            conn.close()

    def test_all_indices_exist(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            indices = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
            expected = {
                "idx_proteins_uniprot",
                "idx_proteins_gene",
                "idx_ppi_exp_source",
                "idx_ppi_nr_protein1",
                "idx_ppi_nr_protein2",
                "idx_ppi_nr_pair",
                "idx_ppi_nr_tier",
                "idx_ppi_nr_source",
                "idx_ppi_nr_unique_source",
                "idx_ppp_protein1",
                "idx_ppp_protein2",
                "idx_ppp_confidence",
                "idx_ppi_splits_fold",
            }
            assert expected.issubset(indices), f"Missing: {expected - indices}"
        finally:
            conn.close()


class TestPPISchema:
    """Test PPI schema constraints and foreign keys."""

    def test_protein_unique_constraint(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('P12345')"
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO proteins (uniprot_accession) VALUES ('P12345')"
                )
        finally:
            conn.close()

    def test_evidence_type_check(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            # Insert required proteins
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('A00001')"
            )
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('B00002')"
            )
            conn.commit()

            # Valid evidence type
            conn.execute(
                "INSERT INTO ppi_negative_results "
                "(protein1_id, protein2_id, evidence_type, confidence_tier, "
                " source_db, source_record_id, extraction_method) "
                "VALUES (1, 2, 'experimental_non_interaction', 'gold', "
                "'intact', 'R001', 'database_direct')"
            )
            conn.commit()

            # Invalid evidence type
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO ppi_negative_results "
                    "(protein1_id, protein2_id, evidence_type, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (1, 2, 'invalid_type', 'gold', "
                    "'intact', 'R002', 'database_direct')"
                )
        finally:
            conn.close()

    def test_confidence_tier_check(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('A00001')"
            )
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('B00002')"
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO ppi_negative_results "
                    "(protein1_id, protein2_id, evidence_type, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (1, 2, 'experimental_non_interaction', 'platinum', "
                    "'intact', 'R001', 'database_direct')"
                )
        finally:
            conn.close()

    def test_canonical_ordering_enforced(self, ppi_db):
        """protein1_id must be < protein2_id."""
        conn = get_connection(ppi_db)
        try:
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('A00001')"
            )
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('B00002')"
            )
            conn.commit()

            # protein1_id=2, protein2_id=1 violates CHECK
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO ppi_negative_results "
                    "(protein1_id, protein2_id, evidence_type, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (2, 1, 'experimental_non_interaction', 'gold', "
                    "'intact', 'R001', 'database_direct')"
                )
        finally:
            conn.close()

    def test_canonical_ordering_equal_ids_rejected(self, ppi_db):
        """protein1_id == protein2_id violates CHECK (strict less-than)."""
        conn = get_connection(ppi_db)
        try:
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('A00001')"
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO ppi_negative_results "
                    "(protein1_id, protein2_id, evidence_type, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (1, 1, 'experimental_non_interaction', 'gold', "
                    "'intact', 'R001', 'database_direct')"
                )
        finally:
            conn.close()

    def test_extraction_method_check(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('A00001')"
            )
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('B00002')"
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO ppi_negative_results "
                    "(protein1_id, protein2_id, evidence_type, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (1, 2, 'experimental_non_interaction', 'gold', "
                    "'intact', 'R001', 'invalid_method')"
                )
        finally:
            conn.close()

    def test_foreign_key_enforcement(self, ppi_db):
        """FK violations should raise errors."""
        conn = get_connection(ppi_db)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO ppi_negative_results "
                    "(protein1_id, protein2_id, evidence_type, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (999, 1000, 'experimental_non_interaction', 'gold', "
                    "'intact', 'R001', 'database_direct')"
                )
        finally:
            conn.close()

    def test_dedup_unique_index(self, ppi_db):
        """Same (pair, experiment, source, record) should fail."""
        conn = get_connection(ppi_db)
        try:
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('A00001')"
            )
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('B00002')"
            )
            conn.commit()

            conn.execute(
                "INSERT INTO ppi_negative_results "
                "(protein1_id, protein2_id, evidence_type, confidence_tier, "
                " source_db, source_record_id, extraction_method) "
                "VALUES (1, 2, 'experimental_non_interaction', 'gold', "
                "'intact', 'R001', 'database_direct')"
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO ppi_negative_results "
                    "(protein1_id, protein2_id, evidence_type, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (1, 2, 'ml_predicted_negative', 'silver', "
                    "'intact', 'R001', 'database_direct')"
                )
        finally:
            conn.close()

    def test_dedup_allows_different_source_records(self, ppi_db):
        """Same pair but different source_record_id is OK."""
        conn = get_connection(ppi_db)
        try:
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('A00001')"
            )
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('B00002')"
            )
            conn.commit()

            conn.execute(
                "INSERT INTO ppi_negative_results "
                "(protein1_id, protein2_id, evidence_type, confidence_tier, "
                " source_db, source_record_id, extraction_method) "
                "VALUES (1, 2, 'experimental_non_interaction', 'gold', "
                "'intact', 'R001', 'database_direct')"
            )
            conn.execute(
                "INSERT INTO ppi_negative_results "
                "(protein1_id, protein2_id, evidence_type, confidence_tier, "
                " source_db, source_record_id, extraction_method) "
                "VALUES (1, 2, 'experimental_non_interaction', 'gold', "
                "'intact', 'R002', 'database_direct')"
            )
            conn.commit()
            count = conn.execute(
                "SELECT COUNT(*) FROM ppi_negative_results"
            ).fetchone()[0]
            assert count == 2
        finally:
            conn.close()

    def test_experiment_source_db_check(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            # Valid source_db
            conn.execute(
                "INSERT INTO ppi_experiments "
                "(source_db, source_experiment_id) "
                "VALUES ('intact', 'EXP001')"
            )
            conn.commit()

            # Invalid source_db
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO ppi_experiments "
                    "(source_db, source_experiment_id) "
                    "VALUES ('invalid_db', 'EXP002')"
                )
        finally:
            conn.close()

    def test_experiment_huri_source_db(self, ppi_db):
        """HuRI is a valid source_db."""
        conn = get_connection(ppi_db)
        try:
            conn.execute(
                "INSERT INTO ppi_experiments "
                "(source_db, source_experiment_id) "
                "VALUES ('huri', 'HI-III-20')"
            )
            conn.commit()
            count = conn.execute(
                "SELECT COUNT(*) FROM ppi_experiments WHERE source_db = 'huri'"
            ).fetchone()[0]
            assert count == 1
        finally:
            conn.close()

    def test_split_strategy_check(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            # Valid strategy
            conn.execute(
                "INSERT INTO ppi_split_definitions "
                "(split_name, split_strategy) "
                "VALUES ('random_v1', 'random')"
            )
            conn.commit()

            # Invalid strategy
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO ppi_split_definitions "
                    "(split_name, split_strategy) "
                    "VALUES ('bad_v1', 'invalid_strategy')"
                )
        finally:
            conn.close()

    def test_pairs_canonical_ordering(self, ppi_db):
        """protein_protein_pairs also enforces protein1 < protein2."""
        conn = get_connection(ppi_db)
        try:
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('A00001')"
            )
            conn.execute(
                "INSERT INTO proteins (uniprot_accession) VALUES ('B00002')"
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO protein_protein_pairs "
                    "(protein1_id, protein2_id, num_experiments, num_sources, "
                    " best_confidence) "
                    "VALUES (2, 1, 1, 1, 'gold')"
                )
        finally:
            conn.close()


class TestRefreshPPIPairs:
    """Test protein_protein_pairs aggregation."""

    def _insert_test_data(self, conn):
        """Insert minimal test data for aggregation testing."""
        # 3 proteins: A(1), B(2), C(3)
        conn.execute(
            "INSERT INTO proteins (uniprot_accession) VALUES ('A00001')"
        )
        conn.execute(
            "INSERT INTO proteins (uniprot_accession) VALUES ('B00002')"
        )
        conn.execute(
            "INSERT INTO proteins (uniprot_accession) VALUES ('C00003')"
        )

        # Experiments
        conn.execute(
            "INSERT INTO ppi_experiments "
            "(source_db, source_experiment_id) "
            "VALUES ('intact', 'EXP001')"
        )
        conn.execute(
            "INSERT INTO ppi_experiments "
            "(source_db, source_experiment_id) "
            "VALUES ('humap', 'HUMAP001')"
        )

        # A-B: 2 results (gold + silver, different sources)
        conn.execute(
            "INSERT INTO ppi_negative_results "
            "(protein1_id, protein2_id, experiment_id, evidence_type, "
            " confidence_tier, source_db, source_record_id, "
            " extraction_method, publication_year, interaction_score) "
            "VALUES (1, 2, 1, 'experimental_non_interaction', 'gold', "
            "'intact', 'R001', 'database_direct', 2020, 0.05)"
        )
        conn.execute(
            "INSERT INTO ppi_negative_results "
            "(protein1_id, protein2_id, experiment_id, evidence_type, "
            " confidence_tier, source_db, source_record_id, "
            " extraction_method, publication_year, interaction_score) "
            "VALUES (1, 2, 2, 'ml_predicted_negative', 'silver', "
            "'humap', 'R002', 'ml_classifier', 2023, 0.01)"
        )

        # A-C: 1 result (bronze)
        conn.execute(
            "INSERT INTO ppi_negative_results "
            "(protein1_id, protein2_id, evidence_type, "
            " confidence_tier, source_db, source_record_id, "
            " extraction_method, publication_year) "
            "VALUES (1, 3, 'low_score_negative', 'bronze', "
            "'string', 'R003', 'score_threshold', 2022)"
        )

        # B-C: 1 result (copper)
        conn.execute(
            "INSERT INTO ppi_negative_results "
            "(protein1_id, protein2_id, evidence_type, "
            " confidence_tier, source_db, source_record_id, "
            " extraction_method, publication_year) "
            "VALUES (2, 3, 'compartment_separated', 'copper', "
            "'string', 'R004', 'score_threshold', 2021)"
        )
        conn.commit()

    def test_refresh_pair_count(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            self._insert_test_data(conn)
            count = refresh_all_ppi_pairs(conn)
            conn.commit()
            assert count == 3  # A-B, A-C, B-C
        finally:
            conn.close()

    def test_best_confidence_aggregation(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            self._insert_test_data(conn)
            refresh_all_ppi_pairs(conn)
            conn.commit()

            row = conn.execute(
                "SELECT best_confidence, num_experiments, num_sources "
                "FROM protein_protein_pairs "
                "WHERE protein1_id = 1 AND protein2_id = 2"
            ).fetchone()
            assert row[0] == "gold"  # best of gold + silver
            assert row[1] == 2  # 2 distinct experiments
            assert row[2] == 2  # 2 distinct sources (intact, humap)
        finally:
            conn.close()

    def test_best_evidence_type_priority(self, ppi_db):
        """experimental_non_interaction should rank above ml_predicted."""
        conn = get_connection(ppi_db)
        try:
            self._insert_test_data(conn)
            refresh_all_ppi_pairs(conn)
            conn.commit()

            row = conn.execute(
                "SELECT best_evidence_type FROM protein_protein_pairs "
                "WHERE protein1_id = 1 AND protein2_id = 2"
            ).fetchone()
            assert row[0] == "experimental_non_interaction"
        finally:
            conn.close()

    def test_score_range(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            self._insert_test_data(conn)
            refresh_all_ppi_pairs(conn)
            conn.commit()

            row = conn.execute(
                "SELECT min_interaction_score, max_interaction_score "
                "FROM protein_protein_pairs "
                "WHERE protein1_id = 1 AND protein2_id = 2"
            ).fetchone()
            assert row[0] == pytest.approx(0.01)
            assert row[1] == pytest.approx(0.05)
        finally:
            conn.close()

    def test_earliest_year(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            self._insert_test_data(conn)
            refresh_all_ppi_pairs(conn)
            conn.commit()

            row = conn.execute(
                "SELECT earliest_year FROM protein_protein_pairs "
                "WHERE protein1_id = 1 AND protein2_id = 2"
            ).fetchone()
            assert row[0] == 2020
        finally:
            conn.close()

    def test_degree_computation_protein1(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            self._insert_test_data(conn)
            refresh_all_ppi_pairs(conn)
            conn.commit()

            # Protein A (id=1) appears as protein1 in A-B and A-C → degree 2
            row = conn.execute(
                "SELECT protein1_degree FROM protein_protein_pairs "
                "WHERE protein1_id = 1 AND protein2_id = 2"
            ).fetchone()
            assert row[0] == 2
        finally:
            conn.close()

    def test_degree_computation_protein2(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            self._insert_test_data(conn)
            refresh_all_ppi_pairs(conn)
            conn.commit()

            # Protein C (id=3) appears as protein2 in A-C and B-C → degree 2
            row = conn.execute(
                "SELECT protein2_degree FROM protein_protein_pairs "
                "WHERE protein1_id = 1 AND protein2_id = 3"
            ).fetchone()
            assert row[0] == 2
        finally:
            conn.close()

    def test_degree_computation_both_sides(self, ppi_db):
        """Protein B (id=2) appears as protein2 in A-B and protein1 in B-C.
        True degree = 2 (partners A and C) regardless of which side it's on."""
        conn = get_connection(ppi_db)
        try:
            self._insert_test_data(conn)
            refresh_all_ppi_pairs(conn)
            conn.commit()

            # In A-B row, protein2 is B → protein2_degree = B's full degree = 2
            row = conn.execute(
                "SELECT protein2_degree FROM protein_protein_pairs "
                "WHERE protein1_id = 1 AND protein2_id = 2"
            ).fetchone()
            assert row[0] == 2

            # In B-C row, protein1 is B → protein1_degree = B's full degree = 2
            row = conn.execute(
                "SELECT protein1_degree FROM protein_protein_pairs "
                "WHERE protein1_id = 2 AND protein2_id = 3"
            ).fetchone()
            assert row[0] == 2
        finally:
            conn.close()

    def test_refresh_is_idempotent(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            self._insert_test_data(conn)
            count1 = refresh_all_ppi_pairs(conn)
            conn.commit()
            count2 = refresh_all_ppi_pairs(conn)
            conn.commit()
            assert count1 == count2
        finally:
            conn.close()

    def test_refresh_clears_split_assignments(self, ppi_db):
        """Refresh should succeed even when split assignments exist (FK)."""
        conn = get_connection(ppi_db)
        try:
            self._insert_test_data(conn)
            refresh_all_ppi_pairs(conn)
            conn.commit()

            # Create a split and assign a pair
            conn.execute(
                "INSERT INTO ppi_split_definitions "
                "(split_name, split_strategy) VALUES ('test_v1', 'random')"
            )
            conn.execute(
                "INSERT INTO ppi_split_assignments (pair_id, split_id, fold) "
                "VALUES (1, 1, 'train')"
            )
            conn.commit()

            # Re-refresh should not crash despite FK to split_assignments
            count = refresh_all_ppi_pairs(conn)
            conn.commit()
            assert count == 3

            # Split assignments should be cleared
            sa_count = conn.execute(
                "SELECT COUNT(*) FROM ppi_split_assignments"
            ).fetchone()[0]
            assert sa_count == 0
        finally:
            conn.close()

    def test_null_experiment_counted(self, ppi_db):
        """Results without experiment_id should use COALESCE sentinel."""
        conn = get_connection(ppi_db)
        try:
            self._insert_test_data(conn)
            refresh_all_ppi_pairs(conn)
            conn.commit()

            # A-C has NULL experiment_id → COALESCE(-1) → counted as 1
            row = conn.execute(
                "SELECT num_experiments FROM protein_protein_pairs "
                "WHERE protein1_id = 1 AND protein2_id = 3"
            ).fetchone()
            assert row[0] == 1
        finally:
            conn.close()

    def test_single_source_count(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            self._insert_test_data(conn)
            refresh_all_ppi_pairs(conn)
            conn.commit()

            # A-C has 1 source (string)
            row = conn.execute(
                "SELECT num_sources FROM protein_protein_pairs "
                "WHERE protein1_id = 1 AND protein2_id = 3"
            ).fetchone()
            assert row[0] == 1
        finally:
            conn.close()
