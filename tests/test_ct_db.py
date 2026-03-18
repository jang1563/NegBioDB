"""Tests for NegBioDB Clinical Trial domain database layer."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from negbiodb_ct.ct_db import (
    create_ct_database,
    get_connection,
    refresh_all_ct_pairs,
    run_ct_migrations,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_ct"


@pytest.fixture
def ct_db(tmp_path):
    """Create a temporary CT database with all migrations applied."""
    db_path = tmp_path / "test_ct.db"
    run_ct_migrations(db_path, MIGRATIONS_DIR)
    return db_path


class TestCTMigrations:
    """Test CT schema creation and migrations."""

    def test_create_ct_database(self, tmp_path):
        db_path = tmp_path / "test.db"
        result = create_ct_database(db_path, MIGRATIONS_DIR)
        assert result == db_path
        assert db_path.exists()

    def test_migration_001_applied(self, ct_db):
        conn = get_connection(ct_db)
        try:
            versions = conn.execute(
                "SELECT version FROM schema_migrations"
            ).fetchall()
            assert ("001",) in versions
        finally:
            conn.close()

    def test_migration_002_applied(self, ct_db):
        conn = get_connection(ct_db)
        try:
            versions = conn.execute(
                "SELECT version FROM schema_migrations"
            ).fetchall()
            assert ("002",) in versions
        finally:
            conn.close()

    def test_idempotent_migrations(self, ct_db):
        """Running migrations twice should not fail."""
        applied = run_ct_migrations(ct_db, MIGRATIONS_DIR)
        assert applied == []  # Nothing new to apply

    def test_all_tables_exist(self, ct_db):
        conn = get_connection(ct_db)
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
                "interventions",
                "conditions",
                "intervention_targets",
                "clinical_trials",
                "trial_interventions",
                "trial_conditions",
                "trial_publications",
                "combination_components",
                "trial_failure_results",
                "intervention_condition_pairs",
                "trial_failure_context",
            }
            assert expected.issubset(tables), f"Missing: {expected - tables}"
        finally:
            conn.close()


class TestCTSchema:
    """Test CT schema constraints and foreign keys."""

    def test_interventions_type_check(self, ct_db):
        conn = get_connection(ct_db)
        try:
            conn.execute(
                "INSERT INTO interventions (intervention_type, intervention_name) "
                "VALUES ('drug', 'Imatinib')"
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO interventions (intervention_type, intervention_name) "
                    "VALUES ('invalid_type', 'Bad')"
                )
        finally:
            conn.close()

    def test_clinical_trial_unique_constraint(self, ct_db):
        conn = get_connection(ct_db)
        try:
            conn.execute(
                "INSERT INTO clinical_trials "
                "(source_db, source_trial_id, overall_status) "
                "VALUES ('clinicaltrials_gov', 'NCT00000001', 'Terminated')"
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO clinical_trials "
                    "(source_db, source_trial_id, overall_status) "
                    "VALUES ('clinicaltrials_gov', 'NCT00000001', 'Completed')"
                )
        finally:
            conn.close()

    def test_same_trial_id_different_source(self, ct_db):
        """Same trial ID from different registries should work."""
        conn = get_connection(ct_db)
        try:
            conn.execute(
                "INSERT INTO clinical_trials "
                "(source_db, source_trial_id, overall_status) "
                "VALUES ('clinicaltrials_gov', 'NCT00000001', 'Terminated')"
            )
            conn.execute(
                "INSERT INTO clinical_trials "
                "(source_db, source_trial_id, overall_status) "
                "VALUES ('eu_ctr', 'NCT00000001', 'Terminated')"
            )
            conn.commit()
            count = conn.execute(
                "SELECT COUNT(*) FROM clinical_trials"
            ).fetchone()[0]
            assert count == 2
        finally:
            conn.close()

    def test_failure_category_check(self, ct_db):
        conn = get_connection(ct_db)
        try:
            # Insert required FK rows
            conn.execute(
                "INSERT INTO interventions (intervention_type, intervention_name) "
                "VALUES ('drug', 'TestDrug')"
            )
            conn.execute(
                "INSERT INTO conditions (condition_name) VALUES ('TestDisease')"
            )
            conn.commit()

            # Valid category
            conn.execute(
                "INSERT INTO trial_failure_results "
                "(intervention_id, condition_id, failure_category, confidence_tier, "
                " source_db, source_record_id, extraction_method) "
                "VALUES (1, 1, 'efficacy', 'gold', 'aact', 'NCT001', 'database_direct')"
            )
            conn.commit()

            # Invalid category
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO trial_failure_results "
                    "(intervention_id, condition_id, failure_category, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (1, 1, 'invalid', 'gold', 'aact', 'NCT002', 'database_direct')"
                )
        finally:
            conn.close()

    def test_confidence_tier_check(self, ct_db):
        conn = get_connection(ct_db)
        try:
            conn.execute(
                "INSERT INTO interventions (intervention_type, intervention_name) "
                "VALUES ('drug', 'TestDrug')"
            )
            conn.execute(
                "INSERT INTO conditions (condition_name) VALUES ('TestDisease')"
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO trial_failure_results "
                    "(intervention_id, condition_id, failure_category, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (1, 1, 'efficacy', 'platinum', 'aact', 'NCT001', 'database_direct')"
                )
        finally:
            conn.close()

    def test_foreign_key_enforcement(self, ct_db):
        """FK violations should raise errors."""
        conn = get_connection(ct_db)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO trial_failure_results "
                    "(intervention_id, condition_id, failure_category, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (999, 999, 'efficacy', 'gold', 'aact', 'NCT001', 'database_direct')"
                )
        finally:
            conn.close()


class TestRefreshCTPairs:
    """Test intervention_condition_pairs aggregation."""

    def _insert_test_data(self, conn):
        """Insert minimal test data for aggregation testing."""
        conn.execute(
            "INSERT INTO interventions (intervention_type, intervention_name) "
            "VALUES ('drug', 'DrugA')"
        )
        conn.execute(
            "INSERT INTO interventions (intervention_type, intervention_name) "
            "VALUES ('drug', 'DrugB')"
        )
        conn.execute(
            "INSERT INTO conditions (condition_name) VALUES ('DiseaseX')"
        )
        conn.execute(
            "INSERT INTO conditions (condition_name) VALUES ('DiseaseY')"
        )
        conn.execute(
            "INSERT INTO clinical_trials "
            "(source_db, source_trial_id, overall_status, trial_phase) "
            "VALUES ('clinicaltrials_gov', 'NCT001', 'Terminated', 'phase_3')"
        )
        conn.execute(
            "INSERT INTO clinical_trials "
            "(source_db, source_trial_id, overall_status, trial_phase) "
            "VALUES ('clinicaltrials_gov', 'NCT002', 'Completed', 'phase_2')"
        )

        # DrugA + DiseaseX: 2 trials (gold + silver)
        conn.execute(
            "INSERT INTO trial_failure_results "
            "(intervention_id, condition_id, trial_id, failure_category, "
            " confidence_tier, source_db, source_record_id, extraction_method, "
            " publication_year, highest_phase_reached) "
            "VALUES (1, 1, 1, 'efficacy', 'gold', 'aact', 'R001', 'database_direct', 2020, 'phase_3')"
        )
        conn.execute(
            "INSERT INTO trial_failure_results "
            "(intervention_id, condition_id, trial_id, failure_category, "
            " confidence_tier, source_db, source_record_id, extraction_method, "
            " publication_year, highest_phase_reached) "
            "VALUES (1, 1, 2, 'efficacy', 'silver', 'aact', 'R002', 'database_direct', 2022, 'phase_2')"
        )

        # DrugA + DiseaseY: 1 trial (bronze)
        conn.execute(
            "INSERT INTO trial_failure_results "
            "(intervention_id, condition_id, failure_category, "
            " confidence_tier, source_db, source_record_id, extraction_method, "
            " publication_year, highest_phase_reached) "
            "VALUES (1, 2, 'safety', 'bronze', 'aact', 'R003', 'nlp_classified', 2021, 'phase_2')"
        )

        # DrugB + DiseaseX: 1 trial (copper)
        conn.execute(
            "INSERT INTO trial_failure_results "
            "(intervention_id, condition_id, failure_category, "
            " confidence_tier, source_db, source_record_id, extraction_method, "
            " publication_year, highest_phase_reached) "
            "VALUES (2, 1, 'enrollment', 'copper', 'aact', 'R004', 'text_mining', 2023, 'phase_1')"
        )
        conn.commit()

    def test_refresh_pair_count(self, ct_db):
        conn = get_connection(ct_db)
        try:
            self._insert_test_data(conn)
            count = refresh_all_ct_pairs(conn)
            conn.commit()
            assert count == 3  # 3 unique intervention-condition pairs
        finally:
            conn.close()

    def test_best_confidence_aggregation(self, ct_db):
        conn = get_connection(ct_db)
        try:
            self._insert_test_data(conn)
            refresh_all_ct_pairs(conn)
            conn.commit()

            row = conn.execute(
                "SELECT best_confidence, num_trials FROM intervention_condition_pairs "
                "WHERE intervention_id = 1 AND condition_id = 1"
            ).fetchone()
            assert row[0] == "gold"  # best of gold + silver
            assert row[1] == 2  # 2 distinct trials
        finally:
            conn.close()

    def test_degree_computation(self, ct_db):
        conn = get_connection(ct_db)
        try:
            self._insert_test_data(conn)
            refresh_all_ct_pairs(conn)
            conn.commit()

            # DrugA tested on 2 conditions → intervention_degree = 2
            row = conn.execute(
                "SELECT intervention_degree FROM intervention_condition_pairs "
                "WHERE intervention_id = 1 AND condition_id = 1"
            ).fetchone()
            assert row[0] == 2

            # DiseaseX tested with 2 drugs → condition_degree = 2
            row = conn.execute(
                "SELECT condition_degree FROM intervention_condition_pairs "
                "WHERE intervention_id = 1 AND condition_id = 1"
            ).fetchone()
            assert row[0] == 2
        finally:
            conn.close()

    def test_refresh_is_idempotent(self, ct_db):
        conn = get_connection(ct_db)
        try:
            self._insert_test_data(conn)
            count1 = refresh_all_ct_pairs(conn)
            conn.commit()
            count2 = refresh_all_ct_pairs(conn)
            conn.commit()
            assert count1 == count2
        finally:
            conn.close()

    def test_highest_phase_ordering(self, ct_db):
        """CASE-based phase ordering should pick phase_3 over not_applicable."""
        conn = get_connection(ct_db)
        try:
            conn.execute(
                "INSERT INTO interventions (intervention_type, intervention_name) "
                "VALUES ('drug', 'DrugX')"
            )
            conn.execute(
                "INSERT INTO conditions (condition_name) VALUES ('DiseaseZ')"
            )
            conn.execute(
                "INSERT INTO clinical_trials "
                "(source_db, source_trial_id, overall_status) "
                "VALUES ('clinicaltrials_gov', 'NCT100', 'Terminated')"
            )
            conn.execute(
                "INSERT INTO clinical_trials "
                "(source_db, source_trial_id, overall_status) "
                "VALUES ('clinicaltrials_gov', 'NCT101', 'Terminated')"
            )
            # Result 1: not_applicable phase
            conn.execute(
                "INSERT INTO trial_failure_results "
                "(intervention_id, condition_id, trial_id, failure_category, "
                " confidence_tier, source_db, source_record_id, extraction_method, "
                " highest_phase_reached) "
                "VALUES (1, 1, 1, 'efficacy', 'bronze', 'aact', 'P001', "
                "'nlp_classified', 'not_applicable')"
            )
            # Result 2: early_phase_1
            conn.execute(
                "INSERT INTO trial_failure_results "
                "(intervention_id, condition_id, trial_id, failure_category, "
                " confidence_tier, source_db, source_record_id, extraction_method, "
                " highest_phase_reached) "
                "VALUES (1, 1, 2, 'efficacy', 'bronze', 'aact', 'P002', "
                "'nlp_classified', 'early_phase_1')"
            )
            conn.commit()

            refresh_all_ct_pairs(conn)
            conn.commit()

            row = conn.execute(
                "SELECT highest_phase_reached FROM intervention_condition_pairs "
                "WHERE intervention_id = 1 AND condition_id = 1"
            ).fetchone()
            # early_phase_1 (rank 2) should beat not_applicable (rank 1)
            assert row[0] == "early_phase_1"
        finally:
            conn.close()


class TestMigration002Fixes:
    """Test migration 002 schema additions."""

    def test_unique_constraint_prevents_duplicates(self, ct_db):
        """Same (intervention, condition, trial, source, record) should fail."""
        conn = get_connection(ct_db)
        try:
            conn.execute(
                "INSERT INTO interventions (intervention_type, intervention_name) "
                "VALUES ('drug', 'DrugU')"
            )
            conn.execute(
                "INSERT INTO conditions (condition_name) VALUES ('DiseaseU')"
            )
            conn.commit()

            conn.execute(
                "INSERT INTO trial_failure_results "
                "(intervention_id, condition_id, failure_category, confidence_tier, "
                " source_db, source_record_id, extraction_method) "
                "VALUES (1, 1, 'efficacy', 'gold', 'aact', 'U001', 'database_direct')"
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO trial_failure_results "
                    "(intervention_id, condition_id, failure_category, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (1, 1, 'safety', 'silver', 'aact', 'U001', 'database_direct')"
                )
        finally:
            conn.close()

    def test_unique_constraint_allows_different_sources(self, ct_db):
        """Same intervention+condition but different source_record_id is OK."""
        conn = get_connection(ct_db)
        try:
            conn.execute(
                "INSERT INTO interventions (intervention_type, intervention_name) "
                "VALUES ('drug', 'DrugV')"
            )
            conn.execute(
                "INSERT INTO conditions (condition_name) VALUES ('DiseaseV')"
            )
            conn.commit()

            conn.execute(
                "INSERT INTO trial_failure_results "
                "(intervention_id, condition_id, failure_category, confidence_tier, "
                " source_db, source_record_id, extraction_method) "
                "VALUES (1, 1, 'efficacy', 'gold', 'aact', 'V001', 'database_direct')"
            )
            conn.execute(
                "INSERT INTO trial_failure_results "
                "(intervention_id, condition_id, failure_category, confidence_tier, "
                " source_db, source_record_id, extraction_method) "
                "VALUES (1, 1, 'efficacy', 'silver', 'aact', 'V002', 'database_direct')"
            )
            conn.commit()
            count = conn.execute(
                "SELECT COUNT(*) FROM trial_failure_results"
            ).fetchone()[0]
            assert count == 2
        finally:
            conn.close()

    def test_interventions_has_inchikey_columns(self, ct_db):
        conn = get_connection(ct_db)
        try:
            conn.execute(
                "INSERT INTO interventions "
                "(intervention_type, intervention_name, inchikey, inchikey_connectivity) "
                "VALUES ('drug', 'InChITest', 'XLYOFNOQVPJJNP-UHFFFAOYSA-N', "
                "'XLYOFNOQVPJJNP')"
            )
            conn.commit()
            row = conn.execute(
                "SELECT inchikey, inchikey_connectivity FROM interventions "
                "WHERE intervention_name = 'InChITest'"
            ).fetchone()
            assert row[0] == "XLYOFNOQVPJJNP-UHFFFAOYSA-N"
            assert row[1] == "XLYOFNOQVPJJNP"
        finally:
            conn.close()

    def test_interventions_molecular_type_check(self, ct_db):
        conn = get_connection(ct_db)
        try:
            # Valid molecular_type
            conn.execute(
                "INSERT INTO interventions "
                "(intervention_type, intervention_name, molecular_type) "
                "VALUES ('drug', 'SmallMol', 'small_molecule')"
            )
            conn.commit()

            # Invalid molecular_type
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO interventions "
                    "(intervention_type, intervention_name, molecular_type) "
                    "VALUES ('drug', 'Bad', 'invalid_type')"
                )
        finally:
            conn.close()

    def test_result_interpretation_check(self, ct_db):
        conn = get_connection(ct_db)
        try:
            conn.execute(
                "INSERT INTO interventions (intervention_type, intervention_name) "
                "VALUES ('drug', 'InterpDrug')"
            )
            conn.execute(
                "INSERT INTO conditions (condition_name) VALUES ('InterpDisease')"
            )
            conn.commit()

            # Valid interpretation
            conn.execute(
                "INSERT INTO trial_failure_results "
                "(intervention_id, condition_id, failure_category, confidence_tier, "
                " source_db, source_record_id, extraction_method, result_interpretation) "
                "VALUES (1, 1, 'efficacy', 'gold', 'aact', 'I001', "
                "'database_direct', 'definitive_negative')"
            )
            conn.commit()

            # Invalid interpretation
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO trial_failure_results "
                    "(intervention_id, condition_id, failure_category, confidence_tier, "
                    " source_db, source_record_id, extraction_method, result_interpretation) "
                    "VALUES (1, 1, 'efficacy', 'gold', 'aact', 'I002', "
                    "'database_direct', 'invalid_interp')"
                )
        finally:
            conn.close()

    def test_termination_type_check(self, ct_db):
        conn = get_connection(ct_db)
        try:
            # Valid termination_type
            conn.execute(
                "INSERT INTO clinical_trials "
                "(source_db, source_trial_id, overall_status, termination_type) "
                "VALUES ('clinicaltrials_gov', 'NCT900', 'Terminated', 'clinical_failure')"
            )
            conn.commit()

            # Invalid termination_type
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO clinical_trials "
                    "(source_db, source_trial_id, overall_status, termination_type) "
                    "VALUES ('clinicaltrials_gov', 'NCT901', 'Terminated', 'invalid')"
                )
        finally:
            conn.close()
