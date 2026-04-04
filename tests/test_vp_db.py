"""Tests for NegBioDB VP (Variant Pathogenicity) database layer.

Tests migration, connection, table creation, schema constraints,
and pair aggregation with conflict/degree handling.
"""

import sqlite3
from pathlib import Path

import pytest

from negbiodb_vp.vp_db import (
    create_vp_database,
    get_connection,
    refresh_all_vp_pairs,
    run_vp_migrations,
)

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_vp"


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary VP database with all migrations applied."""
    db_path = tmp_path / "test_vp.db"
    run_vp_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def conn(tmp_db):
    """Get a connection to the temporary VP database."""
    c = get_connection(tmp_db)
    yield c
    c.close()


# ── Migration tests ───────────────────────────────────────────────────


class TestMigrations:
    def test_migration_creates_all_tables(self, conn):
        """All 11 expected tables should exist after migration."""
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
            "variants",
            "diseases",
            "vp_submissions",
            "vp_negative_results",
            "variant_disease_pairs",
            "vp_split_definitions",
            "vp_split_assignments",
            "vp_cross_domain_genes",
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
        applied = run_vp_migrations(tmp_db, MIGRATIONS_DIR)
        assert applied == [], "No new migrations expected on second run"

    def test_create_vp_database(self, tmp_path):
        """create_vp_database convenience wrapper should work."""
        db_path = tmp_path / "convenience.db"
        result = create_vp_database(db_path, MIGRATIONS_DIR)
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

    def test_variant_locus_unique(self, conn):
        """Variant locus (chr, pos, ref, alt) should be unique."""
        conn.execute(
            """INSERT INTO variants (chromosome, position, ref_allele, alt_allele, variant_type)
            VALUES ('1', 12345, 'A', 'G', 'single nucleotide variant')"""
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO variants (chromosome, position, ref_allele, alt_allele, variant_type)
                VALUES ('1', 12345, 'A', 'G', 'single nucleotide variant')"""
            )

    def test_variant_type_check(self, conn):
        """Invalid variant type should fail."""
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO variants (chromosome, position, ref_allele, alt_allele, variant_type)
                VALUES ('1', 100, 'A', 'G', 'invalid_type')"""
            )

    def test_consequence_type_check(self, conn):
        """Invalid consequence type should fail."""
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO variants (chromosome, position, ref_allele, alt_allele,
                 variant_type, consequence_type)
                VALUES ('1', 100, 'A', 'G', 'single nucleotide variant', 'invalid')"""
            )

    def test_confidence_tier_check(self, conn):
        """Invalid confidence tier should fail."""
        _insert_base_entities(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO vp_negative_results
                (variant_id, disease_id, classification, evidence_type,
                 confidence_tier, source_db, source_record_id, extraction_method)
                VALUES (1, 1, 'benign', 'expert_reviewed', 'platinum',
                        'clinvar', 'test', 'review_status')"""
            )

    def test_confidence_tier_copper_allowed(self, conn):
        """Copper tier should be accepted (4-tier system)."""
        _insert_base_entities(conn)
        conn.execute(
            """INSERT INTO vp_negative_results
            (variant_id, disease_id, classification, evidence_type,
             confidence_tier, source_db, source_record_id, extraction_method)
            VALUES (1, 1, 'benign', 'population_frequency', 'copper',
                    'gnomad', 'chr1:100:A:G', 'population_af_threshold')"""
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM vp_negative_results").fetchone()[0]
        assert count == 1

    def test_evidence_type_check(self, conn):
        """Invalid evidence type should fail."""
        _insert_base_entities(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO vp_negative_results
                (variant_id, disease_id, classification, evidence_type,
                 confidence_tier, source_db, source_record_id, extraction_method)
                VALUES (1, 1, 'benign', 'invalid_type', 'bronze',
                        'clinvar', 'test', 'review_status')"""
            )

    def test_classification_check(self, conn):
        """Invalid classification should fail."""
        _insert_base_entities(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO vp_negative_results
                (variant_id, disease_id, classification, evidence_type,
                 confidence_tier, source_db, source_record_id, extraction_method)
                VALUES (1, 1, 'pathogenic', 'expert_reviewed', 'gold',
                        'clinvar', 'test', 'review_status')"""
            )

    def test_source_db_check(self, conn):
        """Only 'clinvar' and 'gnomad' should be accepted."""
        _insert_base_entities(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO vp_negative_results
                (variant_id, disease_id, classification, evidence_type,
                 confidence_tier, source_db, source_record_id, extraction_method)
                VALUES (1, 1, 'benign', 'expert_reviewed', 'gold',
                        'dbsnp', 'test', 'review_status')"""
            )

    def test_negative_result_dedup(self, conn):
        """Duplicate variant-disease-submission-source should be rejected."""
        _insert_base_entities(conn)
        conn.execute(
            """INSERT INTO vp_submissions (submission_id, scv_accession, variant_id,
             classification, review_status)
            VALUES (1, 'SCV000001', 1, 'benign', 'criteria_provided_single_submitter')"""
        )
        conn.execute(
            """INSERT INTO vp_negative_results
            (variant_id, disease_id, submission_id, classification, evidence_type,
             confidence_tier, source_db, source_record_id, extraction_method)
            VALUES (1, 1, 1, 'benign', 'single_submitter', 'bronze',
                    'clinvar', '12345', 'single_submission')"""
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO vp_negative_results
                (variant_id, disease_id, submission_id, classification, evidence_type,
                 confidence_tier, source_db, source_record_id, extraction_method)
                VALUES (1, 1, 1, 'benign', 'single_submitter', 'bronze',
                        'clinvar', '12345_dup', 'single_submission')"""
            )

    def test_scv_accession_unique(self, conn):
        """SCV accession should be unique."""
        _insert_base_entities(conn)
        conn.execute(
            """INSERT INTO vp_submissions (scv_accession, variant_id,
             classification, review_status)
            VALUES ('SCV000001', 1, 'benign', 'criteria_provided_single_submitter')"""
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO vp_submissions (scv_accession, variant_id,
                 classification, review_status)
                VALUES ('SCV000001', 1, 'likely_benign', 'criteria_provided_single_submitter')"""
            )

    def test_foreign_key_variant(self, conn):
        """FK from vp_negative_results to variants should be enforced."""
        conn.execute(
            "INSERT INTO diseases (disease_id, canonical_name) VALUES (1, 'Test Disease')"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO vp_negative_results
                (variant_id, disease_id, classification, evidence_type,
                 confidence_tier, source_db, source_record_id, extraction_method)
                VALUES (999, 1, 'benign', 'expert_reviewed', 'gold',
                        'clinvar', 'test', 'review_status')"""
            )

    def test_foreign_key_disease(self, conn):
        """FK from vp_negative_results to diseases should be enforced."""
        conn.execute(
            """INSERT INTO variants (variant_id, chromosome, position, ref_allele,
             alt_allele, variant_type)
            VALUES (1, '1', 100, 'A', 'G', 'single nucleotide variant')"""
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO vp_negative_results
                (variant_id, disease_id, classification, evidence_type,
                 confidence_tier, source_db, source_record_id, extraction_method)
                VALUES (1, 999, 'benign', 'expert_reviewed', 'gold',
                        'clinvar', 'test', 'review_status')"""
            )

    def test_clingen_validity_check(self, conn):
        """Invalid ClinGen validity should fail."""
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO genes (gene_symbol, clingen_validity) VALUES ('TEST', 'Invalid')"
            )

    def test_gene_moi_check(self, conn):
        """Invalid gene MOI should fail."""
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO genes (gene_symbol, gene_moi) VALUES ('TEST', 'Invalid')"
            )

    def test_alphamissense_class_check(self, conn):
        """Invalid AlphaMissense class should fail."""
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO variants (chromosome, position, ref_allele, alt_allele,
                 variant_type, alphamissense_class)
                VALUES ('1', 100, 'A', 'G', 'single nucleotide variant', 'invalid')"""
            )

    def test_split_strategy_check(self, conn):
        """Invalid split strategy should fail (VP has 6 including temporal)."""
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO vp_split_definitions (split_name, split_strategy)
                VALUES ('test', 'scaffold')"""
            )

    def test_temporal_split_allowed(self, conn):
        """Temporal split strategy should be accepted (unique to VP)."""
        conn.execute(
            """INSERT INTO vp_split_definitions (split_name, split_strategy)
            VALUES ('temporal_v1', 'temporal')"""
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM vp_split_definitions").fetchone()[0]
        assert count == 1

    def test_cross_domain_bridge_unique(self, conn):
        """Cross-domain bridge entries should be unique per gene+domain+external_id."""
        conn.execute("INSERT INTO genes (gene_id, gene_symbol) VALUES (1, 'TP53')")
        conn.execute(
            """INSERT INTO vp_cross_domain_genes (gene_id, domain, external_id)
            VALUES (1, 'ge', '7157')"""
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO vp_cross_domain_genes (gene_id, domain, external_id)
                VALUES (1, 'ge', '7157')"""
            )

    def test_cross_domain_different_domains(self, conn):
        """Same gene can bridge to multiple domains."""
        conn.execute("INSERT INTO genes (gene_id, gene_symbol) VALUES (1, 'TP53')")
        conn.execute(
            """INSERT INTO vp_cross_domain_genes (gene_id, domain, external_id)
            VALUES (1, 'ge', '7157')"""
        )
        conn.execute(
            """INSERT INTO vp_cross_domain_genes (gene_id, domain, external_id)
            VALUES (1, 'ppi', 'P04637')"""
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM vp_cross_domain_genes").fetchone()[0]
        assert count == 2


# ── Pair aggregation tests ────────────────────────────────────────────


def _insert_base_entities(conn):
    """Insert minimal entities for constraint tests."""
    conn.execute(
        """INSERT OR IGNORE INTO genes (gene_id, entrez_id, gene_symbol)
        VALUES (1, 7157, 'TP53')"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO variants (variant_id, chromosome, position,
         ref_allele, alt_allele, variant_type, gene_id, gnomad_af_global)
        VALUES (1, '17', 7579472, 'G', 'A', 'single nucleotide variant', 1, 0.0001)"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO diseases (disease_id, medgen_cui, canonical_name)
        VALUES (1, 'C0006142', 'Breast cancer')"""
    )
    conn.commit()


def _insert_aggregation_test_data(conn):
    """Insert synthetic test data for pair aggregation tests."""
    # 2 genes
    conn.execute(
        "INSERT INTO genes (gene_id, entrez_id, gene_symbol) VALUES (1, 7157, 'TP53')"
    )
    conn.execute(
        "INSERT INTO genes (gene_id, entrez_id, gene_symbol) VALUES (2, 672, 'BRCA1')"
    )

    # 3 variants (2 for TP53, 1 for BRCA1)
    conn.execute(
        """INSERT INTO variants (variant_id, chromosome, position, ref_allele, alt_allele,
         variant_type, gene_id, gnomad_af_global)
        VALUES (1, '17', 7579472, 'G', 'A', 'single nucleotide variant', 1, 0.15)"""
    )
    conn.execute(
        """INSERT INTO variants (variant_id, chromosome, position, ref_allele, alt_allele,
         variant_type, gene_id, gnomad_af_global)
        VALUES (2, '17', 7579500, 'C', 'T', 'single nucleotide variant', 1, 0.002)"""
    )
    conn.execute(
        """INSERT INTO variants (variant_id, chromosome, position, ref_allele, alt_allele,
         variant_type, gene_id, gnomad_af_global)
        VALUES (3, '17', 43092919, 'A', 'G', 'single nucleotide variant', 2, 0.05)"""
    )

    # 3 diseases
    conn.execute(
        "INSERT INTO diseases (disease_id, medgen_cui, canonical_name) VALUES (1, 'C0006142', 'Breast cancer')"
    )
    conn.execute(
        "INSERT INTO diseases (disease_id, medgen_cui, canonical_name) VALUES (2, 'C0009402', 'Colorectal cancer')"
    )
    conn.execute(
        "INSERT INTO diseases (disease_id, medgen_cui, canonical_name) VALUES (3, 'C0677776', 'Hereditary cancer')"
    )

    # Submissions
    conn.execute(
        """INSERT INTO vp_submissions (submission_id, scv_accession, variant_id,
         submitter_name, classification, review_status, submission_year)
        VALUES (1, 'SCV000001', 1, 'Lab A', 'benign', 'reviewed_by_expert_panel', 2018)"""
    )
    conn.execute(
        """INSERT INTO vp_submissions (submission_id, scv_accession, variant_id,
         submitter_name, classification, review_status, submission_year)
        VALUES (2, 'SCV000002', 1, 'Lab B', 'benign', 'criteria_provided_single_submitter', 2020)"""
    )
    conn.execute(
        """INSERT INTO vp_submissions (submission_id, scv_accession, variant_id,
         submitter_name, classification, review_status, submission_year)
        VALUES (3, 'SCV000003', 2, 'Lab C', 'likely_benign', 'criteria_provided_single_submitter', 2022)"""
    )
    conn.execute(
        """INSERT INTO vp_submissions (submission_id, scv_accession, variant_id,
         submitter_name, classification, review_status, submission_year)
        VALUES (4, 'SCV000004', 3, 'Lab A', 'benign', 'criteria_provided_multiple_submitters_no_conflicts', 2019)"""
    )

    # Negative results:
    # Variant 1 + Disease 1: gold (expert_reviewed), 2 submissions, 2 submitters
    conn.execute(
        """INSERT INTO vp_negative_results
        (variant_id, disease_id, submission_id, classification, evidence_type,
         confidence_tier, source_db, source_record_id, extraction_method,
         submission_year, has_conflict, num_benign_criteria)
        VALUES (1, 1, 1, 'benign', 'expert_reviewed', 'gold',
                'clinvar', '100001', 'review_status', 2018, 0, 3)"""
    )
    conn.execute(
        """INSERT INTO vp_negative_results
        (variant_id, disease_id, submission_id, classification, evidence_type,
         confidence_tier, source_db, source_record_id, extraction_method,
         submission_year, has_conflict, num_benign_criteria)
        VALUES (1, 1, 2, 'benign', 'single_submitter', 'bronze',
                'clinvar', '100002', 'single_submission', 2020, 0, 1)"""
    )

    # Variant 1 + Disease 2: bronze (single_submitter), 1 submission
    conn.execute(
        """INSERT INTO vp_negative_results
        (variant_id, disease_id, submission_id, classification, evidence_type,
         confidence_tier, source_db, source_record_id, extraction_method,
         submission_year, has_conflict, num_benign_criteria)
        VALUES (1, 2, 2, 'benign', 'single_submitter', 'bronze',
                'clinvar', '100003', 'single_submission', 2020, 0, 1)"""
    )

    # Variant 2 + Disease 1: bronze (single_submitter), likely_benign
    conn.execute(
        """INSERT INTO vp_negative_results
        (variant_id, disease_id, submission_id, classification, evidence_type,
         confidence_tier, source_db, source_record_id, extraction_method,
         submission_year, has_conflict, num_benign_criteria)
        VALUES (2, 1, 3, 'likely_benign', 'single_submitter', 'bronze',
                'clinvar', '100004', 'single_submission', 2022, 1, 0)"""
    )

    # Variant 3 + Disease 3: silver (multi_submitter_concordant)
    conn.execute(
        """INSERT INTO vp_negative_results
        (variant_id, disease_id, submission_id, classification, evidence_type,
         confidence_tier, source_db, source_record_id, extraction_method,
         submission_year, has_conflict, num_benign_criteria)
        VALUES (3, 3, 4, 'benign', 'multi_submitter_concordant', 'silver',
                'clinvar', '100005', 'submitter_concordance', 2019, 0, 2)"""
    )

    # Copper-tier gnomAD-only entry: Variant 3 + Disease 1 (population frequency)
    conn.execute(
        """INSERT INTO vp_negative_results
        (variant_id, disease_id, classification, evidence_type,
         confidence_tier, source_db, source_record_id, extraction_method,
         submission_year, has_conflict, num_benign_criteria)
        VALUES (3, 1, 'benign', 'population_frequency', 'copper',
                'gnomad', 'chr17:43092919:A:G', 'population_af_threshold',
                NULL, 0, 0)"""
    )

    conn.commit()


class TestPairAggregation:
    def test_refresh_pair_count(self, conn):
        """Should create correct number of aggregated pairs."""
        _insert_aggregation_test_data(conn)
        count = refresh_all_vp_pairs(conn)
        conn.commit()
        # V1+D1, V1+D2, V2+D1, V3+D3, V3+D1 = 5 pairs
        assert count == 5

    def test_best_confidence_gold(self, conn):
        """Pair with gold + bronze submissions should have best_confidence=gold."""
        _insert_aggregation_test_data(conn)
        refresh_all_vp_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT best_confidence, best_evidence_type, num_submissions, num_submitters
            FROM variant_disease_pairs WHERE variant_id = 1 AND disease_id = 1"""
        ).fetchone()
        assert row[0] == "gold"
        assert row[1] == "expert_reviewed"
        assert row[2] == 2  # num_submissions
        assert row[3] == 2  # num_submitters (Lab A, Lab B)

    def test_best_classification_benign(self, conn):
        """best_classification should be 'benign' when at least one benign submission."""
        _insert_aggregation_test_data(conn)
        refresh_all_vp_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT best_classification
            FROM variant_disease_pairs WHERE variant_id = 1 AND disease_id = 1"""
        ).fetchone()
        assert row[0] == "benign"

    def test_best_classification_likely_benign(self, conn):
        """best_classification should be 'likely_benign' when only likely_benign."""
        _insert_aggregation_test_data(conn)
        refresh_all_vp_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT best_classification
            FROM variant_disease_pairs WHERE variant_id = 2 AND disease_id = 1"""
        ).fetchone()
        assert row[0] == "likely_benign"

    def test_earliest_year(self, conn):
        """earliest_year should be the minimum submission year."""
        _insert_aggregation_test_data(conn)
        refresh_all_vp_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT earliest_year
            FROM variant_disease_pairs WHERE variant_id = 1 AND disease_id = 1"""
        ).fetchone()
        assert row[0] == 2018

    def test_has_conflict_flag(self, conn):
        """has_conflict should propagate from vp_negative_results."""
        _insert_aggregation_test_data(conn)
        refresh_all_vp_pairs(conn)
        conn.commit()
        # Variant 2 + Disease 1 has_conflict = 1
        row = conn.execute(
            """SELECT has_conflict
            FROM variant_disease_pairs WHERE variant_id = 2 AND disease_id = 1"""
        ).fetchone()
        assert row[0] == 1
        # Variant 1 + Disease 1 has_conflict = 0
        row = conn.execute(
            """SELECT has_conflict
            FROM variant_disease_pairs WHERE variant_id = 1 AND disease_id = 1"""
        ).fetchone()
        assert row[0] == 0

    def test_max_population_af(self, conn):
        """max_population_af should come from variants.gnomad_af_global."""
        _insert_aggregation_test_data(conn)
        refresh_all_vp_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT max_population_af
            FROM variant_disease_pairs WHERE variant_id = 1 AND disease_id = 1"""
        ).fetchone()
        assert row[0] == pytest.approx(0.15)

    def test_num_benign_criteria(self, conn):
        """num_benign_criteria should be MAX of submissions."""
        _insert_aggregation_test_data(conn)
        refresh_all_vp_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT num_benign_criteria
            FROM variant_disease_pairs WHERE variant_id = 1 AND disease_id = 1"""
        ).fetchone()
        assert row[0] == 3  # max of 3, 1

    def test_copper_tier_pair(self, conn):
        """Copper-tier gnomAD pair should be aggregated correctly."""
        _insert_aggregation_test_data(conn)
        refresh_all_vp_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT best_confidence, best_evidence_type, num_submitters
            FROM variant_disease_pairs WHERE variant_id = 3 AND disease_id = 1"""
        ).fetchone()
        assert row[0] == "copper"
        assert row[1] == "population_frequency"
        # gnomAD entry has no submission → submitter_name is NULL → COUNT(DISTINCT NULL) = 0
        assert row[2] == 0

    def test_variant_degree(self, conn):
        """Variant 1 should have degree 2 (benign for 2 diseases)."""
        _insert_aggregation_test_data(conn)
        refresh_all_vp_pairs(conn)
        conn.commit()
        row = conn.execute(
            "SELECT variant_degree FROM variant_disease_pairs WHERE variant_id = 1 LIMIT 1"
        ).fetchone()
        assert row[0] == 2

    def test_disease_degree(self, conn):
        """Disease 1 should have degree 3 (3 variants benign for it)."""
        _insert_aggregation_test_data(conn)
        refresh_all_vp_pairs(conn)
        conn.commit()
        row = conn.execute(
            "SELECT disease_degree FROM variant_disease_pairs WHERE disease_id = 1 LIMIT 1"
        ).fetchone()
        assert row[0] == 3  # V1, V2, V3

    def test_refresh_clears_old_pairs(self, conn):
        """Refreshing should delete old pairs and split assignments."""
        _insert_aggregation_test_data(conn)
        refresh_all_vp_pairs(conn)
        conn.commit()

        # Add a split assignment
        conn.execute(
            """INSERT INTO vp_split_definitions
            (split_name, split_strategy) VALUES ('test_split', 'random')"""
        )
        pair_id = conn.execute(
            "SELECT pair_id FROM variant_disease_pairs LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO vp_split_assignments (pair_id, split_id, fold)
            VALUES (?, 1, 'train')""",
            (pair_id,),
        )
        conn.commit()

        # Refresh again
        count = refresh_all_vp_pairs(conn)
        conn.commit()
        assert count == 5

        # Split assignments should be cleared
        sa_count = conn.execute(
            "SELECT COUNT(*) FROM vp_split_assignments"
        ).fetchone()[0]
        assert sa_count == 0

    def test_empty_results(self, conn):
        """Refreshing with no results should produce 0 pairs."""
        count = refresh_all_vp_pairs(conn)
        assert count == 0


# ── Index tests ───────────────────────────────────────────────────────


class TestIndices:
    def test_negative_result_indices(self, conn):
        """Critical indices on vp_negative_results should exist."""
        indices = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list('vp_negative_results')"
            ).fetchall()
        }
        assert "idx_vp_nr_variant" in indices
        assert "idx_vp_nr_disease" in indices
        assert "idx_vp_nr_pair" in indices
        assert "idx_vp_nr_tier" in indices
        assert "idx_vp_nr_unique_source" in indices

    def test_variant_indices(self, conn):
        """Variant lookup indices should exist."""
        indices = {
            row[1]
            for row in conn.execute("PRAGMA index_list('variants')").fetchall()
        }
        assert "idx_variants_locus" in indices
        assert "idx_variants_clinvar" in indices
        assert "idx_variants_gene" in indices

    def test_disease_indices(self, conn):
        """Disease lookup indices should exist."""
        indices = {
            row[1]
            for row in conn.execute("PRAGMA index_list('diseases')").fetchall()
        }
        assert "idx_diseases_medgen" in indices
        assert "idx_diseases_name" in indices

    def test_pair_indices(self, conn):
        """Pair aggregation indices should exist."""
        indices = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list('variant_disease_pairs')"
            ).fetchall()
        }
        assert "idx_vdp_variant" in indices
        assert "idx_vdp_disease" in indices
        assert "idx_vdp_confidence" in indices

    def test_cross_domain_indices(self, conn):
        """Cross-domain bridge indices should exist."""
        indices = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list('vp_cross_domain_genes')"
            ).fetchall()
        }
        assert "idx_vp_bridge_gene" in indices
        assert "idx_vp_bridge_domain" in indices
