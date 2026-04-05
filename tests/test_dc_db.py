"""Tests for NegBioDB DC (Drug Combination Synergy) database layer.

Tests migration, connection, table creation, schema constraints,
symmetric pair ordering, and pair aggregation with degree/target overlap.
"""

import sqlite3
from pathlib import Path

import pytest

from negbiodb_dc.dc_db import (
    classify_synergy,
    create_dc_database,
    get_connection,
    normalize_pair,
    refresh_all_drug_pairs,
    run_dc_migrations,
)

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_dc"


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary DC database with all migrations applied."""
    db_path = tmp_path / "test_dc.db"
    run_dc_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def conn(tmp_db):
    """Get a connection to the temporary DC database."""
    c = get_connection(tmp_db)
    yield c
    c.close()


# ── Migration tests ───────────────────────────────────────────────────


class TestMigrations:
    def test_migration_creates_all_tables(self, conn):
        """All 12 expected tables should exist after migration."""
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {
            "schema_migrations",
            "dataset_versions",
            "compounds",
            "cell_lines",
            "drug_targets",
            "dc_synergy_results",
            "drug_drug_pairs",
            "drug_drug_cell_line_triples",
            "dc_split_definitions",
            "dc_split_assignments",
            "dc_cross_domain_compounds",
            "dc_cross_domain_cell_lines",
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
        applied = run_dc_migrations(tmp_db, MIGRATIONS_DIR)
        assert applied == [], "No new migrations expected on second run"

    def test_create_dc_database(self, tmp_path):
        """create_dc_database convenience wrapper should work."""
        db_path = tmp_path / "convenience.db"
        result = create_dc_database(db_path, MIGRATIONS_DIR)
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


# ── Dataset versions tests ────────────────────────────────────────────


class TestDatasetVersions:
    def test_insert_dataset_version(self, conn):
        """Can insert dataset version tracking records."""
        conn.execute(
            """INSERT INTO dataset_versions (name, version, source_url, row_count)
            VALUES ('drugcomb', '1.4', 'https://zenodo.org/records/18449193', 739964)"""
        )
        conn.commit()
        row = conn.execute(
            "SELECT name, version, row_count FROM dataset_versions WHERE name = 'drugcomb'"
        ).fetchone()
        assert row == ("drugcomb", "1.4", 739964)

    def test_multiple_versions(self, conn):
        """Can track multiple dataset versions."""
        conn.execute(
            "INSERT INTO dataset_versions (name, version) VALUES ('drugcomb', '1.3')"
        )
        conn.execute(
            "INSERT INTO dataset_versions (name, version) VALUES ('drugcomb', '1.4')"
        )
        conn.execute(
            "INSERT INTO dataset_versions (name, version) VALUES ('nci_almanac', 'Nov2017')"
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0]
        assert count == 3


# ── Classify synergy tests ───────────────────────────────────────────


class TestClassifySynergy:
    def test_strongly_synergistic(self):
        assert classify_synergy(15.0) == "strongly_synergistic"

    def test_synergistic(self):
        assert classify_synergy(7.5) == "synergistic"

    def test_additive(self):
        assert classify_synergy(0.0) == "additive"
        assert classify_synergy(5.0) == "additive"
        assert classify_synergy(-5.0) == "additive"

    def test_antagonistic(self):
        assert classify_synergy(-7.5) == "antagonistic"

    def test_strongly_antagonistic(self):
        assert classify_synergy(-15.0) == "strongly_antagonistic"

    def test_none(self):
        assert classify_synergy(None) is None

    def test_boundary_values(self):
        """Test exact boundary values."""
        assert classify_synergy(10.01) == "strongly_synergistic"
        assert classify_synergy(10.0) == "synergistic"  # 10 is ≤ 10
        assert classify_synergy(5.01) == "synergistic"
        assert classify_synergy(5.0) == "additive"  # 5 is ≤ 5
        assert classify_synergy(-5.0) == "additive"  # -5 is ≥ -5
        assert classify_synergy(-5.01) == "antagonistic"
        assert classify_synergy(-10.0) == "antagonistic"  # -10 is ≥ -10
        assert classify_synergy(-10.01) == "strongly_antagonistic"


# ── Normalize pair tests ─────────────────────────────────────────────


class TestNormalizePair:
    def test_already_ordered(self):
        assert normalize_pair(1, 2) == (1, 2)

    def test_reverse_ordered(self):
        assert normalize_pair(5, 3) == (3, 5)

    def test_same_compound_raises(self):
        with pytest.raises(ValueError, match="same compound"):
            normalize_pair(1, 1)


# ── Schema constraint tests ──────────────────────────────────────────


class TestSchemaConstraints:
    def test_compound_name_unique(self, conn):
        """Drug name should be unique."""
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('Aspirin')")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO compounds (drug_name) VALUES ('Aspirin')")

    def test_cell_line_name_unique(self, conn):
        """Cell line name should be unique."""
        conn.execute("INSERT INTO cell_lines (cell_line_name) VALUES ('MCF7')")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO cell_lines (cell_line_name) VALUES ('MCF7')")

    def test_synergy_class_check(self, conn):
        """Invalid synergy class should fail."""
        _insert_base_entities(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO dc_synergy_results
                (compound_a_id, compound_b_id, cell_line_id, synergy_class,
                 source_db)
                VALUES (1, 2, 1, 'invalid_class', 'drugcomb')"""
            )

    def test_confidence_tier_check(self, conn):
        """Invalid confidence tier should fail."""
        _insert_base_entities(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO dc_synergy_results
                (compound_a_id, compound_b_id, cell_line_id, confidence_tier,
                 source_db)
                VALUES (1, 2, 1, 'platinum', 'drugcomb')"""
            )

    def test_evidence_type_check(self, conn):
        """Invalid evidence type should fail."""
        _insert_base_entities(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO dc_synergy_results
                (compound_a_id, compound_b_id, cell_line_id, evidence_type,
                 source_db)
                VALUES (1, 2, 1, 'invalid_evidence', 'drugcomb')"""
            )

    def test_source_db_check(self, conn):
        """Only drugcomb, nci_almanac, az_dream should be accepted."""
        _insert_base_entities(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO dc_synergy_results
                (compound_a_id, compound_b_id, cell_line_id, source_db)
                VALUES (1, 2, 1, 'chembl')"""
            )

    def test_source_db_valid_values(self, conn):
        """All three valid source_db values should be accepted."""
        _insert_base_entities(conn)
        for src in ("drugcomb", "nci_almanac", "az_dream"):
            conn.execute(
                """INSERT INTO dc_synergy_results
                (compound_a_id, compound_b_id, cell_line_id, source_db,
                 synergy_class, confidence_tier)
                VALUES (1, 2, 1, ?, 'additive', 'bronze')""",
                (src,),
            )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM dc_synergy_results").fetchone()[0]
        assert count == 3

    def test_source_db_not_null(self, conn):
        """source_db must not be NULL (NOT NULL constraint)."""
        _insert_base_entities(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO dc_synergy_results
                (compound_a_id, compound_b_id, cell_line_id, source_db)
                VALUES (1, 2, 1, NULL)"""
            )

    def test_copper_tier_allowed(self, conn):
        """Copper tier should be accepted (4-tier system)."""
        _insert_base_entities(conn)
        conn.execute(
            """INSERT INTO dc_synergy_results
            (compound_a_id, compound_b_id, cell_line_id, source_db,
             synergy_class, confidence_tier, evidence_type)
            VALUES (1, 2, 1, 'drugcomb', 'additive', 'copper', 'single_concentration')"""
        )
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM dc_synergy_results WHERE confidence_tier = 'copper'"
        ).fetchone()[0]
        assert count == 1

    def test_pair_ordering_constraint(self, conn):
        """compound_a_id must be < compound_b_id (CHECK constraint)."""
        _insert_base_entities(conn)
        # a_id=2 > b_id=1 should fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO dc_synergy_results
                (compound_a_id, compound_b_id, cell_line_id, source_db)
                VALUES (2, 1, 1, 'drugcomb')"""
            )

    def test_pair_ordering_equal_fails(self, conn):
        """compound_a_id == compound_b_id should fail CHECK constraint."""
        _insert_base_entities(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO dc_synergy_results
                (compound_a_id, compound_b_id, cell_line_id, source_db)
                VALUES (1, 1, 1, 'drugcomb')"""
            )

    def test_foreign_key_compound_a(self, conn):
        """FK from dc_synergy_results to compounds (compound_a_id) should be enforced."""
        conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (2, 'Drug B')")
        conn.execute("INSERT INTO cell_lines (cell_line_id, cell_line_name) VALUES (1, 'MCF7')")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO dc_synergy_results
                (compound_a_id, compound_b_id, cell_line_id, source_db)
                VALUES (1, 2, 1, 'drugcomb')"""
            )

    def test_foreign_key_cell_line(self, conn):
        """FK from dc_synergy_results to cell_lines should be enforced."""
        conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (1, 'Drug A')")
        conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (2, 'Drug B')")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO dc_synergy_results
                (compound_a_id, compound_b_id, cell_line_id, source_db)
                VALUES (1, 2, 999, 'drugcomb')"""
            )

    def test_drug_targets_pk(self, conn):
        """Drug target (compound_id, gene_symbol, source) should be unique PK."""
        conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (1, 'Drug A')")
        conn.execute(
            """INSERT INTO drug_targets (compound_id, gene_symbol, source)
            VALUES (1, 'EGFR', 'chembl')"""
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO drug_targets (compound_id, gene_symbol, source)
                VALUES (1, 'EGFR', 'chembl')"""
            )

    def test_drug_targets_different_source(self, conn):
        """Same target from different sources should be allowed."""
        conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (1, 'Drug A')")
        conn.execute(
            """INSERT INTO drug_targets (compound_id, gene_symbol, source)
            VALUES (1, 'EGFR', 'chembl')"""
        )
        conn.execute(
            """INSERT INTO drug_targets (compound_id, gene_symbol, source)
            VALUES (1, 'EGFR', 'drugbank')"""
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM drug_targets").fetchone()[0]
        assert count == 2

    def test_drug_targets_source_check(self, conn):
        """Invalid drug target source should fail."""
        conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (1, 'Drug A')")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO drug_targets (compound_id, gene_symbol, source)
                VALUES (1, 'EGFR', 'invalid')"""
            )

    def test_split_strategy_check(self, conn):
        """Invalid split strategy should fail."""
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO dc_split_definitions (split_name, split_strategy)
                VALUES ('test', 'temporal')"""
            )

    def test_leave_one_tissue_out_allowed(self, conn):
        """leave_one_tissue_out split strategy should be accepted (novel for DC)."""
        conn.execute(
            """INSERT INTO dc_split_definitions (split_name, split_strategy)
            VALUES ('loto_v1', 'leave_one_tissue_out')"""
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM dc_split_definitions").fetchone()[0]
        assert count == 1

    def test_scaffold_split_allowed(self, conn):
        """Scaffold split strategy should be accepted."""
        conn.execute(
            """INSERT INTO dc_split_definitions (split_name, split_strategy)
            VALUES ('scaffold_v1', 'scaffold')"""
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM dc_split_definitions").fetchone()[0]
        assert count == 1

    def test_consensus_class_check(self, conn):
        """Invalid consensus_class on drug_drug_pairs should fail."""
        _insert_base_entities(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO drug_drug_pairs
                (compound_a_id, compound_b_id, consensus_class)
                VALUES (1, 2, 'invalid')"""
            )

    def test_drug_drug_pairs_ordering(self, conn):
        """drug_drug_pairs CHECK compound_a_id < compound_b_id."""
        _insert_base_entities(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO drug_drug_pairs
                (compound_a_id, compound_b_id, consensus_class)
                VALUES (2, 1, 'additive')"""
            )

    def test_drug_drug_pairs_unique(self, conn):
        """drug_drug_pairs should enforce UNIQUE(compound_a_id, compound_b_id)."""
        _insert_base_entities(conn)
        conn.execute(
            """INSERT INTO drug_drug_pairs
            (compound_a_id, compound_b_id) VALUES (1, 2)"""
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO drug_drug_pairs
                (compound_a_id, compound_b_id) VALUES (1, 2)"""
            )

    def test_cross_domain_compound_bridge_unique(self, conn):
        """Cross-domain compound bridge should be unique per compound+domain+external_id."""
        conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (1, 'Drug A')")
        conn.execute(
            """INSERT INTO dc_cross_domain_compounds (compound_id, domain, external_id)
            VALUES (1, 'dti', 'ABCDEF123')"""
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO dc_cross_domain_compounds (compound_id, domain, external_id)
                VALUES (1, 'dti', 'ABCDEF123')"""
            )

    def test_cross_domain_compound_different_domains(self, conn):
        """Same compound can bridge to multiple domains."""
        conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (1, 'Drug A')")
        conn.execute(
            """INSERT INTO dc_cross_domain_compounds (compound_id, domain, external_id)
            VALUES (1, 'dti', 'ABCDEF123')"""
        )
        conn.execute(
            """INSERT INTO dc_cross_domain_compounds (compound_id, domain, external_id)
            VALUES (1, 'ct', 'Aspirin')"""
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM dc_cross_domain_compounds").fetchone()[0]
        assert count == 2

    def test_cross_domain_cell_line_bridge(self, conn):
        """Cross-domain cell line bridge should work."""
        conn.execute("INSERT INTO cell_lines (cell_line_id, cell_line_name) VALUES (1, 'MCF7')")
        conn.execute(
            """INSERT INTO dc_cross_domain_cell_lines (cell_line_id, domain, external_id)
            VALUES (1, 'ge', 'ACH-000019')"""
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM dc_cross_domain_cell_lines").fetchone()[0]
        assert count == 1


# ── Pair aggregation tests ────────────────────────────────────────────


def _insert_base_entities(conn):
    """Insert minimal entities for constraint tests."""
    conn.execute(
        "INSERT OR IGNORE INTO compounds (compound_id, drug_name) VALUES (1, 'Drug A')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO compounds (compound_id, drug_name) VALUES (2, 'Drug B')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO cell_lines (cell_line_id, cell_line_name) VALUES (1, 'MCF7')"
    )
    conn.commit()


def _insert_aggregation_test_data(conn):
    """Insert synthetic test data for pair aggregation tests.

    Creates:
    - 3 compounds (Drug A, Drug B, Drug C)
    - 2 cell lines (MCF7, A549)
    - Drug targets: Drug A → EGFR, BRAF; Drug B → EGFR, KRAS; Drug C → TP53
    - Synergy results:
        Drug A + Drug B in MCF7: antagonistic (ZIP -8, bronze, drugcomb)
        Drug A + Drug B in MCF7: antagonistic (ZIP -7, bronze, nci_almanac) — multi-source
        Drug A + Drug B in A549: synergistic (ZIP 7, bronze, drugcomb)
        Drug A + Drug C in MCF7: additive (ZIP 1, copper, drugcomb)
    """
    # Compounds
    conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (1, 'Drug A')")
    conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (2, 'Drug B')")
    conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (3, 'Drug C')")

    # Cell lines
    conn.execute(
        "INSERT INTO cell_lines (cell_line_id, cell_line_name, tissue) VALUES (1, 'MCF7', 'breast')"
    )
    conn.execute(
        "INSERT INTO cell_lines (cell_line_id, cell_line_name, tissue) VALUES (2, 'A549', 'lung')"
    )

    # Drug targets (for overlap computation)
    # Drug A: EGFR, BRAF
    conn.execute(
        "INSERT INTO drug_targets (compound_id, gene_symbol, source) VALUES (1, 'EGFR', 'chembl')"
    )
    conn.execute(
        "INSERT INTO drug_targets (compound_id, gene_symbol, source) VALUES (1, 'BRAF', 'chembl')"
    )
    # Drug B: EGFR, KRAS (shares EGFR with Drug A)
    conn.execute(
        "INSERT INTO drug_targets (compound_id, gene_symbol, source) VALUES (2, 'EGFR', 'chembl')"
    )
    conn.execute(
        "INSERT INTO drug_targets (compound_id, gene_symbol, source) VALUES (2, 'KRAS', 'chembl')"
    )
    # Drug C: TP53 (no overlap with Drug A)
    conn.execute(
        "INSERT INTO drug_targets (compound_id, gene_symbol, source) VALUES (3, 'TP53', 'chembl')"
    )

    # Synergy results: Drug A (1) + Drug B (2) — multi-source, multi-cell-line
    conn.execute(
        """INSERT INTO dc_synergy_results
        (compound_a_id, compound_b_id, cell_line_id, zip_score, bliss_score,
         synergy_class, confidence_tier, evidence_type, source_db,
         num_concentrations, has_dose_matrix)
        VALUES (1, 2, 1, -8.0, -6.0, 'antagonistic', 'bronze', 'dose_response_matrix',
                'drugcomb', 5, 1)"""
    )
    conn.execute(
        """INSERT INTO dc_synergy_results
        (compound_a_id, compound_b_id, cell_line_id, zip_score, bliss_score,
         synergy_class, confidence_tier, evidence_type, source_db,
         num_concentrations, has_dose_matrix)
        VALUES (1, 2, 1, -7.0, -5.5, 'antagonistic', 'bronze', 'dose_response_matrix',
                'nci_almanac', 3, 1)"""
    )
    conn.execute(
        """INSERT INTO dc_synergy_results
        (compound_a_id, compound_b_id, cell_line_id, zip_score, bliss_score,
         synergy_class, confidence_tier, evidence_type, source_db,
         num_concentrations, has_dose_matrix)
        VALUES (1, 2, 2, 7.0, 5.0, 'synergistic', 'bronze', 'dose_response_matrix',
                'drugcomb', 5, 1)"""
    )

    # Drug A (1) + Drug C (3) — single source, single cell line
    conn.execute(
        """INSERT INTO dc_synergy_results
        (compound_a_id, compound_b_id, cell_line_id, zip_score, bliss_score,
         synergy_class, confidence_tier, evidence_type, source_db,
         num_concentrations, has_dose_matrix)
        VALUES (1, 3, 1, 1.0, 0.5, 'additive', 'copper', 'single_concentration',
                'drugcomb', 1, 0)"""
    )

    conn.commit()


class TestPairAggregation:
    def test_refresh_pair_count(self, conn):
        """Should create correct number of aggregated pairs."""
        _insert_aggregation_test_data(conn)
        count = refresh_all_drug_pairs(conn)
        conn.commit()
        # Pair(1,2) and Pair(1,3) = 2 pairs
        assert count == 2

    def test_num_cell_lines(self, conn):
        """Pair (1,2) has 2 cell lines (MCF7, A549)."""
        _insert_aggregation_test_data(conn)
        refresh_all_drug_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT num_cell_lines, num_sources, num_measurements
            FROM drug_drug_pairs WHERE compound_a_id = 1 AND compound_b_id = 2"""
        ).fetchone()
        assert row[0] == 2  # MCF7 + A549
        assert row[1] == 2  # drugcomb + nci_almanac
        assert row[2] == 3  # 3 total result rows

    def test_consensus_class_context_dependent(self, conn):
        """Pair (1,2) has mixed results: antagonistic in MCF7, synergistic in A549."""
        _insert_aggregation_test_data(conn)
        refresh_all_drug_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT consensus_class, antagonism_fraction, synergy_fraction
            FROM drug_drug_pairs WHERE compound_a_id = 1 AND compound_b_id = 2"""
        ).fetchone()
        # 2/3 antagonistic, 1/3 synergistic → antagonistic majority
        assert row[0] == "antagonistic"
        assert row[1] == pytest.approx(2.0 / 3.0, abs=0.01)
        assert row[2] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_consensus_class_additive(self, conn):
        """Pair (1,3) is 100% additive."""
        _insert_aggregation_test_data(conn)
        refresh_all_drug_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT consensus_class FROM drug_drug_pairs
            WHERE compound_a_id = 1 AND compound_b_id = 3"""
        ).fetchone()
        assert row[0] == "additive"

    def test_consensus_class_truly_context_dependent(self, conn):
        """A pair with no majority class should be context_dependent."""
        # Insert 3 compounds + 3 cell lines for a balanced 3-way split
        conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (1, 'Drug A')")
        conn.execute("INSERT INTO compounds (compound_id, drug_name) VALUES (2, 'Drug B')")
        conn.execute("INSERT INTO cell_lines (cell_line_id, cell_line_name) VALUES (1, 'CL1')")
        conn.execute("INSERT INTO cell_lines (cell_line_id, cell_line_name) VALUES (2, 'CL2')")
        conn.execute("INSERT INTO cell_lines (cell_line_id, cell_line_name) VALUES (3, 'CL3')")
        # 1 synergistic, 1 antagonistic, 1 additive → no majority
        conn.execute(
            """INSERT INTO dc_synergy_results
            (compound_a_id, compound_b_id, cell_line_id, zip_score,
             synergy_class, confidence_tier, source_db)
            VALUES (1, 2, 1, 8.0, 'synergistic', 'bronze', 'drugcomb')"""
        )
        conn.execute(
            """INSERT INTO dc_synergy_results
            (compound_a_id, compound_b_id, cell_line_id, zip_score,
             synergy_class, confidence_tier, source_db)
            VALUES (1, 2, 2, -8.0, 'antagonistic', 'bronze', 'drugcomb')"""
        )
        conn.execute(
            """INSERT INTO dc_synergy_results
            (compound_a_id, compound_b_id, cell_line_id, zip_score,
             synergy_class, confidence_tier, source_db)
            VALUES (1, 2, 3, 0.0, 'additive', 'bronze', 'drugcomb')"""
        )
        conn.commit()
        refresh_all_drug_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT consensus_class FROM drug_drug_pairs
            WHERE compound_a_id = 1 AND compound_b_id = 2"""
        ).fetchone()
        assert row[0] == "context_dependent"

    def test_best_confidence(self, conn):
        """best_confidence should be the best (lowest ordinal) tier."""
        _insert_aggregation_test_data(conn)
        refresh_all_drug_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT best_confidence FROM drug_drug_pairs
            WHERE compound_a_id = 1 AND compound_b_id = 2"""
        ).fetchone()
        assert row[0] == "bronze"  # All bronze

        row = conn.execute(
            """SELECT best_confidence FROM drug_drug_pairs
            WHERE compound_a_id = 1 AND compound_b_id = 3"""
        ).fetchone()
        assert row[0] == "copper"

    def test_median_zip(self, conn):
        """median_zip (AVG proxy) for pair (1,2) = avg(-8, -7, 7) = -2.67."""
        _insert_aggregation_test_data(conn)
        refresh_all_drug_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT median_zip FROM drug_drug_pairs
            WHERE compound_a_id = 1 AND compound_b_id = 2"""
        ).fetchone()
        assert row[0] == pytest.approx((-8.0 + -7.0 + 7.0) / 3.0, abs=0.1)

    def test_target_overlap(self, conn):
        """Pair (1,2) shares EGFR. Jaccard = 1/3 (EGFR / {EGFR, BRAF, KRAS})."""
        _insert_aggregation_test_data(conn)
        refresh_all_drug_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT num_shared_targets, target_jaccard FROM drug_drug_pairs
            WHERE compound_a_id = 1 AND compound_b_id = 2"""
        ).fetchone()
        assert row[0] == 1  # EGFR shared
        assert row[1] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_target_overlap_none(self, conn):
        """Pair (1,3) has no shared targets."""
        _insert_aggregation_test_data(conn)
        refresh_all_drug_pairs(conn)
        conn.commit()
        row = conn.execute(
            """SELECT num_shared_targets, target_jaccard FROM drug_drug_pairs
            WHERE compound_a_id = 1 AND compound_b_id = 3"""
        ).fetchone()
        assert row[0] == 0
        assert row[1] == pytest.approx(0.0)

    def test_compound_degrees(self, conn):
        """Drug A has 2 partners (B, C). Drug B has 1 partner (A). Drug C has 1 partner (A)."""
        _insert_aggregation_test_data(conn)
        refresh_all_drug_pairs(conn)
        conn.commit()
        # Drug A is in both pairs → degree = 2
        row = conn.execute(
            """SELECT compound_a_degree FROM drug_drug_pairs
            WHERE compound_a_id = 1 AND compound_b_id = 2"""
        ).fetchone()
        assert row[0] == 2  # Drug A has partners B and C

        row = conn.execute(
            """SELECT compound_b_degree FROM drug_drug_pairs
            WHERE compound_a_id = 1 AND compound_b_id = 2"""
        ).fetchone()
        assert row[0] == 1  # Drug B only has partner A

    def test_triples_created(self, conn):
        """Triples should be created for each pair x cell_line combination."""
        _insert_aggregation_test_data(conn)
        refresh_all_drug_pairs(conn)
        conn.commit()
        triple_count = conn.execute(
            "SELECT COUNT(*) FROM drug_drug_cell_line_triples"
        ).fetchone()[0]
        # Pair(1,2): MCF7 + A549 = 2 triples; Pair(1,3): MCF7 = 1 triple
        assert triple_count == 3

    def test_triple_synergy_class(self, conn):
        """Triple for pair(1,2)+MCF7 should be antagonistic (avg ZIP = -7.5)."""
        _insert_aggregation_test_data(conn)
        refresh_all_drug_pairs(conn)
        conn.commit()
        pair_id = conn.execute(
            """SELECT pair_id FROM drug_drug_pairs
            WHERE compound_a_id = 1 AND compound_b_id = 2"""
        ).fetchone()[0]
        row = conn.execute(
            """SELECT best_zip, synergy_class, num_measurements
            FROM drug_drug_cell_line_triples
            WHERE pair_id = ? AND cell_line_id = 1""",
            (pair_id,),
        ).fetchone()
        assert row[0] == pytest.approx(-7.5, abs=0.1)  # avg(-8, -7)
        assert row[1] == "antagonistic"
        assert row[2] == 2  # 2 results for this triple

    def test_triple_synergistic(self, conn):
        """Triple for pair(1,2)+A549 should be synergistic."""
        _insert_aggregation_test_data(conn)
        refresh_all_drug_pairs(conn)
        conn.commit()
        pair_id = conn.execute(
            """SELECT pair_id FROM drug_drug_pairs
            WHERE compound_a_id = 1 AND compound_b_id = 2"""
        ).fetchone()[0]
        row = conn.execute(
            """SELECT best_zip, synergy_class
            FROM drug_drug_cell_line_triples
            WHERE pair_id = ? AND cell_line_id = 2""",
            (pair_id,),
        ).fetchone()
        assert row[0] == pytest.approx(7.0)
        assert row[1] == "synergistic"

    def test_refresh_clears_old_data(self, conn):
        """Refreshing should delete old pairs, triples, and split assignments."""
        _insert_aggregation_test_data(conn)
        refresh_all_drug_pairs(conn)
        conn.commit()

        # Add a split assignment
        conn.execute(
            """INSERT INTO dc_split_definitions
            (split_name, split_strategy) VALUES ('test_split', 'random')"""
        )
        pair_id = conn.execute(
            "SELECT pair_id FROM drug_drug_pairs LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO dc_split_assignments (pair_id, split_id, fold)
            VALUES (?, 1, 'train')""",
            (pair_id,),
        )
        conn.commit()

        # Refresh again
        count = refresh_all_drug_pairs(conn)
        conn.commit()
        assert count == 2

        # Split assignments and triples should be cleared
        sa_count = conn.execute(
            "SELECT COUNT(*) FROM dc_split_assignments"
        ).fetchone()[0]
        assert sa_count == 0

        triple_count = conn.execute(
            "SELECT COUNT(*) FROM drug_drug_cell_line_triples"
        ).fetchone()[0]
        assert triple_count == 3  # Re-created from results

    def test_empty_results(self, conn):
        """Refreshing with no results should produce 0 pairs."""
        count = refresh_all_drug_pairs(conn)
        assert count == 0


# ── Index tests ───────────────────────────────────────────────────────


class TestIndices:
    def test_synergy_result_indices(self, conn):
        """Critical indices on dc_synergy_results should exist."""
        indices = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list('dc_synergy_results')"
            ).fetchall()
        }
        assert "idx_dc_sr_compound_a" in indices
        assert "idx_dc_sr_compound_b" in indices
        assert "idx_dc_sr_pair" in indices
        assert "idx_dc_sr_cell_line" in indices
        assert "idx_dc_sr_tier" in indices
        assert "idx_dc_sr_source" in indices
        assert "idx_dc_sr_class" in indices

    def test_compound_indices(self, conn):
        """Compound lookup indices should exist."""
        indices = {
            row[1]
            for row in conn.execute("PRAGMA index_list('compounds')").fetchall()
        }
        assert "idx_compounds_pubchem" in indices
        assert "idx_compounds_inchikey" in indices
        assert "idx_compounds_chembl" in indices

    def test_cell_line_indices(self, conn):
        """Cell line lookup indices should exist."""
        indices = {
            row[1]
            for row in conn.execute("PRAGMA index_list('cell_lines')").fetchall()
        }
        assert "idx_cell_lines_cosmic" in indices
        assert "idx_cell_lines_depmap" in indices
        assert "idx_cell_lines_tissue" in indices

    def test_pair_indices(self, conn):
        """Pair aggregation indices should exist."""
        indices = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list('drug_drug_pairs')"
            ).fetchall()
        }
        assert "idx_ddp_compound_a" in indices
        assert "idx_ddp_compound_b" in indices
        assert "idx_ddp_confidence" in indices
        assert "idx_ddp_class" in indices

    def test_triple_indices(self, conn):
        """Triple table indices should exist."""
        indices = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list('drug_drug_cell_line_triples')"
            ).fetchall()
        }
        assert "idx_ddclt_pair" in indices
        assert "idx_ddclt_cell_line" in indices

    def test_cross_domain_indices(self, conn):
        """Cross-domain bridge indices should exist."""
        comp_indices = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list('dc_cross_domain_compounds')"
            ).fetchall()
        }
        assert "idx_dc_bridge_compound" in comp_indices
        assert "idx_dc_bridge_compound_domain" in comp_indices

        cl_indices = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list('dc_cross_domain_cell_lines')"
            ).fetchall()
        }
        assert "idx_dc_bridge_cell_line" in cl_indices

    def test_split_indices(self, conn):
        """Split assignment indices should exist."""
        indices = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list('dc_split_assignments')"
            ).fetchall()
        }
        assert "idx_dc_splits_fold" in indices
