"""Tests for NegBioDB database creation and migration runner."""

import sqlite3
from pathlib import Path

import pytest

from negbiodb.db import connect, create_database, get_applied_versions, run_migrations

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

EXPECTED_TABLES = {
    "compounds",
    "targets",
    "assays",
    "negative_results",
    "dti_context",
    "compound_target_pairs",
    "split_definitions",
    "split_assignments",
    "dataset_versions",
    "schema_migrations",
}


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_negbiodb.db"


@pytest.fixture
def migrated_db(db_path):
    create_database(db_path, MIGRATIONS_DIR)
    return db_path


class TestMigrationRunner:

    def test_creates_all_tables(self, migrated_db):
        with connect(migrated_db) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            tables = {row[0] for row in rows}
        assert tables == EXPECTED_TABLES

    def test_idempotent(self, migrated_db):
        applied = run_migrations(migrated_db, MIGRATIONS_DIR)
        assert applied == []

        with connect(migrated_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM schema_migrations"
            ).fetchone()[0]
        assert count == 1

    def test_schema_migrations_records_version(self, migrated_db):
        with connect(migrated_db) as conn:
            versions = get_applied_versions(conn)
        assert "001" in versions

    def test_returns_applied_versions(self, db_path):
        applied = run_migrations(db_path, MIGRATIONS_DIR)
        assert applied == ["001"]


class TestForeignKeys:

    def test_fk_enforcement(self, migrated_db):
        with connect(migrated_db) as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO negative_results "
                    "(compound_id, target_id, result_type, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (99999, 99999, 'hard_negative', 'gold', "
                    "        'chembl', 'TEST001', 'database_direct')"
                )


class TestUniqueConstraints:

    def _insert_compound(self, conn, smiles="CCO",
                         inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N"):
        conn.execute(
            "INSERT INTO compounds (canonical_smiles, inchikey, "
            "inchikey_connectivity) VALUES (?, ?, ?)",
            (smiles, inchikey, inchikey[:14]),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def _insert_target(self, conn, accession="P00533"):
        conn.execute(
            "INSERT INTO targets (uniprot_accession) VALUES (?)",
            (accession,),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def test_duplicate_source_record_rejected(self, migrated_db):
        """NULL assay_id duplicates should be caught by COALESCE index."""
        with connect(migrated_db) as conn:
            cid = self._insert_compound(conn)
            tid = self._insert_target(conn)
            conn.commit()

            conn.execute(
                "INSERT INTO negative_results "
                "(compound_id, target_id, assay_id, result_type, confidence_tier, "
                " source_db, source_record_id, extraction_method) "
                "VALUES (?, ?, NULL, 'hard_negative', 'gold', "
                "        'chembl', 'REC001', 'database_direct')",
                (cid, tid),
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO negative_results "
                    "(compound_id, target_id, assay_id, result_type, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (?, ?, NULL, 'hard_negative', 'gold', "
                    "        'chembl', 'REC001', 'database_direct')",
                    (cid, tid),
                )

    def test_different_source_records_accepted(self, migrated_db):
        with connect(migrated_db) as conn:
            cid = self._insert_compound(conn)
            tid = self._insert_target(conn)
            conn.commit()

            for rec_id in ["REC001", "REC002"]:
                conn.execute(
                    "INSERT INTO negative_results "
                    "(compound_id, target_id, assay_id, result_type, confidence_tier, "
                    " source_db, source_record_id, extraction_method) "
                    "VALUES (?, ?, NULL, 'hard_negative', 'gold', "
                    "        'chembl', ?, 'database_direct')",
                    (cid, tid, rec_id),
                )
            conn.commit()

            count = conn.execute(
                "SELECT COUNT(*) FROM negative_results"
            ).fetchone()[0]
            assert count == 2


class TestConnectionPragmas:

    def test_wal_mode(self, migrated_db):
        with connect(migrated_db) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_on(self, migrated_db):
        with connect(migrated_db) as conn:
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
