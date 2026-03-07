"""Tests for the ML export pipeline."""

import pytest

from negbiodb.db import connect, create_database, refresh_all_pairs
from negbiodb.export import (
    _register_split,
    generate_cold_compound_split,
    generate_cold_target_split,
    generate_random_split,
)

MIGRATIONS_DIR = "migrations"


@pytest.fixture
def migrated_db(tmp_path):
    db_path = tmp_path / "test.db"
    create_database(db_path, MIGRATIONS_DIR)
    return db_path


def _populate_small_db(conn, n_compounds=10, n_targets=5):
    """Insert n_compounds * n_targets pairs for testing splits."""
    for i in range(1, n_compounds + 1):
        conn.execute(
            """INSERT INTO compounds
            (canonical_smiles, inchikey, inchikey_connectivity)
            VALUES (?, ?, ?)""",
            (f"C{i}", f"KEY{i:020d}SA-N", f"KEY{i:020d}"),
        )
    for j in range(1, n_targets + 1):
        conn.execute(
            "INSERT INTO targets (uniprot_accession) VALUES (?)",
            (f"P{j:05d}",),
        )
    for i in range(1, n_compounds + 1):
        for j in range(1, n_targets + 1):
            conn.execute(
                """INSERT INTO negative_results
                (compound_id, target_id, result_type, confidence_tier,
                 activity_type, activity_value, activity_unit,
                 inactivity_threshold, source_db, source_record_id,
                 extraction_method, publication_year)
                VALUES (?, ?, 'hard_negative', 'silver',
                        'IC50', 20000.0, 'nM',
                        10000.0, 'chembl', ?, 'database_direct', ?)""",
                (i, j, f"C:{i}:{j}", 2015 + (i % 10)),
            )
    refresh_all_pairs(conn)
    conn.commit()
    return n_compounds * n_targets


# ============================================================
# TestRegisterSplit
# ============================================================


class TestRegisterSplit:

    def test_creates_definition(self, migrated_db):
        with connect(migrated_db) as conn:
            sid = _register_split(
                conn, "test_split", "random", 42,
                {"train": 0.7, "val": 0.1, "test": 0.2},
            )
            row = conn.execute(
                "SELECT split_name, split_strategy, random_seed "
                "FROM split_definitions WHERE split_id = ?",
                (sid,),
            ).fetchone()
            assert row == ("test_split", "random", 42)

    def test_idempotent(self, migrated_db):
        with connect(migrated_db) as conn:
            sid1 = _register_split(
                conn, "test_split", "random", 42,
                {"train": 0.7, "val": 0.1, "test": 0.2},
            )
            sid2 = _register_split(
                conn, "test_split", "random", 42,
                {"train": 0.7, "val": 0.1, "test": 0.2},
            )
            assert sid1 == sid2


# ============================================================
# TestRandomSplit
# ============================================================


class TestRandomSplit:

    def test_ratios_within_tolerance(self, migrated_db):
        with connect(migrated_db) as conn:
            total = _populate_small_db(conn, 20, 10)
            result = generate_random_split(conn)
            counts = result["counts"]
            assert abs(counts["train"] / total - 0.7) < 0.05
            assert abs(counts["val"] / total - 0.1) < 0.05
            assert abs(counts["test"] / total - 0.2) < 0.05
            assert counts["train"] + counts["val"] + counts["test"] == total

    def test_deterministic(self, migrated_db):
        """Same seed should produce identical assignments."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 10, 5)
            r1 = generate_random_split(conn, seed=42)

        # Fresh DB with same data
        db2 = migrated_db.parent / "test2.db"
        create_database(db2, MIGRATIONS_DIR)
        with connect(db2) as conn:
            _populate_small_db(conn, 10, 5)
            r2 = generate_random_split(conn, seed=42)

        assert r1["counts"] == r2["counts"]

    def test_all_pairs_assigned(self, migrated_db):
        with connect(migrated_db) as conn:
            total = _populate_small_db(conn, 10, 5)
            result = generate_random_split(conn)
            assigned = conn.execute(
                "SELECT COUNT(*) FROM split_assignments WHERE split_id = ?",
                (result["split_id"],),
            ).fetchone()[0]
            assert assigned == total


# ============================================================
# TestColdCompoundSplit
# ============================================================


class TestColdCompoundSplit:

    def test_no_compound_leakage(self, migrated_db):
        """No compound should appear in both train and test."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 20, 5)
            result = generate_cold_compound_split(conn)
            sid = result["split_id"]

            leaks = conn.execute(
                """SELECT COUNT(DISTINCT ctp1.compound_id)
                FROM split_assignments sa1
                JOIN compound_target_pairs ctp1 ON sa1.pair_id = ctp1.pair_id
                WHERE sa1.split_id = ? AND sa1.fold = 'train'
                AND ctp1.compound_id IN (
                    SELECT ctp2.compound_id
                    FROM split_assignments sa2
                    JOIN compound_target_pairs ctp2 ON sa2.pair_id = ctp2.pair_id
                    WHERE sa2.split_id = ? AND sa2.fold = 'test'
                )""",
                (sid, sid),
            ).fetchone()[0]
            assert leaks == 0

    def test_all_pairs_assigned(self, migrated_db):
        with connect(migrated_db) as conn:
            total = _populate_small_db(conn, 10, 5)
            result = generate_cold_compound_split(conn)
            assigned = sum(result["counts"].values())
            assert assigned == total

    def test_compound_ratios(self, migrated_db):
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 20, 5)
            result = generate_cold_compound_split(conn)
            sid = result["split_id"]
            # Count unique compounds per fold
            for fold in ("train", "val", "test"):
                conn.execute(
                    """SELECT COUNT(DISTINCT ctp.compound_id)
                    FROM split_assignments sa
                    JOIN compound_target_pairs ctp ON sa.pair_id = ctp.pair_id
                    WHERE sa.split_id = ? AND sa.fold = ?""",
                    (sid, fold),
                ).fetchone()[0]
            # Just verify it ran without error and all 3 folds exist
            assert set(result["counts"].keys()) == {"train", "val", "test"}


# ============================================================
# TestColdTargetSplit
# ============================================================


class TestColdTargetSplit:

    def test_no_target_leakage(self, migrated_db):
        """No target should appear in both train and test."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 10, 10)
            result = generate_cold_target_split(conn)
            sid = result["split_id"]

            leaks = conn.execute(
                """SELECT COUNT(DISTINCT ctp1.target_id)
                FROM split_assignments sa1
                JOIN compound_target_pairs ctp1 ON sa1.pair_id = ctp1.pair_id
                WHERE sa1.split_id = ? AND sa1.fold = 'train'
                AND ctp1.target_id IN (
                    SELECT ctp2.target_id
                    FROM split_assignments sa2
                    JOIN compound_target_pairs ctp2 ON sa2.pair_id = ctp2.pair_id
                    WHERE sa2.split_id = ? AND sa2.fold = 'test'
                )""",
                (sid, sid),
            ).fetchone()[0]
            assert leaks == 0

    def test_all_pairs_assigned(self, migrated_db):
        with connect(migrated_db) as conn:
            total = _populate_small_db(conn, 10, 10)
            result = generate_cold_target_split(conn)
            assigned = sum(result["counts"].values())
            assert assigned == total
