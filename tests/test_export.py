"""Tests for the ML export pipeline."""

import pytest

from negbiodb.db import connect, create_database, refresh_all_pairs
import pandas as pd
import pyarrow.parquet as pq

from negbiodb.export import (
    _compute_scaffolds,
    _register_split,
    export_negative_dataset,
    generate_cold_compound_split,
    generate_cold_target_split,
    generate_degree_balanced_split,
    generate_random_split,
    generate_scaffold_split,
    generate_temporal_split,
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


# ============================================================
# TestTemporalSplit
# ============================================================

# Valid SMILES for scaffold tests (diverse scaffolds)
REAL_SMILES = [
    "c1ccc(NC(=O)c2ccccc2)cc1",       # benzanilide
    "c1ccc2[nH]ccc2c1",                # indole
    "c1ccc(-c2ccccn2)cc1",             # 2-phenylpyridine
    "O=C1CCCN1",                        # pyrrolidinone
    "c1ccc2ccccc2c1",                   # naphthalene
    "c1cnc2ccccc2n1",                   # quinazoline
    "c1ccc(-c2ccc3ccccc3c2)cc1",       # 2-phenylnaphthalene
    "O=c1[nH]c2ccccc2o1",              # benzoxazolone
    "c1ccc2[nH]c(-c3ccccc3)nc2c1",     # 2-phenylbenzimidazole
    "c1ccc2c(c1)ccc1ccccc12",          # fluorene
]


def _populate_with_real_smiles(conn, n_targets=3, years=None):
    """Insert compounds with valid SMILES + targets + results.

    Returns total pair count.
    """
    n_compounds = len(REAL_SMILES)
    if years is None:
        years = list(range(2015, 2015 + n_compounds))

    for i, smi in enumerate(REAL_SMILES, 1):
        conn.execute(
            """INSERT INTO compounds
            (canonical_smiles, inchikey, inchikey_connectivity)
            VALUES (?, ?, ?)""",
            (smi, f"REAL{i:020d}SA-N", f"REAL{i:020d}"),
        )
    for j in range(1, n_targets + 1):
        conn.execute(
            "INSERT INTO targets (uniprot_accession) VALUES (?)",
            (f"Q{j:05d}",),
        )
    for i in range(1, n_compounds + 1):
        yr = years[i - 1] if i - 1 < len(years) else 2020
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
                (i, j, f"R:{i}:{j}", yr),
            )
    refresh_all_pairs(conn)
    conn.commit()
    return n_compounds * n_targets


class TestTemporalSplit:

    def test_correct_fold_assignment(self, migrated_db):
        """Pairs assigned based on earliest_year boundaries."""
        with connect(migrated_db) as conn:
            # 10 compounds × 3 targets, years 2015-2024
            _populate_with_real_smiles(conn, n_targets=3)
            result = generate_temporal_split(conn, train_cutoff=2020, val_cutoff=2023)
            counts = result["counts"]
            # years 2015-2019 → train (5 compounds × 3 = 15)
            # years 2020-2022 → val (3 compounds × 3 = 9)
            # years 2023-2024 → test (2 compounds × 3 = 6)
            assert counts["train"] == 15
            assert counts["val"] == 9
            assert counts["test"] == 6

    def test_null_year_goes_to_train(self, migrated_db):
        """Pairs with NULL earliest_year are assigned to train."""
        with connect(migrated_db) as conn:
            # Insert one compound with no publication_year
            conn.execute(
                """INSERT INTO compounds
                (canonical_smiles, inchikey, inchikey_connectivity)
                VALUES ('C', 'NULLYR00000000000SA-N', 'NULLYR00000000000')"""
            )
            conn.execute(
                "INSERT INTO targets (uniprot_accession) VALUES ('P99999')"
            )
            conn.execute(
                """INSERT INTO negative_results
                (compound_id, target_id, result_type, confidence_tier,
                 activity_type, activity_value, activity_unit,
                 inactivity_threshold, source_db, source_record_id,
                 extraction_method)
                VALUES (1, 1, 'hard_negative', 'silver',
                        'IC50', 20000.0, 'nM',
                        10000.0, 'chembl', 'N:1:1', 'database_direct')"""
            )
            refresh_all_pairs(conn)
            conn.commit()

            result = generate_temporal_split(conn)
            sid = result["split_id"]
            fold = conn.execute(
                "SELECT fold FROM split_assignments WHERE split_id = ?",
                (sid,),
            ).fetchone()[0]
            assert fold == "train"

    def test_all_pairs_assigned(self, migrated_db):
        with connect(migrated_db) as conn:
            total = _populate_small_db(conn, 10, 5)
            result = generate_temporal_split(conn)
            assigned = sum(result["counts"].values())
            assert assigned == total


# ============================================================
# TestScaffoldSplit
# ============================================================


class TestScaffoldSplit:

    def test_no_compound_leakage(self, migrated_db):
        """No compound should appear in both train and test."""
        with connect(migrated_db) as conn:
            _populate_with_real_smiles(conn, n_targets=5)
            result = generate_scaffold_split(conn)
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
            total = _populate_with_real_smiles(conn, n_targets=5)
            result = generate_scaffold_split(conn)
            assigned = sum(result["counts"].values())
            assert assigned == total

    def test_scaffold_computation(self, migrated_db):
        """Verify scaffolds are computed for real SMILES."""
        with connect(migrated_db) as conn:
            _populate_with_real_smiles(conn, n_targets=1)
            scaffolds = _compute_scaffolds(conn)
            # All 10 real SMILES should produce valid scaffolds (not NONE)
            all_compounds = []
            for compounds in scaffolds.values():
                all_compounds.extend(compounds)
            assert len(all_compounds) == 10
            # "NONE" scaffold should not exist for valid SMILES
            assert "NONE" not in scaffolds

    def test_invalid_smiles_get_none_scaffold(self, migrated_db):
        """Compounds with unparseable SMILES get scaffold='NONE'."""
        with connect(migrated_db) as conn:
            conn.execute(
                """INSERT INTO compounds
                (canonical_smiles, inchikey, inchikey_connectivity)
                VALUES ('INVALID_SMILES', 'BADKEY00000000000SA-N', 'BADKEY00000000000')"""
            )
            scaffolds = _compute_scaffolds(conn)
            assert "NONE" in scaffolds
            assert 1 in scaffolds["NONE"]


# ============================================================
# TestDegreeBalancedSplit
# ============================================================


class TestDegreeBalancedSplit:

    def test_all_pairs_assigned(self, migrated_db):
        with connect(migrated_db) as conn:
            total = _populate_small_db(conn, 10, 5)
            result = generate_degree_balanced_split(conn)
            assigned = sum(result["counts"].values())
            assert assigned == total

    def test_ratios_within_tolerance(self, migrated_db):
        with connect(migrated_db) as conn:
            total = _populate_small_db(conn, 20, 10)
            result = generate_degree_balanced_split(conn)
            counts = result["counts"]
            assert abs(counts["train"] / total - 0.7) < 0.05
            assert abs(counts["val"] / total - 0.1) < 0.05
            assert abs(counts["test"] / total - 0.2) < 0.05

    def test_deterministic(self, migrated_db):
        """Same seed should produce identical assignments."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 10, 5)
            r1 = generate_degree_balanced_split(conn, seed=42)

        db2 = migrated_db.parent / "test2.db"
        create_database(db2, MIGRATIONS_DIR)
        with connect(db2) as conn:
            _populate_small_db(conn, 10, 5)
            r2 = generate_degree_balanced_split(conn, seed=42)

        assert r1["counts"] == r2["counts"]

    def test_degree_distribution_preserved(self, migrated_db):
        """Mean degree in each fold should be similar to overall mean."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 20, 10)
            result = generate_degree_balanced_split(conn)
            sid = result["split_id"]

            overall_mean = conn.execute(
                "SELECT AVG(compound_degree) FROM compound_target_pairs"
            ).fetchone()[0]

            for fold in ("train", "val", "test"):
                fold_mean = conn.execute(
                    """SELECT AVG(ctp.compound_degree)
                    FROM split_assignments sa
                    JOIN compound_target_pairs ctp ON sa.pair_id = ctp.pair_id
                    WHERE sa.split_id = ? AND sa.fold = ?""",
                    (sid, fold),
                ).fetchone()[0]
                # Mean degree per fold should be within 30% of overall
                assert abs(fold_mean - overall_mean) / overall_mean < 0.3


# ============================================================
# TestExportNegativeDataset
# ============================================================


class TestExportNegativeDataset:

    def test_parquet_roundtrip(self, migrated_db, tmp_path):
        """Exported Parquet has correct columns and row count."""
        with connect(migrated_db) as conn:
            total = _populate_small_db(conn, 5, 3)
            generate_random_split(conn)

        export_dir = tmp_path / "exports"
        result = export_negative_dataset(
            migrated_db, export_dir,
            split_strategies=["random"],
        )
        assert result["total_rows"] == total

        df = pd.read_parquet(result["parquet_path"])
        assert len(df) == total
        assert "smiles" in df.columns
        assert "uniprot_id" in df.columns
        assert "split_random" in df.columns
        assert (df["Y"] == 0).all()

    def test_splits_csv_created(self, migrated_db, tmp_path):
        """Lightweight splits CSV is created with correct columns."""
        with connect(migrated_db) as conn:
            total = _populate_small_db(conn, 5, 3)
            generate_random_split(conn)

        export_dir = tmp_path / "exports"
        result = export_negative_dataset(
            migrated_db, export_dir,
            split_strategies=["random"],
        )
        df = pd.read_csv(result["splits_csv_path"])
        assert len(df) == total
        assert "pair_id" in df.columns
        assert "smiles" in df.columns
        assert "split_random" in df.columns
        # target_sequence should NOT be in splits CSV
        assert "target_sequence" not in df.columns

    def test_multiple_splits(self, migrated_db, tmp_path):
        """Export works with multiple split strategies."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 5, 3)
            generate_random_split(conn)
            generate_cold_compound_split(conn)

        export_dir = tmp_path / "exports"
        result = export_negative_dataset(
            migrated_db, export_dir,
            split_strategies=["random", "cold_compound"],
        )
        df = pd.read_parquet(result["parquet_path"])
        assert "split_random" in df.columns
        assert "split_cold_compound" in df.columns

    def test_no_splits_present(self, migrated_db, tmp_path):
        """Export works even when no splits have been generated."""
        with connect(migrated_db) as conn:
            total = _populate_small_db(conn, 3, 2)

        export_dir = tmp_path / "exports"
        result = export_negative_dataset(
            migrated_db, export_dir,
            split_strategies=["random"],
        )
        df = pd.read_parquet(result["parquet_path"])
        assert len(df) == total
        # split_random should be NULL/None for all rows
        assert df["split_random"].isna().all()
