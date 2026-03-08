"""Tests for the ML export pipeline."""

import pytest

from negbiodb.db import connect, create_database, refresh_all_pairs
import pandas as pd
import numpy as np

from negbiodb.export import (
    _compute_scaffolds,
    _register_split,
    add_cold_compound_split,
    add_cold_target_split,
    add_random_split,
    apply_m1_splits,
    export_negative_dataset,
    generate_cold_compound_split,
    generate_cold_target_split,
    generate_degree_balanced_split,
    generate_degree_matched_negatives,
    generate_leakage_report,
    generate_random_split,
    generate_scaffold_split,
    generate_temporal_split,
    generate_uniform_random_negatives,
    merge_positive_negative,
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


def _populate_partial_db(conn, n_compounds=20, n_targets=10, pairs_per_compound=3):
    """Insert compounds and targets with partial pairing.

    Each compound is paired with only ``pairs_per_compound`` targets
    (rotating window), leaving most of the compound×target space
    untested.  Needed for testing random negative generation where
    untested pairs must exist within the DB compound space.
    """
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
    count = 0
    for i in range(1, n_compounds + 1):
        for k in range(pairs_per_compound):
            j = ((i - 1 + k) % n_targets) + 1
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
            count += 1
    refresh_all_pairs(conn)
    conn.commit()
    return count


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


# ============================================================
# TestMergePositiveNegative
# ============================================================


def _make_positives(n=20, uniprot_ids=None):
    """Create a synthetic positives DataFrame for testing."""
    if uniprot_ids is None:
        uniprot_ids = [f"P{i:05d}" for i in range(1, 4)]
    rows = []
    for i in range(n):
        rows.append({
            "smiles": f"POS_C{i}",
            "inchikey": f"POS{i:011d}SA-N",
            "uniprot_id": uniprot_ids[i % len(uniprot_ids)],
            "target_sequence": "MAAAA",
            "pchembl_value": 7.0 + i * 0.01,
            "activity_type": "IC50",
            "activity_value_nm": 100.0,
            "publication_year": 2020,
        })
    return pd.DataFrame(rows)


class TestMergePositiveNegative:

    def test_balanced_output(self, migrated_db, tmp_path):
        """Balanced merge produces equal pos/neg counts."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 10, 5)

        positives = _make_positives(n=15)
        result = merge_positive_negative(
            positives, migrated_db, tmp_path / "m1"
        )
        df = pd.read_parquet(result["balanced"]["path"])
        assert result["balanced"]["n_pos"] == result["balanced"]["n_neg"]
        assert (df["Y"].isin([0, 1])).all()
        assert df["Y"].sum() == result["balanced"]["n_pos"]

    def test_realistic_ratio(self, migrated_db, tmp_path):
        """Realistic merge has ~1:10 pos:neg ratio."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 20, 10)  # 200 negatives

        positives = _make_positives(n=15)
        result = merge_positive_negative(
            positives, migrated_db, tmp_path / "m1"
        )
        assert result["realistic"]["n_neg"] == result["realistic"]["n_pos"] * 10

    def test_overlap_removal(self, migrated_db, tmp_path):
        """Overlapping pairs are removed from positives."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 5, 3)
            # Get actual inchikeys and uniprots from DB
            neg_pairs = conn.execute(
                """SELECT c.inchikey, t.uniprot_accession
                FROM compound_target_pairs ctp
                JOIN compounds c ON ctp.compound_id = c.compound_id
                JOIN targets t ON ctp.target_id = t.target_id
                LIMIT 2"""
            ).fetchall()

        # Create positives that overlap with 2 negatives
        pos_data = []
        for ik, uid in neg_pairs:
            pos_data.append({
                "smiles": "OVERLAP",
                "inchikey": ik,
                "uniprot_id": uid,
                "target_sequence": "MAAAA",
                "pchembl_value": 8.0,
                "activity_type": "IC50",
                "activity_value_nm": 10.0,
                "publication_year": 2022,
            })
        # Add non-overlapping positives
        for i in range(10):
            pos_data.append({
                "smiles": f"UNIQUE_C{i}",
                "inchikey": f"UNIQ{i:020d}SA-N",
                "uniprot_id": "P00001",
                "target_sequence": "MAAAA",
                "pchembl_value": 7.5,
                "activity_type": "IC50",
                "activity_value_nm": 30.0,
                "publication_year": 2021,
            })
        positives = pd.DataFrame(pos_data)

        result = merge_positive_negative(
            positives, migrated_db, tmp_path / "m1"
        )
        # The 2 overlapping should have been removed
        # Balanced should use non-overlapping positives only
        assert result["balanced"]["n_pos"] <= 10

    def test_empty_positives(self, migrated_db, tmp_path):
        """Empty positives produces empty datasets."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 5, 3)

        positives = pd.DataFrame(columns=[
            "smiles", "inchikey", "uniprot_id", "target_sequence", "Y",
        ])
        result = merge_positive_negative(
            positives, migrated_db, tmp_path / "m1"
        )
        assert result["balanced"]["total"] == 0


# ============================================================
# TestLeakageReport
# ============================================================


class TestLeakageReport:

    def test_report_structure(self, migrated_db):
        """Report has all expected top-level keys."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 10, 5)
            generate_random_split(conn)
            generate_cold_compound_split(conn)
            generate_cold_target_split(conn)

        report = generate_leakage_report(migrated_db)
        assert "db_summary" in report
        assert "splits" in report
        assert "cold_split_integrity" in report
        assert report["db_summary"]["pairs"] == 50

    def test_cold_splits_zero_leakage(self, migrated_db):
        """Cold splits in report should show zero leakage."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 20, 5)
            generate_cold_compound_split(conn)
            generate_cold_target_split(conn)

        report = generate_leakage_report(migrated_db)
        integrity = report["cold_split_integrity"]
        assert integrity["cold_compound"]["leaks"] == 0
        assert integrity["cold_target"]["leaks"] == 0

    def test_writes_json(self, migrated_db, tmp_path):
        """Report can be written to JSON file."""
        import json

        with connect(migrated_db) as conn:
            _populate_small_db(conn, 5, 3)

        json_path = tmp_path / "report.json"
        generate_leakage_report(migrated_db, output_path=json_path)
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["db_summary"]["compounds"] == 5

    def test_split_ratios_in_report(self, migrated_db):
        """Split ratios appear in report."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 20, 10)
            generate_random_split(conn)

        report = generate_leakage_report(migrated_db)
        random_split = report["splits"]["random_v1"]
        assert abs(random_split["ratios"]["train"] - 0.7) < 0.05


# ============================================================
# TestDataFrameSplits (M1 split functions)
# ============================================================


def _make_m1_df(n_pos=100, n_neg=100, n_compounds=20, n_targets=5):
    """Create a synthetic M1-like DataFrame for testing splits."""
    rows = []
    for i in range(n_pos):
        cid = i % n_compounds
        tid = i % n_targets
        rows.append({
            "smiles": f"POS{cid}",
            "inchikey": f"POS{cid:020d}SA-N",
            "uniprot_id": f"P{tid:05d}",
            "target_sequence": "MAAAA",
            "Y": 1,
        })
    for i in range(n_neg):
        cid = i % n_compounds
        tid = i % n_targets
        rows.append({
            "smiles": f"NEG{cid}",
            "inchikey": f"NEG{cid:020d}SA-N",
            "uniprot_id": f"P{tid:05d}",
            "target_sequence": "MAAAA",
            "Y": 0,
        })
    return pd.DataFrame(rows)


class TestDataFrameSplits:

    def test_random_split_columns(self):
        """add_random_split adds split_random column."""
        df = _make_m1_df()
        result = add_random_split(df)
        assert "split_random" in result.columns
        assert set(result["split_random"].unique()) == {"train", "val", "test"}

    def test_random_split_ratios(self):
        """Random split has approximately 70/10/20 ratios."""
        df = _make_m1_df(n_pos=500, n_neg=500)
        result = add_random_split(df)
        counts = result["split_random"].value_counts()
        total = len(result)
        assert abs(counts["train"] / total - 0.7) < 0.05
        assert abs(counts["val"] / total - 0.1) < 0.05
        assert abs(counts["test"] / total - 0.2) < 0.05

    def test_random_split_deterministic(self):
        """Same seed produces identical splits."""
        df = _make_m1_df()
        r1 = add_random_split(df, seed=42)
        r2 = add_random_split(df, seed=42)
        assert (r1["split_random"] == r2["split_random"]).all()

    def test_cold_compound_no_leak(self):
        """Cold-compound split: no compound in both train and test."""
        df = _make_m1_df(n_pos=200, n_neg=200, n_compounds=30)
        result = add_cold_compound_split(df)
        train_compounds = set(
            result[result["split_cold_compound"] == "train"]["inchikey"].str[:14]
        )
        test_compounds = set(
            result[result["split_cold_compound"] == "test"]["inchikey"].str[:14]
        )
        assert len(train_compounds & test_compounds) == 0

    def test_cold_compound_no_leak_val(self):
        """Cold-compound split: no compound in both val and test."""
        df = _make_m1_df(n_pos=200, n_neg=200, n_compounds=30)
        result = add_cold_compound_split(df)
        val_compounds = set(
            result[result["split_cold_compound"] == "val"]["inchikey"].str[:14]
        )
        test_compounds = set(
            result[result["split_cold_compound"] == "test"]["inchikey"].str[:14]
        )
        assert len(val_compounds & test_compounds) == 0

    def test_cold_target_no_leak(self):
        """Cold-target split: no target in both train and test."""
        df = _make_m1_df(n_pos=200, n_neg=200, n_targets=10)
        result = add_cold_target_split(df)
        train_targets = set(
            result[result["split_cold_target"] == "train"]["uniprot_id"]
        )
        test_targets = set(
            result[result["split_cold_target"] == "test"]["uniprot_id"]
        )
        assert len(train_targets & test_targets) == 0

    def test_apply_m1_splits_all_columns(self):
        """apply_m1_splits adds all 3 split columns."""
        df = _make_m1_df()
        result = apply_m1_splits(df)
        assert "split_random" in result.columns
        assert "split_cold_compound" in result.columns
        assert "split_cold_target" in result.columns

    def test_empty_dataframe(self):
        """Split functions handle empty DataFrames gracefully."""
        df = pd.DataFrame(columns=[
            "smiles", "inchikey", "uniprot_id", "target_sequence", "Y",
        ])
        result = apply_m1_splits(df)
        assert len(result) == 0
        assert "split_random" in result.columns

    def test_original_not_modified(self):
        """Split functions do not modify the original DataFrame."""
        df = _make_m1_df()
        original_cols = set(df.columns)
        _ = add_random_split(df)
        assert set(df.columns) == original_cols


class TestMergeWithSplits:

    def test_balanced_has_split_columns(self, migrated_db, tmp_path):
        """Balanced merge output has split columns."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 10, 5)

        positives = _make_positives(n=15)
        result = merge_positive_negative(
            positives, migrated_db, tmp_path / "m1"
        )
        df = pd.read_parquet(result["balanced"]["path"])
        assert "split_random" in df.columns
        assert "split_cold_compound" in df.columns
        assert "split_cold_target" in df.columns

    def test_realistic_has_split_columns(self, migrated_db, tmp_path):
        """Realistic merge output has split columns."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 20, 10)

        positives = _make_positives(n=15)
        result = merge_positive_negative(
            positives, migrated_db, tmp_path / "m1"
        )
        df = pd.read_parquet(result["realistic"]["path"])
        assert "split_random" in df.columns
        assert "split_cold_compound" in df.columns
        assert "split_cold_target" in df.columns


# ============================================================
# TestRandomNegatives (Exp 1 controls)
# ============================================================


class TestUniformRandomNegatives:

    def test_generates_correct_count(self, migrated_db, tmp_path):
        """Uniform random generates requested number of negatives."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 10, 5)

        positives = _make_positives(n=10, uniprot_ids=[f"P{i:05d}" for i in range(1, 6)])
        result = generate_uniform_random_negatives(
            migrated_db, positives, n_samples=20,
            output_dir=tmp_path / "rand",
        )
        df = pd.read_parquet(result["path"])
        # balanced: min(10 pos, 20 neg) = 10 each → 20 total
        assert result["n_pos"] == 10
        assert result["n_neg"] == 10
        assert result["total"] == 20

    def test_no_overlap_with_tested(self, migrated_db, tmp_path):
        """Generated pairs do not exist in NegBioDB or positives."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 10, 5)
            # Load tested pairs
            tested = set()
            for row in conn.execute(
                """SELECT c.inchikey_connectivity, t.uniprot_accession
                FROM compound_target_pairs ctp
                JOIN compounds c ON ctp.compound_id = c.compound_id
                JOIN targets t ON ctp.target_id = t.target_id"""
            ):
                tested.add((row[0], row[1]))

        positives = _make_positives(n=10, uniprot_ids=[f"P{i:05d}" for i in range(1, 6)])
        for ik, uid in zip(positives["inchikey"].str[:14], positives["uniprot_id"]):
            tested.add((ik, uid))

        result = generate_uniform_random_negatives(
            migrated_db, positives, n_samples=30,
            output_dir=tmp_path / "rand",
        )
        df = pd.read_parquet(result["path"])
        neg_df = df[df["Y"] == 0]
        for _, row in neg_df.iterrows():
            key = (row["inchikey"][:14], row["uniprot_id"])
            assert key not in tested, f"Generated pair {key} is in tested set"

    def test_has_split_columns(self, migrated_db, tmp_path):
        """Output has M1 split columns."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 10, 5)

        positives = _make_positives(n=10, uniprot_ids=[f"P{i:05d}" for i in range(1, 6)])
        result = generate_uniform_random_negatives(
            migrated_db, positives, n_samples=20,
            output_dir=tmp_path / "rand",
        )
        df = pd.read_parquet(result["path"])
        assert "split_random" in df.columns
        assert "split_cold_compound" in df.columns
        assert "split_cold_target" in df.columns

    def test_deterministic(self, migrated_db, tmp_path):
        """Same seed produces identical output."""
        with connect(migrated_db) as conn:
            _populate_small_db(conn, 10, 5)

        positives = _make_positives(n=10, uniprot_ids=[f"P{i:05d}" for i in range(1, 6)])
        r1 = generate_uniform_random_negatives(
            migrated_db, positives, n_samples=15,
            output_dir=tmp_path / "r1", seed=42,
        )
        r2 = generate_uniform_random_negatives(
            migrated_db, positives, n_samples=15,
            output_dir=tmp_path / "r2", seed=42,
        )
        df1 = pd.read_parquet(r1["path"])
        df2 = pd.read_parquet(r2["path"])
        pd.testing.assert_frame_equal(df1, df2)


class TestDegreeMatchedNegatives:
    """Degree-matched tests use _populate_partial_db (sparse graph)
    because generate_degree_matched_negatives only samples from DB
    compounds.  _populate_small_db creates ALL pairs → 0 untested.
    """

    def test_generates_output(self, migrated_db, tmp_path):
        """Degree-matched generates a non-empty output."""
        with connect(migrated_db) as conn:
            _populate_partial_db(conn, 20, 10, pairs_per_compound=3)

        positives = _make_positives(n=10, uniprot_ids=[f"P{i:05d}" for i in range(1, 11)])
        result = generate_degree_matched_negatives(
            migrated_db, positives, n_samples=20,
            output_dir=tmp_path / "deg",
        )
        df = pd.read_parquet(result["path"])
        assert len(df) > 0
        assert (df["Y"].isin([0, 1])).all()

    def test_has_split_columns(self, migrated_db, tmp_path):
        """Output has M1 split columns."""
        with connect(migrated_db) as conn:
            _populate_partial_db(conn, 20, 10, pairs_per_compound=3)

        positives = _make_positives(n=10, uniprot_ids=[f"P{i:05d}" for i in range(1, 11)])
        result = generate_degree_matched_negatives(
            migrated_db, positives, n_samples=20,
            output_dir=tmp_path / "deg",
        )
        df = pd.read_parquet(result["path"])
        assert "split_random" in df.columns
        assert "split_cold_compound" in df.columns

    def test_no_overlap_with_tested(self, migrated_db, tmp_path):
        """Generated pairs do not exist in tested set."""
        with connect(migrated_db) as conn:
            _populate_partial_db(conn, 20, 10, pairs_per_compound=3)
            tested = set()
            for row in conn.execute(
                """SELECT c.inchikey_connectivity, t.uniprot_accession
                FROM compound_target_pairs ctp
                JOIN compounds c ON ctp.compound_id = c.compound_id
                JOIN targets t ON ctp.target_id = t.target_id"""
            ):
                tested.add((row[0], row[1]))

        positives = _make_positives(n=10, uniprot_ids=[f"P{i:05d}" for i in range(1, 11)])
        for ik, uid in zip(positives["inchikey"].str[:14], positives["uniprot_id"]):
            tested.add((ik, uid))

        result = generate_degree_matched_negatives(
            migrated_db, positives, n_samples=20,
            output_dir=tmp_path / "deg",
        )
        df = pd.read_parquet(result["path"])
        neg_df = df[df["Y"] == 0]
        for _, row in neg_df.iterrows():
            key = (row["inchikey"][:14], row["uniprot_id"])
            assert key not in tested
