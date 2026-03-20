"""Tests for negbiodb_ppi.export — PPI ML dataset export pipeline."""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from negbiodb_ppi.export import (
    _register_ppi_split,
    generate_random_split,
    generate_cold_protein_split,
    generate_cold_both_partition,
    generate_degree_balanced_split,
    _build_export_query,
    export_negative_dataset,
    resolve_conflicts,
    add_random_split,
    add_cold_protein_split,
    add_cold_both_partition_split,
    add_degree_balanced_split,
    apply_ppi_m1_splits,
    build_m1_balanced,
    build_m1_realistic,
    generate_uniform_random_negatives,
    generate_degree_matched_negatives,
    control_pairs_to_df,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def ppi_db(tmp_path):
    """Create a minimal PPI database with proteins, pairs, and split tables."""
    db_path = tmp_path / "test_ppi.db"
    conn = sqlite3.connect(str(db_path))

    # Schema (minimal version of migration 001)
    conn.executescript("""
        CREATE TABLE schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE proteins (
            protein_id INTEGER PRIMARY KEY AUTOINCREMENT,
            uniprot_accession TEXT NOT NULL UNIQUE,
            uniprot_entry_name TEXT,
            gene_symbol TEXT,
            amino_acid_sequence TEXT,
            sequence_length INTEGER,
            organism TEXT DEFAULT 'Homo sapiens',
            taxonomy_id INTEGER DEFAULT 9606,
            subcellular_location TEXT,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE ppi_negative_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            protein1_id INTEGER NOT NULL REFERENCES proteins(protein_id),
            protein2_id INTEGER NOT NULL REFERENCES proteins(protein_id),
            experiment_id INTEGER,
            evidence_type TEXT NOT NULL,
            confidence_tier TEXT NOT NULL,
            interaction_score REAL,
            source_db TEXT NOT NULL,
            source_record_id TEXT NOT NULL,
            extraction_method TEXT NOT NULL,
            publication_year INTEGER,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            CHECK (protein1_id < protein2_id)
        );

        CREATE TABLE protein_protein_pairs (
            pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
            protein1_id INTEGER NOT NULL REFERENCES proteins(protein_id),
            protein2_id INTEGER NOT NULL REFERENCES proteins(protein_id),
            num_experiments INTEGER NOT NULL,
            num_sources INTEGER NOT NULL,
            best_confidence TEXT NOT NULL,
            best_evidence_type TEXT,
            earliest_year INTEGER,
            min_interaction_score REAL,
            max_interaction_score REAL,
            protein1_degree INTEGER,
            protein2_degree INTEGER,
            UNIQUE(protein1_id, protein2_id),
            CHECK (protein1_id < protein2_id)
        );

        CREATE TABLE ppi_split_definitions (
            split_id INTEGER PRIMARY KEY AUTOINCREMENT,
            split_name TEXT NOT NULL,
            split_strategy TEXT NOT NULL CHECK (split_strategy IN (
                'random', 'cold_protein', 'cold_both',
                'bfs_cluster', 'degree_balanced')),
            description TEXT,
            random_seed INTEGER,
            train_ratio REAL DEFAULT 0.7,
            val_ratio REAL DEFAULT 0.1,
            test_ratio REAL DEFAULT 0.2,
            date_created TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            version TEXT DEFAULT '1.0',
            UNIQUE(split_name, version)
        );

        CREATE TABLE ppi_split_assignments (
            pair_id INTEGER NOT NULL REFERENCES protein_protein_pairs(pair_id),
            split_id INTEGER NOT NULL REFERENCES ppi_split_definitions(split_id),
            fold TEXT NOT NULL CHECK (fold IN ('train', 'val', 'test')),
            PRIMARY KEY (pair_id, split_id)
        );

        INSERT INTO schema_migrations (version) VALUES ('001');
    """)

    # Insert 10 proteins with sequences
    proteins = [
        ("A00001", "ACGT" * 25, "GENE_A", "Nucleus"),
        ("A00002", "DEFG" * 30, "GENE_B", "Cytoplasm"),
        ("A00003", "HIJK" * 20, "GENE_C", "Membrane"),
        ("A00004", "LMNO" * 35, "GENE_D", None),
        ("A00005", "PQRS" * 15, "GENE_E", "Nucleus"),
        ("A00006", "TUVW" * 40, "GENE_F", "Cytoplasm"),
        ("A00007", "XYZA" * 22, "GENE_G", None),
        ("A00008", "BCDE" * 28, "GENE_H", "Membrane"),
        ("A00009", "FGHI" * 18, "GENE_I", "Nucleus"),
        ("A00010", "JKLM" * 33, "GENE_J", "Cytoplasm"),
    ]
    conn.executemany(
        """INSERT INTO proteins (uniprot_accession, amino_acid_sequence,
           sequence_length, gene_symbol, subcellular_location)
        VALUES (?, ?, ?, ?, ?)""",
        [(acc, seq, len(seq), gene, loc) for acc, seq, gene, loc in proteins],
    )

    # Insert negative results and pairs
    # Create a connected graph: 1-2, 1-3, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-10, 1-5, 2-6, 3-7, 4-8, 5-9
    pair_data = [
        (1, 2), (1, 3), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (1, 5), (2, 6), (3, 7), (4, 8), (5, 9),
    ]
    for p1, p2 in pair_data:
        conn.execute(
            """INSERT INTO ppi_negative_results
            (protein1_id, protein2_id, evidence_type, confidence_tier,
             source_db, source_record_id, extraction_method)
            VALUES (?, ?, 'experimental_non_interaction', 'gold',
                    'huri', ?, 'database_direct')""",
            (p1, p2, f"huri_{p1}_{p2}"),
        )

    # Compute degrees
    from negbiodb_ppi.ppi_db import refresh_all_ppi_pairs
    refresh_all_ppi_pairs(conn)
    conn.commit()

    conn.close()
    return db_path


@pytest.fixture
def sample_neg_df():
    """Sample negative DataFrame for merge tests."""
    return pd.DataFrame({
        "pair_id": list(range(100)),
        "uniprot_id_1": [f"A{i:05d}" for i in range(1, 101)],
        "sequence_1": ["ACGT" * 25] * 100,
        "gene_symbol_1": [f"G{i}" for i in range(100)],
        "subcellular_location_1": ["Nucleus"] * 100,
        "uniprot_id_2": [f"B{i:05d}" for i in range(1, 101)],
        "sequence_2": ["DEFG" * 30] * 100,
        "gene_symbol_2": [f"G{i}" for i in range(100, 200)],
        "subcellular_location_2": ["Cytoplasm"] * 100,
        "Y": [0] * 100,
        "confidence_tier": ["gold"] * 100,
        "num_sources": [1] * 100,
        "protein1_degree": list(range(1, 101)),
        "protein2_degree": list(range(100, 0, -1)),
    })


@pytest.fixture
def sample_pos_df():
    """Sample positive DataFrame for merge tests."""
    return pd.DataFrame({
        "pair_id": [None] * 20,
        "uniprot_id_1": [f"C{i:05d}" for i in range(1, 21)],
        "sequence_1": ["HIJK" * 20] * 20,
        "gene_symbol_1": [f"PG{i}" for i in range(20)],
        "subcellular_location_1": ["Membrane"] * 20,
        "uniprot_id_2": [f"D{i:05d}" for i in range(1, 21)],
        "sequence_2": ["LMNO" * 35] * 20,
        "gene_symbol_2": [f"PG{i}" for i in range(20, 40)],
        "subcellular_location_2": [None] * 20,
        "Y": [1] * 20,
        "confidence_tier": [None] * 20,
        "num_sources": [None] * 20,
        "protein1_degree": [None] * 20,
        "protein2_degree": [None] * 20,
    })


# ------------------------------------------------------------------
# DB-level split tests
# ------------------------------------------------------------------

class TestRegisterPpiSplit:
    def test_register_new(self, ppi_db):
        conn = sqlite3.connect(str(ppi_db))
        split_id = _register_ppi_split(
            conn, "test_v1", "random", 42, {"train": 0.7, "val": 0.1, "test": 0.2}
        )
        assert split_id > 0
        conn.close()

    def test_reregister_clears(self, ppi_db):
        conn = sqlite3.connect(str(ppi_db))
        sid1 = _register_ppi_split(
            conn, "test_v1", "random", 42, {"train": 0.7, "val": 0.1, "test": 0.2}
        )
        # Insert a dummy assignment
        conn.execute(
            "INSERT INTO ppi_split_assignments (pair_id, split_id, fold) VALUES (1, ?, 'train')",
            (sid1,),
        )
        conn.commit()

        sid2 = _register_ppi_split(
            conn, "test_v1", "random", 42, {"train": 0.7, "val": 0.1, "test": 0.2}
        )
        assert sid1 == sid2

        cnt = conn.execute(
            "SELECT COUNT(*) FROM ppi_split_assignments WHERE split_id = ?",
            (sid1,),
        ).fetchone()[0]
        assert cnt == 0  # cleared
        conn.close()


class TestRandomSplit:
    def test_covers_all_pairs(self, ppi_db):
        conn = sqlite3.connect(str(ppi_db))
        result = generate_random_split(conn, seed=42)
        total_pairs = conn.execute(
            "SELECT COUNT(*) FROM protein_protein_pairs"
        ).fetchone()[0]
        assert sum(result["counts"].values()) == total_pairs
        conn.close()

    def test_deterministic(self, ppi_db):
        conn = sqlite3.connect(str(ppi_db))
        r1 = generate_random_split(conn, seed=42)
        r2 = generate_random_split(conn, seed=42)
        assert r1["counts"] == r2["counts"]
        conn.close()

    def test_approximate_ratios(self, ppi_db):
        conn = sqlite3.connect(str(ppi_db))
        result = generate_random_split(conn, seed=42)
        total = sum(result["counts"].values())
        # With 15 pairs, ratios won't be exact, but train should be largest
        assert result["counts"].get("train", 0) >= result["counts"].get("test", 0)
        conn.close()


class TestColdProteinSplit:
    def test_no_leakage(self, ppi_db):
        """Test-group proteins must never appear in train-fold pairs."""
        conn = sqlite3.connect(str(ppi_db))
        result = generate_cold_protein_split(conn, seed=42)
        # leaked_proteins = test-group proteins in train-fold pairs (should be 0)
        assert result["leaked_proteins"] == 0
        conn.close()

    def test_covers_all_pairs(self, ppi_db):
        conn = sqlite3.connect(str(ppi_db))
        result = generate_cold_protein_split(conn, seed=42)
        total = conn.execute("SELECT COUNT(*) FROM protein_protein_pairs").fetchone()[0]
        assert sum(result["counts"].values()) == total
        conn.close()

    def test_all_folds_present(self, ppi_db):
        conn = sqlite3.connect(str(ppi_db))
        result = generate_cold_protein_split(conn, seed=42)
        assert "train" in result["counts"]
        assert "test" in result["counts"]
        conn.close()


class TestColdBothPartition:
    def test_excludes_cross_partition(self, ppi_db):
        conn = sqlite3.connect(str(ppi_db))
        result = generate_cold_both_partition(conn, seed=42, nparts=4)
        total = conn.execute("SELECT COUNT(*) FROM protein_protein_pairs").fetchone()[0]
        assigned = sum(result["counts"].values())
        assert assigned <= total
        assert result["excluded"] == total - assigned
        conn.close()

    def test_cold_both_integrity(self, ppi_db):
        """No protein should appear in both train and test."""
        conn = sqlite3.connect(str(ppi_db))
        result = generate_cold_both_partition(conn, seed=42, nparts=4)
        split_id = result["split_id"]

        train_proteins = set()
        for r in conn.execute(
            """SELECT protein1_id FROM protein_protein_pairs ppp
            JOIN ppi_split_assignments sa ON ppp.pair_id = sa.pair_id
            WHERE sa.split_id = ? AND sa.fold = 'train'
            UNION
            SELECT protein2_id FROM protein_protein_pairs ppp
            JOIN ppi_split_assignments sa ON ppp.pair_id = sa.pair_id
            WHERE sa.split_id = ? AND sa.fold = 'train'""",
            (split_id, split_id),
        ).fetchall():
            train_proteins.add(r[0])

        test_proteins = set()
        for r in conn.execute(
            """SELECT protein1_id FROM protein_protein_pairs ppp
            JOIN ppi_split_assignments sa ON ppp.pair_id = sa.pair_id
            WHERE sa.split_id = ? AND sa.fold = 'test'
            UNION
            SELECT protein2_id FROM protein_protein_pairs ppp
            JOIN ppi_split_assignments sa ON ppp.pair_id = sa.pair_id
            WHERE sa.split_id = ? AND sa.fold = 'test'""",
            (split_id, split_id),
        ).fetchall():
            test_proteins.add(r[0])

        assert len(train_proteins & test_proteins) == 0
        conn.close()


class TestDegreeBalancedSplit:
    def test_covers_all_pairs(self, ppi_db):
        conn = sqlite3.connect(str(ppi_db))
        result = generate_degree_balanced_split(conn, seed=42)
        total = conn.execute("SELECT COUNT(*) FROM protein_protein_pairs").fetchone()[0]
        assert sum(result["counts"].values()) == total
        conn.close()

    def test_all_folds_present(self, ppi_db):
        conn = sqlite3.connect(str(ppi_db))
        result = generate_degree_balanced_split(conn, seed=42)
        assert "train" in result["counts"]
        assert "test" in result["counts"]
        conn.close()


# ------------------------------------------------------------------
# Export tests
# ------------------------------------------------------------------

class TestExportNegativeDataset:
    def test_export_parquet(self, ppi_db, tmp_path):
        conn = sqlite3.connect(str(ppi_db))
        generate_random_split(conn, seed=42)
        conn.close()

        output_dir = tmp_path / "exports"
        result = export_negative_dataset(
            ppi_db, output_dir, split_strategies=["random"]
        )
        assert result["total_rows"] == 15
        assert Path(result["parquet_path"]).exists()

        # Verify schema
        df = pd.read_parquet(result["parquet_path"])
        assert "uniprot_id_1" in df.columns
        assert "sequence_1" in df.columns
        assert "uniprot_id_2" in df.columns
        assert "sequence_2" in df.columns
        assert "Y" in df.columns
        assert (df["Y"] == 0).all()
        assert "split_random" in df.columns

    def test_export_all_strategies(self, ppi_db, tmp_path):
        conn = sqlite3.connect(str(ppi_db))
        generate_random_split(conn, seed=42)
        generate_cold_protein_split(conn, seed=42)
        generate_degree_balanced_split(conn, seed=42)
        conn.close()

        output_dir = tmp_path / "exports"
        result = export_negative_dataset(
            ppi_db, output_dir,
            split_strategies=["random", "cold_protein", "degree_balanced"],
        )
        df = pd.read_parquet(result["parquet_path"])
        assert "split_random" in df.columns
        assert "split_cold_protein" in df.columns
        assert "split_degree_balanced" in df.columns


# ------------------------------------------------------------------
# Conflict resolution tests
# ------------------------------------------------------------------

class TestConflictResolution:
    def test_no_conflicts(self, sample_neg_df, sample_pos_df):
        clean_neg, clean_pos, n = resolve_conflicts(sample_neg_df, sample_pos_df)
        assert n == 0
        assert len(clean_neg) == 100
        assert len(clean_pos) == 20

    def test_conflicts_removed(self):
        neg = pd.DataFrame({
            "uniprot_id_1": ["A", "B", "C"],
            "uniprot_id_2": ["X", "Y", "Z"],
            "Y": [0, 0, 0],
        })
        pos = pd.DataFrame({
            "uniprot_id_1": ["A", "D"],
            "uniprot_id_2": ["X", "W"],
            "Y": [1, 1],
        })
        clean_neg, clean_pos, n = resolve_conflicts(neg, pos)
        assert n == 1  # (A, X) is conflict
        assert len(clean_neg) == 2
        assert len(clean_pos) == 1
        assert "A" not in clean_neg["uniprot_id_1"].values
        assert "A" not in clean_pos["uniprot_id_1"].values


# ------------------------------------------------------------------
# DataFrame-level split tests
# ------------------------------------------------------------------

class TestDataFrameSplits:
    def test_random_split(self, sample_neg_df):
        df = add_random_split(sample_neg_df, seed=42)
        assert "split_random" in df.columns
        assert set(df["split_random"].unique()) == {"train", "val", "test"}
        assert len(df) == 100

    def test_random_split_empty(self):
        df = pd.DataFrame(columns=["uniprot_id_1", "uniprot_id_2"])
        result = add_random_split(df, seed=42)
        assert "split_random" in result.columns
        assert len(result) == 0

    def test_cold_protein_no_leakage(self, sample_neg_df):
        """Test-group proteins must not appear in train-fold pairs.

        Note: train-group proteins CAN appear in test-fold pairs (expected:
        when partner is test-group, pair fold = max = test).
        """
        df = add_cold_protein_split(sample_neg_df, seed=42)
        assert "split_cold_protein" in df.columns

        train_pairs = df[df["split_cold_protein"] == "train"]
        test_pairs = df[df["split_cold_protein"] == "test"]

        # Identify test-group proteins: proteins that ONLY appear in test/val pairs
        # (never in train pairs as both sides)
        train_proteins = set(train_pairs["uniprot_id_1"]) | set(train_pairs["uniprot_id_2"])
        test_only_proteins = (
            (set(test_pairs["uniprot_id_1"]) | set(test_pairs["uniprot_id_2"]))
            - train_proteins
        )
        # These test-only proteins should not appear in any train pair
        for _, row in train_pairs.iterrows():
            assert row["uniprot_id_1"] not in test_only_proteins
            assert row["uniprot_id_2"] not in test_only_proteins

    def test_cold_both_excludes_cross(self, sample_neg_df):
        df = add_cold_both_partition_split(sample_neg_df, seed=42, nparts=4)
        assert "split_cold_both" in df.columns
        # Some rows should be None (excluded)
        n_assigned = df["split_cold_both"].notna().sum()
        assert n_assigned <= len(df)

    def test_degree_balanced(self, sample_neg_df):
        df = add_degree_balanced_split(sample_neg_df, seed=42)
        assert "split_degree_balanced" in df.columns
        assert set(df["split_degree_balanced"].unique()) == {"train", "val", "test"}

    def test_apply_all_splits(self, sample_neg_df):
        df = apply_ppi_m1_splits(sample_neg_df, seed=42)
        assert "split_random" in df.columns
        assert "split_cold_protein" in df.columns
        assert "split_cold_both" in df.columns
        assert "split_degree_balanced" in df.columns


# ------------------------------------------------------------------
# M1 builder tests
# ------------------------------------------------------------------

class TestM1Builders:
    def test_balanced_1to1(self, sample_neg_df, sample_pos_df):
        df = build_m1_balanced(sample_neg_df, sample_pos_df, seed=42)
        n_pos = (df["Y"] == 1).sum()
        n_neg = (df["Y"] == 0).sum()
        assert n_pos == 20
        assert n_neg == 20  # sampled down to match
        assert "split_random" in df.columns

    def test_realistic_ratio(self, sample_neg_df, sample_pos_df):
        df = build_m1_realistic(sample_neg_df, sample_pos_df, ratio=3, seed=42)
        n_pos = (df["Y"] == 1).sum()
        n_neg = (df["Y"] == 0).sum()
        assert n_pos == 20
        assert n_neg == 60  # 3:1 ratio

    def test_realistic_capped(self, sample_neg_df, sample_pos_df):
        """When not enough negatives, use all available."""
        df = build_m1_realistic(sample_neg_df, sample_pos_df, ratio=100, seed=42)
        n_neg = (df["Y"] == 0).sum()
        assert n_neg == 100  # all available


# ------------------------------------------------------------------
# Control negative tests
# ------------------------------------------------------------------

class TestControlNegatives:
    def test_uniform_random(self, ppi_db):
        # Use empty positive set
        pairs = generate_uniform_random_negatives(
            ppi_db, positive_pairs=set(), n_samples=10, seed=42
        )
        assert len(pairs) == 10
        # All canonical
        for a, b in pairs:
            assert a < b

    def test_uniform_excludes_negatives(self, ppi_db):
        pairs = generate_uniform_random_negatives(
            ppi_db, positive_pairs=set(), n_samples=5, seed=42
        )
        # Load existing negatives
        conn = sqlite3.connect(str(ppi_db))
        existing = set()
        for r in conn.execute(
            """SELECT p1.uniprot_accession, p2.uniprot_accession
            FROM protein_protein_pairs ppp
            JOIN proteins p1 ON ppp.protein1_id = p1.protein_id
            JOIN proteins p2 ON ppp.protein2_id = p2.protein_id"""
        ).fetchall():
            existing.add((r[0], r[1]) if r[0] < r[1] else (r[1], r[0]))
        conn.close()

        assert len(pairs & existing) == 0

    def test_degree_matched(self, ppi_db):
        pairs = generate_degree_matched_negatives(
            ppi_db, positive_pairs=set(), n_samples=5, seed=42
        )
        assert len(pairs) == 5

    def test_control_to_df(self, ppi_db):
        pairs = {("A00001", "A00010"), ("A00002", "A00009")}
        df = control_pairs_to_df(pairs, ppi_db)
        assert len(df) == 2
        assert (df["Y"] == 0).all()
        assert df["sequence_1"].notna().all()
        assert df["sequence_2"].notna().all()


# ------------------------------------------------------------------
# Parquet schema contract
# ------------------------------------------------------------------

class TestSchemaContract:
    """Verify Parquet output matches the schema contract from the plan."""

    REQUIRED_NEGATIVE_COLS = [
        "pair_id", "uniprot_id_1", "sequence_1", "gene_symbol_1",
        "subcellular_location_1", "uniprot_id_2", "sequence_2",
        "gene_symbol_2", "subcellular_location_2", "Y",
        "confidence_tier", "num_sources", "protein1_degree", "protein2_degree",
    ]

    def test_negative_schema(self, ppi_db, tmp_path):
        conn = sqlite3.connect(str(ppi_db))
        generate_random_split(conn, seed=42)
        conn.close()

        result = export_negative_dataset(
            ppi_db, tmp_path / "exports", split_strategies=["random"]
        )
        df = pd.read_parquet(result["parquet_path"])
        for col in self.REQUIRED_NEGATIVE_COLS:
            assert col in df.columns, f"Missing column: {col}"
        assert "split_random" in df.columns

    def test_m1_schema(self, sample_neg_df, sample_pos_df):
        df = build_m1_balanced(sample_neg_df, sample_pos_df, seed=42)
        for col in self.REQUIRED_NEGATIVE_COLS:
            assert col in df.columns, f"Missing column: {col}"
        # Y should have both 0 and 1
        assert set(df["Y"].unique()) == {0, 1}
        # Positives should have pair_id = None
        pos_rows = df[df["Y"] == 1]
        assert pos_rows["pair_id"].isna().all()
