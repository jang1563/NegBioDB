"""Tests for VP ML export pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from negbiodb_vp.export import (
    generate_cold_both_split,
    generate_cold_disease_split,
    generate_cold_gene_split,
    generate_degree_balanced_split,
    generate_random_split,
    generate_temporal_split,
    export_vp_dataset,
)
from negbiodb_vp.vp_db import get_connection, refresh_all_vp_pairs, run_vp_migrations

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_vp"


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_vp.db"
    run_vp_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def conn(tmp_db):
    c = get_connection(tmp_db)
    yield c
    c.close()


def _seed_export_data(conn, n_genes=5, n_diseases=3, n_variants=20):
    """Seed database with test data for export testing."""
    # Insert genes
    for i in range(1, n_genes + 1):
        conn.execute(
            "INSERT INTO genes (gene_id, gene_symbol) VALUES (?, ?)",
            (i, f"GENE{i}"),
        )

    # Insert diseases
    for i in range(1, n_diseases + 1):
        conn.execute(
            "INSERT INTO diseases (disease_id, canonical_name) VALUES (?, ?)",
            (i, f"Disease_{i}"),
        )

    # Insert variants
    for i in range(1, n_variants + 1):
        gene_id = (i % n_genes) + 1
        conn.execute(
            """INSERT INTO variants (variant_id, chromosome, position, ref_allele,
               alt_allele, variant_type, gene_id, consequence_type)
            VALUES (?, '1', ?, 'A', 'G', 'single nucleotide variant', ?, 'missense')""",
            (i, 1000 + i, gene_id),
        )

    # Insert submissions (with all required NOT NULL fields)
    for i in range(1, n_variants + 1):
        conn.execute(
            """INSERT INTO vp_submissions
               (submission_id, variant_id, classification, review_status, submitter_name)
            VALUES (?, ?, 'benign', 'criteria_provided_single_submitter', 'Lab1')""",
            (i, i),
        )

    # Insert negative results (with all required NOT NULL fields)
    years = list(range(2015, 2025))
    for i in range(1, n_variants + 1):
        disease_id = (i % n_diseases) + 1
        year = years[i % len(years)]
        conn.execute(
            """INSERT INTO vp_negative_results
               (result_id, variant_id, disease_id, submission_id, source_db,
                classification, evidence_type, confidence_tier, submission_year,
                source_record_id, extraction_method)
            VALUES (?, ?, ?, ?, 'clinvar', 'benign', 'single_submitter', 'bronze', ?,
                    ?, 'single_submission')""",
            (i, i, disease_id, i, year, f"VCV{i:06d}"),
        )

    conn.commit()
    refresh_all_vp_pairs(conn)
    conn.commit()


def _count_split_assignments(conn, split_name):
    """Count assignments for a split by name (joins definitions)."""
    return conn.execute(
        """SELECT COUNT(*) FROM vp_split_assignments sa
           JOIN vp_split_definitions sd ON sa.split_id = sd.split_id
           WHERE sd.split_name = ?""",
        (split_name,),
    ).fetchone()[0]


def _get_split_folds(conn, split_name):
    """Get distinct folds for a split by name."""
    rows = conn.execute(
        """SELECT DISTINCT sa.fold FROM vp_split_assignments sa
           JOIN vp_split_definitions sd ON sa.split_id = sd.split_id
           WHERE sd.split_name = ?""",
        (split_name,),
    ).fetchall()
    return {r[0] for r in rows}


class TestGenerateRandomSplit:
    def test_creates_split(self, conn):
        _seed_export_data(conn)
        generate_random_split(conn, seed=42)

        assert _count_split_assignments(conn, "random_s42") > 0

    def test_all_folds_assigned(self, conn):
        _seed_export_data(conn)
        generate_random_split(conn, seed=42)

        assert _get_split_folds(conn, "random_s42") == {"train", "val", "test"}

    def test_covers_all_pairs(self, conn):
        _seed_export_data(conn)
        generate_random_split(conn, seed=42)

        n_pairs = conn.execute("SELECT COUNT(*) FROM variant_disease_pairs").fetchone()[0]
        n_assigned = _count_split_assignments(conn, "random_s42")
        assert n_assigned == n_pairs


class TestGenerateColdGeneSplit:
    def test_no_gene_leakage(self, conn):
        _seed_export_data(conn, n_genes=10, n_variants=40)
        generate_cold_gene_split(conn, seed=42)

        train_genes = conn.execute("""
            SELECT DISTINCT v.gene_id FROM vp_split_assignments sa
            JOIN vp_split_definitions sd ON sa.split_id = sd.split_id
            JOIN variant_disease_pairs vdp ON sa.pair_id = vdp.pair_id
            JOIN variants v ON vdp.variant_id = v.variant_id
            WHERE sd.split_name = 'cold_gene_s42' AND sa.fold = 'train'
        """).fetchall()
        test_genes = conn.execute("""
            SELECT DISTINCT v.gene_id FROM vp_split_assignments sa
            JOIN vp_split_definitions sd ON sa.split_id = sd.split_id
            JOIN variant_disease_pairs vdp ON sa.pair_id = vdp.pair_id
            JOIN variants v ON vdp.variant_id = v.variant_id
            WHERE sd.split_name = 'cold_gene_s42' AND sa.fold = 'test'
        """).fetchall()

        train_set = {r[0] for r in train_genes}
        test_set = {r[0] for r in test_genes}
        assert not (train_set & test_set), "Gene leakage detected between train and test"


class TestGenerateColdDiseaseSplit:
    def test_no_disease_leakage(self, conn):
        _seed_export_data(conn, n_diseases=8, n_variants=40)
        generate_cold_disease_split(conn, seed=42)

        train_diseases = conn.execute("""
            SELECT DISTINCT vdp.disease_id FROM vp_split_assignments sa
            JOIN vp_split_definitions sd ON sa.split_id = sd.split_id
            JOIN variant_disease_pairs vdp ON sa.pair_id = vdp.pair_id
            WHERE sd.split_name = 'cold_disease_s42' AND sa.fold = 'train'
        """).fetchall()
        test_diseases = conn.execute("""
            SELECT DISTINCT vdp.disease_id FROM vp_split_assignments sa
            JOIN vp_split_definitions sd ON sa.split_id = sd.split_id
            JOIN variant_disease_pairs vdp ON sa.pair_id = vdp.pair_id
            WHERE sd.split_name = 'cold_disease_s42' AND sa.fold = 'test'
        """).fetchall()

        train_set = {r[0] for r in train_diseases}
        test_set = {r[0] for r in test_diseases}
        assert not (train_set & test_set), "Disease leakage detected"


class TestGenerateTemporalSplit:
    def test_temporal_ordering(self, conn):
        _seed_export_data(conn)
        generate_temporal_split(conn)

        assert _count_split_assignments(conn, "temporal") > 0

    def test_temporal_folds(self, conn):
        _seed_export_data(conn)
        generate_temporal_split(conn)

        folds = _get_split_folds(conn, "temporal")
        assert "train" in folds  # years 2015-2019 → train


class TestGenerateDegreeBalancedSplit:
    def test_creates_split(self, conn):
        _seed_export_data(conn)
        generate_degree_balanced_split(conn, seed=42)

        assert _count_split_assignments(conn, "degree_balanced_s42") > 0

    def test_all_folds(self, conn):
        _seed_export_data(conn)
        generate_degree_balanced_split(conn, seed=42)

        folds = _get_split_folds(conn, "degree_balanced_s42")
        assert folds == {"train", "val", "test"}


class TestGenerateColdBothSplit:
    def test_creates_split(self, conn):
        _seed_export_data(conn, n_genes=10, n_diseases=8, n_variants=40)
        generate_cold_both_split(conn, seed=42)

        assert _count_split_assignments(conn, "cold_both_s42") > 0


class TestExportDataset:
    def test_export_parquet(self, conn, tmp_path):
        _seed_export_data(conn)
        generate_random_split(conn, seed=42)

        output = tmp_path / "vp_export.parquet"
        n_rows = export_vp_dataset(conn, output)

        assert output.exists()
        assert n_rows > 0

        df = pd.read_parquet(output)
        assert "pair_id" in df.columns
        assert "variant_id" in df.columns
        assert "disease_id" in df.columns
        assert "gene_symbol" in df.columns
        assert f"split_random_s42" in df.columns
