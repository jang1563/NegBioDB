"""Tests for GE ML export module.

Tests split generation, conflict resolution, and control negatives
using a small synthetic database.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from negbiodb_depmap.depmap_db import get_connection, run_ge_migrations, refresh_all_ge_pairs
from negbiodb_depmap.export import (
    build_ge_m1,
    build_ge_m2,
    export_ge_negatives,
    generate_cold_both_split,
    generate_cold_cell_line_split,
    generate_cold_gene_split,
    generate_degree_balanced_split,
    generate_random_split,
    generate_uniform_random_negatives,
)

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_depmap"


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_ge.db"
    run_ge_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def seeded_db(tmp_db):
    """DB with 3 genes × 3 cell lines = 6 negative pairs."""
    conn = get_connection(tmp_db)
    # Genes
    conn.execute("INSERT INTO genes (gene_id, entrez_id, gene_symbol, is_reference_nonessential) VALUES (1, 1001, 'GeneA', 1)")
    conn.execute("INSERT INTO genes (gene_id, entrez_id, gene_symbol) VALUES (2, 1002, 'GeneB')")
    conn.execute("INSERT INTO genes (gene_id, entrez_id, gene_symbol, is_common_essential) VALUES (3, 1003, 'GeneC', 1)")

    # Cell lines
    conn.execute("INSERT INTO cell_lines (cell_line_id, model_id, lineage) VALUES (1, 'ACH-001', 'Lung')")
    conn.execute("INSERT INTO cell_lines (cell_line_id, model_id, lineage) VALUES (2, 'ACH-002', 'Breast')")
    conn.execute("INSERT INTO cell_lines (cell_line_id, model_id, lineage) VALUES (3, 'ACH-003', 'Skin')")

    # Screen
    conn.execute(
        """INSERT INTO ge_screens (screen_id, source_db, depmap_release, screen_type, algorithm)
        VALUES (1, 'depmap', 'test', 'crispr', 'Chronos')"""
    )

    # 6 negative results (GeneA in 3 CLs, GeneB in 3 CLs)
    for gene_id in [1, 2]:
        for cl_id in [1, 2, 3]:
            effect = 0.05 if gene_id == 1 else -0.4
            conn.execute(
                """INSERT INTO ge_negative_results
                (gene_id, cell_line_id, screen_id, gene_effect_score, dependency_probability,
                 evidence_type, confidence_tier, source_db, source_record_id, extraction_method)
                VALUES (?, ?, 1, ?, 0.1, 'crispr_nonessential', 'silver',
                        'depmap', ?, 'score_threshold')""",
                (gene_id, cl_id, effect, f"r{gene_id}_{cl_id}"),
            )

    conn.commit()
    refresh_all_ge_pairs(conn)
    conn.commit()
    conn.close()
    return tmp_db


# ── Split tests ───────────────────────────────────────────────────────────


class TestRandomSplit:
    def test_all_pairs_assigned(self, seeded_db):
        conn = get_connection(seeded_db)
        try:
            result = generate_random_split(conn, seed=42)
            total = sum(result["counts"].values())
            assert total == 6
        finally:
            conn.close()

    def test_fold_coverage(self, seeded_db):
        conn = get_connection(seeded_db)
        try:
            result = generate_random_split(conn, seed=42)
            assert set(result["counts"].keys()).issubset({"train", "val", "test"})
        finally:
            conn.close()


class TestColdGeneSplit:
    def test_all_pairs_assigned(self, seeded_db):
        conn = get_connection(seeded_db)
        try:
            result = generate_cold_gene_split(conn, seed=42)
            total = sum(result["counts"].values())
            assert total == 6
        finally:
            conn.close()

    def test_no_gene_leakage(self, seeded_db):
        """Test genes in train should not appear in test."""
        conn = get_connection(seeded_db)
        try:
            result = generate_cold_gene_split(conn, seed=42)
            split_id = result["split_id"]

            train_genes = {
                r[0] for r in conn.execute(
                    """SELECT DISTINCT p.gene_id
                    FROM ge_split_assignments sa
                    JOIN gene_cell_pairs p ON sa.pair_id = p.pair_id
                    WHERE sa.split_id = ? AND sa.fold = 'train'""",
                    (split_id,),
                ).fetchall()
            }
            test_genes = {
                r[0] for r in conn.execute(
                    """SELECT DISTINCT p.gene_id
                    FROM ge_split_assignments sa
                    JOIN gene_cell_pairs p ON sa.pair_id = p.pair_id
                    WHERE sa.split_id = ? AND sa.fold = 'test'""",
                    (split_id,),
                ).fetchall()
            }
            assert len(train_genes & test_genes) == 0
        finally:
            conn.close()


class TestColdCellLineSplit:
    def test_all_pairs_assigned(self, seeded_db):
        conn = get_connection(seeded_db)
        try:
            result = generate_cold_cell_line_split(conn, seed=42)
            total = sum(result["counts"].values())
            assert total == 6
        finally:
            conn.close()

    def test_no_cell_line_leakage(self, seeded_db):
        conn = get_connection(seeded_db)
        try:
            result = generate_cold_cell_line_split(conn, seed=42)
            split_id = result["split_id"]

            train_cls = {
                r[0] for r in conn.execute(
                    """SELECT DISTINCT p.cell_line_id
                    FROM ge_split_assignments sa
                    JOIN gene_cell_pairs p ON sa.pair_id = p.pair_id
                    WHERE sa.split_id = ? AND sa.fold = 'train'""",
                    (split_id,),
                ).fetchall()
            }
            test_cls = {
                r[0] for r in conn.execute(
                    """SELECT DISTINCT p.cell_line_id
                    FROM ge_split_assignments sa
                    JOIN gene_cell_pairs p ON sa.pair_id = p.pair_id
                    WHERE sa.split_id = ? AND sa.fold = 'test'""",
                    (split_id,),
                ).fetchall()
            }
            assert len(train_cls & test_cls) == 0
        finally:
            conn.close()


class TestColdBothSplit:
    def test_all_pairs_assigned(self, seeded_db):
        conn = get_connection(seeded_db)
        try:
            result = generate_cold_both_split(conn, seed=42)
            total = sum(result["counts"].values())
            assert total == 6
        finally:
            conn.close()


class TestDegreeBalancedSplit:
    def test_all_pairs_assigned(self, seeded_db):
        conn = get_connection(seeded_db)
        try:
            result = generate_degree_balanced_split(conn, seed=42)
            total = sum(result["counts"].values())
            assert total == 6
        finally:
            conn.close()


# ── Export tests ──────────────────────────────────────────────────────────


class TestExport:
    def test_export_parquet(self, seeded_db, tmp_path):
        conn = get_connection(seeded_db)
        try:
            generate_random_split(conn, seed=42)
            out = tmp_path / "export" / "pairs.parquet"
            count = export_ge_negatives(conn, out)
            assert count == 6
            assert out.exists()
            df = pd.read_parquet(out)
            assert "gene_symbol" in df.columns
            assert "model_id" in df.columns
            assert "split_random_v1" in df.columns
        finally:
            conn.close()


# ── M1/M2 tests ──────────────────────────────────────────────────────────


class TestBuildM1:
    def test_basic_m1(self, seeded_db):
        conn = get_connection(seeded_db)
        try:
            pos_df = pd.DataFrame({
                "gene_id": [3, 3],
                "cell_line_id": [1, 2],
                "essentiality_type": ["common_essential", "common_essential"],
            })
            neg_df = pd.DataFrame({
                "gene_id": [1, 1, 2],
                "cell_line_id": [1, 2, 1],
            })
            result = build_ge_m1(conn, pos_df, neg_df, balanced=True, ratio=1.0)
            assert (result["label"] == 1).sum() == 2
            assert (result["label"] == 0).sum() == 2
        finally:
            conn.close()

    def test_conflict_resolution(self, seeded_db):
        conn = get_connection(seeded_db)
        try:
            # Gene 1 in CL 1 appears in both positive and negative
            pos_df = pd.DataFrame({
                "gene_id": [1, 3],
                "cell_line_id": [1, 1],
                "essentiality_type": ["selective_essential", "common_essential"],
            })
            neg_df = pd.DataFrame({
                "gene_id": [1, 2],
                "cell_line_id": [1, 1],
            })
            result = build_ge_m1(conn, pos_df, neg_df, balanced=False)
            # (1, 1) conflict → removed from both
            assert len(result) == 2  # gene3-cl1 pos + gene2-cl1 neg
        finally:
            conn.close()


class TestBuildM2:
    def test_three_classes(self, seeded_db):
        conn = get_connection(seeded_db)
        try:
            pos_df = pd.DataFrame({
                "gene_id": [3, 2],
                "cell_line_id": [1, 3],
                "essentiality_type": ["common_essential", "selective_essential"],
            })
            neg_df = pd.DataFrame({
                "gene_id": [1],
                "cell_line_id": [1],
            })
            result = build_ge_m2(conn, pos_df, neg_df)
            assert (result["label"] == 0).sum() == 1  # common essential
            assert (result["label"] == 1).sum() == 1  # selective essential
            assert (result["label"] == 2).sum() == 1  # non-essential
        finally:
            conn.close()


# ── Control negatives ─────────────────────────────────────────────────────


class TestControlNegatives:
    def test_uniform_random(self, seeded_db):
        conn = get_connection(seeded_db)
        try:
            result = generate_uniform_random_negatives(conn, n_samples=5, seed=42)
            assert len(result) <= 5
            assert "neg_source" in result.columns
            assert all(result["neg_source"] == "uniform_random")
        finally:
            conn.close()

    def test_no_overlap_with_existing(self, seeded_db):
        conn = get_connection(seeded_db)
        try:
            existing = set()
            for row in conn.execute(
                "SELECT gene_id, cell_line_id FROM gene_cell_pairs"
            ).fetchall():
                existing.add((row[0], row[1]))

            result = generate_uniform_random_negatives(conn, n_samples=5, seed=42)
            for _, r in result.iterrows():
                assert (r["gene_id"], r["cell_line_id"]) not in existing
        finally:
            conn.close()
