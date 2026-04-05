"""Tests for DC domain ML export module.

Tests all 6 split strategies, parquet export, and M1/M2 label builders.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from negbiodb_dc.dc_db import get_connection, refresh_all_drug_pairs, run_dc_migrations
from negbiodb_dc.export import (
    build_dc_m1_labels,
    build_dc_m2_labels,
    export_dc_dataset,
    generate_all_splits,
    generate_cold_both_split,
    generate_cold_cell_line_split,
    generate_cold_compound_split,
    generate_leave_one_tissue_out_split,
    generate_random_split,
    generate_scaffold_split,
)

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_dc"


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_dc.db"
    run_dc_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def conn(tmp_db):
    c = get_connection(tmp_db)
    yield c
    c.close()


def _seed_rich_data(conn, n_compounds=10, n_cell_lines=5, n_pairs=20):
    """Seed a reasonably-sized DB for split testing."""
    # Compounds with SMILES
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
        "OC1=CC=CC=C1",  # Phenol
        "C1=CC=C(C=C1)O",  # Also phenol variant
        "CCO",  # Ethanol
        "CCCO",  # Propanol
        "CC(=O)O",  # Acetic acid
        "CC",  # Ethane
    ]
    for i in range(n_compounds):
        smiles = smiles_list[i] if i < len(smiles_list) else f"C{'C' * i}"
        conn.execute(
            "INSERT INTO compounds (drug_name, canonical_smiles) VALUES (?, ?)",
            (f"Drug{i}", smiles),
        )

    tissues = ["Breast", "Lung", "Colon", "Brain", "Kidney"]
    for i in range(n_cell_lines):
        conn.execute(
            "INSERT INTO cell_lines (cell_line_name, tissue) VALUES (?, ?)",
            (f"CL{i}", tissues[i % len(tissues)]),
        )

    # Drug targets
    genes = ["TP53", "EGFR", "BRAF", "KRAS", "PIK3CA"]
    for i in range(min(n_compounds, 5)):
        for g in genes[: (i % 3) + 1]:
            try:
                conn.execute(
                    "INSERT INTO drug_targets (compound_id, gene_symbol, source) "
                    "VALUES (?, ?, 'dgidb')",
                    (i + 1, g),
                )
            except sqlite3.IntegrityError:
                pass

    # Synergy results — create pairs
    classes = ["antagonistic", "synergistic", "additive",
               "strongly_antagonistic", "strongly_synergistic"]
    tiers = ["gold", "silver", "bronze", "copper"]
    rng = np.random.RandomState(42)
    inserted = 0

    for a in range(1, n_compounds + 1):
        for b in range(a + 1, n_compounds + 1):
            if inserted >= n_pairs:
                break
            cl_id = rng.randint(1, n_cell_lines + 1)
            cls = classes[rng.randint(0, len(classes))]
            tier = tiers[rng.randint(0, len(tiers))]
            zip_score = rng.uniform(-15, 15)
            conn.execute(
                """INSERT INTO dc_synergy_results
                (compound_a_id, compound_b_id, cell_line_id,
                 zip_score, bliss_score, synergy_class, confidence_tier,
                 evidence_type, source_db)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'dose_response_matrix', 'drugcomb')""",
                (a, b, cl_id, zip_score, zip_score * 0.8, cls, tier),
            )
            inserted += 1
        if inserted >= n_pairs:
            break

    conn.commit()
    refresh_all_drug_pairs(conn)


# ── Random split ────────────────────────────────────────────────────


class TestRandomSplit:
    def test_basic(self, conn):
        _seed_rich_data(conn)
        split_id = generate_random_split(conn, seed=42)
        assert split_id > 0

        folds = conn.execute(
            "SELECT fold, COUNT(*) FROM dc_split_assignments WHERE split_id = ? GROUP BY fold",
            (split_id,),
        ).fetchall()
        fold_dict = dict(folds)
        assert "train" in fold_dict
        assert "test" in fold_dict

    def test_all_pairs_assigned(self, conn):
        _seed_rich_data(conn)
        split_id = generate_random_split(conn, seed=42)
        n_pairs = conn.execute("SELECT COUNT(*) FROM drug_drug_pairs").fetchone()[0]
        n_assigned = conn.execute(
            "SELECT COUNT(*) FROM dc_split_assignments WHERE split_id = ?",
            (split_id,),
        ).fetchone()[0]
        assert n_assigned == n_pairs

    def test_reproducible(self, conn):
        _seed_rich_data(conn)
        s1 = generate_random_split(conn, seed=42)
        folds1 = dict(conn.execute(
            "SELECT pair_id, fold FROM dc_split_assignments WHERE split_id = ?", (s1,)
        ).fetchall())

        s2 = generate_random_split(conn, seed=42)
        folds2 = dict(conn.execute(
            "SELECT pair_id, fold FROM dc_split_assignments WHERE split_id = ?", (s2,)
        ).fetchall())
        assert folds1 == folds2

    def test_empty_db(self, conn):
        split_id = generate_random_split(conn, seed=42)
        assert split_id > 0
        n = conn.execute(
            "SELECT COUNT(*) FROM dc_split_assignments WHERE split_id = ?",
            (split_id,),
        ).fetchone()[0]
        assert n == 0


# ── Cold compound split ─────────────────────────────────────────────


class TestColdCompoundSplit:
    def test_basic(self, conn):
        _seed_rich_data(conn)
        split_id = generate_cold_compound_split(conn, seed=42)
        folds = conn.execute(
            "SELECT fold, COUNT(*) FROM dc_split_assignments WHERE split_id = ? GROUP BY fold",
            (split_id,),
        ).fetchall()
        fold_dict = dict(folds)
        assert "train" in fold_dict

    def test_cold_property(self, conn):
        """Each test pair has at least one compound not in any train pair."""
        _seed_rich_data(conn)
        split_id = generate_cold_compound_split(conn, seed=42)

        train_pairs = conn.execute(
            """SELECT p.compound_a_id, p.compound_b_id
            FROM dc_split_assignments sa
            JOIN drug_drug_pairs p ON sa.pair_id = p.pair_id
            WHERE sa.split_id = ? AND sa.fold = 'train'""",
            (split_id,),
        ).fetchall()
        train_compounds = set()
        for a, b in train_pairs:
            train_compounds.add(a)
            train_compounds.add(b)

        test_pairs = conn.execute(
            """SELECT p.compound_a_id, p.compound_b_id
            FROM dc_split_assignments sa
            JOIN drug_drug_pairs p ON sa.pair_id = p.pair_id
            WHERE sa.split_id = ? AND sa.fold = 'test'""",
            (split_id,),
        ).fetchall()

        # Each test pair should have at least one compound NOT in train
        # (max-rank: pair in test ⟹ at least one compound has rank=2)
        for a, b in test_pairs:
            assert a not in train_compounds or b not in train_compounds, (
                f"Both compounds ({a}, {b}) appear in train — not cold"
            )


# ── Cold cell line split ────────────────────────────────────────────


class TestColdCellLineSplit:
    def test_basic(self, conn):
        _seed_rich_data(conn)
        split_id = generate_cold_cell_line_split(conn, seed=42)
        n = conn.execute(
            "SELECT COUNT(*) FROM dc_split_assignments WHERE split_id = ?",
            (split_id,),
        ).fetchone()[0]
        assert n > 0


# ── Cold both split ─────────────────────────────────────────────────


class TestColdBothSplit:
    def test_basic(self, conn):
        _seed_rich_data(conn)
        split_id = generate_cold_both_split(conn, seed=42)
        folds = conn.execute(
            "SELECT fold, COUNT(*) FROM dc_split_assignments WHERE split_id = ? GROUP BY fold",
            (split_id,),
        ).fetchall()
        fold_dict = dict(folds)
        assert "train" in fold_dict


# ── Scaffold split ──────────────────────────────────────────────────


class TestScaffoldSplit:
    def test_basic(self, conn):
        _seed_rich_data(conn)
        split_id = generate_scaffold_split(conn, seed=42)
        n = conn.execute(
            "SELECT COUNT(*) FROM dc_split_assignments WHERE split_id = ?",
            (split_id,),
        ).fetchone()[0]
        assert n > 0

    def test_all_pairs_assigned(self, conn):
        _seed_rich_data(conn)
        split_id = generate_scaffold_split(conn, seed=42)
        n_pairs = conn.execute("SELECT COUNT(*) FROM drug_drug_pairs").fetchone()[0]
        n_assigned = conn.execute(
            "SELECT COUNT(*) FROM dc_split_assignments WHERE split_id = ?",
            (split_id,),
        ).fetchone()[0]
        assert n_assigned == n_pairs


# ── Leave-one-tissue-out split ──────────────────────────────────────


class TestLeaveOneTissueOutSplit:
    def test_basic(self, conn):
        _seed_rich_data(conn)
        split_id = generate_leave_one_tissue_out_split(conn, seed=42)
        folds = conn.execute(
            "SELECT fold, COUNT(*) FROM dc_split_assignments WHERE split_id = ? GROUP BY fold",
            (split_id,),
        ).fetchall()
        fold_dict = dict(folds)
        assert "test" in fold_dict
        assert "train" in fold_dict

    def test_specific_tissue(self, conn):
        _seed_rich_data(conn)
        split_id = generate_leave_one_tissue_out_split(
            conn, held_out_tissue="Breast", seed=42
        )
        n = conn.execute(
            "SELECT COUNT(*) FROM dc_split_assignments WHERE split_id = ?",
            (split_id,),
        ).fetchone()[0]
        assert n > 0


# ── Generate all splits ─────────────────────────────────────────────


class TestGenerateAllSplits:
    def test_all_six_strategies(self, conn):
        _seed_rich_data(conn)
        splits = generate_all_splits(conn, seed=42)
        assert len(splits) == 6
        assert "random" in splits
        assert "cold_compound" in splits
        assert "cold_cell_line" in splits
        assert "cold_both" in splits
        assert "scaffold" in splits
        assert "leave_one_tissue_out" in splits


# ── Parquet export ──────────────────────────────────────────────────


class TestExportDcDataset:
    def test_basic_export(self, conn, tmp_path):
        _seed_rich_data(conn)
        generate_all_splits(conn, seed=42)

        output = tmp_path / "dc_export.parquet"
        n = export_dc_dataset(conn, output)
        assert n > 0
        assert output.exists()

        df = pd.read_parquet(output)
        assert "pair_id" in df.columns
        assert "drug_a_name" in df.columns
        assert "consensus_class" in df.columns
        # Should have split columns
        split_cols = [c for c in df.columns if c.startswith("split_")]
        assert len(split_cols) == 6

    def test_export_without_splits(self, conn, tmp_path):
        _seed_rich_data(conn)
        output = tmp_path / "dc_no_splits.parquet"
        n = export_dc_dataset(conn, output)
        assert n > 0

    def test_empty_db(self, conn, tmp_path):
        output = tmp_path / "dc_empty.parquet"
        n = export_dc_dataset(conn, output)
        assert n == 0


# ── M1/M2 label builders ───────────────────────────────────────────


class TestLabelBuilders:
    def test_m1_labels(self):
        df = pd.DataFrame({
            "consensus_class": [
                "antagonistic", "additive", "synergistic", "context_dependent"
            ]
        })
        labels = build_dc_m1_labels(df)
        assert labels.tolist() == [0, 0, 1, 0]

    def test_m1_negative_dominant(self):
        """Most combinations are negative (antagonistic/additive) → label 0."""
        df = pd.DataFrame({
            "consensus_class": ["antagonistic"] * 6 + ["additive"] * 3 + ["synergistic"]
        })
        labels = build_dc_m1_labels(df)
        assert sum(labels == 0) == 9
        assert sum(labels == 1) == 1

    def test_m2_labels(self):
        df = pd.DataFrame({
            "consensus_class": ["antagonistic", "additive", "synergistic"]
        })
        labels = build_dc_m2_labels(df)
        assert labels.tolist() == [0, 1, 2]

    def test_m2_context_dependent_is_nan(self):
        df = pd.DataFrame({
            "consensus_class": ["context_dependent"]
        })
        labels = build_dc_m2_labels(df)
        assert labels.isna().all()

    def test_m2_mixed(self):
        df = pd.DataFrame({
            "consensus_class": [
                "antagonistic", "synergistic", "context_dependent", "additive"
            ]
        })
        labels = build_dc_m2_labels(df)
        assert labels.iloc[0] == 0
        assert labels.iloc[1] == 2
        assert pd.isna(labels.iloc[2])
        assert labels.iloc[3] == 1
