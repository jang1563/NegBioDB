"""Tests for DC domain feature engineering module.

Tests molecular descriptors, Morgan fingerprints, Tanimoto similarity,
tabular feature assembly, and DeepSynergy feature construction.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from negbiodb_dc.dc_db import get_connection, run_dc_migrations
from negbiodb_dc.dc_features import (
    DESCRIPTOR_NAMES,
    MISSING_SENTINEL,
    build_deepsynergy_features,
    build_tabular_features,
    compute_mol_descriptors,
    compute_morgan_fp,
    compute_tanimoto,
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


def _seed_basic_data(conn):
    """Seed compounds, cell lines, targets, pairs for feature tests."""
    conn.execute(
        "INSERT INTO compounds (drug_name, canonical_smiles) "
        "VALUES ('Aspirin', 'CC(=O)OC1=CC=CC=C1C(=O)O')"
    )
    conn.execute(
        "INSERT INTO compounds (drug_name, canonical_smiles) "
        "VALUES ('Ibuprofen', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O')"
    )
    conn.execute(
        "INSERT INTO compounds (drug_name, canonical_smiles) "
        "VALUES ('Caffeine', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C')"
    )
    conn.execute("INSERT INTO cell_lines (cell_line_name, tissue) VALUES ('MCF7', 'Breast')")
    conn.execute("INSERT INTO cell_lines (cell_line_name, tissue) VALUES ('A549', 'Lung')")

    # Drug targets for overlap features
    conn.execute(
        "INSERT INTO drug_targets (compound_id, gene_symbol, source) "
        "VALUES (1, 'COX1', 'dgidb')"
    )
    conn.execute(
        "INSERT INTO drug_targets (compound_id, gene_symbol, source) "
        "VALUES (1, 'COX2', 'dgidb')"
    )
    conn.execute(
        "INSERT INTO drug_targets (compound_id, gene_symbol, source) "
        "VALUES (2, 'COX2', 'dgidb')"
    )

    # Synergy results (to create pairs)
    conn.execute(
        """INSERT INTO dc_synergy_results
        (compound_a_id, compound_b_id, cell_line_id,
         zip_score, bliss_score, synergy_class, confidence_tier,
         evidence_type, source_db)
        VALUES (1, 2, 1, -8.0, -6.0, 'antagonistic', 'bronze',
                'dose_response_matrix', 'drugcomb')"""
    )
    conn.execute(
        """INSERT INTO dc_synergy_results
        (compound_a_id, compound_b_id, cell_line_id,
         zip_score, bliss_score, synergy_class, confidence_tier,
         evidence_type, source_db)
        VALUES (1, 3, 2, 7.0, 5.0, 'synergistic', 'bronze',
                'dose_response_matrix', 'drugcomb')"""
    )
    conn.commit()

    # Refresh pairs
    from negbiodb_dc.dc_db import refresh_all_drug_pairs
    refresh_all_drug_pairs(conn)


# ── Molecular descriptors ──────────────────────────────────────────


class TestComputeMolDescriptors:
    def test_valid_smiles(self):
        desc = compute_mol_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
        assert desc.shape == (8,)
        assert not np.any(np.isnan(desc))
        # MW of aspirin ~180.16
        assert 170 < desc[0] < 190

    def test_none_smiles(self):
        desc = compute_mol_descriptors(None)
        assert desc.shape == (8,)
        assert np.all(np.isnan(desc))

    def test_empty_smiles(self):
        desc = compute_mol_descriptors("")
        assert desc.shape == (8,)
        assert np.all(np.isnan(desc))

    def test_invalid_smiles(self):
        desc = compute_mol_descriptors("INVALID_SMILES_XXX")
        assert desc.shape == (8,)
        assert np.all(np.isnan(desc))

    def test_descriptor_names_match(self):
        assert len(DESCRIPTOR_NAMES) == 8


# ── Morgan fingerprint ─────────────────────────────────────────────


class TestComputeMorganFp:
    def test_valid_smiles(self):
        fp = compute_morgan_fp("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert fp.shape == (2048,)
        assert fp.dtype == np.float32
        assert np.sum(fp) > 0  # At least some bits set

    def test_custom_bits(self):
        fp = compute_morgan_fp("CC", n_bits=512)
        assert fp.shape == (512,)

    def test_none_smiles(self):
        fp = compute_morgan_fp(None)
        assert fp.shape == (2048,)
        assert np.sum(fp) == 0  # All zeros

    def test_empty_smiles(self):
        fp = compute_morgan_fp("")
        assert np.sum(fp) == 0

    def test_binary_values(self):
        fp = compute_morgan_fp("CCO")
        unique_vals = set(fp.tolist())
        assert unique_vals.issubset({0.0, 1.0})


# ── Tanimoto similarity ────────────────────────────────────────────


class TestComputeTanimoto:
    def test_identical(self):
        fp = compute_morgan_fp("CCO")
        assert compute_tanimoto(fp, fp) == pytest.approx(1.0)

    def test_different(self):
        fp_a = compute_morgan_fp("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
        fp_b = compute_morgan_fp("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # Caffeine
        sim = compute_tanimoto(fp_a, fp_b)
        assert 0.0 <= sim <= 1.0
        assert sim < 1.0  # Different molecules

    def test_zero_vectors(self):
        zero = np.zeros(2048, dtype=np.float32)
        assert compute_tanimoto(zero, zero) == 0.0

    def test_no_overlap(self):
        a = np.zeros(10, dtype=np.float32)
        b = np.zeros(10, dtype=np.float32)
        a[0] = 1.0
        b[5] = 1.0
        assert compute_tanimoto(a, b) == pytest.approx(0.0)

    def test_complete_overlap(self):
        a = np.array([1, 1, 0, 0], dtype=np.float32)
        b = np.array([1, 1, 0, 0], dtype=np.float32)
        assert compute_tanimoto(a, b) == pytest.approx(1.0)


# ── Tabular features ───────────────────────────────────────────────


class TestBuildTabularFeatures:
    def test_basic_shape(self, conn):
        _seed_basic_data(conn)
        X, names, pair_ids = build_tabular_features(conn)
        assert X.ndim == 2
        assert X.shape[0] == 2  # 2 pairs
        assert len(names) == X.shape[1]
        assert len(pair_ids) == 2

    def test_feature_names_present(self, conn):
        _seed_basic_data(conn)
        X, names, pair_ids = build_tabular_features(conn)
        # Drug A descriptors
        assert "drug_a_molecular_weight" in names
        # Drug B descriptors
        assert "drug_b_logp" in names
        # Target overlap
        assert "shared_targets" in names
        assert "tanimoto_similarity" in names
        # Pair stats
        assert "num_cell_lines" in names

    def test_with_pair_filter(self, conn):
        _seed_basic_data(conn)
        pair_ids_all = [r[0] for r in conn.execute(
            "SELECT pair_id FROM drug_drug_pairs"
        ).fetchall()]

        X, names, pids = build_tabular_features(conn, pair_ids=[pair_ids_all[0]])
        assert X.shape[0] == 1
        assert pids == [pair_ids_all[0]]

    def test_sentinel_mode(self, conn):
        _seed_basic_data(conn)
        X, names, pair_ids = build_tabular_features(conn, use_sentinel=True)
        # No NaN should remain when sentinel is used
        assert not np.any(np.isnan(X))

    def test_empty_db(self, conn):
        X, names, pair_ids = build_tabular_features(conn)
        assert X.shape == (0, 0)
        assert names == []
        assert pair_ids == []

    def test_descriptor_values_reasonable(self, conn):
        _seed_basic_data(conn)
        X, names, pair_ids = build_tabular_features(conn)
        # Find drug_a_molecular_weight column
        mw_idx = names.index("drug_a_molecular_weight")
        assert X[0, mw_idx] > 50  # Molecular weight should be positive


# ── DeepSynergy features ───────────────────────────────────────────


class TestBuildDeepSynergyFeatures:
    def test_basic_shape(self, conn):
        _seed_basic_data(conn)
        X, pair_ids = build_deepsynergy_features(conn)
        assert X.shape == (2, 4096)  # 2 pairs × (2048 + 2048)
        assert len(pair_ids) == 2

    def test_custom_fp_bits(self, conn):
        _seed_basic_data(conn)
        X, pair_ids = build_deepsynergy_features(conn, fp_bits=512)
        assert X.shape == (2, 1024)  # 2 × (512 + 512)

    def test_binary_values(self, conn):
        _seed_basic_data(conn)
        X, pair_ids = build_deepsynergy_features(conn)
        unique_vals = set(X.flatten().tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_with_pair_filter(self, conn):
        _seed_basic_data(conn)
        pair_ids_all = [r[0] for r in conn.execute(
            "SELECT pair_id FROM drug_drug_pairs"
        ).fetchall()]

        X, pids = build_deepsynergy_features(conn, pair_ids=[pair_ids_all[0]])
        assert X.shape[0] == 1

    def test_empty_db(self, conn):
        X, pair_ids = build_deepsynergy_features(conn)
        assert X.shape[0] == 0
        assert pair_ids == []
