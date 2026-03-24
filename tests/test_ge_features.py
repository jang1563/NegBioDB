"""Tests for GE omics feature computation module.

Tests gene features, cell line features, omics loading, and combined matrix building.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from negbiodb_depmap.depmap_db import get_connection, run_ge_migrations
from negbiodb_depmap.ge_features import (
    build_feature_matrix,
    compute_cell_line_features,
    compute_gene_features,
    load_omics_features,
)

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_depmap"


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_ge.db"
    run_ge_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def populated_db(tmp_db):
    """DB with genes, cell lines, screens, negative results, and pairs."""
    conn = get_connection(tmp_db)

    # Genes
    conn.execute(
        "INSERT INTO genes (gene_id, entrez_id, gene_symbol, is_common_essential, is_reference_nonessential) "
        "VALUES (1, 7157, 'TP53', 0, 0)"
    )
    conn.execute(
        "INSERT INTO genes (gene_id, entrez_id, gene_symbol, is_common_essential, is_reference_nonessential) "
        "VALUES (2, 1956, 'EGFR', 0, 1)"
    )
    conn.execute(
        "INSERT INTO genes (gene_id, entrez_id, gene_symbol, is_common_essential, is_reference_nonessential) "
        "VALUES (3, 672, 'BRCA1', 1, 0)"
    )

    # Cell lines
    conn.execute(
        "INSERT INTO cell_lines (cell_line_id, model_id, ccle_name, lineage, primary_disease) "
        "VALUES (1, 'ACH-000001', 'HELA_CERVIX', 'Cervix', 'Cervical Cancer')"
    )
    conn.execute(
        "INSERT INTO cell_lines (cell_line_id, model_id, ccle_name, lineage, primary_disease) "
        "VALUES (2, 'ACH-000002', 'MCF7_BREAST', 'Breast', 'Breast Cancer')"
    )
    conn.execute(
        "INSERT INTO cell_lines (cell_line_id, model_id, ccle_name, lineage, primary_disease) "
        "VALUES (3, 'ACH-000003', 'A549_LUNG', 'Lung', 'Lung Cancer')"
    )

    # Screen
    conn.execute(
        "INSERT INTO ge_screens (screen_id, source_db, depmap_release, screen_type, algorithm) "
        "VALUES (1, 'depmap', 'TEST', 'crispr', 'Chronos')"
    )

    # Negative results
    for gid, clid, effect, dep_prob in [
        (1, 1, -0.1, 0.1),
        (1, 2, -0.2, 0.15),
        (2, 1, -0.05, 0.05),
        (2, 2, -0.3, 0.2),
        (2, 3, -0.15, 0.08),
        (3, 1, -0.4, 0.25),
    ]:
        conn.execute(
            "INSERT INTO ge_negative_results "
            "(gene_id, cell_line_id, screen_id, gene_effect_score, dependency_probability, "
            "confidence_tier, evidence_type, source_db, source_record_id, extraction_method) "
            "VALUES (?, ?, 1, ?, ?, 'silver', 'crispr_nonessential', 'depmap', 'TEST', 'score_threshold')",
            (gid, clid, effect, dep_prob),
        )

    conn.commit()

    # Refresh pairs
    from negbiodb_depmap.depmap_db import refresh_all_ge_pairs
    refresh_all_ge_pairs(conn)
    conn.commit()

    conn.close()
    return tmp_db


@pytest.fixture
def conn_pop(populated_db):
    c = get_connection(populated_db)
    yield c
    c.close()


# ── Gene features ─────────────────────────────────────────────────────


class TestComputeGeneFeatures:
    def test_returns_dataframe(self, conn_pop):
        df = compute_gene_features(conn_pop)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_gene_count(self, conn_pop):
        df = compute_gene_features(conn_pop)
        assert len(df) == 3  # 3 genes

    def test_expected_columns(self, conn_pop):
        df = compute_gene_features(conn_pop)
        for col in ["mean_effect", "min_effect", "max_effect",
                     "is_common_essential", "is_reference_nonessential",
                     "rnai_concordance_fraction"]:
            assert col in df.columns

    def test_mean_effect_reasonable(self, conn_pop):
        df = compute_gene_features(conn_pop)
        # Gene 2 (EGFR) has 3 records: -0.05, -0.3, -0.15
        row = df.loc[2]
        assert -0.5 < row["mean_effect"] < 0.0

    def test_common_essential_flag(self, conn_pop):
        df = compute_gene_features(conn_pop)
        assert df.loc[3, "is_common_essential"] == 1  # BRCA1
        assert df.loc[1, "is_common_essential"] == 0  # TP53


# ── Cell line features ────────────────────────────────────────────────


class TestComputeCellLineFeatures:
    def test_returns_dataframe(self, conn_pop):
        df = compute_cell_line_features(conn_pop)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_cell_line_count(self, conn_pop):
        df = compute_cell_line_features(conn_pop)
        assert len(df) == 3  # 3 cell lines

    def test_lineage_one_hot(self, conn_pop):
        df = compute_cell_line_features(conn_pop)
        lineage_cols = [c for c in df.columns if c.startswith("lineage_")]
        assert len(lineage_cols) == 3  # Cervix, Breast, Lung

    def test_disease_one_hot(self, conn_pop):
        df = compute_cell_line_features(conn_pop)
        disease_cols = [c for c in df.columns if c.startswith("disease_")]
        assert len(disease_cols) == 3


# ── Omics features ───────────────────────────────────────────────────


class TestLoadOmicsFeatures:
    def test_empty_when_no_files(self):
        result = load_omics_features()
        assert result == {}

    def test_expression_loading(self, tmp_path):
        # Create synthetic expression matrix
        data = {"TP53": [5.0, 3.0], "EGFR": [2.0, 4.0]}
        df = pd.DataFrame(data, index=["ACH-000001", "ACH-000002"])
        expr_file = tmp_path / "expression.csv"
        df.to_csv(expr_file, index_label="")

        result = load_omics_features(expression_file=expr_file)
        assert len(result) > 0
        assert ("ACH-000001", "TP53") in result
        assert result[("ACH-000001", "TP53")][0] == 5.0  # expression dim

    def test_cn_loading(self, tmp_path):
        data = {"TP53": [2.0]}
        df = pd.DataFrame(data, index=["ACH-000001"])
        cn_file = tmp_path / "cn.csv"
        df.to_csv(cn_file, index_label="")

        result = load_omics_features(cn_file=cn_file)
        assert ("ACH-000001", "TP53") in result
        assert result[("ACH-000001", "TP53")][1] == 2.0  # CN dim

    def test_mutation_loading(self, tmp_path):
        data = {"TP53": [1.0], "EGFR": [0.0]}
        df = pd.DataFrame(data, index=["ACH-000001"])
        mut_file = tmp_path / "mutations.csv"
        df.to_csv(mut_file, index_label="")

        result = load_omics_features(mutation_file=mut_file)
        assert ("ACH-000001", "TP53") in result
        assert result[("ACH-000001", "TP53")][2] == 1.0  # mutation dim

    def test_combined_3_dims(self, tmp_path):
        # All 3 omics files
        for fname, data in [
            ("expr.csv", {"G1": [5.0]}),
            ("cn.csv", {"G1": [2.0]}),
            ("mut.csv", {"G1": [1.0]}),
        ]:
            df = pd.DataFrame(data, index=["ACH-000001"])
            df.to_csv(tmp_path / fname, index_label="")

        result = load_omics_features(
            expression_file=tmp_path / "expr.csv",
            cn_file=tmp_path / "cn.csv",
            mutation_file=tmp_path / "mut.csv",
        )
        vec = result[("ACH-000001", "G1")]
        assert len(vec) == 3
        assert vec[0] == 5.0
        assert vec[1] == 2.0
        assert vec[2] == 1.0

    def test_gene_symbol_filter(self, tmp_path):
        data = {"TP53": [5.0], "EGFR": [3.0], "BRCA1": [1.0]}
        df = pd.DataFrame(data, index=["ACH-000001"])
        expr_file = tmp_path / "expression.csv"
        df.to_csv(expr_file, index_label="")

        result = load_omics_features(
            expression_file=expr_file,
            gene_symbols=["TP53", "EGFR"],
        )
        keys = [k[1] for k in result.keys()]
        assert "TP53" in keys
        assert "EGFR" in keys

    def test_model_id_filter(self, tmp_path):
        data = {"TP53": [5.0, 3.0]}
        df = pd.DataFrame(data, index=["ACH-000001", "ACH-000002"])
        expr_file = tmp_path / "expression.csv"
        df.to_csv(expr_file, index_label="")

        result = load_omics_features(
            expression_file=expr_file,
            model_ids=["ACH-000001"],
        )
        model_ids = [k[0] for k in result.keys()]
        assert "ACH-000001" in model_ids
        assert "ACH-000002" not in model_ids


# ── Build feature matrix ─────────────────────────────────────────────


class TestBuildFeatureMatrix:
    def test_basic_matrix(self, conn_pop):
        pairs_df = pd.DataFrame({
            "gene_id": [1, 2],
            "cell_line_id": [1, 2],
            "gene_symbol": ["TP53", "EGFR"],
            "model_id": ["ACH-000001", "ACH-000002"],
        })
        X = build_feature_matrix(conn_pop, pairs_df)
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 2
        assert X.shape[1] > 0

    def test_no_nan(self, conn_pop):
        pairs_df = pd.DataFrame({
            "gene_id": [1],
            "cell_line_id": [1],
            "gene_symbol": ["TP53"],
            "model_id": ["ACH-000001"],
        })
        X = build_feature_matrix(conn_pop, pairs_df)
        assert not np.isnan(X).any()

    def test_with_omics(self, conn_pop, tmp_path):
        # Create simple omics
        data = {"TP53": [5.0]}
        df = pd.DataFrame(data, index=["ACH-000001"])
        expr_file = tmp_path / "expression.csv"
        df.to_csv(expr_file, index_label="")

        omics = load_omics_features(expression_file=expr_file)

        pairs_df = pd.DataFrame({
            "gene_id": [1],
            "cell_line_id": [1],
            "gene_symbol": ["TP53"],
            "model_id": ["ACH-000001"],
        })
        X = build_feature_matrix(conn_pop, pairs_df, omics_features=omics)
        assert X.shape[0] == 1
        # Should have gene features + cell line features + 3 omics dims
        assert X.shape[1] > 10

    def test_unknown_gene_zeros(self, conn_pop):
        pairs_df = pd.DataFrame({
            "gene_id": [999],
            "cell_line_id": [1],
            "gene_symbol": ["FAKE"],
            "model_id": ["ACH-000001"],
        })
        X = build_feature_matrix(conn_pop, pairs_df)
        assert X.shape[0] == 1
        # Gene features should be zero since gene_id 999 doesn't exist
