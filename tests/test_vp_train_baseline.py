"""Tests for the VP training harness input preparation."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts_vp"))

from train_vp_baseline import compute_metrics, load_dataset_frame, load_esm2_inputs, load_gnn_inputs


@pytest.fixture
def vp_parquet(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "pair_id": [1, 2, 3],
            "variant_id": [11, 22, 33],
            "disease_id": [101, 102, 103],
            "gene_symbol": ["GENE1", "GENE2", "MISSING"],
            "hgvs_protein": ["p.Ala1Asp", "p.Ala2Asp", "p.Ala3Asp"],
            "consequence_type": ["missense", "missense", "missense"],
            "disease_name": ["D1", "D2", "D3"],
            "medgen_cui": ["C1", "C2", "C3"],
            "Y": [0, 1, 0],
            "confidence_tier": ["bronze", "silver", "bronze"],
            "num_submissions": [1, 2, 1],
            "num_submitters": [1, 2, 1],
            "has_conflict": [0, 0, 0],
            "feature_a": [1.0, np.nan, 3.0],
            "feature_b": [0.1, 0.2, 0.3],
            "split_random": ["train", "val", "test"],
        }
    )
    path = tmp_path / "vp_export.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def esm2_parquet(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "variant_id": [11, 33],
            "esm2_0": [0.5, 1.5],
            "esm2_1": [0.6, 1.6],
        }
    )
    path = tmp_path / "esm2.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def gene_graph(tmp_path: Path) -> Path:
    graph = {
        "gene_to_idx": {"GENE1": 0, "GENE2": 1},
        "node_features": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
    }
    path = tmp_path / "graph.pkl"
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    return path


class TestLoadDatasetFrame:
    def test_feature_columns_exclude_meta_and_split(self, vp_parquet: Path):
        df, feature_cols = load_dataset_frame(vp_parquet)
        assert list(df["pair_id"]) == [1, 2, 3]
        assert feature_cols == ["feature_a", "feature_b"]


class TestLoadEsm2Inputs:
    def test_joins_embeddings_and_zero_fills_missing(self, vp_parquet: Path, esm2_parquet: Path):
        df, feature_cols = load_dataset_frame(vp_parquet)
        (
            X_tab_train,
            X_esm_train,
            y_train,
            X_tab_val,
            X_esm_val,
            y_val,
            X_tab_test,
            X_esm_test,
            y_test,
            esm_cols,
        ) = load_esm2_inputs(df, feature_cols, "split_random", esm2_parquet)

        assert esm_cols == ["esm2_0", "esm2_1"]
        assert X_tab_train.shape == (1, 2)
        assert X_esm_train.shape == (1, 2)
        assert y_train.tolist() == [0]
        assert X_esm_train[0].tolist() == pytest.approx([0.5, 0.6])

        # Missing embedding for variant_id=22 should be zero-filled and NaN tabular should use sentinel.
        assert X_tab_val[0, 0] == pytest.approx(-1.0)
        assert X_esm_val[0].tolist() == pytest.approx([0.0, 0.0])
        assert y_val.tolist() == [1]

        assert X_esm_test[0].tolist() == pytest.approx([1.5, 1.6])
        assert y_test.tolist() == [0]


class TestLoadGnnInputs:
    def test_maps_known_and_missing_gene_indices(self, vp_parquet: Path, gene_graph: Path):
        df, feature_cols = load_dataset_frame(vp_parquet)
        (
            X_tab_train,
            gene_idx_train,
            y_train,
            X_tab_val,
            gene_idx_val,
            y_val,
            X_tab_test,
            gene_idx_test,
            y_test,
            gene_features,
            edge_index,
        ) = load_gnn_inputs(df, feature_cols, "split_random", gene_graph)

        assert X_tab_train.shape == (1, 2)
        assert gene_idx_train.tolist() == [0]
        assert gene_idx_val.tolist() == [1]
        assert gene_idx_test.tolist() == [-1]
        assert gene_features.shape == (2, 2)
        assert edge_index.shape == (2, 2)
        assert y_test.tolist() == [0]


class TestComputeMetrics:
    def test_binary_accepts_probability_matrix(self):
        metrics = compute_metrics(
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 0, 1]),
            np.array(
                [
                    [0.9, 0.1],
                    [0.2, 0.8],
                    [0.7, 0.3],
                    [0.1, 0.9],
                ]
            ),
        )
        assert metrics["auroc"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_multiclass_probability_matrix(self):
        metrics = compute_metrics(
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            np.array(
                [
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8],
                ]
            ),
        )
        assert metrics["auroc"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
