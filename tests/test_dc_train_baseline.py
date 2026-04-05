"""Tests for DC domain training script and ML models.

Tests XGBoost, MLP, DeepSynergy, and DrugCombGNN model training/prediction.
"""

from pathlib import Path

import numpy as np
import pytest


# ── Helper functions (must be defined before skipif decorators) ─────


def _check_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


def _check_torch_geometric():
    try:
        import torch_geometric
        return True
    except ImportError:
        return False


# ── XGBoost tests ───────────────────────────────────────────────────


class TestXGBoostDC:
    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n, 10).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        return X[:70], y[:70], X[70:85], y[70:85], X[85:], y[85:]

    def test_binary_train_predict(self, data):
        from negbiodb_dc.models.xgboost_dc import train_xgboost_dc, predict_xgboost_dc

        X_train, y_train, X_val, y_val, X_test, y_test = data
        model = train_xgboost_dc(X_train, y_train, X_val, y_val, task="binary")
        y_pred, y_prob = predict_xgboost_dc(model, X_test, task="binary")

        assert y_pred.shape == (15,)
        assert y_prob.shape == (15, 2)
        assert set(y_pred.tolist()).issubset({0, 1})

    def test_multiclass_train_predict(self, data):
        from negbiodb_dc.models.xgboost_dc import train_xgboost_dc, predict_xgboost_dc

        X_train, y_train, X_val, y_val, X_test, y_test = data
        # Create 3 classes
        y_train_3 = y_train.copy()
        y_train_3[y_train_3 == 0] = np.where(
            np.random.RandomState(0).random(sum(y_train_3 == 0)) > 0.5,
            0, 2,
        )
        y_val_3 = y_val.copy()
        y_val_3[y_val_3 == 0] = 2

        model = train_xgboost_dc(X_train, y_train_3, X_val, y_val_3, task="multiclass")
        y_pred, y_prob = predict_xgboost_dc(model, X_test, task="multiclass")
        assert y_pred.shape == (15,)

    def test_save_load(self, data, tmp_path):
        from negbiodb_dc.models.xgboost_dc import (
            load_xgboost_model, save_xgboost_model, train_xgboost_dc,
            predict_xgboost_dc,
        )

        X_train, y_train, X_val, y_val, X_test, y_test = data
        model = train_xgboost_dc(X_train, y_train, task="binary")
        path = tmp_path / "model.json"
        save_xgboost_model(model, path)
        assert path.exists()

        loaded = load_xgboost_model(path)
        y1, _ = predict_xgboost_dc(model, X_test)
        y2, _ = predict_xgboost_dc(loaded, X_test)
        np.testing.assert_array_equal(y1, y2)


# ── MLP tests ──────────────────────────────────────────────────────


class TestMLPDC:
    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n, 10).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        return X[:70], y[:70], X[70:85], y[70:85], X[85:], y[85:]

    @pytest.mark.skipif(
        not _check_torch(), reason="torch not available"
    )
    def test_binary_train(self, data):
        from negbiodb_dc.models.mlp_dc import train_mlp_dc

        X_train, y_train, X_val, y_val, X_test, y_test = data
        model, history = train_mlp_dc(
            X_train, y_train, X_val, y_val,
            n_classes=2, epochs=3, batch_size=32,
        )
        assert len(history["train_loss"]) == 3

    @pytest.mark.skipif(
        not _check_torch(), reason="torch not available"
    )
    def test_forward_pass(self, data):
        import torch
        from negbiodb_dc.models.mlp_dc import DCMLP

        model = DCMLP(input_dim=10, n_classes=2)
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5, 2)


# ── DeepSynergy tests ──────────────────────────────────────────────


class TestDeepSynergyDC:
    @pytest.mark.skipif(
        not _check_torch(), reason="torch not available"
    )
    def test_forward_pass(self):
        import torch
        from negbiodb_dc.models.deepsynergy_dc import DeepSynergyDC

        model = DeepSynergyDC(input_dim=100, n_classes=2)
        x = torch.randn(5, 100)
        out = model(x)
        assert out.shape == (5, 2)

    @pytest.mark.skipif(
        not _check_torch(), reason="torch not available"
    )
    def test_binary_train(self):
        from negbiodb_dc.models.deepsynergy_dc import train_deepsynergy_dc

        rng = np.random.RandomState(42)
        X = rng.randn(100, 100).astype(np.float32)
        y = rng.randint(0, 2, 100)

        model, history = train_deepsynergy_dc(
            X[:80], y[:80], X[80:], y[80:],
            n_classes=2, epochs=2, batch_size=32,
        )
        assert len(history["train_loss"]) == 2


# ── DrugCombGNN tests ──────────────────────────────────────────────


class TestDrugCombGNN:
    @pytest.mark.skipif(
        not _check_torch_geometric(), reason="torch_geometric not available"
    )
    def test_mol_to_graph(self):
        from negbiodb_dc.models.drugcomb_gnn import mol_to_graph

        graph = mol_to_graph("CCO")
        assert graph is not None
        assert graph.x.shape[0] == 3  # 3 atoms in ethanol
        assert graph.x.shape[1] == 9  # 9 features

    @pytest.mark.skipif(
        not _check_torch_geometric(), reason="torch_geometric not available"
    )
    def test_invalid_smiles(self):
        from negbiodb_dc.models.drugcomb_gnn import mol_to_graph

        graph = mol_to_graph("INVALID")
        assert graph is None

    @pytest.mark.skipif(
        not _check_torch_geometric(), reason="torch_geometric not available"
    )
    def test_prepare_graph_pairs(self):
        from negbiodb_dc.models.drugcomb_gnn import prepare_graph_pairs

        ga, gb, valid = prepare_graph_pairs(
            ["CCO", "INVALID", "CC"],
            ["CC", "CCO", "CCO"],
        )
        assert len(ga) == 2  # First and third are valid
        assert len(gb) == 2
        assert valid == [True, False, True]


# ── Metric computation ──────────────────────────────────────────────


class TestComputeMetrics:
    def test_binary_metrics(self):
        from scripts_dc.train_dc_baseline import compute_metrics

        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_prob = np.array([0.2, 0.6, 0.8, 0.9, 0.3])

        m = compute_metrics(y_true, y_pred, y_prob)
        assert "auroc" in m
        assert "mcc" in m
        assert 0 <= m["auroc"] <= 1
        assert m["n_test"] == 5

    def test_single_class(self):
        from scripts_dc.train_dc_baseline import compute_metrics

        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3])

        m = compute_metrics(y_true, y_pred, y_prob)
        assert m["auroc"] is None
        assert "single class" in m.get("note", "")
