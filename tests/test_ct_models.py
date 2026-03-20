"""Tests for CT model definitions (ct_models.py).

3 test classes:
  TestCTMLP: 4 tests — output shapes, gradient flow, dropout
  TestCTGNNTab: 4 tests — output shapes, placeholder graph, gradient flow
  TestModelFactory: 2 tests — correct model type, unknown raises ValueError
"""

import numpy as np
import pytest
import torch

from negbiodb_ct.ct_features import (
    CONDITION_DIM,
    DRUG_TAB_DIM,
    M2_TRIAL_DIM,
    TOTAL_M1_DIM,
    TOTAL_M2_DIM,
)
from negbiodb_ct.ct_models import (
    CT_GNN_Tab,
    CT_MLP,
    GNN_TAB_DIM_M1,
    GNN_TAB_DIM_M2,
    build_ct_model,
)

# Skip GNN tests if torch_geometric not installed
torch_geometric = pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data

from negbiodb.models.graphdta import NODE_FEATURE_DIM


def _make_dummy_graph(n_atoms: int = 5) -> Data:
    """Create a dummy molecular graph for testing."""
    x = torch.randn(n_atoms, NODE_FEATURE_DIM)
    # Simple chain graph: 0-1-2-...-n
    edges = []
    for i in range(n_atoms - 1):
        edges.extend([[i, i + 1], [i + 1, i]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def _make_single_node_graph() -> Data:
    """Placeholder graph: single node, no edges."""
    x = torch.zeros(1, NODE_FEATURE_DIM)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


# ============================================================================
# TestCTMLP
# ============================================================================


class TestCTMLP:
    """Test CT_MLP forward pass and properties."""

    def test_m1_output_shape(self):
        """M1 binary: output shape (B,)."""
        model = CT_MLP(input_dim=TOTAL_M1_DIM, num_classes=1)
        x = torch.randn(4, TOTAL_M1_DIM)
        out = model(x)
        assert out.shape == (4,)

    def test_m2_output_shape(self):
        """M2 multiclass: output shape (B, 8)."""
        model = CT_MLP(input_dim=TOTAL_M2_DIM, num_classes=8)
        x = torch.randn(4, TOTAL_M2_DIM)
        out = model(x)
        assert out.shape == (4, 8)

    def test_gradient_flows(self):
        """Gradient should flow through all parameters."""
        model = CT_MLP(input_dim=TOTAL_M1_DIM, num_classes=1)
        x = torch.randn(2, TOTAL_M1_DIM)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_dropout_effect(self):
        """Dropout should cause different outputs in train vs eval mode."""
        model = CT_MLP(input_dim=TOTAL_M1_DIM, num_classes=1, dropout=0.5)
        x = torch.randn(8, TOTAL_M1_DIM)

        model.train()
        # Run multiple times to avoid the unlikely case of identical outputs
        train_outputs = [model(x).detach() for _ in range(5)]
        # At least some should differ due to dropout
        differs = any(
            not torch.allclose(train_outputs[0], train_outputs[i])
            for i in range(1, 5)
        )
        assert differs, "Dropout should cause variation in train mode"

        model.eval()
        eval_out1 = model(x).detach()
        eval_out2 = model(x).detach()
        torch.testing.assert_close(eval_out1, eval_out2)


# ============================================================================
# TestCTGNNTab
# ============================================================================


class TestCTGNNTab:
    """Test CT_GNN_Tab forward pass."""

    def test_m1_output_shape(self):
        """M1 binary: output (B,)."""
        model = CT_GNN_Tab(tab_dim=GNN_TAB_DIM_M1, num_classes=1)
        graphs = [_make_dummy_graph(5), _make_dummy_graph(3)]
        batch = Batch.from_data_list(graphs)
        tab = torch.randn(2, GNN_TAB_DIM_M1)
        out = model(batch, tab)
        assert out.shape == (2,)

    def test_m2_output_shape(self):
        """M2 multiclass: output (B, 8)."""
        model = CT_GNN_Tab(tab_dim=GNN_TAB_DIM_M2, num_classes=8)
        graphs = [_make_dummy_graph(4), _make_dummy_graph(6)]
        batch = Batch.from_data_list(graphs)
        tab = torch.randn(2, GNN_TAB_DIM_M2)
        out = model(batch, tab)
        assert out.shape == (2, 8)

    def test_single_node_placeholder(self):
        """Single-node placeholder graph should not crash."""
        model = CT_GNN_Tab(tab_dim=GNN_TAB_DIM_M1, num_classes=1)
        graphs = [_make_single_node_graph(), _make_dummy_graph(3)]
        batch = Batch.from_data_list(graphs)
        tab = torch.randn(2, GNN_TAB_DIM_M1)
        out = model(batch, tab)
        assert out.shape == (2,)
        assert not torch.any(torch.isnan(out))

    def test_gradient_flows(self):
        """Gradient should flow through GNN and tabular encoder."""
        model = CT_GNN_Tab(tab_dim=GNN_TAB_DIM_M1, num_classes=1)
        graphs = [_make_dummy_graph(5)]
        batch = Batch.from_data_list(graphs)
        tab = torch.randn(1, GNN_TAB_DIM_M1)
        out = model(batch, tab)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ============================================================================
# TestModelFactory
# ============================================================================


class TestModelFactory:
    """Test build_ct_model factory function."""

    def test_correct_model_types(self):
        """Factory should return correct model types."""
        mlp = build_ct_model("mlp", task="m1")
        assert isinstance(mlp, CT_MLP)

        gnn = build_ct_model("gnn", task="m2")
        assert isinstance(gnn, CT_GNN_Tab)

    def test_unknown_model_raises(self):
        """Unknown model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            build_ct_model("transformer", task="m1")
