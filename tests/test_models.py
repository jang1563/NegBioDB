"""Tests for ML baseline models (DeepDTA, GraphDTA, DrugBAN).

PyTorch required for all tests. torch-geometric required for GraphDTA/DrugBAN.
Tests that require unavailable packages are skipped automatically.
"""

import pytest

# Skip entire module if torch not installed
torch = pytest.importorskip("torch", reason="requires torch")

import torch.nn as nn
import torch.optim as optim

from negbiodb.models.deepdta import (
    AA_VOCAB,
    MAX_SEQ_LEN,
    MAX_SMILES_LEN,
    SMILES_VOCAB,
    DeepDTA,
    seq_to_tensor,
    smiles_to_tensor,
)

try:
    from torch_geometric.data import Batch, Data

    from negbiodb.models.graphdta import (
        GraphDTA,
        NODE_FEATURE_DIM,
        smiles_to_graph,
    )
    from negbiodb.models.drugban import DrugBAN

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

requires_pyg = pytest.mark.skipif(
    not HAS_TORCH_GEOMETRIC, reason="requires torch_geometric"
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TINY_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
TINY_SEQS = [
    "MKTLLLTLVVVTIVCLDLGYT",  # 21 AA
    "ACDEFGHIKLMNPQRSTVWXY",  # 21 AA
    "ACGTHKLMNPQRSTVWXYABC",  # 21 AA
    "MKTLLTLVVTIVCLDLGYTAC",  # 21 AA
]
TINY_LABELS = torch.tensor([1.0, 0.0, 1.0, 0.0])


@pytest.fixture
def tiny_drug_tokens() -> torch.Tensor:
    return smiles_to_tensor(TINY_SMILES)


@pytest.fixture
def tiny_target_tokens() -> torch.Tensor:
    return seq_to_tensor(TINY_SEQS)


@pytest.fixture
def tiny_labels() -> torch.Tensor:
    return TINY_LABELS


@pytest.fixture
def tiny_drug_graphs():
    """Build a PyG Batch from tiny SMILES. Skipped if no PyG."""
    if not HAS_TORCH_GEOMETRIC:
        pytest.skip("requires torch_geometric")
    graphs = []
    for smi in TINY_SMILES:
        g = smiles_to_graph(smi)
        assert g is not None, f"SMILES '{smi}' failed to parse"
        graphs.append(g)
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# Tokenization utilities
# ---------------------------------------------------------------------------


class TestTokenization:
    def test_smiles_to_tensor_shape(self, tiny_drug_tokens):
        assert tiny_drug_tokens.shape == (len(TINY_SMILES), MAX_SMILES_LEN)

    def test_smiles_to_tensor_dtype(self, tiny_drug_tokens):
        assert tiny_drug_tokens.dtype == torch.long

    def test_smiles_truncation(self):
        long_smiles = ["C" * 200]
        t = smiles_to_tensor(long_smiles, max_len=MAX_SMILES_LEN)
        assert t.shape[1] == MAX_SMILES_LEN

    def test_smiles_padding(self):
        short = ["CC"]
        t = smiles_to_tensor(short, max_len=MAX_SMILES_LEN)
        # Positions beyond len("CC") should be 0 (padding)
        assert t[0, 2:].sum().item() == 0

    def test_seq_to_tensor_shape(self, tiny_target_tokens):
        assert tiny_target_tokens.shape == (len(TINY_SEQS), MAX_SEQ_LEN)

    def test_seq_unknown_char(self):
        t = seq_to_tensor(["Z"])  # Z not in AA_VOCAB
        assert t[0, 0].item() == 0  # mapped to padding/unknown


# ---------------------------------------------------------------------------
# DeepDTA
# ---------------------------------------------------------------------------


class TestDeepDTA:
    def test_forward_shape(self, tiny_drug_tokens, tiny_target_tokens):
        model = DeepDTA()
        model.eval()
        with torch.no_grad():
            out = model(tiny_drug_tokens, tiny_target_tokens)
        assert out.shape == (len(TINY_SMILES),), f"Expected ({len(TINY_SMILES)},), got {out.shape}"

    def test_output_is_logit(self, tiny_drug_tokens, tiny_target_tokens):
        """Output should be raw logits (not bounded to [0,1])."""
        model = DeepDTA()
        model.eval()
        with torch.no_grad():
            out = model(tiny_drug_tokens, tiny_target_tokens)
        # BCEWithLogitsLoss expects raw logits; sigmoid may exceed no specific range
        assert out.dtype == torch.float32

    def test_sigmoid_output_bounded(self, tiny_drug_tokens, tiny_target_tokens):
        """After sigmoid, output must be in (0, 1)."""
        model = DeepDTA()
        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(tiny_drug_tokens, tiny_target_tokens))
        assert (out >= 0).all() and (out <= 1).all()

    def test_gradient_flow(self, tiny_drug_tokens, tiny_target_tokens, tiny_labels):
        model = DeepDTA()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        out = model(tiny_drug_tokens, tiny_target_tokens)
        loss = criterion(out, tiny_labels)
        loss.backward()
        # Check at least one gradient is non-zero
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert any(g.abs().sum().item() > 0 for g in grads)

    def test_single_sample(self):
        """Forward pass with batch size 1."""
        model = DeepDTA()
        model.eval()
        d = smiles_to_tensor(["CCO"])
        t = seq_to_tensor(["MKTLL"])
        with torch.no_grad():
            out = model(d, t)
        assert out.shape == (1,)

    def test_training_loss_decreases(self, tiny_drug_tokens, tiny_target_tokens, tiny_labels):
        """3 gradient steps should reduce BCE loss."""
        torch.manual_seed(0)
        model = DeepDTA()
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.BCEWithLogitsLoss()

        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            out = model(tiny_drug_tokens, tiny_target_tokens)
            loss = criterion(out, tiny_labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_invalid_smiles_graceful(self):
        """Unknown chars should map to padding (0), not crash."""
        weird = ["∞∞∞"]
        t = smiles_to_tensor(weird)
        assert t[0].sum().item() == 0  # All unknown → all zero


# ---------------------------------------------------------------------------
# GraphDTA
# ---------------------------------------------------------------------------


class TestSmilesToGraph:
    @requires_pyg
    def test_valid_smiles(self):
        g = smiles_to_graph("CCO")
        assert g is not None
        assert g.x.shape[1] == NODE_FEATURE_DIM
        assert g.edge_index.shape[0] == 2

    @requires_pyg
    def test_invalid_smiles_returns_none(self):
        g = smiles_to_graph("not_a_valid_smiles_!!!")
        assert g is None

    @requires_pyg
    def test_benzene_undirected(self):
        g = smiles_to_graph("c1ccccc1")
        assert g is not None
        # 6 atoms, 6 bonds undirected = 12 directed edges
        assert g.edge_index.shape[1] == 12

    @requires_pyg
    def test_single_atom_no_edges(self):
        g = smiles_to_graph("[Na+]")
        assert g is not None
        assert g.edge_index.shape[1] == 0


@requires_pyg
class TestGraphDTA:
    def test_forward_shape(self, tiny_drug_graphs, tiny_target_tokens):
        model = GraphDTA()
        model.eval()
        with torch.no_grad():
            out = model(tiny_drug_graphs, tiny_target_tokens)
        assert out.shape == (len(TINY_SMILES),)

    def test_sigmoid_output_bounded(self, tiny_drug_graphs, tiny_target_tokens):
        model = GraphDTA()
        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(tiny_drug_graphs, tiny_target_tokens))
        assert (out >= 0).all() and (out <= 1).all()

    def test_gradient_flow(self, tiny_drug_graphs, tiny_target_tokens, tiny_labels):
        model = GraphDTA()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        out = model(tiny_drug_graphs, tiny_target_tokens)
        loss = criterion(out, tiny_labels)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert any(g.abs().sum().item() > 0 for g in grads)

    def test_training_loss_decreases(self, tiny_drug_graphs, tiny_target_tokens, tiny_labels):
        torch.manual_seed(0)
        model = GraphDTA()
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.BCEWithLogitsLoss()
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            out = model(tiny_drug_graphs, tiny_target_tokens)
            loss = criterion(out, tiny_labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


# ---------------------------------------------------------------------------
# DrugBAN
# ---------------------------------------------------------------------------


@requires_pyg
class TestDrugBAN:
    def test_forward_shape(self, tiny_drug_graphs, tiny_target_tokens):
        model = DrugBAN()
        model.eval()
        with torch.no_grad():
            out = model(tiny_drug_graphs, tiny_target_tokens)
        assert out.shape == (len(TINY_SMILES),)

    def test_sigmoid_output_bounded(self, tiny_drug_graphs, tiny_target_tokens):
        model = DrugBAN()
        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(tiny_drug_graphs, tiny_target_tokens))
        assert (out >= 0).all() and (out <= 1).all()

    def test_bcn_attention_shape(self, tiny_drug_graphs, tiny_target_tokens):
        """BCN should produce (B, gnn_hidden + target_dim) output."""
        from negbiodb.models.drugban import _BatchedBCN

        gnn_hidden = 16
        target_dim = 8
        B = len(TINY_SMILES)
        # Mock drug node features: 10 nodes total, 4+3+2+1 per graph
        drug_x = torch.randn(10, gnn_hidden)
        batch_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
        target_h = torch.randn(B, MAX_SEQ_LEN, target_dim)

        bcn = _BatchedBCN(gnn_hidden, target_dim)
        out = bcn(drug_x, batch_idx, target_h)
        assert out.shape == (B, gnn_hidden + target_dim)

    def test_gradient_flow(self, tiny_drug_graphs, tiny_target_tokens, tiny_labels):
        model = DrugBAN()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        out = model(tiny_drug_graphs, tiny_target_tokens)
        loss = criterion(out, tiny_labels)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert any(g.abs().sum().item() > 0 for g in grads)

    def test_training_loss_decreases(self, tiny_drug_graphs, tiny_target_tokens, tiny_labels):
        torch.manual_seed(0)
        model = DrugBAN()
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.BCEWithLogitsLoss()
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            out = model(tiny_drug_graphs, tiny_target_tokens)
            loss = criterion(out, tiny_labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
