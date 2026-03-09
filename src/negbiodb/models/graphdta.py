"""GraphDTA: Drug-Target Affinity Prediction via Graph Neural Networks.

Reference: Nguyen et al., 2020 (arXiv:2003.06751).
Architecture: GCN for drug molecular graph + 1D-CNN for protein sequence.

Requires: torch-geometric (install with `pip install negbiodb[ml]`).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from negbiodb.models.deepdta import (
    AA_VOCAB_SIZE,
    MAX_SEQ_LEN,
    _CNNEncoder,
    seq_to_tensor,
)

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GCNConv, global_max_pool

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    warnings.warn(
        "torch_geometric not found. GraphDTA and DrugBAN require `pip install negbiodb[ml]`.",
        stacklevel=1,
    )

# --- Atom feature constants --------------------------------------------------
# One-hot feature sizes (must match smiles_to_graph)
ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
    "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd",
    "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In",
    "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
]  # 44 atom types
ATOM_DEGREES = list(range(11))  # 0-10 → 11 values
ATOM_H_COUNTS = list(range(11))  # 0-10 → 11 values
ATOM_VALENCES = list(range(11))  # 0-10 → 11 values
# _one_hot returns len(choices)+1 (includes unknown bucket).
# Total: (44+1) + (11+1) + (11+1) + (11+1) + 1 (aromaticity) = 82
NODE_FEATURE_DIM = 82


def _one_hot(value: int | str, choices: list) -> list[int]:
    """Return one-hot vector; last element is 1 if value not in choices."""
    vec = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    vec[idx] = 1
    return vec


def smiles_to_graph(smiles: str) -> "Data | None":
    """Convert a SMILES string to a PyG Data object.

    Returns None if the SMILES is invalid.

    Node features (82-dim):
        (44+1) atom type + (11+1) degree + (11+1) H-count + (11+1) valence + 1 aromaticity
        Each _one_hot adds +1 unknown bucket, giving 82 total.
    """
    if not HAS_TORCH_GEOMETRIC:
        raise RuntimeError("torch_geometric required for GraphDTA.")
    try:
        from rdkit import Chem
    except ImportError as e:
        raise RuntimeError("rdkit required for graph construction.") from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = []
    for atom in mol.GetAtoms():
        feat = (
            _one_hot(atom.GetSymbol(), ATOM_TYPES)
            + _one_hot(atom.GetDegree(), ATOM_DEGREES)
            + _one_hot(atom.GetTotalNumHs(), ATOM_H_COUNTS)
            + _one_hot(atom.GetImplicitValence(), ATOM_VALENCES)
            + [int(atom.GetIsAromatic())]
        )
        node_features.append(feat)

    if len(node_features) == 0:
        return None  # Mol with no atoms — treat as invalid

    x = torch.tensor(node_features, dtype=torch.float)  # (N_atoms, 82)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]  # undirected

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


class _GCNDrugEncoder(nn.Module):
    """3-layer GCN encoder for molecular graphs."""

    def __init__(self, in_dim: int = NODE_FEATURE_DIM, hidden: int = 256) -> None:
        super().__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise RuntimeError("torch_geometric required.")
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        return global_max_pool(h, batch)  # (B, hidden)


class GraphDTA(nn.Module):
    """GraphDTA model for binary DTI prediction (GCN variant).

    Args:
        gnn_hidden: Hidden dimension for GCN layers.
        target_embed_dim: Embedding dimension for amino acid characters.
        target_filters: Filter counts for target 1D-CNN.
        target_kernels: Kernel sizes for target 1D-CNN.
        fc_dims: Fully-connected layer sizes after concat.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        gnn_hidden: int = 256,
        target_embed_dim: int = 128,
        target_filters: tuple[int, int, int] = (32, 64, 96),
        target_kernels: tuple[int, int, int] = (4, 8, 12),
        fc_dims: tuple[int, int] = (1024, 512),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise RuntimeError("torch_geometric required for GraphDTA.")
        self.drug_encoder = _GCNDrugEncoder(NODE_FEATURE_DIM, gnn_hidden)
        self.target_encoder = _CNNEncoder(
            AA_VOCAB_SIZE, target_embed_dim, target_filters, target_kernels
        )

        target_out_dim = target_filters[-1]
        concat_dim = gnn_hidden + target_out_dim

        fc_layers: list[nn.Module] = []
        in_dim = concat_dim
        for out_dim in fc_dims:
            fc_layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        fc_layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(
        self,
        drug_graph: "Batch",
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            drug_graph: PyG Batch of molecular graphs (x, edge_index, batch).
            target_tokens: (B, MAX_SEQ_LEN) integer token ids.

        Returns:
            (B,) raw logits.
        """
        d = self.drug_encoder(drug_graph.x, drug_graph.edge_index, drug_graph.batch)
        t = self.target_encoder(target_tokens)
        h = torch.cat([d, t], dim=1)
        return self.fc(h).squeeze(-1)
