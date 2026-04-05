"""Graph Neural Network for DC domain drug combination prediction.

Architecture:
  Drug A graph → GCN(3 layers) → global_mean_pool → drug_a_embed (128)
  Drug B graph → GCN(3 layers) → global_mean_pool → drug_b_embed (128)
  Concat(drug_a_embed, drug_b_embed) → FC(256→128→64→n_classes)

Requires: torch_geometric
Inspired by DeepDDS and DDoS architectures.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GCNConv, global_mean_pool

    _PYGEOM_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    _PYGEOM_AVAILABLE = False


# Atom feature dimensions: [atomic_num(118), degree(5), charge(1), aromatic(1),
#                           num_h(5), hybridization(6)] = 136 via one-hot
_ATOM_FEATURES_DIM = 9  # simplified: atomic_num, degree, formal_charge,
# num_hs, aromatic, in_ring, hybridization, valence, mass


def mol_to_graph(smiles: str) -> Data | None:
    """Convert SMILES to PyG Data object with atom/bond features."""
    if not _PYGEOM_AVAILABLE:
        raise ImportError("torch_geometric required: pip install torch_geometric")

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumImplicitHs(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetHybridization().real,
            atom.GetTotalValence(),
            atom.GetMass() / 100.0,  # Normalize
        ]
        atom_features.append(features)

    if not atom_features:
        return None

    x = torch.tensor(atom_features, dtype=torch.float32)

    # Bond edges (bidirectional)
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


if _PYGEOM_AVAILABLE:

    class DrugEncoder(nn.Module):
        """GCN-based drug molecular graph encoder."""

        def __init__(self, in_dim: int = 9, hidden_dim: int = 128, out_dim: int = 128):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, out_dim)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
            return global_mean_pool(x, batch)

    class DrugCombGNN(nn.Module):
        """GNN model for drug combination synergy prediction."""

        def __init__(
            self,
            atom_dim: int = 9,
            drug_embed_dim: int = 128,
            n_classes: int = 2,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.drug_encoder = DrugEncoder(atom_dim, drug_embed_dim, drug_embed_dim)
            self.classifier = nn.Sequential(
                nn.Linear(drug_embed_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

        def forward(self, data_a: Batch, data_b: Batch) -> torch.Tensor:
            embed_a = self.drug_encoder(data_a.x, data_a.edge_index, data_a.batch)
            embed_b = self.drug_encoder(data_b.x, data_b.edge_index, data_b.batch)
            combined = torch.cat([embed_a, embed_b], dim=1)
            return self.classifier(combined)


def prepare_graph_pairs(
    smiles_a: list[str],
    smiles_b: list[str],
) -> tuple[list[Data], list[Data], list[bool]]:
    """Convert SMILES pairs to PyG Data lists.

    Returns:
        (graphs_a, graphs_b, valid_mask): Valid entries where both SMILES parsed.
    """
    graphs_a = []
    graphs_b = []
    valid_mask = []

    for sa, sb in zip(smiles_a, smiles_b):
        ga = mol_to_graph(sa) if sa else None
        gb = mol_to_graph(sb) if sb else None
        if ga is not None and gb is not None:
            graphs_a.append(ga)
            graphs_b.append(gb)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    return graphs_a, graphs_b, valid_mask


def train_drugcomb_gnn(
    graphs_a: list[Data],
    graphs_b: list[Data],
    y_train: np.ndarray,
    val_graphs_a: list[Data] | None = None,
    val_graphs_b: list[Data] | None = None,
    y_val: np.ndarray | None = None,
    n_classes: int = 2,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cpu",
) -> tuple:
    """Train DrugCombGNN model.

    Returns (model, history_dict).
    """
    if not _PYGEOM_AVAILABLE:
        raise ImportError("torch_geometric required: pip install torch_geometric")

    model = DrugCombGNN(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = len(graphs_a)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

    has_val = val_graphs_a is not None and y_val is not None
    if has_val:
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_train).tolist()
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = indices[start:end]

            batch_a = Batch.from_data_list([graphs_a[i] for i in batch_idx]).to(device)
            batch_b = Batch.from_data_list([graphs_b[i] for i in batch_idx]).to(device)
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            logits = model(batch_a, batch_b)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        if has_val:
            model.eval()
            with torch.no_grad():
                va = Batch.from_data_list(val_graphs_a).to(device)
                vb = Batch.from_data_list(val_graphs_b).to(device)
                val_logits = model(va, vb)
                val_loss = criterion(val_logits, y_val_t).item()
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
