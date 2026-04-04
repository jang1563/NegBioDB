"""VariantGNN: Gene interaction graph GCN + tabular features for VP prediction.

Architecture:
  Pre-computed STRING v12.0 gene interaction graph (combined_score > 700).
  GCN encoder: GCNConv(5→64) → GCNConv(64→32) → per-gene 32-dim embedding.
  Prediction head: Concat(gene_emb_32, tabular_56) → FC(88→128→64→1).

Genes not in STRING get zero gene_embedding (32 dims) + tabular features only.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv
except ImportError:
    torch = None
    nn = None
    GCNConv = None

GENE_FEATURE_DIM = 5  # pLI, LOEUF, missense_z, variant_count, mean_consequence_severity
GENE_EMBED_DIM = 32
TABULAR_DIM = 56


class GeneGCN(nn.Module):
    """2-layer GCN encoder on gene interaction graph."""

    def __init__(self, in_dim: int = GENE_FEATURE_DIM, hidden_dim: int = 64, out_dim: int = GENE_EMBED_DIM):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x, edge_index))
        h = self.relu(self.conv2(h, edge_index))
        return h  # (N_genes, out_dim)


class VariantGNN(nn.Module):
    """Gene graph GCN + tabular features → FC prediction head."""

    def __init__(
        self,
        gene_in_dim: int = GENE_FEATURE_DIM,
        gene_hidden: int = 64,
        gene_embed_dim: int = GENE_EMBED_DIM,
        tabular_dim: int = TABULAR_DIM,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gcn = GeneGCN(gene_in_dim, gene_hidden, gene_embed_dim)
        concat_dim = gene_embed_dim + tabular_dim  # 88

        self.fc = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

        self.gene_embed_dim = gene_embed_dim

    def forward(
        self,
        tabular: torch.Tensor,  # (B, 56)
        gene_embeddings: torch.Tensor,  # (B, 32) — pre-looked-up per variant
    ) -> torch.Tensor:
        x = torch.cat([gene_embeddings, tabular], dim=1)
        return self.fc(x)

    def precompute_gene_embeddings(
        self,
        gene_features: torch.Tensor,  # (N_genes, 5)
        edge_index: torch.Tensor,  # (2, E)
    ) -> torch.Tensor:
        """Run GCN forward pass on full gene graph. Returns (N_genes, 32)."""
        self.eval()
        with torch.no_grad():
            return self.gcn(gene_features, edge_index)


def train_variant_gnn(
    X_tab_train: np.ndarray,
    gene_idx_train: np.ndarray,
    y_train: np.ndarray,
    gene_features: np.ndarray,
    edge_index: np.ndarray,
    X_tab_val: np.ndarray | None = None,
    gene_idx_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    n_classes: int = 2,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cpu",
) -> tuple[VariantGNN, dict]:
    """Train VariantGNN model.

    Args:
        X_tab_train: Tabular features (N, 56)
        gene_idx_train: Gene indices into the graph node table (-1 for missing)
        y_train: Labels
        gene_features: Gene node features (N_genes, 5) for GCN
        edge_index: Graph edge index (2, E) for GCN
    """
    if torch is None:
        raise ImportError("torch and torch_geometric required")

    model = VariantGNN(
        tabular_dim=X_tab_train.shape[1],
        n_classes=n_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    X_tab_t = torch.tensor(X_tab_train, dtype=torch.float32).to(device)
    gene_idx_t = torch.tensor(gene_idx_train, dtype=torch.long).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    gf_t = torch.tensor(gene_features, dtype=torch.float32).to(device)
    ei_t = torch.tensor(edge_index, dtype=torch.long).to(device)

    has_val = X_tab_val is not None and y_val is not None
    if has_val:
        X_tab_vt = torch.tensor(X_tab_val, dtype=torch.float32).to(device)
        gene_idx_vt = torch.tensor(gene_idx_val, dtype=torch.long).to(device)
        y_vt = torch.tensor(y_val, dtype=torch.long).to(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    n_train = len(X_tab_t)

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_train, device=device)
        gene_emb_all = model.gcn(gf_t, ei_t)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = indices[start:end]
            batch_gene_emb = _lookup_gene_embeddings(gene_emb_all, gene_idx_t[idx])

            optimizer.zero_grad()
            logits = model(X_tab_t[idx], batch_gene_emb)
            loss = criterion(logits, y_t[idx])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        if has_val:
            model.eval()
            with torch.no_grad():
                val_gene_emb_all = model.gcn(gf_t, ei_t)
                val_gene_emb = _lookup_gene_embeddings(val_gene_emb_all, gene_idx_vt)
                val_logits = model(X_tab_vt, val_gene_emb)
                val_loss = criterion(val_logits, y_vt).item()
                val_acc = (val_logits.argmax(1) == y_vt).float().mean().item()

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            scheduler.step(val_loss)

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


def _lookup_gene_embeddings(
    gene_emb_all: torch.Tensor,
    gene_idx: torch.Tensor,
) -> torch.Tensor:
    """Gather per-example gene embeddings, using zeros for missing genes."""
    out = torch.zeros((gene_idx.shape[0], gene_emb_all.shape[1]), device=gene_emb_all.device)
    valid = gene_idx >= 0
    if valid.any():
        out[valid] = gene_emb_all[gene_idx[valid]]
    return out


def predict_variant_gnn(
    model: VariantGNN,
    X_tab: np.ndarray,
    gene_idx: np.ndarray,
    gene_features: np.ndarray,
    edge_index: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels/probabilities with a trained VariantGNN."""
    if torch is None:
        raise ImportError("torch and torch_geometric required")

    model = model.to(device)
    model.eval()

    X_tab_t = torch.tensor(X_tab, dtype=torch.float32).to(device)
    gene_idx_t = torch.tensor(gene_idx, dtype=torch.long).to(device)
    gf_t = torch.tensor(gene_features, dtype=torch.float32).to(device)
    ei_t = torch.tensor(edge_index, dtype=torch.long).to(device)

    with torch.no_grad():
        gene_emb_all = model.gcn(gf_t, ei_t)
        gene_emb = _lookup_gene_embeddings(gene_emb_all, gene_idx_t)
        logits = model(X_tab_t, gene_emb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

    return preds, probs[:, 1] if probs.shape[1] == 2 else probs
