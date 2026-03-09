"""DrugBAN: Bilinear Attention Network for Drug-Target Interaction.

Reference: Bai et al., 2023 (arXiv:2303.06429).
Architecture: GCN for drug graph + CNN for target + Bilinear Cross Network (BCN)
for pairwise drug-target interaction modelling.

Requires: torch-geometric (install with `pip install negbiodb[ml]`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from negbiodb.models.deepdta import (
    AA_VOCAB_SIZE,
    _CNNEncoder,
)
from negbiodb.models.graphdta import (
    HAS_TORCH_GEOMETRIC,
    NODE_FEATURE_DIM,
    _GCNDrugEncoder,
)

if HAS_TORCH_GEOMETRIC:
    from torch_geometric.data import Batch
    from torch_geometric.nn import global_max_pool


class _BCN(nn.Module):
    """Bilinear Cross Network (BCN) for drug–target interaction.

    Given drug node features D ∈ R^{Nd × d} and target residue features
    T ∈ R^{Nt × d}, computes a cross-attention output:
        A = softmax(D · W_a · T^T)          # (Nd × Nt) attention map
        drug_ctx  = mean(A^T · D, dim=0)    # (d,)  target-attended drug
        tgt_ctx   = mean(T, dim=0)          # (d,)  pooled target
    Returns concat of drug_ctx and tgt_ctx: (2d,)
    """

    def __init__(self, drug_dim: int, target_dim: int) -> None:
        super().__init__()
        self.W_a = nn.Parameter(torch.empty(drug_dim, target_dim))
        nn.init.xavier_uniform_(self.W_a)

    def forward(
        self,
        drug_nodes: torch.Tensor,    # (Nd, drug_dim)
        target_nodes: torch.Tensor,  # (Nt, target_dim)
    ) -> torch.Tensor:
        # Attention map: (Nd, Nt)
        scores = drug_nodes @ self.W_a @ target_nodes.t()
        A = torch.softmax(scores, dim=-1)
        # Target-attended drug context: (drug_dim,)
        drug_ctx = (A.t() @ drug_nodes).mean(dim=0)
        # Pooled target context: (target_dim,)
        tgt_ctx = target_nodes.mean(dim=0)
        return torch.cat([drug_ctx, tgt_ctx], dim=0)  # (drug_dim + target_dim,)


class _BatchedBCN(nn.Module):
    """BCN applied to a batch using packed drug node representations."""

    def __init__(self, drug_dim: int, target_dim: int) -> None:
        super().__init__()
        self.bcn = _BCN(drug_dim, target_dim)

    def forward(
        self,
        drug_x: torch.Tensor,     # (total_nodes, drug_dim) — all graphs packed
        batch_idx: torch.Tensor,  # (total_nodes,) — graph index per node
        target_h: torch.Tensor,   # (B, Nt, target_dim) — packed target features
    ) -> torch.Tensor:
        outputs = []
        B = target_h.size(0)
        for i in range(B):
            mask = batch_idx == i
            d_nodes = drug_x[mask]           # (Nd_i, drug_dim)
            t_nodes = target_h[i]            # (Nt, target_dim)
            outputs.append(self.bcn(d_nodes, t_nodes))
        return torch.stack(outputs, dim=0)   # (B, drug_dim + target_dim)


class _CNNEncoder2D(nn.Module):
    """1D-CNN encoder that returns all-position features (not pooled).

    Returns (B, T, C) so BCN can use per-residue features.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: tuple[int, int, int] = (64, 64, 64),
        kernel_sizes: tuple[int, int, int] = (4, 8, 12),
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        layers: list[nn.Module] = []
        in_ch = embed_dim
        for n_filters, k in zip(num_filters, kernel_sizes):
            layers += [
                nn.Conv1d(in_ch, n_filters, kernel_size=k, padding=k // 2),
                nn.ReLU(),
            ]
            in_ch = n_filters
        self.conv = nn.Sequential(*layers)
        self.out_dim = num_filters[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) → (B, L, D) → (B, D, L) → conv → (B, C, L) → (B, L, C)
        h = self.embed(x).permute(0, 2, 1)
        h = self.conv(h)
        return h.permute(0, 2, 1)  # (B, L, C)


class DrugBAN(nn.Module):
    """DrugBAN model for binary DTI prediction.

    Args:
        gnn_hidden: Hidden dimension for GCN drug encoder.
        target_embed_dim: Embedding dimension for AA characters.
        target_filters: Filter counts for target CNN.
        target_kernels: Kernel sizes for target CNN.
        fc_dims: Fully-connected layer sizes after BCN.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        gnn_hidden: int = 64,
        target_embed_dim: int = 128,
        target_filters: tuple[int, int, int] = (64, 64, 64),
        target_kernels: tuple[int, int, int] = (4, 8, 12),
        fc_dims: tuple[int, int] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise RuntimeError("torch_geometric required for DrugBAN.")
        self.drug_encoder = _GCNDrugEncoder(NODE_FEATURE_DIM, gnn_hidden)
        self.target_encoder = _CNNEncoder2D(
            AA_VOCAB_SIZE, target_embed_dim, target_filters, target_kernels
        )
        self.bcn = _BatchedBCN(gnn_hidden, target_filters[-1])

        # Also use pooled drug representation as residual
        concat_dim = gnn_hidden + target_filters[-1]

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
            drug_graph: PyG Batch (x, edge_index, batch).
            target_tokens: (B, MAX_SEQ_LEN) integer token ids.

        Returns:
            (B,) raw logits.
        """
        # Drug: per-node GCN features (not pooled) for BCN
        drug_x = drug_graph.x
        edge_index = drug_graph.edge_index
        batch_idx = drug_graph.batch

        h = F.relu(self.drug_encoder.conv1(drug_x, edge_index))
        h = F.relu(self.drug_encoder.conv2(h, edge_index))
        drug_node_feats = F.relu(self.drug_encoder.conv3(h, edge_index))  # (total_nodes, gnn_hidden)

        # Target: per-residue CNN features for BCN
        target_node_feats = self.target_encoder(target_tokens)  # (B, L, C)

        # BCN interaction
        interaction = self.bcn(drug_node_feats, batch_idx, target_node_feats)  # (B, gnn_hidden+C)

        return self.fc(interaction).squeeze(-1)  # (B,)
