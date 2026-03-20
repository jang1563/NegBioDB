"""CT domain ML model definitions.

Models:
  CT_MLP: MLP for tabular features (CT-M1 binary, CT-M2 multiclass)
  CT_GNN_Tab: GNN for drug graph + tabular condition/trial features

Reuses _GCNDrugEncoder and smiles_to_graph from negbiodb.models.graphdta.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from negbiodb_ct.ct_features import (
    CONDITION_DIM,
    DRUG_TAB_DIM,
    M2_TRIAL_DIM,
    TOTAL_M1_DIM,
    TOTAL_M2_DIM,
)

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GCNConv, global_max_pool

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    warnings.warn(
        "torch_geometric not found. CT_GNN_Tab requires `pip install negbiodb[ml]`.",
        stacklevel=1,
    )

# Import GCN drug encoder from DTI domain
from negbiodb.models.graphdta import NODE_FEATURE_DIM, _GCNDrugEncoder


# ---------------------------------------------------------------------------
# CT_MLP
# ---------------------------------------------------------------------------


class CT_MLP(nn.Module):
    """MLP for CT-M1 (binary) or CT-M2 (multiclass classification).

    Args:
        input_dim: Input feature dimension (1044 for M1, 1066 for M2).
        num_classes: 1 for binary (M1), 8 for multiclass (M2).
        hidden_dims: Tuple of hidden layer sizes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) feature tensor.

        Returns:
            (B,) raw logits for binary, (B, num_classes) for multiclass.
        """
        out = self.fc(x)
        if self.num_classes == 1:
            return out.squeeze(-1)  # (B,)
        return out  # (B, num_classes)


# ---------------------------------------------------------------------------
# CT_GNN_Tab
# ---------------------------------------------------------------------------


class CT_GNN_Tab(nn.Module):
    """GNN encoder for drug molecular graph + tabular condition/trial features.

    Drug: _GCNDrugEncoder (3-layer GCN, 128-dim) → 128-dim graph embedding
    Tab: FC(tab_dim → 64) → ReLU
    Concat: (128 + 64) = 192 → FC(256) → ReLU → Dropout → FC(128) → ReLU → Dropout → FC(out)

    Args:
        tab_dim: Tabular feature dimension (14 for M1, 36 for M2).
        num_classes: 1 for binary, 8 for multiclass.
        gnn_hidden: GCN hidden dimension (128, smaller than DTI's 256).
        fc_dims: FC layer sizes after concat.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        tab_dim: int,
        num_classes: int = 1,
        gnn_hidden: int = 128,
        fc_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise RuntimeError("torch_geometric required for CT_GNN_Tab.")
        self.num_classes = num_classes

        # Drug graph encoder (reuse DTI GCN, smaller hidden)
        self.drug_encoder = _GCNDrugEncoder(NODE_FEATURE_DIM, gnn_hidden)

        # Tabular encoder
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.ReLU(),
        )

        # Classification head
        concat_dim = gnn_hidden + 64
        layers: list[nn.Module] = []
        in_dim = concat_dim
        for f_dim in fc_dims:
            layers.extend([nn.Linear(in_dim, f_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = f_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(
        self,
        drug_graph: "Batch",
        tab_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            drug_graph: PyG Batch of molecular graphs (x, edge_index, batch).
            tab_features: (B, tab_dim) tabular features.

        Returns:
            (B,) raw logits for binary, (B, num_classes) for multiclass.
        """
        d = self.drug_encoder(drug_graph.x, drug_graph.edge_index, drug_graph.batch)
        t = self.tab_encoder(tab_features)
        h = torch.cat([d, t], dim=1)
        out = self.fc(h)
        if self.num_classes == 1:
            return out.squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Tab dims for GNN+Tab
GNN_TAB_DIM_M1 = DRUG_TAB_DIM + CONDITION_DIM  # 13 + 1 = 14
GNN_TAB_DIM_M2 = GNN_TAB_DIM_M1 + M2_TRIAL_DIM  # 14 + 22 = 36


def build_ct_model(
    model_name: str,
    task: str = "m1",
    num_classes: int | None = None,
    **kwargs,
) -> nn.Module:
    """Factory function to create CT models.

    Args:
        model_name: "mlp" or "gnn"
        task: "m1" or "m2"
        num_classes: override (default: 1 for M1, 8 for M2)
        **kwargs: passed to model constructor

    Returns:
        nn.Module instance
    """
    if num_classes is None:
        num_classes = 1 if task == "m1" else 8

    if model_name == "mlp":
        input_dim = TOTAL_M1_DIM if task == "m1" else TOTAL_M2_DIM
        return CT_MLP(input_dim=input_dim, num_classes=num_classes, **kwargs)
    elif model_name == "gnn":
        tab_dim = GNN_TAB_DIM_M1 if task == "m1" else GNN_TAB_DIM_M2
        return CT_GNN_Tab(tab_dim=tab_dim, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Choose 'mlp' or 'gnn'.")
