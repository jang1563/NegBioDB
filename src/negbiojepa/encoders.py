"""Negative-JEPA encoders: TabularEncoder, MolGraphEncoder, SequenceEncoder,
PerceiverFusion, and UnifiedEncoder.

Architecture references:
  - TabularEncoder:   T-JEPA (ICLR 2025, arXiv:2410.05016) — regularization tokens
  - MolGraphEncoder:  Graph-JEPA (TMLR, arXiv:2309.16014) + _GCNDrugEncoder from graphdta.py
  - SequenceEncoder:  _ResidueEncoder from pipr.py + ESM2 projection for VP domain
  - PerceiverFusion:  GeneJepa (bioRxiv:2025.10.14.682378) + pre-norm LayerNorm
  - UnifiedEncoder:   Routes domain-specific modalities through sub-encoders → PerceiverFusion

All sub-encoders project to shared embed_dim=D (default 256).
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from negbiojepa.config import JEPAConfig

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GCNConv, GlobalAttention

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    warnings.warn(
        "torch_geometric not found. MolGraphEncoder requires `pip install negbiodb[ml]`.",
        stacklevel=1,
    )

# Node feature dimension: must match graphdta.NODE_FEATURE_DIM = 82
# (44+1 atom types + (11+1)×3 degree/H/valence + 1 aromaticity)
_NODE_FEATURE_DIM = 82

# ESM2-650M embedding dimension (from negbiodb_vp/models/esm2_vp.py)
_ESM2_DIM = 1280


# ─── TabularEncoder ───────────────────────────────────────────────────────────

class TabularEncoder(nn.Module):
    """T-JEPA-style encoder for heterogeneous tabular features.

    Each feature value is independently projected to a D-dimensional token
    (feature tokenizer), positional embeddings are added, and masked positions
    are replaced with learnable regularization tokens before a Transformer.

    Key T-JEPA design choices:
    - Feature tokenizer: each scalar feature → D-dim token (not concatenated)
    - Regularization tokens: replace masked positions (not zeros or learnable MAE tokens)
    - Augmentation-free: masking is the only source of stochasticity
    """

    def __init__(
        self,
        max_features: int = 300,
        embed_dim: int = 256,
        depth: int = 2,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.max_features = max_features
        self.embed_dim = embed_dim

        # Project each scalar feature to D-dim token
        self.feature_tokenizer = nn.Linear(1, embed_dim)

        # Positional embedding for feature indices (not sequence position)
        self.position_embed = nn.Embedding(max_features, embed_dim)

        # Regularization tokens: one per feature position (T-JEPA §3.2)
        self.reg_tokens = nn.Parameter(torch.randn(max_features, embed_dim) * 0.02)

        # Transformer encoder (Pre-LN via batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, F) float32 tabular features, zero-padded to max_features.
            mask: (B, F) bool, True = position is masked (replaced by reg_token).

        Returns:
            (B, F, D) per-feature representations.
        """
        B, F = x.shape
        assert F <= self.max_features, f"F={F} exceeds max_features={self.max_features}"

        # Feature tokenization: (B, F, 1) → (B, F, D)
        tokens = self.feature_tokenizer(x.unsqueeze(-1))

        # Add positional embeddings for feature indices
        tokens = tokens + self.position_embed.weight[:F].unsqueeze(0)  # broadcast over B

        # Replace masked positions with learnable regularization tokens (T-JEPA)
        if mask is not None:
            # Must iterate per-sample: each sample has an independent mask pattern.
            # reg_tokens shape: (max_features, D); index by feature position.
            reg = self.reg_tokens[:F]  # (F, D)
            for b in range(B):
                m = mask[b]            # (F,) bool
                if m.any():
                    tokens[b, m] = reg[m]

        return self.transformer(tokens)  # (B, F, D)


# ─── MolGraphEncoder ──────────────────────────────────────────────────────────

class MolGraphEncoder(nn.Module):
    """3-layer GCN encoder for molecular graphs with optional subgraph masking.

    Adapted from _GCNDrugEncoder in negbiodb/models/graphdta.py.
    Key differences:
    - GlobalAttention pooling (differentiable, not global_max_pool)
    - Subgraph masking: masked nodes are zeroed before message passing
    - Projects to shared embed_dim D
    """

    def __init__(
        self,
        in_features: int = _NODE_FEATURE_DIM,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("MolGraphEncoder requires torch_geometric. "
                              "Install with: pip install negbiodb[ml]")
        super().__init__()
        # in_features=82 matches NODE_FEATURE_DIM in negbiodb/models/graphdta.py
        dims = [in_features] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(n_layers)
        ])
        # GlobalAttention: differentiable learned pooling
        self.attention_pool = GlobalAttention(nn.Linear(hidden_dim, 1))
        self.project = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        data: "Batch",
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            data:      PyG Batch (x: (N, 82), edge_index: (2, E), batch: (N,))
            node_mask: (N,) bool — True = masked nodes (zeroed before GCN)

        Returns:
            (B, D) graph-level embeddings.
        """
        x = data.x.float()

        # Subgraph masking: zero out masked node features (Graph-JEPA style)
        if node_mask is not None:
            x = x.clone()
            x[node_mask] = 0.0

        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))

        graph_emb = self.attention_pool(x, data.batch)  # (B, hidden_dim)
        return self.project(graph_emb)                  # (B, D)


# ─── SequenceEncoder ──────────────────────────────────────────────────────────

class SequenceEncoder(nn.Module):
    """CNN encoder for amino acid / SMILES sequences.

    Base architecture mirrors _ResidueEncoder from negbiodb_ppi/models/pipr.py:
    same-padding Conv1d layers that preserve sequence length.

    Two input paths:
    1. Token IDs (seq): Embedding → Conv1d × n_layers → mean-pool → Linear → (B, D)
    2. Pre-computed ESM2 embeddings (esm2_emb): Linear(1280, D) → (B, D)
       Used for VP domain where ESM2-650M embeddings are precomputed on HPC.
    """

    def __init__(
        self,
        vocab_size: int = 22,  # AA_VOCAB_SIZE from deepdta.py (21 AA + padding)
        max_len: int = 1200,
        embed_dim: int = 256,
        n_filters: int = 128,
        kernel_size: int = 7,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Same-padding CNN layers (mirrors _ResidueEncoder)
        layers: list[nn.Module] = []
        in_ch = embed_dim
        for _ in range(n_layers):
            pad = kernel_size // 2
            layers += [
                nn.Conv1d(in_ch, n_filters, kernel_size=kernel_size, padding=pad),
                nn.ReLU(),
            ]
            in_ch = n_filters
        self.conv = nn.Sequential(*layers)
        self.project_seq = nn.Linear(n_filters, embed_dim)

        # ESM2 fallback path (VP domain)
        self.project_esm2 = nn.Linear(_ESM2_DIM, embed_dim)

    def forward(
        self,
        seq: Optional[torch.Tensor] = None,
        esm2_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            seq:      (B, L) int64 token ids, or None.
            esm2_emb: (B, 1280) float32 pre-computed embeddings, or None.

        Returns:
            (B, D) sequence embedding.

        Exactly one of seq or esm2_emb must be provided.
        """
        if esm2_emb is not None:
            return self.project_esm2(esm2_emb)  # (B, D)

        if seq is None:
            raise ValueError("SequenceEncoder requires either seq or esm2_emb")

        h = self.embed(seq)                # (B, L, E)
        h = h.permute(0, 2, 1)            # (B, E, L)
        h = self.conv(h)                   # (B, n_filters, L)
        h = h.mean(dim=2)                  # (B, n_filters) — mean-pool over length
        return self.project_seq(h)         # (B, D)


# ─── PerceiverFusion ──────────────────────────────────────────────────────────

class PerceiverFusion(nn.Module):
    """Perceiver-style cross-attention fusion for multi-modal biological inputs.

    Design (from GeneJepa, bioRxiv:2025.10.14.682378):
    - Fixed number of latent queries regardless of number/length of input modalities
    - Cross-attention: queries attend to all available modality tokens
    - Self-attention: latent queries refine their representation
    - Domain token: injected into KV stream to provide domain-specific bias
    - Pre-norm pattern (LayerNorm before attention, not after — more stable)

    Output: (B, n_latents, D) — same shape regardless of which modalities are present.
    This is the representation that JEPAPredictor and the classification head consume.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_latents: int = 16,
        n_domains: int = 8,
        depth: int = 2,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.embed_dim = embed_dim

        # Learnable latent query bank
        self.latent_queries = nn.Parameter(torch.randn(n_latents, embed_dim) * 0.02)

        # Domain-specific token injected into KV stream
        self.domain_tokens = nn.Embedding(n_domains, embed_dim)

        # Alternating cross-attention + self-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.0)
            for _ in range(depth)
        ])
        # Pre-norm for cross-attention queries (standard Perceiver pattern)
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(depth)
        ])
        # Self-attention on latent queries (TransformerEncoderLayer already has LN)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=4 * embed_dim,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])

    def forward(
        self,
        modality_tokens: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            modality_tokens: (B, M, D) concatenated sub-encoder outputs.
            domain_id:       (B,) int64 domain index.

        Returns:
            (B, n_latents, D) fixed-size latent representation.
        """
        B = modality_tokens.shape[0]

        # Prepend domain token to KV stream
        domain_tok = self.domain_tokens(domain_id).unsqueeze(1)  # (B, 1, D)
        kv = torch.cat([domain_tok, modality_tokens], dim=1)     # (B, M+1, D)

        # Expand latent queries to batch
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # (B, n_latents, D)

        for cross_attn, norm, self_attn in zip(
            self.cross_attn_layers, self.cross_attn_norms, self.self_attn_layers
        ):
            # Pre-norm cross-attention: queries attend to all modality tokens
            queries = queries + cross_attn(norm(queries), kv, kv)[0]
            # Self-attention refines latent queries (LayerNorm inside TransformerEncoderLayer)
            queries = self_attn(queries)

        return queries  # (B, n_latents, D)


# ─── UnifiedEncoder ───────────────────────────────────────────────────────────

class UnifiedEncoder(nn.Module):
    """Multi-modal encoder for all 8 NegBioDB domains.

    Routes each batch through the appropriate sub-encoders based on available
    modality keys, then fuses via PerceiverFusion to produce a fixed-size
    (B, n_latents, D) representation suitable for JEPA prediction.

    Batch dict keys:
      tabular_A  (required): (B, F) float32
      tabular_B  (optional): (B, F) float32
      tab_mask_A (optional): (B, F) bool
      tab_mask_B (optional): (B, F) bool
      graph_A    (optional): PyG Batch — DTI drug, DC drug_A, CP compound
      graph_B    (optional): PyG Batch — DC drug_B only
      node_mask_A (optional): (N_A,) bool
      node_mask_B (optional): (N_B,) bool
      seq_A      (optional): (B, L) int64
      seq_B      (optional): (B, L) int64
      esm2_A     (optional): (B, 1280) float32 — VP domain
      domain_id  (required): (B,) int64
    """

    def __init__(self, cfg: JEPAConfig) -> None:
        super().__init__()
        D = cfg.embed_dim

        self.tabular_enc = TabularEncoder(
            max_features=cfg.tabular_max_features,
            embed_dim=D,
            depth=cfg.tabular_depth,
            n_heads=cfg.tabular_n_heads,
        )
        if HAS_TORCH_GEOMETRIC:
            self.mol_graph_enc: Optional[MolGraphEncoder] = MolGraphEncoder(
                in_features=_NODE_FEATURE_DIM,
                hidden_dim=128,
                embed_dim=D,
                n_layers=3,
            )
        else:
            self.mol_graph_enc = None

        self.sequence_enc = SequenceEncoder(embed_dim=D)

        self.fusion = PerceiverFusion(
            embed_dim=D,
            n_latents=cfg.perceiver_n_latents,
            n_domains=cfg.n_domains,
            depth=cfg.perceiver_depth,
            n_heads=cfg.tabular_n_heads,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with keys as described in class docstring.

        Returns:
            (B, n_latents, D) latent representation.
        """
        B = batch["tabular_A"].shape[0]
        tokens: list[torch.Tensor] = []

        # ── Tabular (always available) ───────────────────────────────────────
        tab_A = self.tabular_enc(batch["tabular_A"], batch.get("tab_mask_A"))
        # tab_A: (B, F, D) → mean over F → (B, D) → unsqueeze → (B, 1, D)
        tokens.append(tab_A.mean(dim=1, keepdim=True))

        if "tabular_B" in batch:
            tab_B = self.tabular_enc(batch["tabular_B"], batch.get("tab_mask_B"))
            tokens.append(tab_B.mean(dim=1, keepdim=True))

        # ── Molecular graphs (DTI, DC, CP) ───────────────────────────────────
        if "graph_A" in batch and self.mol_graph_enc is not None:
            g_A_valid = self.mol_graph_enc(batch["graph_A"], batch.get("node_mask_A"))
            # g_A_valid: (n_valid, D) — scatter back to (B, D)
            g_A = torch.zeros(B, g_A_valid.shape[-1], device=g_A_valid.device)
            if "has_graph_A" in batch:
                g_A[batch["has_graph_A"]] = g_A_valid
            else:
                g_A = g_A_valid  # all samples have graphs
            tokens.append(g_A.unsqueeze(1))  # (B, 1, D)

        if "graph_B" in batch and self.mol_graph_enc is not None:
            g_B_valid = self.mol_graph_enc(batch["graph_B"], batch.get("node_mask_B"))
            g_B = torch.zeros(B, g_B_valid.shape[-1], device=g_B_valid.device)
            if "has_graph_B" in batch:
                g_B[batch["has_graph_B"]] = g_B_valid
            else:
                g_B = g_B_valid
            tokens.append(g_B.unsqueeze(1))

        # ── Sequences (DTI, PPI, VP) ──────────────────────────────────────────
        if "seq_A" in batch or "esm2_A" in batch:
            s_A = self.sequence_enc(
                seq=batch.get("seq_A"),
                esm2_emb=batch.get("esm2_A"),
            )
            tokens.append(s_A.unsqueeze(1))  # (B, D) → (B, 1, D)

        if "seq_B" in batch:
            s_B = self.sequence_enc(seq=batch["seq_B"])
            tokens.append(s_B.unsqueeze(1))

        # ── Fuse ─────────────────────────────────────────────────────────────
        all_tokens = torch.cat(tokens, dim=1)          # (B, M, D)
        return self.fusion(all_tokens, batch["domain_id"])  # (B, n_latents, D)


def build_encoder_pair(cfg: JEPAConfig) -> tuple[UnifiedEncoder, UnifiedEncoder]:
    """Create a (context_encoder, target_encoder) pair sharing the same architecture.

    For Option A (SIGReg / no EMA): returns two independent encoders — both receive
    gradients. The target encoder starts with the same weights but diverges during training.

    For Option B (VICReg + EMA): target_encoder weights are never directly optimized;
    they are updated only by EMAUpdater. Call this function and then immediately copy
    context weights to target before training starts.
    """
    context_enc = UnifiedEncoder(cfg)
    target_enc = UnifiedEncoder(cfg)
    if cfg.use_ema:
        # Initialize target with identical weights — EMA will maintain them
        target_enc.load_state_dict(context_enc.state_dict())
        for p in target_enc.parameters():
            p.requires_grad_(False)
    return context_enc, target_enc
