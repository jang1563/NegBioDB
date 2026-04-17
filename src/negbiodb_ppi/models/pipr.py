"""PIPR: PPI Prediction via Residue-level Cross-attention.

Inspired by Chen et al., Bioinformatics 2019. Adapted for symmetric PPI:
  - Shared residue-level CNN encoder (weight sharing → f(A,B) = f(B,A))
  - Cross-attention between protein1 and protein2 residue representations
  - Attention-weighted pooling → FC → logit

Architecture:
  shared_encoder(seq) → residue embeddings (B, L, D)
  cross_attention(res1, res2) → attended vectors
  pool + concat → FC → logit
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from negbiodb.models.deepdta import AA_VOCAB_SIZE, _AA_LUT, _as_int64_tensor

MAX_SEQ_LEN = 1000


def seq_to_tensor(seqs: list[str], max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """Encode amino acid sequences to integer tensor (batch × max_len)."""
    result = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, seq in enumerate(seqs):
        s = seq[:max_len]
        codes = np.frombuffer(s.encode("ascii", errors="replace"), dtype=np.uint8)
        result[i, : len(codes)] = _AA_LUT[codes]
    return _as_int64_tensor(result)


class _ResidueEncoder(nn.Module):
    """Residue-level CNN encoder that preserves sequence length."""

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        kernel_size: int = 7,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(AA_VOCAB_SIZE, embed_dim, padding_idx=0)

        layers: list[nn.Module] = []
        in_ch = embed_dim
        for _ in range(n_layers):
            # Same-padding to preserve length
            pad = kernel_size // 2
            layers += [
                nn.Conv1d(in_ch, hidden_dim, kernel_size=kernel_size, padding=pad),
                nn.ReLU(),
            ]
            in_ch = hidden_dim
        self.conv = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) integer token ids.

        Returns:
            (B, L, D) residue-level representations.
        """
        h = self.embed(x)              # (B, L, E)
        h = h.permute(0, 2, 1)         # (B, E, L)
        h = self.conv(h)               # (B, D, L)
        return h.permute(0, 2, 1)      # (B, L, D)


class _CrossAttention(nn.Module):
    """Scaled dot-product cross-attention between two residue sequences."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(
        self,
        res1: torch.Tensor,
        res2: torch.Tensor,
        mask1: torch.Tensor | None = None,
        mask2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            res1: (B, L1, D) residue representations of protein 1.
            res2: (B, L2, D) residue representations of protein 2.
            mask1: (B, L1) bool mask (True = valid) for protein 1.
            mask2: (B, L2) bool mask (True = valid) for protein 2.

        Returns:
            (attended_1, attended_2): each (B, D) pooled vectors.
        """
        Q1 = self.query(res1)  # (B, L1, D)
        K2 = self.key(res2)    # (B, L2, D)
        V2 = self.value(res2)  # (B, L2, D)

        # Attention: res1 attends to res2
        attn_12 = torch.bmm(Q1, K2.transpose(1, 2)) / self.scale  # (B, L1, L2)
        if mask2 is not None:
            attn_12 = attn_12.masked_fill(~mask2.unsqueeze(1), float("-inf"))
        attn_12 = F.softmax(attn_12, dim=-1)
        ctx_12 = torch.bmm(attn_12, V2)  # (B, L1, D)

        # Pool ctx_12 with mask1
        if mask1 is not None:
            ctx_12 = ctx_12 * mask1.unsqueeze(-1).float()
            pooled_1 = ctx_12.sum(dim=1) / mask1.float().sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled_1 = ctx_12.mean(dim=1)  # (B, D)

        # Symmetric: res2 attends to res1
        Q2 = self.query(res2)
        K1 = self.key(res1)
        V1 = self.value(res1)

        attn_21 = torch.bmm(Q2, K1.transpose(1, 2)) / self.scale
        if mask1 is not None:
            attn_21 = attn_21.masked_fill(~mask1.unsqueeze(1), float("-inf"))
        attn_21 = F.softmax(attn_21, dim=-1)
        ctx_21 = torch.bmm(attn_21, V1)  # (B, L2, D)

        if mask2 is not None:
            ctx_21 = ctx_21 * mask2.unsqueeze(-1).float()
            pooled_2 = ctx_21.sum(dim=1) / mask2.float().sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled_2 = ctx_21.mean(dim=1)

        return pooled_1, pooled_2


class PIPR(nn.Module):
    """PIPR model for PPI binary prediction.

    Args:
        embed_dim: AA embedding dimension.
        hidden_dim: CNN hidden and attention dimension.
        kernel_size: CNN kernel size (with same-padding).
        n_conv_layers: Number of residue-level CNN layers.
        fc_dims: FC layer sizes after cross-attention pooling.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        kernel_size: int = 7,
        n_conv_layers: int = 2,
        fc_dims: tuple[int, int] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Shared encoder for symmetry
        self.encoder = _ResidueEncoder(embed_dim, hidden_dim, kernel_size, n_conv_layers)
        self.cross_attn = _CrossAttention(hidden_dim)

        # FC head: symmetric [pooled_1 + pooled_2, |pooled_1 - pooled_2|]
        concat_dim = hidden_dim * 2
        fc_layers: list[nn.Module] = []
        in_dim = concat_dim
        for out_dim in fc_dims:
            fc_layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        fc_layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(
        self, seq1_tokens: torch.Tensor, seq2_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            seq1_tokens, seq2_tokens: (B, L) integer token ids.

        Returns:
            (B,) raw logits.
        """
        # Compute padding masks (non-zero = valid)
        mask1 = seq1_tokens != 0  # (B, L)
        mask2 = seq2_tokens != 0

        res1 = self.encoder(seq1_tokens)  # (B, L, D)
        res2 = self.encoder(seq2_tokens)

        pooled_1, pooled_2 = self.cross_attn(res1, res2, mask1, mask2)
        h = torch.cat([pooled_1 + pooled_2, torch.abs(pooled_1 - pooled_2)], dim=1)

        return self.fc(h).squeeze(-1)
