"""SiameseCNN: Shared-weight CNN encoder for PPI binary prediction.

Simple sequence-only baseline. Shared encoder ensures f(A,B) = f(B,A)
symmetry appropriate for PPI (both entities are proteins).

Architecture:
  shared_encoder(seq) → emb → Conv1D×3 → AdaptiveMaxPool
  [enc(seq1) + enc(seq2), |enc(seq1) - enc(seq2)|] → FC → logit
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Reuse AA vocabulary from DTI DeepDTA
from negbiodb.models.deepdta import AA_VOCAB_SIZE, _AA_LUT

MAX_SEQ_LEN = 1000  # covers 95%+ of human proteins


def seq_to_tensor(seqs: list[str], max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """Encode amino acid sequences to integer tensor (batch × max_len)."""
    result = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, seq in enumerate(seqs):
        s = seq[:max_len]
        codes = np.frombuffer(s.encode("ascii", errors="replace"), dtype=np.uint8)
        result[i, : len(codes)] = _AA_LUT[codes]
    return torch.frombuffer(result.tobytes(), dtype=torch.int64).reshape(
        len(seqs), max_len
    ).clone()


class _ProteinEncoder(nn.Module):
    """1D-CNN encoder for protein sequences."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_filters: tuple[int, int, int] = (64, 96, 128),
        kernel_sizes: tuple[int, int, int] = (4, 8, 12),
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(AA_VOCAB_SIZE, embed_dim, padding_idx=0)
        layers: list[nn.Module] = []
        in_ch = embed_dim
        for n_filters, k in zip(num_filters, kernel_sizes):
            layers += [nn.Conv1d(in_ch, n_filters, kernel_size=k), nn.ReLU()]
            in_ch = n_filters
        layers.append(nn.AdaptiveMaxPool1d(1))
        self.conv = nn.Sequential(*layers)
        self.out_dim = num_filters[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x).permute(0, 2, 1)  # (B, D, L)
        h = self.conv(h)                     # (B, C, 1)
        return h.squeeze(-1)                 # (B, C)


class SiameseCNN(nn.Module):
    """Siamese CNN for PPI binary prediction.

    Args:
        embed_dim: Embedding dimension for amino acids.
        num_filters: Filter counts for 3 conv layers.
        kernel_sizes: Kernel sizes for 3 conv layers.
        fc_dims: FC layer sizes after concatenation.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_filters: tuple[int, int, int] = (64, 96, 128),
        kernel_sizes: tuple[int, int, int] = (4, 8, 12),
        fc_dims: tuple[int, int] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Shared encoder (weight tying → symmetry)
        self.encoder = _ProteinEncoder(embed_dim, num_filters, kernel_sizes)

        enc_dim = self.encoder.out_dim
        # Symmetric aggregation: [enc1 + enc2, |enc1 - enc2|]
        concat_dim = enc_dim * 2

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
            seq1_tokens: (B, MAX_SEQ_LEN) integer token ids for protein 1.
            seq2_tokens: (B, MAX_SEQ_LEN) integer token ids for protein 2.

        Returns:
            (B,) raw logits.
        """
        e1 = self.encoder(seq1_tokens)  # (B, C)
        e2 = self.encoder(seq2_tokens)  # (B, C)
        h = torch.cat([e1 + e2, torch.abs(e1 - e2)], dim=1)  # (B, 2C)
        return self.fc(h).squeeze(-1)                          # (B,)
