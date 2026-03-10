"""DeepDTA: Drug-Target Affinity Prediction via CNNs.

Reference: Öztürk et al., 2018 (arXiv:1801.10193).
Architecture: Dual 1D-CNN encoders for SMILES and protein sequence.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Vocabulary for SMILES characters (75 unique + 1 padding index 0).
# Note: Öztürk et al. 2018 used 64 chars; we use 75 to cover a broader chemical space.
SMILES_CHARS = (
    "#%()+-./0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]abcdefghijklmnopqrstuvwxyz"
)
SMILES_VOCAB: dict[str, int] = {c: i + 1 for i, c in enumerate(SMILES_CHARS)}
SMILES_VOCAB_SIZE = len(SMILES_VOCAB) + 1  # +1 for padding index 0

# Vocabulary for amino acid single-letter codes (20 standard + X = 21, + 1 padding index 0)
AA_CHARS = "ACDEFGHIKLMNPQRSTVWXY"
AA_VOCAB: dict[str, int] = {c: i + 1 for i, c in enumerate(AA_CHARS)}
AA_VOCAB_SIZE = len(AA_VOCAB) + 1  # +1 for padding index 0

# Max sequence lengths (from original DeepDTA Davis benchmark)
MAX_SMILES_LEN = 85
MAX_SEQ_LEN = 1200

# ASCII lookup tables for fast vectorised tokenisation (size 256 covers all byte values).
# Non-vocab characters map to 0 (padding index) by default.
_SMILES_LUT: np.ndarray = np.zeros(256, dtype=np.int64)
for _c, _idx in SMILES_VOCAB.items():
    _SMILES_LUT[ord(_c)] = _idx

_AA_LUT: np.ndarray = np.zeros(256, dtype=np.int64)
for _c, _idx in AA_VOCAB.items():
    _AA_LUT[ord(_c)] = _idx


def smiles_to_tensor(smiles: list[str], max_len: int = MAX_SMILES_LEN) -> torch.Tensor:
    """Encode SMILES strings to integer tensor (batch × max_len).

    Uses numpy vectorised indexing (~50x faster than element-wise Python loops).
    Non-ASCII or unknown characters map to 0 (padding).
    """
    result = np.zeros((len(smiles), max_len), dtype=np.int64)
    for i, smi in enumerate(smiles):
        s = smi[:max_len]
        codes = np.frombuffer(s.encode("ascii", errors="replace"), dtype=np.uint8)
        result[i, : len(codes)] = _SMILES_LUT[codes]
    # torch.frombuffer bypasses the broken torch-numpy bridge (torch 2.2.2 + NumPy 2.x).
    return torch.frombuffer(result.tobytes(), dtype=torch.int64).reshape(len(smiles), max_len).clone()


def seq_to_tensor(seqs: list[str], max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """Encode amino acid sequences to integer tensor (batch × max_len).

    Uses numpy vectorised indexing (~50x faster than element-wise Python loops).
    Non-ASCII or unknown amino acid characters map to 0 (padding).
    """
    result = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, seq in enumerate(seqs):
        s = seq[:max_len]
        codes = np.frombuffer(s.encode("ascii", errors="replace"), dtype=np.uint8)
        result[i, : len(codes)] = _AA_LUT[codes]
    # torch.frombuffer bypasses the broken torch-numpy bridge (torch 2.2.2 + NumPy 2.x).
    return torch.frombuffer(result.tobytes(), dtype=torch.int64).reshape(len(seqs), max_len).clone()


class _CNNEncoder(nn.Module):
    """1D-CNN encoder for sequential inputs (SMILES or protein sequence)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: tuple[int, int, int] = (32, 64, 96),
        kernel_sizes: tuple[int, int, int] = (4, 6, 8),
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        layers: list[nn.Module] = []
        in_ch = embed_dim
        for n_filters, k in zip(num_filters, kernel_sizes):
            layers += [nn.Conv1d(in_ch, n_filters, kernel_size=k), nn.ReLU()]
            in_ch = n_filters
        layers.append(nn.AdaptiveMaxPool1d(1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) → embed → (B, L, D) → permute → (B, D, L)
        h = self.embed(x).permute(0, 2, 1)
        h = self.conv(h)          # (B, C, 1)
        return h.squeeze(-1)      # (B, C)


class DeepDTA(nn.Module):
    """DeepDTA model for binary DTI prediction.

    Args:
        drug_embed_dim: Embedding dimension for SMILES characters.
        target_embed_dim: Embedding dimension for amino acid characters.
        drug_filters: Tuple of (n_filters_1, n_filters_2, n_filters_3) for drug CNN.
        drug_kernels: Tuple of kernel sizes for drug CNN.
        target_filters: Tuple of filter counts for target CNN.
        target_kernels: Tuple of kernel sizes for target CNN.
        fc_dims: Sizes of fully-connected layers after concat.
        dropout: Dropout rate in FC layers.
    """

    def __init__(
        self,
        drug_embed_dim: int = 128,
        target_embed_dim: int = 128,
        drug_filters: tuple[int, int, int] = (32, 64, 96),
        drug_kernels: tuple[int, int, int] = (4, 6, 8),
        target_filters: tuple[int, int, int] = (32, 64, 96),
        target_kernels: tuple[int, int, int] = (4, 8, 12),
        fc_dims: tuple[int, int, int] = (1024, 1024, 512),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_encoder = _CNNEncoder(
            SMILES_VOCAB_SIZE, drug_embed_dim, drug_filters, drug_kernels
        )
        self.target_encoder = _CNNEncoder(
            AA_VOCAB_SIZE, target_embed_dim, target_filters, target_kernels
        )

        drug_out_dim = drug_filters[-1]
        target_out_dim = target_filters[-1]
        concat_dim = drug_out_dim + target_out_dim

        fc_layers: list[nn.Module] = []
        in_dim = concat_dim
        for out_dim in fc_dims:
            fc_layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        fc_layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(
        self, drug_tokens: torch.Tensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            drug_tokens: (B, MAX_SMILES_LEN) integer token ids.
            target_tokens: (B, MAX_SEQ_LEN) integer token ids.

        Returns:
            (B,) raw logits (before sigmoid; use BCEWithLogitsLoss for training).
        """
        d = self.drug_encoder(drug_tokens)      # (B, 96)
        t = self.target_encoder(target_tokens)  # (B, 96)
        h = torch.cat([d, t], dim=1)            # (B, 192)
        return self.fc(h).squeeze(-1)           # (B,)
