"""MLPFeatures: Hand-crafted feature MLP for PPI binary prediction.

Uses interpretable features rather than raw sequences:
  - AA composition (20-dim × 2 proteins)
  - Sequence length × 2
  - Network degree × 2
  - Length ratio
  - Subcellular location co-occurrence (one-hot)

Simple 3-layer MLP. Cheapest to train, most interpretable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Standard 20 amino acids
_AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {c: i for i, c in enumerate(_AA_LETTERS)}

# Known subcellular locations (top-10 + "other")
SUBCELLULAR_LOCATIONS = [
    "Nucleus", "Cytoplasm", "Membrane", "Cell membrane",
    "Mitochondrion", "Endoplasmic reticulum", "Golgi apparatus",
    "Secreted", "Cell junction", "Cytoskeleton", "other",
]
_LOC_TO_IDX = {loc: i for i, loc in enumerate(SUBCELLULAR_LOCATIONS)}
N_LOCATIONS = len(SUBCELLULAR_LOCATIONS)


def compute_aa_composition(seq: str) -> list[float]:
    """Compute 20-dim amino acid frequency vector."""
    if not seq:
        return [0.0] * 20
    counts = [0] * 20
    total = 0
    for c in seq:
        idx = _AA_TO_IDX.get(c)
        if idx is not None:
            counts[idx] += 1
            total += 1
    if total == 0:
        return [0.0] * 20
    return [c / total for c in counts]


def encode_subcellular(loc: str | None) -> list[float]:
    """Encode subcellular location as one-hot vector."""
    vec = [0.0] * N_LOCATIONS
    if loc is None:
        return vec
    idx = _LOC_TO_IDX.get(loc, _LOC_TO_IDX["other"])
    vec[idx] = 1.0
    return vec


def extract_features(
    seq1: str,
    seq2: str,
    degree1: float,
    degree2: float,
    loc1: str | None,
    loc2: str | None,
) -> list[float]:
    """Extract feature vector for a protein pair.

    Returns:
        Feature vector of length 20+20+2+2+1+11+11 = 67.
    """
    aa1 = compute_aa_composition(seq1)  # 20
    aa2 = compute_aa_composition(seq2)  # 20
    len1 = len(seq1) if seq1 else 0
    len2 = len(seq2) if seq2 else 0
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
    loc1_vec = encode_subcellular(loc1)  # 11
    loc2_vec = encode_subcellular(loc2)  # 11

    return (
        aa1 + aa2                                      # 40
        + [len1 / 1000.0, len2 / 1000.0]             # 2 (normalized)
        + [degree1 / 1000.0, degree2 / 1000.0]       # 2 (normalized)
        + [len_ratio]                                  # 1
        + loc1_vec + loc2_vec                          # 22
    )


FEATURE_DIM = 67  # 20+20+2+2+1+11+11


class MLPFeatures(nn.Module):
    """MLP classifier using hand-crafted protein pair features.

    Args:
        input_dim: Feature vector dimensionality.
        hidden_dims: Sizes of hidden layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: tuple[int, int, int] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, FEATURE_DIM) float tensor.

        Returns:
            (B,) raw logits.
        """
        return self.net(features).squeeze(-1)
