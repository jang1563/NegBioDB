"""Loss functions for Negative-JEPA.

Two regularization options to ablate (research/24_negative_jepa_design.md §3.5):
  Option A — SIGReg  (LeJEPA, arXiv:2511.08544): provably optimal, no EMA required
  Option B — VICReg  (V-JEPA style, empirically validated in EchoJEPA, GeneJepa)

Combined loss:
  L = SmoothL1(predicted, target) + λ * reg(context_repr)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from negbiojepa.config import JEPAConfig


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (LeJEPA, arXiv:2511.08544).

    Constrains the embedding distribution to an isotropic Gaussian — this is the
    provably optimal distribution for minimizing downstream prediction risk.

    Implementation: random projection (sketch) of the batch covariance, then MSE
    against the identity matrix. ~50 lines, no EMA, no stop-gradient needed.

    When used as the regularizer, do NOT detach the target representations —
    gradients must flow through both encoder branches (LeJEPA requirement).
    """

    def __init__(self, lmbda: float = 1.0, sketch_dim: int = 64) -> None:
        super().__init__()
        self.lmbda = lmbda
        self.sketch_dim = sketch_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, N, D) latent representations from the context encoder.

        Returns:
            Scalar regularization loss.
        """
        z_flat = z.reshape(-1, z.shape[-1])          # (B*N, D)
        z_centered = z_flat - z_flat.mean(dim=0)

        D = z_flat.shape[-1]
        # Random projection for scalability (avoids O(D²) covariance matrix)
        proj = torch.randn(D, self.sketch_dim, device=z.device) / math.sqrt(self.sketch_dim)
        z_proj = z_centered @ proj                    # (B*N, sketch_dim)

        # Empirical covariance of projected embeddings
        cov = (z_proj.T @ z_proj) / (z_proj.shape[0] - 1)
        target = torch.eye(self.sketch_dim, device=z.device)

        return self.lmbda * F.mse_loss(cov, target)


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance Regularization (V-JEPA / SimSiam style).

    Used by EchoJEPA (arXiv:2602.02603) and GeneJepa (bioRxiv:2025.10.14.682378) —
    empirically validated on biological data.

    Note: We only apply Variance and Covariance terms (not the Invariance term),
    because the prediction loss already provides the invariance objective.

    When used as the regularizer, the target representations MUST be detached
    (use_ema=True path in NegJEPALoss) to prevent gradient from flowing into the
    EMA target encoder.
    """

    def __init__(
        self,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        var_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.var_threshold = var_threshold

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, N, D) latent representations from the context encoder.

        Returns:
            Scalar regularization loss.
        """
        z_flat = z.reshape(-1, z.shape[-1])  # (B*N, D)

        # Variance term: hinge loss encouraging std > threshold per dimension
        std = z_flat.std(dim=0)
        var_loss = F.relu(self.var_threshold - std).mean()

        # Covariance term: off-diagonal covariance should be zero
        z_centered = z_flat - z_flat.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (z_centered.shape[0] - 1)
        # Zero out diagonal, penalize off-diagonal elements
        D = z_flat.shape[-1]
        cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D

        return self.lambda_var * var_loss + self.lambda_cov * cov_loss


class NegJEPALoss(nn.Module):
    """Combined Negative-JEPA training loss.

    L = SmoothL1(predicted, target) + reg(context_repr)

    The `use_ema` flag controls gradient flow through the target representations:
    - use_ema=False (Option A / SIGReg): target is live encoder output → no detach
      → gradients flow through both branches (LeJEPA requirement)
    - use_ema=True  (Option B / VICReg): target is EMA-frozen encoder output → detach
      → prevents gradients reaching the frozen target encoder
    """

    def __init__(self, cfg: JEPAConfig) -> None:
        super().__init__()
        self.prediction_loss = nn.SmoothL1Loss()
        if cfg.reg_type == "sigreg":
            self.reg_loss: nn.Module = SIGReg(
                lmbda=cfg.sigreg_lambda,
                sketch_dim=cfg.sigreg_sketch_dim,
            )
        elif cfg.reg_type == "vicreg":
            self.reg_loss = VICRegLoss(
                lambda_var=cfg.vicreg_lambda_var,
                lambda_cov=cfg.vicreg_lambda_cov,
            )
        else:
            raise ValueError(f"Unknown reg_type: {cfg.reg_type!r}")

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        context_repr: torch.Tensor,
        use_ema: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            predicted:     (B, K, D) predictor output at K masked latent positions.
            target:        (B, K, D) target encoder representations at same positions.
            context_repr:  (B, n_latents, D) full context encoder output (for reg loss).
            use_ema:       If True, detach target before prediction loss (EMA path).
                           If False, no detach (SIGReg/LeJEPA path).

        Returns:
            (total_loss, metrics_dict) where metrics_dict has keys 'pred' and 'reg'.
        """
        # use_ema=True (Option B): target comes from stopped EMA encoder → detach
        # use_ema=False (Option A): target is part of the live graph → do NOT detach
        target_for_loss = target.detach() if use_ema else target

        pred_loss = self.prediction_loss(predicted, target_for_loss)
        reg_loss = self.reg_loss(context_repr)
        total = pred_loss + reg_loss

        return total, {"pred": pred_loss.item(), "reg": reg_loss.item()}


def check_collapse(
    z: torch.Tensor,
    min_rank_ratio: float = 0.1,
    min_std: float = 0.01,
) -> tuple[bool, str]:
    """Detect representation collapse during pretraining.

    Two collapse modes:
    1. Point collapse: all representations converge to the same point (std → 0)
    2. Dimensional collapse: representations span a low-rank subspace (rank → 0)

    Returns:
        (collapsed: bool, message: str)
    """
    z_flat = z.detach().reshape(-1, z.shape[-1])

    # Point collapse: mean standard deviation across dimensions
    per_dim_std = z_flat.std(dim=0)
    if per_dim_std.mean().item() < min_std:
        return True, f"Point collapse: mean std = {per_dim_std.mean().item():.6f}"

    # Dimensional collapse: effective rank of the representation
    # torch.svd deprecated since PyTorch 2.0; use torch.linalg.svd instead
    try:
        _, S, _ = torch.linalg.svd(z_flat - z_flat.mean(dim=0), full_matrices=False)
        effective_rank = int((S > S[0] * 0.01).sum().item())
        rank_ratio = effective_rank / z_flat.shape[-1]
    except torch._C._LinAlgError:
        # SVD can fail on ill-conditioned matrices (early training); skip rank check
        rank_ratio = 1.0
        effective_rank = z_flat.shape[-1]

    if rank_ratio < min_rank_ratio:
        return True, f"Dimensional collapse: rank ratio = {rank_ratio:.3f} ({effective_rank}/{z_flat.shape[-1]})"

    return False, f"OK (std={per_dim_std.mean().item():.4f}, rank_ratio={rank_ratio:.3f})"
