"""JEPAPredictor and EMAUpdater for Negative-JEPA.

JEPAPredictor: predicts target encoder representations from context encoder output.
EMAUpdater:    exponential moving average for Option B (VICReg + EMA) training.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from negbiojepa.config import JEPAConfig


class JEPAPredictor(nn.Module):
    """Predicts target encoder's latent representations from context encoder output.

    Intentionally smaller than the encoder (default 4 layers vs encoder 2+2) to
    prevent trivial solutions where the predictor memorizes the identity mapping.

    At masked latent positions, the predictor receives a learnable mask token plus
    positional embedding instead of the context encoder's actual output — forcing it
    to predict from surrounding (unmasked) context.

    Ref: I-JEPA (arXiv:2301.08243), V-JEPA (arXiv:2404.08471)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_latents: int = 16,
        depth: int = 4,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_latents = n_latents

        # Learnable mask token broadcast to masked positions
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional embedding for latent positions
        self.position_embed = nn.Embedding(n_latents, embed_dim)

        # Transformer blocks (Pre-LN for stability)
        self.blocks = nn.ModuleList([
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
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        context_repr: torch.Tensor,
        mask_positions: list[int],
    ) -> torch.Tensor:
        """
        Args:
            context_repr:   (B, n_latents, D) from context encoder.
            mask_positions: list of latent position indices to predict.

        Returns:
            (B, len(mask_positions), D) predicted target representations.
        """
        B = context_repr.shape[0]
        x = context_repr.clone()

        # Replace masked positions with mask_token + positional embedding
        for pos in mask_positions:
            x[:, pos] = (
                self.mask_token.squeeze(0).squeeze(0)      # (D,)
                + self.position_embed.weight[pos]           # (D,)
            )

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, mask_positions]  # (B, len(mask_positions), D)

    @classmethod
    def from_config(cls, cfg: JEPAConfig) -> "JEPAPredictor":
        return cls(
            embed_dim=cfg.embed_dim,
            n_latents=cfg.perceiver_n_latents,
            depth=cfg.predictor_depth,
            n_heads=cfg.predictor_n_heads,
        )


class EMAUpdater:
    """Exponential Moving Average for the target encoder (Option B / VICReg + EMA).

    The EMA decay is annealed from base_decay to final_decay following a cosine
    schedule, as in V-JEPA and BYOL. Higher decay later in training → more stable
    targets as the context encoder converges.

    IMPORTANT: Only instantiate EMAUpdater when cfg.use_ema=True. When using
    LeJEPA/SIGReg (cfg.use_ema=False), the target encoder receives gradients
    normally — EMA would break the LeJEPA training invariant.
    """

    def __init__(
        self,
        base_decay: float = 0.996,
        final_decay: float = 0.999,
    ) -> None:
        self.base_decay = base_decay
        self.final_decay = final_decay

    def get_decay(self, step: int, total_steps: int) -> float:
        """Cosine-annealed EMA decay."""
        progress = step / max(total_steps, 1)
        return self.final_decay - (self.final_decay - self.base_decay) * (
            1 + math.cos(math.pi * progress)
        ) / 2

    @torch.no_grad()
    def update(
        self,
        online_encoder: nn.Module,
        target_encoder: nn.Module,
        step: int,
        total_steps: int,
    ) -> None:
        """Update target encoder as EMA of online encoder.

        Args:
            online_encoder: the context encoder (trained via backprop)
            target_encoder: the EMA target encoder (no gradients)
            step:           current global training step
            total_steps:    total steps in training run
        """
        decay = self.get_decay(step, total_steps)
        for online_p, target_p in zip(
            online_encoder.parameters(), target_encoder.parameters()
        ):
            target_p.data.mul_(decay).add_(online_p.data, alpha=1.0 - decay)
