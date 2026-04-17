"""Multi-level masking strategies for Negative-JEPA.

Four masking types (design doc §3.6):
1. Entity masking  (50%): mask entire entity A or entity B
2. Feature masking (50%): mask random tabular feature subsets (T-JEPA style)
3. Subgraph masking(20%): mask random molecular graph nodes (Graph-JEPA style)
4. Cross-domain masking (10%): predict across domain boundaries — TODO in Phase 3

Types are NOT mutually exclusive: feature + subgraph masking can be combined.
"""
from __future__ import annotations

from typing import Optional

import torch


class MultiLevelMasker:
    """Generate multi-level masks for a batch dict.

    Used by NegJEPATrainer._train_step to produce masks for each forward pass.
    The context encoder sees the masked batch; the target encoder sees the full batch.
    """

    def __init__(
        self,
        entity_ratio: float = 0.5,
        feature_ratio: float = 0.5,
        subgraph_ratio: float = 0.2,
        cross_domain_ratio: float = 0.1,
    ) -> None:
        self.entity_ratio = entity_ratio
        self.feature_ratio = feature_ratio
        self.subgraph_ratio = subgraph_ratio
        self.cross_domain_ratio = cross_domain_ratio

    def generate_masks(self, batch: dict) -> dict:
        """Generate all masks for a given batch.

        Args:
            batch: dict with at minimum 'tabular_A' key (B, F).

        Returns:
            masks dict with keys:
              entity_mask  (B,) bool — True = this sample has entity masking applied
              mask_side    (B,) int  — 0=mask entity A, 1=mask entity B
              tab_mask_A   (B, F) bool
              tab_mask_B   (B, F) bool
              node_mask_A  (N_A,) bool — only if 'graph_A' in batch
              latent_mask_positions: list[int] — which predictor latent positions to predict
        """
        B = batch["tabular_A"].shape[0]
        F = batch["tabular_A"].shape[1]
        device = batch["tabular_A"].device
        masks: dict = {}

        # ── Entity masking ────────────────────────────────────────────────────
        # For 50% of samples, the predictor must reconstruct entity A or entity B
        # entirely from the other entity's context — the primary JEPA objective.
        entity_mask = torch.rand(B, device=device) < self.entity_ratio
        masks["entity_mask"] = entity_mask
        masks["mask_side"] = torch.randint(0, 2, (B,), device=device)  # 0=A, 1=B

        # ── Feature masking (T-JEPA style) ────────────────────────────────────
        # Each feature is independently masked with probability feature_ratio.
        # Masked positions are replaced by learnable reg_tokens in TabularEncoder.
        masks["tab_mask_A"] = torch.rand(B, F, device=device) < self.feature_ratio
        masks["tab_mask_B"] = torch.rand(B, F, device=device) < self.feature_ratio

        # ── Subgraph masking (Graph-JEPA style) ───────────────────────────────
        if "graph_A" in batch:
            n_nodes_A = batch["graph_A"].x.shape[0]
            masks["node_mask_A"] = torch.rand(n_nodes_A, device=device) < self.subgraph_ratio

        if "graph_B" in batch:
            n_nodes_B = batch["graph_B"].x.shape[0]
            masks["node_mask_B"] = torch.rand(n_nodes_B, device=device) < self.subgraph_ratio

        # ── Latent mask positions ─────────────────────────────────────────────
        # Which PerceiverFusion latent positions the predictor must reconstruct.
        # For entity masking: mask half the latent positions (predictor predicts entity B's context)
        # For feature masking: mask a random subset of positions
        # Simple default: mask positions [n_latents//2:] — first half sees context, second predicts
        # The actual n_latents is determined by the encoder config at runtime.
        masks["_entity_mask_applied"] = entity_mask  # alias for trainer use

        # TODO (Phase 3): Cross-domain masking
        # For the 10% cross-domain objective, the DataLoader must pair samples from two
        # different domains in the same batch slot (e.g. DTI drug at positions 0-49
        # alongside PPI protein at positions 0-44, zero-padded to 300). The context
        # encoder processes domain_id=0 (DTI) and the target encoder processes domain_id=1
        # (PPI); the predictor maps between them using the shared D=256 latent space.
        # This requires a multi-domain collation step beyond what generate_masks covers —
        # deferred to dataset.py Phase 3 implementation.

        return masks

    def apply_entity_mask_to_batch(self, batch: dict, masks: dict) -> dict:
        """Return a copy of batch with entity masking applied.

        For samples where entity_mask[b]=True and mask_side[b]=0: zero out tabular_A
        (and graph_A / seq_A if present). For mask_side[b]=1: zero out tabular_B.

        This produces the 'context batch' seen by the context encoder.
        The original batch (unmasked) is seen by the target encoder.
        """
        masked = {k: v for k, v in batch.items()}  # shallow copy

        entity_mask = masks["entity_mask"]    # (B,)
        mask_side = masks["mask_side"]        # (B,) 0=A, 1=B

        B = batch["tabular_A"].shape[0]

        # Mask entity A (side=0)
        mask_A = entity_mask & (mask_side == 0)
        if mask_A.any():
            tab_A = batch["tabular_A"].clone()
            tab_A[mask_A] = 0.0
            masked["tabular_A"] = tab_A
            # Also zero seq_A and graph_A signal (handled downstream by zeroing tokens)
            if "seq_A" in batch:
                seq_A = batch["seq_A"].clone()
                seq_A[mask_A] = 0
                masked["seq_A"] = seq_A
            if "esm2_A" in batch:
                esm2_A = batch["esm2_A"].clone()
                esm2_A[mask_A] = 0.0
                masked["esm2_A"] = esm2_A

        # Mask entity B (side=1)
        mask_B = entity_mask & (mask_side == 1)
        if mask_B.any() and "tabular_B" in batch:
            tab_B = batch["tabular_B"].clone()
            tab_B[mask_B] = 0.0
            masked["tabular_B"] = tab_B
            if "seq_B" in batch:
                seq_B = batch["seq_B"].clone()
                seq_B[mask_B] = 0
                masked["seq_B"] = seq_B

        return masked
