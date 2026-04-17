"""NegJEPATrainer (pretraining) and NegJEPAFinetuner (supervised fine-tuning).

Pretraining loop:
  Phase A — Self-supervised JEPA objective (masking + prediction in latent space).
  No labels used. Options A (SIGReg) and B (VICReg+EMA) supported via cfg.use_ema.

Fine-tuning loop:
  Phase B — Supervised classification with frozen or trainable encoder.
  Uses BCEWithLogitsLoss + AdamW + cosine LR + early stopping on val MCC.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from negbiojepa.config import JEPAConfig
from negbiojepa.losses import NegJEPALoss, check_collapse
from negbiojepa.masking import MultiLevelMasker
from negbiojepa.predictor import EMAUpdater


def _cosine_lr_schedule(
    optimizer: optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
) -> None:
    """In-place LR update (linear warmup + cosine decay)."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ─── NegJEPATrainer ───────────────────────────────────────────────────────────

class NegJEPATrainer:
    """Self-supervised JEPA pretraining loop.

    Handles both Option A (SIGReg, no EMA) and Option B (VICReg + EMA).
    Logs metrics and checks for representation collapse every cfg.log_every steps.
    Saves checkpoints every cfg.save_every epochs and keeps best.pt.
    """

    def __init__(
        self,
        cfg: JEPAConfig,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.context_enc = context_encoder.to(device)
        self.target_enc = target_encoder.to(device)
        self.predictor = predictor.to(device)
        self.device = device

        self.loss_fn = NegJEPALoss(cfg)
        self.masker = MultiLevelMasker(
            entity_ratio=cfg.entity_mask_ratio,
            feature_ratio=cfg.feature_mask_ratio,
            subgraph_ratio=cfg.subgraph_mask_ratio,
        )
        self.ema_updater: Optional[EMAUpdater] = (
            EMAUpdater(cfg.ema_base_decay, cfg.ema_final_decay) if cfg.use_ema else None
        )

        # Only optimize context encoder + predictor
        params = list(self.context_enc.parameters()) + list(self.predictor.parameters())
        self.optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.global_step = 0
        self.best_loss = float("inf")
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def _move_batch(self, batch: dict) -> dict:
        """Move all tensors in batch to self.device."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            elif hasattr(v, "to"):  # PyG Batch
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _train_step(self, batch: dict, total_steps: int) -> dict:
        """Single pretraining step.

        1. Generate masks
        2. Apply entity masking to produce context batch
        3. Forward context_encoder(masked_batch) → z_context
        4. Forward target_encoder(full_batch)    → z_target  [detached if use_ema]
        5. Determine which latent positions to predict (entity-masked half)
        6. Forward predictor(z_context, mask_positions) → z_pred
        7. Compute loss, backprop, clip gradients, optimizer step
        8. Update EMA (if Option B)
        """
        self.optimizer.zero_grad()
        batch = self._move_batch(batch)

        # Generate masks
        masks = self.masker.generate_masks(batch)
        context_batch = self.masker.apply_entity_mask_to_batch(batch, masks)

        # Add tabular masks to context batch
        context_batch["tab_mask_A"] = masks["tab_mask_A"]
        if "tabular_B" in batch:
            context_batch["tab_mask_B"] = masks["tab_mask_B"]

        # Propagate node masks so MolGraphEncoder applies subgraph masking
        if "node_mask_A" in masks:
            context_batch["node_mask_A"] = masks["node_mask_A"]
        if "node_mask_B" in masks:
            context_batch["node_mask_B"] = masks["node_mask_B"]

        # Encode context (masked)
        z_context = self.context_enc(context_batch)   # (B, n_latents, D)

        # Encode target (full, unmasked)
        if self.cfg.use_ema:
            with torch.no_grad():
                z_target = self.target_enc(batch)     # (B, n_latents, D)
        else:
            z_target = self.target_enc(batch)         # gradients flow through both

        # Which latent positions to predict: second half (predictor learns from first half)
        n_latents = z_context.shape[1]
        mask_positions = list(range(n_latents // 2, n_latents))

        # Predict target latents
        z_pred = self.predictor(z_context, mask_positions)         # (B, K, D)
        z_target_masked = z_target[:, mask_positions]              # (B, K, D)

        # Compute loss
        loss, metrics = self.loss_fn(
            z_pred, z_target_masked, z_context, use_ema=self.cfg.use_ema
        )

        loss.backward()

        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.context_enc.parameters()) + list(self.predictor.parameters()),
                self.cfg.grad_clip,
            )

        self.optimizer.step()

        # EMA update (Option B only)
        if self.ema_updater is not None:
            self.ema_updater.update(
                self.context_enc, self.target_enc,
                self.global_step, total_steps,
            )

        self.global_step += 1
        return {"loss": loss.item(), **metrics}

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        total_steps: int,
    ) -> dict:
        """Run one epoch; return averaged metrics."""
        self.context_enc.train()
        self.predictor.train()
        if not self.cfg.use_ema:
            self.target_enc.train()

        _cosine_lr_schedule(
            self.optimizer, epoch, self.cfg.epochs, self.cfg.warmup_epochs, self.cfg.lr
        )

        epoch_metrics: dict = {"loss": [], "pred": [], "reg": []}
        for step, batch in enumerate(dataloader):
            step_metrics = self._train_step(batch, total_steps)
            for k in epoch_metrics:
                epoch_metrics[k].append(step_metrics[k])

            # Collapse detection every log_every steps
            if self.global_step % self.cfg.log_every == 0:
                with torch.no_grad():
                    sample_batch = self._move_batch(batch)
                    z = self.context_enc(sample_batch)
                    collapsed, msg = check_collapse(z)
                    if collapsed:
                        print(f"[Step {self.global_step}] WARNING: Collapse detected: {msg}", flush=True)
                    else:
                        avg_loss = float(np.mean(epoch_metrics["loss"][-self.cfg.log_every:]))
                        print(f"[Epoch {epoch} Step {self.global_step}] "
                              f"loss={avg_loss:.4f} collapse={msg}", flush=True)

        return {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

    def fit(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Full pretraining loop across all epochs."""
        total_steps = self.cfg.epochs * len(dataloader)
        for epoch in range(self.cfg.epochs):
            metrics = self.train_epoch(dataloader, epoch, total_steps)
            print(f"[Epoch {epoch}] loss={metrics['loss']:.4f} "
                  f"pred={metrics['pred']:.4f} reg={metrics['reg']:.4f}", flush=True)

            # Save periodic checkpoint
            if (epoch + 1) % self.cfg.save_every == 0:
                self.save(os.path.join(self.cfg.output_dir, f"checkpoint_epoch{epoch+1}.pt"))

            # Save best checkpoint
            if metrics["loss"] < self.best_loss:
                self.best_loss = metrics["loss"]
                self.save(os.path.join(self.cfg.output_dir, "best.pt"))

    def save(self, path: str) -> None:
        """Save encoder + predictor + optimizer state."""
        torch.save({
            "context_enc": self.context_enc.state_dict(),
            "target_enc": self.target_enc.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "cfg": self.cfg,
        }, path)
        print(f"Saved checkpoint: {path}", flush=True)

    @classmethod
    def load(
        cls,
        path: str,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
        device: torch.device,
    ) -> "NegJEPATrainer":
        """Restore trainer from checkpoint."""
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["cfg"]
        trainer = cls(cfg, context_encoder, target_encoder, predictor, device)
        trainer.context_enc.load_state_dict(ckpt["context_enc"])
        trainer.target_enc.load_state_dict(ckpt["target_enc"])
        trainer.predictor.load_state_dict(ckpt["predictor"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer.global_step = ckpt["global_step"]
        trainer.best_loss = ckpt["best_loss"]
        return trainer


# ─── NegJEPAFinetuner ─────────────────────────────────────────────────────────

class NegJEPAFinetuner:
    """Supervised fine-tuning on top of a pretrained JEPA encoder.

    Adds a linear classification head: Linear(embed_dim * n_latents, 1).
    Uses BCEWithLogitsLoss, AdamW, cosine LR, early stopping on val MCC.
    Evaluates with all 7 NegBioDB metrics: AUROC, MCC, LogAUC, F1, P, R, ACC.
    """

    def __init__(
        self,
        cfg: JEPAConfig,
        pretrained_encoder: nn.Module,
        domain: str,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.domain = domain
        self.device = device

        self.encoder = pretrained_encoder.to(device)
        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        head_in = cfg.embed_dim * cfg.perceiver_n_latents
        self.head = nn.Linear(head_in, 1).to(device)

        params = list(self.head.parameters())
        if not cfg.freeze_encoder:
            params += list(self.encoder.parameters())
        self.optimizer = optim.AdamW(params, lr=cfg.ft_lr, weight_decay=cfg.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()

    def _encode(self, batch: dict) -> torch.Tensor:
        """Encode batch → (B, embed_dim * n_latents) for classification."""
        z = self.encoder(batch)                 # (B, n_latents, D)
        return z.reshape(z.shape[0], -1)        # (B, n_latents * D)

    def _move_batch(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            elif hasattr(v, "to"):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def train_epoch(self, train_loader) -> float:
        self.encoder.train() if not self.cfg.freeze_encoder else self.encoder.eval()
        self.head.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()
            feats = self._encode(batch)
            logits = self.head(feats).squeeze(-1)
            labels = batch["label"].float()
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(train_loader), 1)

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        """Return all 7 NegBioDB metrics."""
        self.encoder.eval()
        self.head.eval()
        all_logits, all_labels = [], []
        for batch in loader:
            batch = self._move_batch(batch)
            feats = self._encode(batch)
            logits = self.head(feats).squeeze(-1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())

        if not all_logits:
            return {}

        logits_np = np.concatenate(all_logits)
        labels_np = np.concatenate(all_labels)
        probs = 1 / (1 + np.exp(-logits_np))  # sigmoid
        preds = (probs >= 0.5).astype(int)

        _trapz = getattr(np, "trapezoid", np.trapz)

        try:
            auroc = roc_auc_score(labels_np, probs)
        except Exception:
            auroc = float("nan")

        try:
            mcc = matthews_corrcoef(labels_np, preds)
        except Exception:
            mcc = float("nan")

        # LogAUC[0.001, 0.1]
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(labels_np, probs)
            mask = (fpr >= 0.001) & (fpr <= 0.1)
            if mask.sum() >= 2:
                log_fpr = np.log10(fpr[mask])
                log_auc = float(_trapz(tpr[mask], log_fpr) / (np.log10(0.1) - np.log10(0.001)))
            else:
                log_auc = float("nan")
        except Exception:
            log_auc = float("nan")

        return {
            "auroc": float(auroc),
            "mcc": float(mcc),
            "log_auc": float(log_auc),
            "f1": float(f1_score(labels_np, preds, zero_division=0)),
            "precision": float(precision_score(labels_np, preds, zero_division=0)),
            "recall": float(recall_score(labels_np, preds, zero_division=0)),
            "accuracy": float(accuracy_score(labels_np, preds)),
        }

    def fit(
        self,
        train_loader,
        val_loader,
        test_loader,
    ) -> dict:
        """Fine-tune with early stopping; return test metrics."""
        best_val_mcc = -float("inf")
        best_state = copy.deepcopy(self.head.state_dict())
        patience_counter = 0

        for epoch in range(self.cfg.ft_epochs):
            _cosine_lr_schedule(
                self.optimizer, epoch, self.cfg.ft_epochs, self.cfg.ft_warmup_epochs, self.cfg.ft_lr
            )
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_mcc = val_metrics.get("mcc", -float("inf"))

            print(f"[FT Epoch {epoch}] loss={train_loss:.4f} val_mcc={val_mcc:.4f}")

            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                best_state = copy.deepcopy(self.head.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.ft_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.head.load_state_dict(best_state)
        test_metrics = self.evaluate(test_loader)
        return test_metrics

    def save_results(self, metrics: dict, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"domain": self.domain, **metrics}, f, indent=2)
        print(f"Results saved: {output_path}")
