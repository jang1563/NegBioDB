"""Evaluation utilities for Negative-JEPA.

linear_probe():   frozen encoder + linear head — measures representation quality
knn_eval():       k-NN classification in latent space
anomaly_score():  prediction error as negative-result score (unsupervised)

All functions follow the 7-metric convention from NegBioDB:
  AUROC, MCC, LogAUC[0.001, 0.1], F1, Precision, Recall, Accuracy
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier


def _encode_all(
    encoder: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_latents: int,
    embed_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode all batches and return (features, labels) arrays.

    Features are flattened latent queries: (N, n_latents * embed_dim).
    """
    encoder.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
                elif hasattr(v, "to"):
                    batch[k] = v.to(device)
            z = encoder(batch)                        # (B, n_latents, D)
            feats = z.reshape(z.shape[0], -1).cpu().numpy()
            all_feats.append(feats)
            all_labels.append(batch["label"].cpu().numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


def _compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Compute all 7 NegBioDB metrics from binary labels and prediction probabilities."""
    _trapz = getattr(np, "trapezoid", np.trapz)
    preds = (probs >= 0.5).astype(int)

    try:
        auroc = float(roc_auc_score(labels, probs))
    except Exception:
        auroc = float("nan")

    try:
        mcc = float(matthews_corrcoef(labels, preds))
    except Exception:
        mcc = float("nan")

    # LogAUC[0.001, 0.1]
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, probs)
        mask = (fpr >= 0.001) & (fpr <= 0.1)
        if mask.sum() >= 2:
            log_auc = float(
                _trapz(tpr[mask], np.log10(fpr[mask]))
                / (np.log10(0.1) - np.log10(0.001))
            )
        else:
            log_auc = float("nan")
    except Exception:
        log_auc = float("nan")

    return {
        "auroc": auroc,
        "mcc": mcc,
        "log_auc": log_auc,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }


def linear_probe(
    encoder: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    embed_dim: int,
    n_latents: int,
    device: torch.device,
    n_epochs: int = 20,
    lr: float = 1e-3,
) -> dict:
    """Linear probing: freeze encoder, train linear head.

    Measures representation quality without any fine-tuning signal from the encoder.
    Good linear probe performance → encoder has learned discriminative representations.

    Returns test metrics dict.
    """
    head_in = embed_dim * n_latents
    head = nn.Linear(head_in, 1).to(device)
    optimizer = optim.AdamW(head.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    for epoch in range(n_epochs):
        head.train()
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
                elif hasattr(v, "to"):
                    batch[k] = v.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                z = encoder(batch)
            feats = z.reshape(z.shape[0], -1)
            logits = head(feats).squeeze(-1)
            loss = criterion(logits, batch["label"].float())
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    head.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
                elif hasattr(v, "to"):
                    batch[k] = v.to(device)
            z = encoder(batch)
            feats = z.reshape(z.shape[0], -1)
            logits = head(feats).squeeze(-1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-logits_np))
    return _compute_metrics(labels_np, probs)


def knn_eval(
    encoder: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    embed_dim: int,
    n_latents: int,
    device: torch.device,
    k: int = 5,
) -> dict:
    """k-NN classification in latent space.

    No training required — tests whether representations form discriminative clusters.
    Closer to unsupervised evaluation than linear probing.
    """
    train_feats, train_labels = _encode_all(encoder, train_loader, device, n_latents, embed_dim)
    test_feats, test_labels = _encode_all(encoder, test_loader, device, n_latents, embed_dim)

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(train_feats, train_labels)
    probs = knn.predict_proba(test_feats)[:, 1]
    return _compute_metrics(test_labels, probs)


def anomaly_score(
    context_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_latents: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample prediction error as anomaly score.

    Negative results hypothesis: genuine negative results (non-interactions) should
    produce higher prediction error in latent space — violations of learned biological
    expectations (the V-JEPA parallel for biology).

    Masking strategy at inference: entity masking only (mask entity B, context = A).
    This is the cleanest signal: can entity A's context predict entity B's representation?

    See design doc Section 6.4 for full inference procedure.

    Args:
        context_encoder: context encoder (forward = masked input)
        target_encoder:  target encoder (forward = full unmasked input)
        predictor:       JEPA predictor
        test_loader:     DataLoader (batches with 'label' key)
        device:          compute device
        n_latents:       number of latent queries in PerceiverFusion

    Returns:
        (scores, labels) arrays of shape (N,).
        Higher score = model predicts this is anomalous (i.e., a negative result).
    """
    from negbiojepa.masking import MultiLevelMasker

    masker = MultiLevelMasker(entity_ratio=1.0, feature_ratio=0.0, subgraph_ratio=0.0)
    # Force all samples to mask entity B (mask_side=1)
    mask_positions = list(range(n_latents // 2, n_latents))

    context_encoder.eval()
    target_encoder.eval()
    predictor.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
                elif hasattr(v, "to"):
                    batch[k] = v.to(device)

            # Step 1: mask entity B in context batch
            B = batch["tabular_A"].shape[0]
            F_A = batch["tabular_A"].shape[1]
            masks = {
                "entity_mask": torch.ones(B, dtype=torch.bool, device=device),
                "mask_side": torch.ones(B, dtype=torch.long, device=device),  # 1 = mask B
                "tab_mask_A": torch.zeros(B, F_A, dtype=torch.bool, device=device),
            }
            # tab_mask_B only for pair domains (DTI, PPI, DC, CP); GE/CT/VP/MD have no entity B
            if "tabular_B" in batch:
                masks["tab_mask_B"] = torch.zeros(B, batch["tabular_B"].shape[1], dtype=torch.bool, device=device)
            context_batch = masker.apply_entity_mask_to_batch(batch, masks)

            # Step 2: encode
            z_context = context_encoder(context_batch)   # (B, n_latents, D)
            z_target = target_encoder(batch)              # (B, n_latents, D)

            # Step 3: predict
            z_pred = predictor(z_context, mask_positions)        # (B, K, D)
            z_target_masked = z_target[:, mask_positions]        # (B, K, D)

            # Step 4: prediction error — L2 norm, mean over latent positions
            error = (z_pred - z_target_masked).pow(2).sum(dim=-1).sqrt()  # (B, K)
            score = error.mean(dim=1)                                       # (B,)

            all_scores.append(score.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)
