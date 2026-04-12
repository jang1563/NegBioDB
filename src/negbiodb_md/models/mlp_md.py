"""MLP baseline for MD metabolite-disease biomarker classification.

Uses full 2068-dim feature vectors (ECFP4 + physicochemical + metadata).
StandardScaler applied before MLP to handle mixed feature scales.

Two tasks:
  MD-M1: Binary — is_significant (0/1)
  MD-M2: Multi-class — disease_category (5-class)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def train_mlp_md(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    task: str = "binary",
    hidden_sizes: tuple[int, ...] = (512, 256, 128),
    dropout: float = 0.3,
    lr: float = 1e-3,
    max_epochs: int = 100,
    batch_size: int = 512,
    patience: int = 10,
    seed: int = 42,
    scale_features: bool = True,
):
    """Train MLP classifier for MD with optional StandardScaler preprocessing.

    Args:
        X_train, y_train: Training data.
        X_val, y_val:     Validation data (for early stopping).
        task:             'binary' (MD-M1) or 'multiclass' (MD-M2).
        hidden_sizes:     Sizes of hidden layers.
        dropout:          Dropout rate.
        lr:               Learning rate.
        max_epochs:       Maximum epochs.
        batch_size:       Mini-batch size.
        patience:         Early stopping patience (epochs without val improvement).
        seed:             Random seed.
        scale_features:   Apply StandardScaler before training (recommended: True).

    Returns:
        Tuple of (trained model, scaler or None).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = None
    if scale_features:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        if X_val is not None and len(X_val) > 0:
            X_val = scaler.transform(X_val).astype(np.float32)

    n_classes = len(np.unique(y_train))
    out_dim = 1 if task == "binary" else n_classes

    # Build MLP
    layers: list[nn.Module] = []
    in_dim = X_train.shape[1]
    for h in hidden_sizes:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    if task == "binary":
        layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)

    # Loss
    if task == "binary":
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DataLoaders
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32 if task == "binary" else torch.long)
    train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    has_val = X_val is not None and len(X_val) > 0
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            if task == "binary":
                loss = criterion(out.squeeze(), yb)
            else:
                loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                X_v = torch.tensor(X_val, dtype=torch.float32)
                y_v = torch.tensor(y_val, dtype=torch.float32 if task == "binary" else torch.long)
                out_v = model(X_v)
                if task == "binary":
                    val_loss = criterion(out_v.squeeze(), y_v).item()
                else:
                    val_loss = criterion(out_v, y_v).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d (val_loss=%.4f)", epoch + 1, best_val_loss)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info("MLP %s trained: %d samples, %d features", task, len(X_train), X_train.shape[1])
    return model, scaler


def evaluate_mlp_md(
    model,
    scaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "binary",
) -> dict:
    """Evaluate MLP model and return metrics dict."""
    import torch

    if scaler is not None and len(X_test) > 0:
        X_test = scaler.transform(X_test).astype(np.float32)

    if len(X_test) == 0:
        return {"mcc": None, "accuracy": None, "auroc": None, "n_test": 0}

    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        out = model(X_t)
        if task == "binary":
            y_prob = out.squeeze().numpy()
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_pred = out.argmax(dim=1).numpy()
            y_prob = torch.softmax(out, dim=1).numpy()

    try:
        from sklearn.metrics import (
            matthews_corrcoef, accuracy_score, roc_auc_score,
            average_precision_score,
        )
        mcc = matthews_corrcoef(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        metrics: dict = {"mcc": float(mcc), "accuracy": float(acc), "n_test": len(y_test)}

        if task == "binary" and len(np.unique(y_test)) > 1:
            metrics["auroc"] = float(roc_auc_score(y_test, y_prob))
            metrics["auprc"] = float(average_precision_score(y_test, y_prob))
        else:
            metrics["auroc"] = None
            metrics["auprc"] = None
    except Exception as exc:
        logger.warning("MLP evaluation error: %s", exc)
        metrics = {"mcc": None, "accuracy": None, "n_test": len(y_test)}

    return metrics
