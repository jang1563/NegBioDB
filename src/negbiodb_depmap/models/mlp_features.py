"""MLP on concatenated gene + cell line + omics features for GE prediction.

Simple 3-layer MLP. Expected to underperform XGBoost on this tabular data
but included for completeness and to match the PPI domain model set.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GEMLP(nn.Module):
    """Simple 3-layer MLP for gene essentiality prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GEDataset(torch.utils.data.Dataset):
    """Dataset for GE MLP training."""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_mlp_ge(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    n_classes: int = 2,
    hidden_dim: int = 256,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[GEMLP, dict]:
    """Train MLP for GE classification.

    Returns:
        (model, history_dict)
    """
    import numpy as np

    input_dim = X_train.shape[1]
    model = GEMLP(input_dim, hidden_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = GEDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        val_ds = GEDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False
        )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        if val_loader:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    val_loss += criterion(logits, y_batch).item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += len(y_batch)

            avg_val_loss = val_loss / max(len(val_loader), 1)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(correct / max(total, 1))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
