"""DeepSynergy-style DNN for DC domain drug combination prediction.

Architecture: FC(4096→2048→1024→512→n_classes) with BatchNorm + Dropout.
Input: Drug A Morgan FP (2048) ⊕ Drug B Morgan FP (2048) = 4096-dim.

Reference: Preuer et al. "DeepSynergy: predicting anti-cancer drug synergy
with Deep Learning" (Bioinformatics, 2018).
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class DeepSynergyDC(nn.Module):
    """DeepSynergy-style network for drug combination prediction."""

    def __init__(
        self,
        input_dim: int = 4096,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_deepsynergy_dc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    n_classes: int = 2,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cpu",
) -> tuple[DeepSynergyDC, dict]:
    """Train DeepSynergy-style model for DC classification.

    Args:
        X_train: (n, 4096) concatenated Morgan FP for Drug A + Drug B.
        y_train: Labels.
        X_val, y_val: Validation data.
        n_classes: 2 for M1, 3 for M2.
        epochs, batch_size, lr, patience: Training hyperparameters.
        device: 'cpu' or 'cuda'.

    Returns:
        (model, history_dict).
    """
    if torch is None:
        raise ImportError("torch is required: pip install torch")

    input_dim = X_train.shape[1]
    model = DeepSynergyDC(input_dim, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

    has_val = X_val is not None and y_val is not None
    if has_val:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    n_train = len(X_train_t)

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = indices[start:end]
            x_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        if has_val:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
