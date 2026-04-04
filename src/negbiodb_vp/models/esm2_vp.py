"""ESM2-VP model: ESM2-650M embedding + tabular features for VP prediction.

Architecture:
  Pre-computed ESM2 embeddings (1280-dim) concatenated with tabular features (56-dim).
  Concat(1336) → FC(512) → FC(256) → FC(1)

ESM2 embeddings are pre-computed on HPC and loaded from parquet at training time.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

ESM2_DIM = 1280
TABULAR_DIM = 56


class ESM2VP(nn.Module):
    """ESM2 embedding + tabular features → FC prediction head."""

    def __init__(
        self,
        tabular_dim: int = TABULAR_DIM,
        esm2_dim: int = ESM2_DIM,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        concat_dim = tabular_dim + esm2_dim  # 1336
        self.net = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(
        self,
        tabular: torch.Tensor,  # (B, 56)
        esm2_emb: torch.Tensor,  # (B, 1280)
    ) -> torch.Tensor:
        x = torch.cat([tabular, esm2_emb], dim=1)
        return self.net(x)


def train_esm2_vp(
    X_tab_train: np.ndarray,
    X_esm_train: np.ndarray,
    y_train: np.ndarray,
    X_tab_val: np.ndarray | None = None,
    X_esm_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    n_classes: int = 2,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cpu",
) -> tuple[ESM2VP, dict]:
    """Train ESM2-VP model.

    Args:
        X_tab_train: Tabular features (N, 56)
        X_esm_train: ESM2 embeddings (N, 1280)
        y_train: Labels
    """
    if torch is None:
        raise ImportError("torch is required: pip install torch")

    model = ESM2VP(
        tabular_dim=X_tab_train.shape[1],
        esm2_dim=X_esm_train.shape[1],
        n_classes=n_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    X_tab_t = torch.tensor(X_tab_train, dtype=torch.float32).to(device)
    X_esm_t = torch.tensor(X_esm_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)

    has_val = X_tab_val is not None and y_val is not None
    if has_val:
        X_tab_vt = torch.tensor(X_tab_val, dtype=torch.float32).to(device)
        X_esm_vt = torch.tensor(X_esm_val, dtype=torch.float32).to(device)
        y_vt = torch.tensor(y_val, dtype=torch.long).to(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    n_train = len(X_tab_t)

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = indices[start:end]

            optimizer.zero_grad()
            logits = model(X_tab_t[idx], X_esm_t[idx])
            loss = criterion(logits, y_t[idx])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        if has_val:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_tab_vt, X_esm_vt)
                val_loss = criterion(val_logits, y_vt).item()
                val_acc = (val_logits.argmax(1) == y_vt).float().mean().item()

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            scheduler.step(val_loss)

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
