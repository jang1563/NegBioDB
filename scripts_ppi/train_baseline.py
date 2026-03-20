#!/usr/bin/env python3
"""Unified training harness for PPI baseline models.

Supports SiameseCNN, PIPR, MLPFeatures on PPI M1 binary task.

Usage:
    PYTHONPATH=src python scripts_ppi/train_baseline.py \
        --model siamese_cnn \
        --split random \
        --negative negbiodb \
        --dataset balanced \
        --epochs 100 --patience 10 \
        --batch_size 256 --lr 0.001 --seed 42

Outputs:
    results/ppi_baselines/{model}_{dataset}_{split}_{negative}_seed{seed}/
        best.pt          — best model checkpoint (val LogAUC)
        results.json     — test-set metrics
        training_log.csv — per-epoch train/val metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

# Parquet filenames by (dataset, negative) key
_DATASET_MAP: dict[tuple[str, str], str] = {
    ("balanced", "negbiodb"):       "ppi_m1_balanced.parquet",
    ("realistic", "negbiodb"):      "ppi_m1_realistic.parquet",
    ("balanced", "uniform_random"): "ppi_m1_uniform_random.parquet",
    ("balanced", "degree_matched"): "ppi_m1_degree_matched.parquet",
    ("balanced", "ddb"):            "ppi_m1_balanced_ddb.parquet",
}

# Split column name by split type
_SPLIT_COL_MAP: dict[str, str] = {
    "random":        "split_random",
    "cold_protein":  "split_cold_protein",
    "cold_both":     "split_cold_both",
    "ddb":           "split_degree_balanced",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _resolve_dataset_file(dataset: str, split: str, negative: str) -> str | None:
    """Resolve parquet filename for a valid experiment configuration."""
    if split == "ddb":
        if negative != "negbiodb":
            return None
        return _DATASET_MAP.get((dataset, "ddb"))
    if negative in {"uniform_random", "degree_matched"} and dataset != "balanced":
        return None
    return _DATASET_MAP.get((dataset, negative))


def _build_run_name(model: str, dataset: str, split: str, negative: str, seed: int) -> str:
    return f"{model}_{dataset}_{split}_{negative}_seed{seed}"


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_results_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, allow_nan=False)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PPIDataset:
    """PPI dataset backed by a Parquet file."""

    def __init__(
        self,
        parquet_path: Path,
        split_col: str,
        fold: str,
        model_type: str,
    ) -> None:
        self.model_type = model_type
        df_full = pd.read_parquet(parquet_path)

        # Recompute pair degrees from the full merged graph to avoid
        # pos/neg asymmetry (DB negatives have degree, positives/controls don't).
        if model_type == "mlp_features":
            all_ids = pd.concat([df_full["uniprot_id_1"], df_full["uniprot_id_2"]])
            degree_map = all_ids.value_counts()
            df_full["protein1_degree"] = df_full["uniprot_id_1"].map(degree_map).astype(float)
            df_full["protein2_degree"] = df_full["uniprot_id_2"].map(degree_map).astype(float)

        # For cold_both, NaN fold = excluded pairs (dropped by == comparison)
        self.df = df_full[df_full[split_col] == fold].reset_index(drop=True)

        before = len(self.df)
        self.df = self.df.dropna(subset=["sequence_1", "sequence_2"]).reset_index(drop=True)
        if len(self.df) < before:
            logger.warning("Dropped %d rows with NaN sequences", before - len(self.df))
        logger.info("Fold '%s': %d rows (label 1: %d)", fold, len(self.df), (self.df["Y"] == 1).sum())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = float(row["Y"])
        if self.model_type == "mlp_features":
            return (
                row["sequence_1"], row["sequence_2"],
                0.0 if pd.isna(row.get("protein1_degree")) else float(row["protein1_degree"]),
                0.0 if pd.isna(row.get("protein2_degree")) else float(row["protein2_degree"]),
                row.get("subcellular_location_1"),
                row.get("subcellular_location_2"),
                label,
            )
        else:
            return row["sequence_1"], row["sequence_2"], label


def _collate_sequence_pair(batch, device):
    """Collate for SiameseCNN and PIPR (sequence-based models)."""
    import torch
    from negbiodb_ppi.models.siamese_cnn import seq_to_tensor

    seqs1, seqs2, labels = zip(*batch)
    return (
        seq_to_tensor(list(seqs1)).to(device),
        seq_to_tensor(list(seqs2)).to(device),
        torch.tensor(labels, dtype=torch.float32).to(device),
    )


def _collate_features(batch, device):
    """Collate for MLPFeatures (feature-based model)."""
    import torch
    from negbiodb_ppi.models.mlp_features import extract_features

    features_list = []
    labels = []
    for seq1, seq2, deg1, deg2, loc1, loc2, label in batch:
        features_list.append(extract_features(seq1, seq2, deg1, deg2, loc1, loc2))
        labels.append(label)

    return (
        torch.tensor(features_list, dtype=torch.float32).to(device),
        None,  # placeholder for consistency
        torch.tensor(labels, dtype=torch.float32).to(device),
    )


def make_dataloader(dataset: PPIDataset, batch_size: int, shuffle: bool, device):
    from torch.utils.data import DataLoader

    if dataset.model_type == "mlp_features":
        collate_fn = lambda b: _collate_features(b, device)  # noqa: E731
    else:
        collate_fn = lambda b: _collate_sequence_pair(b, device)  # noqa: E731

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _run_epoch(model, loader, criterion, optimizer, device, train: bool, model_type: str):
    import torch
    model.train(train)
    total_loss = 0.0
    all_labels: list[float] = []
    all_preds: list[float] = []
    n_batches = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            if model_type == "mlp_features":
                features, _, labels = batch
                logits = model(features)
            else:
                seq1_tokens, seq2_tokens, labels = batch
                logits = model(seq1_tokens, seq2_tokens)

            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(torch.sigmoid(logits).detach().cpu().tolist())
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, np.array(all_labels), np.array(all_preds)


def _compute_val_metric(y_true, y_score):
    from negbiodb.metrics import log_auc
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return log_auc(y_true, y_score)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    patience: int,
    lr: float,
    output_dir: Path,
    device,
    model_type: str,
):
    import torch
    import torch.nn as nn

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_metric = float("-inf")
    patience_counter = 0
    training_log = []

    for epoch in range(1, epochs + 1):
        train_loss, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, model_type=model_type
        )
        val_loss, val_y, val_pred = _run_epoch(
            model, val_loader, criterion, optimizer, device, train=False, model_type=model_type
        )
        val_metric = _compute_val_metric(val_y, val_pred)

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_log_auc": val_metric}
        training_log.append(row)

        logger.info(
            "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_LogAUC=%.4f",
            epoch, train_loss, val_loss, val_metric,
        )

        if not np.isnan(val_metric) and val_metric > best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
            logger.info("  -> Saved best checkpoint (val_LogAUC=%.4f)", best_val_metric)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
        writer.writeheader()
        writer.writerows(training_log)
    logger.info("Training log saved -> %s", log_path)

    return best_val_metric


def evaluate(model, test_loader, checkpoint_path: Path, device, model_type: str) -> dict[str, float]:
    import torch
    from negbiodb.metrics import compute_all_metrics

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    _, y_true, y_score = _run_epoch(
        model, test_loader, criterion, None, device, train=False, model_type=model_type
    )

    if len(np.unique(y_true)) < 2:
        logger.warning("Test set has only one class — metrics will be NaN.")
        return {k: float("nan") for k in ["auroc", "auprc", "mcc", "log_auc", "bedroc", "ef_1pct", "ef_5pct"]}

    return compute_all_metrics(y_true, y_score)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(model_type: str):
    if model_type == "siamese_cnn":
        from negbiodb_ppi.models.siamese_cnn import SiameseCNN
        return SiameseCNN()
    elif model_type == "pipr":
        from negbiodb_ppi.models.pipr import PIPR
        return PIPR()
    elif model_type == "mlp_features":
        from negbiodb_ppi.models.mlp_features import MLPFeatures
        return MLPFeatures()
    else:
        raise ValueError(f"Unknown model: {model_type!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a PPI baseline model.")
    parser.add_argument("--model", required=True,
                        choices=["siamese_cnn", "pipr", "mlp_features"])
    parser.add_argument("--split", required=True, choices=list(_SPLIT_COL_MAP))
    parser.add_argument("--negative", required=True,
                        choices=["negbiodb", "uniform_random", "degree_matched"])
    parser.add_argument("--dataset", default="balanced",
                        choices=["balanced", "realistic"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=Path,
                        default=ROOT / "exports" / "ppi")
    parser.add_argument("--output_dir", type=Path,
                        default=ROOT / "results" / "ppi_baselines")
    args = parser.parse_args(argv)

    set_seed(args.seed)

    if args.split == "ddb" and args.dataset != "balanced":
        logger.error("DDB split only supported for dataset=balanced.")
        return 1

    filename = _resolve_dataset_file(args.dataset, args.split, args.negative)
    if filename is None:
        logger.error(
            "Invalid combination: dataset=%s, split=%s, negative=%s",
            args.dataset, args.split, args.negative,
        )
        return 1

    parquet_path = args.data_dir / filename
    if not parquet_path.exists():
        logger.error("Dataset not found: %s", parquet_path)
        logger.error("Run `scripts_ppi/prepare_exp_data.py` first.")
        return 1

    split_col = _SPLIT_COL_MAP[args.split]

    run_name = _build_run_name(
        args.model, args.dataset, args.split, args.negative, args.seed
    )
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run: %s -> %s", run_name, out_dir)

    # Device
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    except ImportError:
        logger.error("torch not found.")
        return 1
    logger.info("Device: %s", device)

    # Build datasets
    train_ds = PPIDataset(parquet_path, split_col, "train", args.model)
    val_ds   = PPIDataset(parquet_path, split_col, "val",   args.model)
    test_ds  = PPIDataset(parquet_path, split_col, "test",  args.model)

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        logger.error("Empty split. Check split_col=%s in %s.", split_col, parquet_path.name)
        return 1

    train_loader = make_dataloader(train_ds, args.batch_size, shuffle=True,  device=device)
    val_loader   = make_dataloader(val_ds,   args.batch_size, shuffle=False, device=device)
    test_loader  = make_dataloader(test_ds,  args.batch_size, shuffle=False, device=device)

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s | params: %d", args.model, n_params)

    best_val = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, patience=args.patience, lr=args.lr,
        output_dir=out_dir, device=device, model_type=args.model,
    )

    checkpoint = out_dir / "best.pt"
    if not checkpoint.exists():
        logger.warning("No checkpoint saved — all epochs produced NaN val LogAUC.")
        null_metrics = {k: None for k in [
            "log_auc", "auprc", "bedroc", "ef_1pct", "ef_5pct", "mcc", "auroc",
        ]}
        results = {
            "run_name": run_name,
            "model": args.model,
            "split": args.split,
            "negative": args.negative,
            "dataset": args.dataset,
            "seed": args.seed,
            "best_val_log_auc": None,
            "test_metrics": null_metrics,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
        }
        results_path = out_dir / "results.json"
        write_results_json(results_path, results)
        logger.info("Null results saved → %s", results_path)
        return 0

    test_metrics = evaluate(model, test_loader, checkpoint, device, args.model)

    results = {
        "run_name": run_name,
        "model": args.model,
        "split": args.split,
        "negative": args.negative,
        "dataset": args.dataset,
        "seed": args.seed,
        "best_val_log_auc": best_val,
        "test_metrics": test_metrics,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }
    results_path = out_dir / "results.json"
    write_results_json(results_path, results)

    logger.info("Results saved -> %s", results_path)
    for metric, value in test_metrics.items():
        logger.info("  %s: %.4f", metric, value if value is not None else float("nan"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
