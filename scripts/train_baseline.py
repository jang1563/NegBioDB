#!/usr/bin/env python3
"""Unified training harness for DTI baseline models.

Supports DeepDTA, GraphDTA, DrugBAN on the NegBioDB M1 binary task.

Usage:
    python scripts/train_baseline.py \\
        --model deepdta \\
        --split random \\
        --negative negbiodb \\
        --dataset balanced \\
        --epochs 100 --patience 10 \\
        --batch_size 256 --lr 0.001 --seed 42 \\
        --data_dir exports/ \\
        --output_dir results/baselines/

Outputs:
    results/baselines/{model}_{dataset}_{split}_{negative}_seed{seed}/
        best.pt          — best model checkpoint (val LogAUC)
        results.json     — test-set metrics (7 metrics)
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
    ("balanced", "negbiodb"):       "negbiodb_m1_balanced.parquet",
    ("realistic", "negbiodb"):      "negbiodb_m1_realistic.parquet",
    ("balanced", "uniform_random"): "negbiodb_m1_uniform_random.parquet",
    ("balanced", "degree_matched"): "negbiodb_m1_degree_matched.parquet",
    ("balanced", "ddb"):            "negbiodb_m1_balanced_ddb.parquet",
}

# Split column name by split type
_SPLIT_COL_MAP: dict[str, str] = {
    "random":         "split_random",
    "cold_compound":  "split_cold_compound",
    "cold_target":    "split_cold_target",
    "ddb":            "split_degree_balanced",
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
    """Resolve the parquet filename for a valid experiment configuration."""
    if split == "ddb":
        if negative != "negbiodb":
            return None
        return _DATASET_MAP.get((dataset, "ddb"))
    if negative in {"uniform_random", "degree_matched"} and dataset != "balanced":
        return None
    return _DATASET_MAP.get((dataset, negative))


def _build_run_name(model: str, dataset: str, split: str, negative: str, seed: int) -> str:
    """Build a unique output directory name for a training run."""
    return f"{model}_{dataset}_{split}_{negative}_seed{seed}"


def _json_safe(value):
    """Convert NaN/Inf values to JSON-safe nulls."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_results_json(path: Path, payload: dict) -> None:
    """Write strict JSON results, normalising non-finite floats to null."""
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, allow_nan=False)


def _prepare_graph_cache(parquet_path: Path, cache_path: Path) -> dict[str, object]:
    """Load or build a graph cache that covers every SMILES in parquet_path."""
    import torch
    from negbiodb.models.graphdta import smiles_to_graph

    smiles_series = pd.read_parquet(parquet_path, columns=["smiles"])["smiles"].dropna()
    smiles_list = smiles_series.unique().tolist()

    cache: dict[str, object]
    if cache_path.exists():
        logger.info("Loading graph cache (once): %s", cache_path)
        cache = torch.load(cache_path, weights_only=False)
        logger.info("Graph cache loaded: %d entries", len(cache))
    else:
        cache = {}

    missing = [smi for smi in smiles_list if smi not in cache]
    if missing:
        logger.info("Backfilling graph cache for %d missing SMILES...", len(missing))
        failed = 0
        for smi in missing:
            graph = smiles_to_graph(smi)
            cache[smi] = graph
            if graph is None:
                failed += 1
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        logger.info(
            "Saved graph cache → %s (%d total entries, %d failed parses)",
            cache_path,
            len(cache),
            failed,
        )

    return cache


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DTIDataset:
    """Lazy-loading DTI dataset backed by a Parquet file."""

    def __init__(
        self,
        parquet_path: Path,
        split_col: str,
        fold: str,
        model_type: str,
        graph_cache_path: "Path | dict | None" = None,
    ) -> None:
        self.model_type = model_type
        df_full = pd.read_parquet(parquet_path)
        self.df = df_full[df_full[split_col] == fold].reset_index(drop=True)
        before = len(self.df)
        self.df = self.df.dropna(subset=["smiles", "target_sequence"]).reset_index(drop=True)
        if len(self.df) < before:
            logger.warning("Dropped %d rows with NaN smiles/target_sequence", before - len(self.df))
        logger.info("Fold '%s': %d rows (label 1: %d)", fold, len(self.df), (self.df["Y"] == 1).sum())

        self._graphs: dict[str, object] | None = None
        if model_type in ("graphdta", "drugban"):
            self._load_graphs(graph_cache_path)

    def _load_graphs(self, cache_path: "Path | dict | None") -> None:
        try:
            import torch
            from negbiodb.models.graphdta import smiles_to_graph
        except ImportError as e:
            raise RuntimeError("torch_geometric required for GraphDTA/DrugBAN.") from e

        # Accept a pre-loaded dict to avoid reloading the cache for each fold.
        if isinstance(cache_path, dict):
            self._graphs = cache_path
            return

        smiles_list = self.df["smiles"].unique().tolist()

        if cache_path and cache_path.exists():
            logger.info("Loading graph cache: %s", cache_path)
            self._graphs = torch.load(cache_path, weights_only=False)
            missing = [smi for smi in smiles_list if smi not in self._graphs]
            if missing:
                logger.info("Backfilling %d missing SMILES into graph cache...", len(missing))
                failed = 0
                for smi in missing:
                    g = smiles_to_graph(smi)
                    self._graphs[smi] = g
                    if g is None:
                        failed += 1
                torch.save(self._graphs, cache_path)
                logger.info("Graph cache updated. %d failed parses.", failed)
        else:
            logger.info("Building graph cache for %d unique SMILES...", len(smiles_list))
            self._graphs = {}
            failed = 0
            for smi in smiles_list:
                g = smiles_to_graph(smi)
                self._graphs[smi] = g
                if g is None:
                    failed += 1
            logger.info("Graph cache built. %d failed parses.", failed)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._graphs, cache_path)
                logger.info("Saved graph cache → %s", cache_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = float(row["Y"])
        if self.model_type == "deepdta":
            return row["smiles"], row["target_sequence"], label
        else:
            graph = self._graphs.get(row["smiles"])  # type: ignore[union-attr]
            return graph, row["smiles"], row["target_sequence"], label


def _collate_deepdta(batch, device):
    from negbiodb.models.deepdta import smiles_to_tensor, seq_to_tensor
    import torch
    smiles, seqs, labels = zip(*batch)
    return (
        smiles_to_tensor(list(smiles)).to(device),
        seq_to_tensor(list(seqs)).to(device),
        torch.tensor(labels, dtype=torch.float32).to(device),
    )


def _collate_graph(batch, device):
    import torch
    from torch_geometric.data import Batch
    from negbiodb.models.deepdta import seq_to_tensor

    graphs, smiles_list, seqs, labels = zip(*batch)
    # Replace None graphs with a minimal placeholder (single isolated node)
    from negbiodb.models.graphdta import NODE_FEATURE_DIM
    from torch_geometric.data import Data
    placeholder = Data(
        x=torch.zeros(1, NODE_FEATURE_DIM),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
    )
    graphs_clean = [g if g is not None else placeholder for g in graphs]

    drug_batch = Batch.from_data_list(graphs_clean).to(device)
    target_tokens = seq_to_tensor(list(seqs)).to(device)
    label_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    return drug_batch, target_tokens, label_tensor


def make_dataloader(dataset: DTIDataset, batch_size: int, shuffle: bool, device):
    import torch
    from torch.utils.data import DataLoader

    if dataset.model_type == "deepdta":
        collate_fn = lambda b: _collate_deepdta(b, device)  # noqa: E731
    else:
        collate_fn = lambda b: _collate_graph(b, device)  # noqa: E731

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


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    import torch
    model.train(train)
    total_loss = 0.0
    all_labels: list[float] = []
    all_preds: list[float] = []
    n_batches = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            inputs, targets_or_seq, labels = batch
            logits = model(inputs, targets_or_seq)

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
    """Primary validation metric: LogAUC[0.001, 0.1]."""
    from negbiodb.metrics import log_auc
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return log_auc(y_true, y_score)


def train(
    model,
    train_loader,
    val_loader,
    epochs: int,
    patience: int,
    lr: float,
    output_dir: Path,
    device,
):
    import torch
    import torch.nn as nn

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_metric = float("-inf")
    patience_counter = 0
    training_log = []

    for epoch in range(1, epochs + 1):
        train_loss, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_y, val_pred = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
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
            logger.info("  ↳ Saved best checkpoint (val_LogAUC=%.4f)", best_val_metric)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # Save training log
    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
        writer.writeheader()
        writer.writerows(training_log)
    logger.info("Training log saved → %s", log_path)

    return best_val_metric


def evaluate(model, test_loader, checkpoint_path: Path, device) -> dict[str, float]:
    import torch
    from negbiodb.metrics import compute_all_metrics

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    _, y_true, y_score = _run_epoch(model, test_loader, criterion, None, device, train=False)

    if len(np.unique(y_true)) < 2:
        logger.warning("Test set has only one class — metrics will be NaN.")
        return {k: float("nan") for k in ["auroc", "auprc", "mcc", "log_auc", "bedroc", "ef_1pct", "ef_5pct"]}

    return compute_all_metrics(y_true, y_score)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(model_type: str):
    if model_type == "deepdta":
        from negbiodb.models.deepdta import DeepDTA
        return DeepDTA()
    elif model_type == "graphdta":
        from negbiodb.models.graphdta import GraphDTA
        return GraphDTA()
    elif model_type == "drugban":
        from negbiodb.models.drugban import DrugBAN
        return DrugBAN()
    else:
        raise ValueError(f"Unknown model: {model_type!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a DTI baseline model.")
    parser.add_argument("--model", required=True, choices=["deepdta", "graphdta", "drugban"])
    parser.add_argument("--split", required=True, choices=list(_SPLIT_COL_MAP))
    parser.add_argument("--negative", required=True,
                        choices=["negbiodb", "uniform_random", "degree_matched"])
    parser.add_argument("--dataset", default="balanced", choices=["balanced", "realistic"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=Path, default=ROOT / "exports")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "baselines")
    args = parser.parse_args(argv)

    set_seed(args.seed)

    if args.split == "ddb" and args.dataset != "balanced":
        logger.error("DDB split is only supported for dataset=balanced.")
        return 1

    filename = _resolve_dataset_file(args.dataset, args.split, args.negative)
    if filename is None:
        available = list(_DATASET_MAP.keys())
        logger.error(
            "Dataset combination not supported: dataset=%s, split=%s, negative=%s. "
            "Available: %s",
            args.dataset, args.split, args.negative, available,
        )
        return 1

    parquet_path = args.data_dir / filename
    if not parquet_path.exists():
        logger.error("Dataset file not found: %s", parquet_path)
        logger.error("Run `python scripts/prepare_exp_data.py` first.")
        return 1

    split_col = _SPLIT_COL_MAP[args.split]

    # Output directory
    run_name = _build_run_name(
        args.model, args.dataset, args.split, args.negative, args.seed
    )
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run: %s → %s", run_name, out_dir)

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
        logger.error("torch not found. Install with: pip install negbiodb[ml]")
        return 1
    logger.info("Device: %s", device)

    # Graph cache — load once and share across all three folds to avoid tripling RAM usage.
    graph_cache: "Path | dict | None" = None
    if args.model in ("graphdta", "drugban"):
        cache_path = args.data_dir / "graph_cache.pt"
        graph_cache = _prepare_graph_cache(parquet_path, cache_path)

    # Build datasets
    train_ds = DTIDataset(parquet_path, split_col, "train", args.model, graph_cache)
    val_ds   = DTIDataset(parquet_path, split_col, "val",   args.model, graph_cache)
    test_ds  = DTIDataset(parquet_path, split_col, "test",  args.model, graph_cache)

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        logger.error("Empty split detected. Check split_col=%s in %s.", split_col, parquet_path.name)
        return 1

    # Adjust batch size for DrugBAN (higher memory)
    batch_size = args.batch_size
    if args.model == "drugban" and batch_size > 128:
        logger.info("Reducing batch_size from %d to 128 for DrugBAN (memory limit).", batch_size)
        batch_size = 128

    train_loader = make_dataloader(train_ds, batch_size, shuffle=True,  device=device)
    val_loader   = make_dataloader(val_ds,   batch_size, shuffle=False, device=device)
    test_loader  = make_dataloader(test_ds,  batch_size, shuffle=False, device=device)

    # Build and move model
    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s | params: %d", args.model, n_params)

    # Train
    best_val = train(
        model, train_loader, val_loader,
        epochs=args.epochs, patience=args.patience, lr=args.lr,
        output_dir=out_dir, device=device,
    )

    # Evaluate on test set
    checkpoint = out_dir / "best.pt"
    if not checkpoint.exists():
        logger.error("No checkpoint saved — all epochs produced NaN val LogAUC.")
        return 1

    test_metrics = evaluate(model, test_loader, checkpoint, device)

    # Save results
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

    logger.info("Test metrics:")
    for k, v in test_metrics.items():
        logger.info("  %-15s = %.4f", k, v if not np.isnan(v) else float("nan"))
    logger.info("Results saved → %s", results_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
