#!/usr/bin/env python3
"""Unified training harness for CT baseline models.

Supports XGBoost, MLP, GNN+Tab on CT-M1 (binary) and CT-M2 (7/8-way classification).

Usage:
    python scripts_ct/train_ct_baseline.py \\
        --model xgboost \\
        --task m1 \\
        --split random \\
        --dataset balanced \\
        --negative negbiodb \\
        --epochs 100 --patience 15 \\
        --batch_size 256 --lr 0.001 --seed 42 \\
        --data_dir exports/ct/ \\
        --output_dir results/ct_baselines/

Outputs:
    results/ct_baselines/{model}_{task}_{dataset}_{split}_{negative}_seed{seed}/
        best.pt|best.json   — checkpoint (pt for MLP/GNN, json for XGBoost)
        results.json         — test-set metrics
        training_log.csv     — per-epoch log (MLP/GNN only)
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

# ---------------------------------------------------------------------------
# Dataset maps
# ---------------------------------------------------------------------------

# M1 parquets by (dataset, negative)
_M1_DATASET_MAP: dict[tuple[str, str], str] = {
    ("balanced", "negbiodb"): "negbiodb_ct_m1_balanced.parquet",
    ("realistic", "negbiodb"): "negbiodb_ct_m1_realistic.parquet",
    ("smiles_only", "negbiodb"): "negbiodb_ct_m1_smiles_only.parquet",
    ("balanced", "uniform_random"): "negbiodb_ct_m1_uniform_random.parquet",
    ("balanced", "degree_matched"): "negbiodb_ct_m1_degree_matched.parquet",
}

# M2 has a single parquet
_M2_PARQUET = "negbiodb_ct_m2.parquet"

# Pre-computed split columns in M1 parquets
_M1_PRECOMPUTED_SPLITS = {"random", "cold_drug", "cold_condition"}

# Failure category mapping (must match ct_export.CATEGORY_TO_INT)
CATEGORY_TO_INT: dict[str, int] = {
    "efficacy": 0,
    "enrollment": 1,
    "other": 2,
    "strategic": 3,
    "safety": 4,
    "design": 5,
    "regulatory": 6,
    "pharmacokinetic": 7,
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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


def _json_safe(value):
    """Convert NaN/Inf to JSON-safe nulls."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def write_results_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, allow_nan=False)


# ---------------------------------------------------------------------------
# Data Loading + Split Resolution
# ---------------------------------------------------------------------------


def _resolve_m1_parquet(dataset: str, negative: str) -> str | None:
    return _M1_DATASET_MAP.get((dataset, negative))


def _get_split_column(task: str, split: str) -> str:
    """Return the split column name in the parquet."""
    return f"split_{split}"


def _compute_runtime_split(df: pd.DataFrame, split: str, seed: int) -> pd.Series:
    """Compute temporal/scaffold/degree_balanced splits at runtime for M1.

    M1 parquets may have NaN pair_id for success rows (from CTO).
    We assign synthetic integer IDs for the split functions.
    """
    from negbiodb_ct.ct_export import (
        generate_ct_degree_balanced_split,
        generate_ct_scaffold_split,
        generate_ct_temporal_split,
    )

    # Use synthetic IDs: M1 pair_id can have NaN for success rows
    df = df.copy()
    synthetic_id_col = "_split_id"
    df[synthetic_id_col] = range(len(df))
    # Temporarily set as pair_id/result_id for split functions
    orig_id_col = "pair_id" if "pair_id" in df.columns else "result_id"
    orig_ids = df[orig_id_col].copy()
    df[orig_id_col] = df[synthetic_id_col]

    if split == "temporal":
        fold_map = generate_ct_temporal_split(df)
    elif split == "scaffold":
        fold_map = generate_ct_scaffold_split(df, seed=seed)
    elif split == "degree_balanced":
        fold_map = generate_ct_degree_balanced_split(df, seed=seed)
    else:
        raise ValueError(f"Cannot compute runtime split for: {split}")

    # Map back using synthetic IDs
    return pd.Series(df[synthetic_id_col].map(fold_map).values, index=df.index)


def load_data(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split data for the given task/split/dataset/negative combo.

    Returns (train_df, val_df, test_df).
    """
    if args.task == "m1":
        filename = _resolve_m1_parquet(args.dataset, args.negative)
        if filename is None:
            raise ValueError(
                f"M1 dataset combo not found: dataset={args.dataset}, negative={args.negative}"
            )
        parquet_path = args.data_dir / filename
        if not parquet_path.exists():
            raise FileNotFoundError(f"M1 parquet not found: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        logger.info("Loaded M1 data: %d rows from %s", len(df), parquet_path.name)

        # Get split column
        split_col = _get_split_column(args.task, args.split)
        if split_col in df.columns:
            logger.info("Using pre-computed split: %s", split_col)
        elif args.split in _M1_PRECOMPUTED_SPLITS:
            raise ValueError(f"Split column {split_col} not found in {parquet_path.name}")
        else:
            # Runtime split for temporal/scaffold/degree_balanced
            if args.split == "temporal" and args.negative != "negbiodb":
                raise ValueError(
                    f"Temporal split requires negbiodb negatives "
                    f"(earliest_completion_year column), got --negative={args.negative}"
                )
            logger.info("Computing runtime split: %s", args.split)
            df[split_col] = _compute_runtime_split(df, args.split, args.seed)
            # Drop rows with None fold (e.g. scaffold split for non-SMILES)
            before = len(df)
            df = df.dropna(subset=[split_col]).reset_index(drop=True)
            if len(df) < before:
                logger.info("Dropped %d rows with NULL fold", before - len(df))

    else:  # m2
        parquet_path = args.data_dir / _M2_PARQUET
        if not parquet_path.exists():
            raise FileNotFoundError(f"M2 parquet not found: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        logger.info("Loaded M2 data: %d rows from %s", len(df), parquet_path.name)
        split_col = _get_split_column(args.task, args.split)
        if split_col not in df.columns:
            raise ValueError(f"Split column {split_col} not in {parquet_path.name}")

    # Split into folds
    train_df = df[df[split_col] == "train"].reset_index(drop=True)
    val_df = df[df[split_col] == "val"].reset_index(drop=True)
    test_df = df[df[split_col] == "test"].reset_index(drop=True)

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_m1_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute M1 binary metrics (9 total: 7 DTI + accuracy + F1)."""
    from negbiodb.metrics import compute_all_metrics
    from sklearn.metrics import accuracy_score, f1_score

    if len(np.unique(y_true)) < 2:
        logger.warning("Test set has only one class — metrics will be NaN.")
        keys = ["auroc", "auprc", "mcc", "log_auc", "bedroc", "ef_1pct", "ef_5pct", "accuracy", "f1"]
        return {k: float("nan") for k in keys}

    base = compute_all_metrics(y_true, y_score)
    y_pred = (y_score >= 0.5).astype(int)
    base["accuracy"] = float(accuracy_score(y_true, y_pred))
    base["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return base


def compute_m2_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute M2 multiclass metrics."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
    )

    n_classes = len(CATEGORY_TO_INT)  # always 8
    all_labels = list(range(n_classes))
    per_class_acc = {}
    for c in all_labels:
        mask = y_true == c
        cat_name = [k for k, v in CATEGORY_TO_INT.items() if v == c]
        cat_name = cat_name[0] if cat_name else str(c)
        if mask.sum() > 0:
            per_class_acc[cat_name] = float((y_pred[mask] == c).mean())
        else:
            per_class_acc[cat_name] = float("nan")

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=all_labels)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=all_labels)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=all_labels).tolist(),
    }


# ---------------------------------------------------------------------------
# XGBoost Training
# ---------------------------------------------------------------------------


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: str,
    output_dir: Path,
    seed: int,
) -> dict:
    """Train and evaluate XGBoost."""
    import xgboost as xgb

    from negbiodb_ct.ct_features import build_xgboost_features

    train_X = build_xgboost_features(train_df, task)
    val_X = build_xgboost_features(val_df, task)
    test_X = build_xgboost_features(test_df, task)

    if task == "m1":
        train_y = train_df["Y"].values.astype(np.float32)
        val_y = val_df["Y"].values.astype(np.float32)
        test_y = test_df["Y"].values.astype(np.float32)
    else:
        train_y = train_df["failure_category_int"].values.astype(np.int64)
        val_y = val_df["failure_category_int"].values.astype(np.int64)
        test_y = test_df["failure_category_int"].values.astype(np.int64)

    params: dict = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 500,
        "early_stopping_rounds": 20,
        "tree_method": "hist",
        "random_state": seed,
        "verbosity": 0,
    }

    if task == "m1":
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        n_neg = (train_y == 0).sum()
        n_pos = (train_y == 1).sum()
        if n_pos > 0:
            params["scale_pos_weight"] = float(n_neg / n_pos)
    else:
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
        params["num_class"] = 8

    logger.info("XGBoost params: %s", {k: v for k, v in params.items() if k != "verbosity"})

    model = xgb.XGBClassifier(**params)
    model.fit(
        train_X, train_y,
        eval_set=[(val_X, val_y)],
        verbose=False,
    )

    # Save model
    model_path = output_dir / "best.json"
    model.save_model(str(model_path))
    logger.info("Saved XGBoost model → %s", model_path)

    # Evaluate
    if task == "m1":
        y_score = model.predict_proba(test_X)[:, 1]
        test_metrics = compute_m1_metrics(test_y, y_score)
    else:
        y_proba = model.predict_proba(test_X)
        y_pred = np.argmax(y_proba, axis=1)
        test_metrics = compute_m2_metrics(test_y, y_pred)

    return test_metrics


# ---------------------------------------------------------------------------
# MLP/GNN PyTorch Datasets + Collate
# ---------------------------------------------------------------------------


class CTTabularDataset:
    """Tabular dataset for MLP."""

    def __init__(self, df: pd.DataFrame, task: str) -> None:
        from negbiodb_ct.ct_features import build_mlp_features
        self.X = build_mlp_features(df, task).astype(np.float32)
        if task == "m1":
            self.y = df["Y"].values.astype(np.float32)
        else:
            self.y = df["failure_category_int"].values.astype(np.int64)
        self.task = task

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class CTGraphDataset:
    """Graph + tabular dataset for GNN+Tab. SMILES-only rows."""

    def __init__(
        self,
        df: pd.DataFrame,
        task: str,
        graph_cache: dict,
    ) -> None:
        from negbiodb_ct.ct_features import build_gnn_tab_features

        # Filter to SMILES-only
        has_smiles = df["smiles"].notna()
        self.df = df[has_smiles].reset_index(drop=True)
        if len(self.df) < len(df):
            logger.info(
                "GNN dataset: dropped %d rows without SMILES (%d → %d)",
                len(df) - len(self.df), len(df), len(self.df),
            )

        self.smiles = self.df["smiles"].tolist()
        self.tab = build_gnn_tab_features(self.df, task).astype(np.float32)
        if task == "m1":
            self.y = self.df["Y"].values.astype(np.float32)
        else:
            self.y = self.df["failure_category_int"].values.astype(np.int64)
        self.task = task
        self.graph_cache = graph_cache

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        graph = self.graph_cache.get(self.smiles[idx])
        return graph, self.tab[idx], self.y[idx]


def _collate_mlp(batch, device):
    import torch
    X_list, y_list = zip(*batch)
    X = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(y_list)).to(device)
    return X, y


def _collate_gnn(batch, device):
    import torch
    from torch_geometric.data import Batch, Data

    from negbiodb.models.graphdta import NODE_FEATURE_DIM

    graphs, tabs, labels = zip(*batch)

    # Placeholder for None graphs
    placeholder = Data(
        x=torch.zeros(1, NODE_FEATURE_DIM),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
    )
    graphs_clean = [g if g is not None else placeholder for g in graphs]

    drug_batch = Batch.from_data_list(graphs_clean).to(device)
    tab_tensor = torch.tensor(np.array(tabs), dtype=torch.float32).to(device)
    label_tensor = torch.tensor(np.array(labels)).to(device)
    return drug_batch, tab_tensor, label_tensor


def _prepare_ct_graph_cache(parquet_path: Path, cache_path: Path) -> dict:
    """Build/load graph cache for CT GNN training."""
    import torch
    from negbiodb.models.graphdta import smiles_to_graph

    smiles_series = pd.read_parquet(parquet_path, columns=["smiles"])["smiles"].dropna()
    smiles_list = smiles_series.unique().tolist()

    cache: dict = {}
    if cache_path.exists():
        logger.info("Loading graph cache: %s", cache_path)
        cache = torch.load(cache_path, weights_only=False)
        logger.info("Graph cache loaded: %d entries", len(cache))

    missing = [smi for smi in smiles_list if smi not in cache]
    if missing:
        logger.info("Building graphs for %d missing SMILES...", len(missing))
        failed = 0
        for smi in missing:
            g = smiles_to_graph(smi)
            cache[smi] = g
            if g is None:
                failed += 1
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        logger.info("Saved graph cache → %s (%d total, %d failed)", cache_path, len(cache), failed)

    return cache


# ---------------------------------------------------------------------------
# MLP/GNN Training Loop
# ---------------------------------------------------------------------------


def _run_epoch_mlp(model, loader, criterion, optimizer, device, is_train: bool, task: str):
    """Run one epoch for MLP."""
    import torch
    model.train(is_train)
    total_loss = 0.0
    all_labels = []
    all_preds = []
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            logits = model(X)
            if task == "m1":
                loss = criterion(logits, y.float())
            else:
                loss = criterion(logits, y.long())

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_labels.extend(y.cpu().tolist())
            if task == "m1":
                all_preds.extend(torch.sigmoid(logits).detach().cpu().tolist())
            else:
                all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, np.array(all_labels), np.array(all_preds)


def _run_epoch_gnn(model, loader, criterion, optimizer, device, is_train: bool, task: str):
    """Run one epoch for GNN+Tab."""
    import torch
    model.train(is_train)
    total_loss = 0.0
    all_labels = []
    all_preds = []
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for drug_batch, tab, y in loader:
            logits = model(drug_batch, tab)
            if task == "m1":
                loss = criterion(logits, y.float())
            else:
                loss = criterion(logits, y.long())

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_labels.extend(y.cpu().tolist())
            if task == "m1":
                all_preds.extend(torch.sigmoid(logits).detach().cpu().tolist())
            else:
                all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, np.array(all_labels), np.array(all_preds)


def _compute_val_metric(y_true, y_pred, task: str) -> float:
    """Compute primary validation metric."""
    if task == "m1":
        from negbiodb.metrics import auroc as compute_auroc
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return compute_auroc(y_true, y_pred)
    else:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true.astype(int), y_pred.astype(int), average="macro", zero_division=0))


def train_neural(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: str,
    output_dir: Path,
    args,
) -> dict:
    """Train and evaluate MLP or GNN+Tab."""
    import torch
    import torch.nn as nn

    from negbiodb_ct.ct_models import build_ct_model

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Build datasets
    if model_name == "mlp":
        train_ds = CTTabularDataset(train_df, task)
        val_ds = CTTabularDataset(val_df, task)
        test_ds = CTTabularDataset(test_df, task)

        from torch.utils.data import DataLoader
        collate_fn = lambda b: _collate_mlp(b, device)  # noqa: E731
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        run_epoch = _run_epoch_mlp
    else:
        # GNN: build graph cache
        parquet_name = _resolve_m1_parquet(args.dataset, args.negative) if task == "m1" else _M2_PARQUET
        parquet_path = args.data_dir / parquet_name
        cache_path = args.data_dir / "ct_graph_cache.pt"
        graph_cache = _prepare_ct_graph_cache(parquet_path, cache_path)

        train_ds = CTGraphDataset(train_df, task, graph_cache)
        val_ds = CTGraphDataset(val_df, task, graph_cache)
        test_ds = CTGraphDataset(test_df, task, graph_cache)

        from torch.utils.data import DataLoader
        collate_fn = lambda b: _collate_gnn(b, device)  # noqa: E731
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        run_epoch = _run_epoch_gnn

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Empty split detected (likely SMILES filtering for GNN)")

    # Build model
    model = build_ct_model(model_name, task).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s | task: %s | params: %d", model_name, task, n_params)

    # Loss function
    if task == "m1":
        criterion = nn.BCEWithLogitsLoss()
    else:
        # Compute class weights from actual training data
        # For GNN, use SMILES-only subset (matches CTGraphDataset filtering)
        if model_name == "gnn":
            train_labels = train_ds.y
        else:
            train_labels = train_df["failure_category_int"].values
        class_counts = np.bincount(train_labels, minlength=8).astype(np.float64)
        # pharmacokinetic (idx 7) may have 0 records → weight 0.0
        weights = np.zeros(8, dtype=np.float64)
        for c in range(8):
            if class_counts[c] > 0:
                weights[c] = class_counts.sum() / (8 * class_counts[c])
            else:
                weights[c] = 0.0
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info("M2 class weights: %s", {k: f"{weights[v]:.2f}" for k, v in CATEGORY_TO_INT.items()})

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_metric = float("-inf")
    patience_counter = 0
    training_log = []

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, True, task)
        val_loss, val_y, val_pred = run_epoch(model, val_loader, criterion, None, device, False, task)
        val_metric = _compute_val_metric(val_y, val_pred, task)

        metric_name = "val_auroc" if task == "m1" else "val_macro_f1"
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, metric_name: val_metric}
        training_log.append(row)

        logger.info(
            "Epoch %3d | train_loss=%.4f | val_loss=%.4f | %s=%.4f",
            epoch, train_loss, val_loss, metric_name, val_metric,
        )

        if not np.isnan(val_metric) and val_metric > best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
            logger.info("  Saved best checkpoint (%s=%.4f)", metric_name, best_val_metric)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                break

    # Save training log
    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
        writer.writeheader()
        writer.writerows(training_log)
    logger.info("Training log saved → %s", log_path)

    # Evaluate on test set
    checkpoint = output_dir / "best.pt"
    if not checkpoint.exists():
        logger.error("No checkpoint saved — all epochs produced NaN val metric.")
        return {}

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    _, test_y, test_pred = run_epoch(model, test_loader, criterion, None, device, False, task)

    if task == "m1":
        test_metrics = compute_m1_metrics(test_y, test_pred)
    else:
        test_metrics = compute_m2_metrics(test_y.astype(int), test_pred.astype(int))

    # Return actual dataset sizes (may differ from df sizes for GNN SMILES filtering)
    test_metrics["_n_train"] = len(train_ds)
    test_metrics["_n_val"] = len(val_ds)
    test_metrics["_n_test"] = len(test_ds)
    return test_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CT baseline model.")
    parser.add_argument("--model", required=True, choices=["xgboost", "mlp", "gnn"])
    parser.add_argument("--task", required=True, choices=["m1", "m2"])
    parser.add_argument("--split", required=True,
                        choices=["random", "cold_drug", "cold_condition",
                                 "temporal", "scaffold", "degree_balanced"])
    parser.add_argument("--dataset", default="balanced",
                        choices=["balanced", "realistic", "smiles_only"])
    parser.add_argument("--negative", default="negbiodb",
                        choices=["negbiodb", "uniform_random", "degree_matched"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=Path, default=ROOT / "exports" / "ct")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "ct_baselines")
    args = parser.parse_args(argv)

    set_seed(args.seed)

    # Build run name
    run_name = f"{args.model}_{args.task}_{args.dataset}_{args.split}_{args.negative}_seed{args.seed}"
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run: %s → %s", run_name, out_dir)

    # Load data
    try:
        train_df, val_df, test_df = load_data(args)
    except (ValueError, FileNotFoundError) as e:
        logger.error("%s", e)
        return 1

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        logger.error("Empty split detected.")
        return 1

    # Train
    if args.model == "xgboost":
        test_metrics = train_xgboost(train_df, val_df, test_df, args.task, out_dir, args.seed)
    else:
        test_metrics = train_neural(args.model, train_df, val_df, test_df, args.task, out_dir, args)

    if not test_metrics:
        logger.warning("No test metrics produced (single-class val/test set?).")
        null_metrics = {k: None for k in (
            ["auroc", "auprc", "mcc", "log_auc", "bedroc", "ef_1pct", "ef_5pct", "accuracy", "f1"]
            if args.task == "m1" else
            ["macro_f1", "weighted_f1", "mcc", "accuracy"]
        )}
        results = {
            "run_name": run_name,
            "model": args.model,
            "task": args.task,
            "split": args.split,
            "negative": args.negative,
            "dataset": args.dataset,
            "seed": args.seed,
            "test_metrics": null_metrics,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        }
        results_path = out_dir / "results.json"
        write_results_json(results_path, results)
        logger.info("Null results saved → %s", results_path)
        return 0

    # Save results — use actual dataset sizes from neural models (GNN filters SMILES)
    n_train = test_metrics.pop("_n_train", len(train_df))
    n_val = test_metrics.pop("_n_val", len(val_df))
    n_test = test_metrics.pop("_n_test", len(test_df))
    results = {
        "run_name": run_name,
        "model": args.model,
        "task": args.task,
        "split": args.split,
        "negative": args.negative,
        "dataset": args.dataset,
        "seed": args.seed,
        "test_metrics": test_metrics,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
    results_path = out_dir / "results.json"
    write_results_json(results_path, results)

    logger.info("Test metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            logger.info("  %-20s = %.4f", k, v if np.isfinite(v) else float("nan"))
        else:
            logger.info("  %-20s = %s", k, str(v)[:80])
    logger.info("Results saved → %s", results_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
