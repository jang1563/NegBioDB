#!/usr/bin/env python3
"""Train VP baseline models (XGBoost, MLP, ESM2-VP, VariantGNN).

Usage:
    PYTHONPATH=src python scripts_vp/train_vp_baseline.py \
        --model xgboost --dataset m1_balanced --split random --seed 42

    PYTHONPATH=src python scripts_vp/train_vp_baseline.py \
        --model mlp --dataset m1_realistic --split cold_gene --seed 42

Output:
    results/vp_baselines/{model}_{dataset}_{split}_seed{seed}/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


META_COLS = {
    # Identifiers
    "pair_id", "variant_id", "disease_id", "clinvar_variation_id", "rs_id", "entrez_id",
    # Variant descriptors (strings, not numeric features)
    "chromosome", "ref_allele", "alt_allele", "variant_type",
    "hgvs_coding", "hgvs_protein", "consequence_type",
    "alphamissense_class",
    # Gene/disease descriptors
    "gene_symbol", "disease_name", "medgen_cui", "hgnc_id",
    "clingen_validity", "gene_moi",
    # Submission metadata
    "confidence_tier", "best_evidence_type", "best_classification",
    "num_submissions", "num_submitters", "has_conflict",
    # Label
    "Y",
}


def load_dataset_frame(parquet_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load a VP parquet export and return the dataframe plus feature columns."""
    df = pd.read_parquet(parquet_path)

    split_cols = {c for c in df.columns if c.startswith("split_")}
    feature_cols = [c for c in df.columns if c not in META_COLS and c not in split_cols]
    return df, feature_cols


def load_data(parquet_path: Path, split_col: str) -> tuple:
    """Load parquet and split into train/val/test."""
    df, feature_cols = load_dataset_frame(parquet_path)
    X = df[feature_cols].values.astype(np.float32)
    y = df["Y"].values.astype(int)

    train_mask, val_mask, test_mask = get_split_masks(df, split_col)

    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask],
        feature_cols,
    )


def get_split_masks(df: pd.DataFrame, split_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test boolean masks for a split column."""
    train_mask = (df[split_col] == "train").to_numpy()
    val_mask = (df[split_col] == "val").to_numpy()
    test_mask = (df[split_col] == "test").to_numpy()
    return train_mask, val_mask, test_mask


def _sanitize_dense_features(array: np.ndarray, fill_value: float) -> np.ndarray:
    """Replace NaN values for dense neural-network inputs."""
    return np.nan_to_num(array.astype(np.float32), nan=fill_value)


def load_esm2_inputs(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_col: str,
    embeddings_path: Path,
) -> tuple[np.ndarray, ...]:
    """Merge precomputed ESM2 embeddings with exported VP rows."""
    emb_df = pd.read_parquet(embeddings_path)
    esm_cols = [c for c in emb_df.columns if c.startswith("esm2_")]
    if "variant_id" not in emb_df.columns or not esm_cols:
        raise ValueError("ESM2 embeddings parquet must contain variant_id and esm2_* columns")

    merged = df.merge(emb_df[["variant_id", *esm_cols]], on="variant_id", how="left")
    merged[esm_cols] = merged[esm_cols].fillna(0.0)

    X_tab = _sanitize_dense_features(merged[feature_cols].to_numpy(), fill_value=-1.0)
    X_esm = merged[esm_cols].to_numpy(dtype=np.float32)
    y = merged["Y"].to_numpy(dtype=int)
    train_mask, val_mask, test_mask = get_split_masks(merged, split_col)

    return (
        X_tab[train_mask], X_esm[train_mask], y[train_mask],
        X_tab[val_mask], X_esm[val_mask], y[val_mask],
        X_tab[test_mask], X_esm[test_mask], y[test_mask],
        esm_cols,
    )


def load_gnn_inputs(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_col: str,
    graph_path: Path,
) -> tuple[np.ndarray, ...]:
    """Attach graph gene indices to exported VP rows."""
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    gene_to_idx = graph["gene_to_idx"]
    gene_features = np.asarray(graph["node_features"], dtype=np.float32)
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)

    gene_idx = df["gene_symbol"].map(gene_to_idx).fillna(-1).astype(int).to_numpy()
    X_tab = _sanitize_dense_features(df[feature_cols].to_numpy(), fill_value=-1.0)
    y = df["Y"].to_numpy(dtype=int)
    train_mask, val_mask, test_mask = get_split_masks(df, split_col)

    return (
        X_tab[train_mask], gene_idx[train_mask], y[train_mask],
        X_tab[val_mask], gene_idx[val_mask], y[val_mask],
        X_tab[test_mask], gene_idx[test_mask], y[test_mask],
        gene_features, edge_index,
    )


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute classification metrics for binary or multiclass tasks."""
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        return {
            "auroc": None, "auprc": None, "mcc": None,
            "f1": None, "accuracy": None,
            "n_test": int(len(y_true)),
            "note": "single class in test set",
        }

    y_prob_arr = np.asarray(y_prob)
    if n_classes == 2:
        if y_prob_arr.ndim == 2:
            y_prob_arr = y_prob_arr[:, 1]
        auroc = float(roc_auc_score(y_true, y_prob_arr))
        auprc = float(average_precision_score(y_true, y_prob_arr))
        f1 = float(f1_score(y_true, y_pred, zero_division=0.0))
    else:
        if y_prob_arr.ndim != 2:
            raise ValueError("Multiclass metrics require probability matrix of shape (N, C)")
        classes = np.unique(y_true)
        y_bin = label_binarize(y_true, classes=classes)
        auroc = float(roc_auc_score(y_bin, y_prob_arr, multi_class="ovr", average="macro"))
        auprc = float(average_precision_score(y_bin, y_prob_arr, average="macro"))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": f1,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train VP baseline model.")
    parser.add_argument("--model", required=True, choices=["xgboost", "mlp", "esm2", "gnn"])
    parser.add_argument("--dataset", required=True, help="e.g., m1_balanced, m1_realistic")
    parser.add_argument("--split", required=True,
                        choices=["random", "cold_gene", "cold_disease", "cold_both",
                                 "degree_balanced", "temporal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "exports" / "vp_ml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "vp_baselines")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    # ESM2-specific
    parser.add_argument("--esm2-embeddings", type=Path, default=None,
                        help="Path to ESM2 embeddings parquet")
    # GNN-specific
    parser.add_argument("--gene-graph", type=Path, default=None,
                        help="Path to gene graph pickle")
    args = parser.parse_args(argv)

    parquet_path = args.data_dir / f"negbiodb_vp_{args.dataset}.parquet"
    if not parquet_path.exists():
        logger.error("Dataset not found: %s", parquet_path)
        return 1

    # Split columns in the parquet use seed-suffixed names (e.g., split_random_s42, split_temporal).
    # Try exact match first, then look for a matching prefix using schema metadata only.
    split_col = f"split_{args.split}"
    import pyarrow.parquet as pq
    all_cols = pq.read_schema(parquet_path).names
    if split_col not in all_cols:
        candidates = [c for c in all_cols if c.startswith(f"split_{args.split}")]
        if len(candidates) == 1:
            split_col = candidates[0]
            logger.info("Resolved split column: %s -> %s", f"split_{args.split}", split_col)
        elif len(candidates) > 1:
            logger.error("Ambiguous split columns for '%s': %s", args.split, candidates)
            return 1
        else:
            logger.error("Split column 'split_%s' not found. Available: %s",
                         args.split, [c for c in all_cols if c.startswith("split_")])
            return 1
    run_name = f"{args.model}_{args.dataset}_{args.split}_seed{args.seed}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Training %s ===", run_name)

    df, feature_cols = load_dataset_frame(parquet_path)
    X_train, y_train, X_val, y_val, X_test, y_test, _ = load_data(parquet_path, split_col)
    logger.info("Train: %d, Val: %d, Test: %d", len(y_train), len(y_val), len(y_test))

    n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))

    if args.model == "xgboost":
        from negbiodb_vp.models.xgboost_vp import train_xgboost_vp, predict_xgboost_vp

        task = "binary" if n_classes == 2 else "multiclass"
        model = train_xgboost_vp(X_train, y_train, X_val, y_val, task=task, seed=args.seed)
        y_pred, y_prob = predict_xgboost_vp(model, X_test)

    elif args.model == "mlp":
        from negbiodb_vp.models.mlp_features import train_mlp_vp

        # Replace NaN with sentinel for MLP
        X_train_c = np.nan_to_num(X_train, nan=-1.0)
        X_val_c = np.nan_to_num(X_val, nan=-1.0)
        X_test_c = np.nan_to_num(X_test, nan=-1.0)

        model, history = train_mlp_vp(
            X_train_c, y_train, X_val_c, y_val,
            n_classes=n_classes, epochs=args.epochs, device=args.device,
        )
        import torch
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test_c, dtype=torch.float32).to(args.device)
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1)
            y_prob = probs[:, 1].cpu().numpy() if n_classes == 2 else probs.cpu().numpy()
            y_pred = logits.argmax(1).cpu().numpy()

    elif args.model == "esm2":
        if args.esm2_embeddings is None:
            logger.error("ESM2 training requires --esm2-embeddings")
            return 1
        if not args.esm2_embeddings.exists():
            logger.error("ESM2 embeddings not found: %s", args.esm2_embeddings)
            return 1

        from negbiodb_vp.models.esm2_vp import train_esm2_vp
        import torch

        (
            X_tab_train, X_esm_train, y_train,
            X_tab_val, X_esm_val, y_val,
            X_tab_test, X_esm_test, y_test,
            esm_cols,
        ) = load_esm2_inputs(df, feature_cols, split_col, args.esm2_embeddings)
        logger.info("Loaded %d ESM2 dimensions from %s", len(esm_cols), args.esm2_embeddings)

        model, history = train_esm2_vp(
            X_tab_train, X_esm_train, y_train,
            X_tab_val, X_esm_val, y_val,
            n_classes=n_classes, epochs=args.epochs, device=args.device,
        )
        model.eval()
        with torch.no_grad():
            X_tab_t = torch.tensor(X_tab_test, dtype=torch.float32).to(args.device)
            X_esm_t = torch.tensor(X_esm_test, dtype=torch.float32).to(args.device)
            logits = model(X_tab_t, X_esm_t)
            probs = torch.softmax(logits, dim=1)
            y_prob = probs[:, 1].cpu().numpy() if n_classes == 2 else probs.cpu().numpy()
            y_pred = logits.argmax(1).cpu().numpy()

    elif args.model == "gnn":
        if args.gene_graph is None:
            logger.error("GNN training requires --gene-graph")
            return 1
        if not args.gene_graph.exists():
            logger.error("Gene graph not found: %s", args.gene_graph)
            return 1

        from negbiodb_vp.models.variant_gnn import predict_variant_gnn, train_variant_gnn

        (
            X_tab_train, gene_idx_train, y_train,
            X_tab_val, gene_idx_val, y_val,
            X_tab_test, gene_idx_test, y_test,
            gene_features, edge_index,
        ) = load_gnn_inputs(df, feature_cols, split_col, args.gene_graph)

        model, history = train_variant_gnn(
            X_tab_train, gene_idx_train, y_train,
            gene_features, edge_index,
            X_tab_val, gene_idx_val, y_val,
            n_classes=n_classes, epochs=args.epochs, device=args.device,
        )
        y_pred, y_prob = predict_variant_gnn(
            model, X_tab_test, gene_idx_test, gene_features, edge_index, device=args.device,
        )

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics.update({
        "task": "vp_binary" if n_classes == 2 else f"vp_{n_classes}way",
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    })
    if args.model in {"mlp", "esm2", "gnn"}:
        metrics["epochs_ran"] = len(history.get("train_loss", []))

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Results: AUROC=%.4f, MCC=%.4f, F1=%.4f",
                metrics.get("auroc") or 0, metrics.get("mcc") or 0, metrics.get("f1") or 0)
    logger.info("Saved to %s", results_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
