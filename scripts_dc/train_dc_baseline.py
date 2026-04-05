#!/usr/bin/env python3
"""Train DC baseline models (XGBoost, MLP, DeepSynergy, DrugCombGNN).

Usage:
    PYTHONPATH=src python scripts_dc/train_dc_baseline.py \
        --model xgboost --task m1 --split random --seed 42

    PYTHONPATH=src python scripts_dc/train_dc_baseline.py \
        --model deepsynergy --task m1 --split cold_compound --seed 42 --device cuda

Output:
    results/dc_baselines/{model}_{task}_{split}_seed{seed}/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
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

# Columns that are metadata, not features
META_COLS = {
    "pair_id", "compound_a_id", "compound_b_id",
    "drug_a_name", "drug_b_name", "smiles_a", "smiles_b",
    "pubchem_cid_a", "pubchem_cid_b",
    "consensus_class", "confidence_tier", "Y",
}

# Split column prefix for matching
SPLIT_PREFIX = "split_"


def load_dataset(parquet_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load DC parquet export and return DataFrame plus feature column names."""
    df = pd.read_parquet(parquet_path)
    split_cols = {c for c in df.columns if c.startswith(SPLIT_PREFIX)}
    feature_cols = [c for c in df.columns if c not in META_COLS and c not in split_cols]
    return df, feature_cols


def get_split_masks(df: pd.DataFrame, split_col: str):
    """Return train/val/test boolean masks for a split column."""
    train_mask = (df[split_col] == "train").to_numpy()
    val_mask = (df[split_col] == "val").to_numpy()
    test_mask = (df[split_col] == "test").to_numpy()
    return train_mask, val_mask, test_mask


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
            raise ValueError("Multiclass metrics require probability matrix (N, C)")
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
    parser = argparse.ArgumentParser(description="Train DC baseline model.")
    parser.add_argument("--model", required=True,
                        choices=["xgboost", "mlp", "deepsynergy", "gnn"])
    parser.add_argument("--task", required=True, choices=["m1", "m2"],
                        help="m1=binary (antag+add vs syn), m2=3-class")
    parser.add_argument("--split", required=True,
                        choices=["random", "cold_compound", "cold_cell_line",
                                 "cold_both", "scaffold", "leave_one_tissue_out"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path,
                        default=PROJECT_ROOT / "exports" / "dc_ml")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "dc_baselines")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Use first 200 rows for quick validation")
    args = parser.parse_args(argv)

    parquet_path = args.data_dir / f"negbiodb_dc_{args.task}.parquet"
    if not parquet_path.exists():
        logger.error("Dataset not found: %s", parquet_path)
        return 1

    # Find split column by partial match
    df, feature_cols = load_dataset(parquet_path)
    split_candidates = [c for c in df.columns if c.startswith(SPLIT_PREFIX)
                        and args.split in c]
    if not split_candidates:
        logger.error("No split column matching '%s' found. Available: %s",
                     args.split, [c for c in df.columns if c.startswith(SPLIT_PREFIX)])
        return 1
    split_col = split_candidates[0]

    if args.smoke_test:
        df = df.head(200)

    run_name = f"{args.model}_{args.task}_{args.split}_seed{args.seed}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Training %s ===", run_name)

    # Prepare labels
    from negbiodb_dc.export import build_dc_m1_labels, build_dc_m2_labels
    if args.task == "m1":
        df["Y"] = build_dc_m1_labels(df)
        n_classes = 2
    else:
        df["Y"] = build_dc_m2_labels(df)
        n_classes = 3

    # Drop rows with missing labels
    valid_mask = df["Y"].notna()
    df = df[valid_mask].copy()
    df["Y"] = df["Y"].astype(int)

    train_mask, val_mask, test_mask = get_split_masks(df, split_col)
    X = df[feature_cols].values.astype(np.float32)
    y = df["Y"].values

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info("Train: %d, Val: %d, Test: %d", len(y_train), len(y_val), len(y_test))

    if args.model == "xgboost":
        from negbiodb_dc.models.xgboost_dc import train_xgboost_dc, predict_xgboost_dc

        task_type = "binary" if n_classes == 2 else "multiclass"
        model = train_xgboost_dc(X_train, y_train, X_val, y_val,
                                 task=task_type, seed=args.seed)
        y_pred, y_prob = predict_xgboost_dc(model, X_test, task=task_type)

    elif args.model == "mlp":
        from negbiodb_dc.models.mlp_dc import train_mlp_dc
        import torch

        X_train_c = np.nan_to_num(X_train, nan=-1.0)
        X_val_c = np.nan_to_num(X_val, nan=-1.0)
        X_test_c = np.nan_to_num(X_test, nan=-1.0)

        model, history = train_mlp_dc(
            X_train_c, y_train, X_val_c, y_val,
            n_classes=n_classes, epochs=args.epochs, device=args.device,
        )
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test_c, dtype=torch.float32).to(args.device)
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1)
            y_prob = probs[:, 1].cpu().numpy() if n_classes == 2 else probs.cpu().numpy()
            y_pred = logits.argmax(1).cpu().numpy()

    elif args.model == "deepsynergy":
        from negbiodb_dc.models.deepsynergy_dc import train_deepsynergy_dc
        import torch

        # DeepSynergy uses raw Morgan FP concatenation (Drug A + Drug B)
        # Need to rebuild FP features from SMILES
        from negbiodb_dc.dc_features import compute_morgan_fp

        fp_bits = 2048
        smiles_a = df["smiles_a"].tolist()
        smiles_b = df["smiles_b"].tolist()

        X_ds = np.zeros((len(df), fp_bits * 2), dtype=np.float32)
        for i, (sa, sb) in enumerate(zip(smiles_a, smiles_b)):
            X_ds[i, :fp_bits] = compute_morgan_fp(sa, fp_bits)
            X_ds[i, fp_bits:] = compute_morgan_fp(sb, fp_bits)

        model, history = train_deepsynergy_dc(
            X_ds[train_mask], y_train,
            X_ds[val_mask], y_val,
            n_classes=n_classes, epochs=args.epochs, device=args.device,
        )
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_ds[test_mask], dtype=torch.float32).to(args.device)
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1)
            y_prob = probs[:, 1].cpu().numpy() if n_classes == 2 else probs.cpu().numpy()
            y_pred = logits.argmax(1).cpu().numpy()

    elif args.model == "gnn":
        from negbiodb_dc.models.drugcomb_gnn import (
            prepare_graph_pairs, train_drugcomb_gnn,
        )
        import torch

        smiles_a = df["smiles_a"].tolist()
        smiles_b = df["smiles_b"].tolist()
        graphs_a, graphs_b, valid = prepare_graph_pairs(smiles_a, smiles_b)

        valid_arr = np.array(valid)
        # Filter to valid graph pairs
        df_valid = df[valid_arr].copy()
        y_valid = df_valid["Y"].values
        train_m = (df_valid[split_col] == "train").to_numpy()
        val_m = (df_valid[split_col] == "val").to_numpy()
        test_m = (df_valid[split_col] == "test").to_numpy()

        train_idx = np.where(train_m)[0].tolist()
        val_idx = np.where(val_m)[0].tolist()
        test_idx = np.where(test_m)[0].tolist()

        model, history = train_drugcomb_gnn(
            [graphs_a[i] for i in train_idx],
            [graphs_b[i] for i in train_idx],
            y_valid[train_m],
            [graphs_a[i] for i in val_idx],
            [graphs_b[i] for i in val_idx],
            y_valid[val_m],
            n_classes=n_classes, epochs=args.epochs, device=args.device,
        )

        from torch_geometric.data import Batch
        model.eval()
        with torch.no_grad():
            ba = Batch.from_data_list([graphs_a[i] for i in test_idx]).to(args.device)
            bb = Batch.from_data_list([graphs_b[i] for i in test_idx]).to(args.device)
            logits = model(ba, bb)
            probs = torch.softmax(logits, dim=1)
            y_prob = probs[:, 1].cpu().numpy() if n_classes == 2 else probs.cpu().numpy()
            y_pred = logits.argmax(1).cpu().numpy()

        y_test = y_valid[test_m]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics.update({
        "task": f"dc_{args.task}",
        "model": args.model,
        "split": args.split,
        "seed": args.seed,
        "n_train": int(len(y_train)) if args.model != "gnn" else int(sum(train_m)),
        "n_val": int(len(y_val)) if args.model != "gnn" else int(sum(val_m)),
    })
    if args.model in {"mlp", "deepsynergy", "gnn"}:
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
