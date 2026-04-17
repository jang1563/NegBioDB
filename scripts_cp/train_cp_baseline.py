#!/usr/bin/env python3
"""Train CP baseline models on exported Cell Painting benchmark datasets."""

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
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "exports" / "cp_ml"
RESULTS_DIR = PROJECT_ROOT / "results" / "cp_baselines"

LABEL_META_COLS = {
    "cp_result_id",
    "compound_id",
    "cell_line_id",
    "assay_context_id",
    "batch_id",
    "dose",
    "dose_unit",
    "timepoint_h",
    "num_observations",
    "num_valid_observations",
    "dmso_distance_mean",
    "replicate_reproducibility",
    "viability_ratio",
    "outcome_label",
    "confidence_tier",
    "has_orthogonal_evidence",
    "compound_name",
    "canonical_smiles",
    "inchikey",
    "inchikey_connectivity",
    "cell_line_name",
    "tissue",
    "disease",
    "batch_name",
    "Y",
}

FEATURE_META_COLS = {"cp_result_id", "feature_source", "storage_uri"}


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_label_frame(data_dir: Path, task: str) -> pd.DataFrame:
    name = "negbiodb_cp_pairs.parquet" if task == "m1" else "negbiodb_cp_m2.parquet"
    return pd.read_parquet(data_dir / name)


def _load_feature_frame(data_dir: Path, feature_set: str) -> pd.DataFrame:
    if feature_set == "profile":
        return pd.read_parquet(data_dir / "negbiodb_cp_profile_features.parquet")
    if feature_set == "image":
        return pd.read_parquet(data_dir / "negbiodb_cp_image_features.parquet")

    profile = pd.read_parquet(data_dir / "negbiodb_cp_profile_features.parquet").copy()
    image = pd.read_parquet(data_dir / "negbiodb_cp_image_features.parquet").copy()
    profile_cols = {
        col: (f"profile_{col}" if col not in FEATURE_META_COLS else col)
        for col in profile.columns
    }
    image_cols = {
        col: (f"image_{col}" if col not in FEATURE_META_COLS else col)
        for col in image.columns
    }
    return profile.rename(columns=profile_cols).merge(
        image.rename(columns=image_cols),
        on="cp_result_id",
        how="outer",
    )


def load_joined_dataset(data_dir: Path, task: str, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    labels = _load_label_frame(data_dir, task)
    features = _load_feature_frame(data_dir, feature_set)
    df = labels.merge(features, on="cp_result_id", how="left")
    feature_cols = [
        col for col in df.columns
        if col not in LABEL_META_COLS
        and not col.startswith("split_")
        and col not in FEATURE_META_COLS
        and np.issubdtype(df[col].dtype, np.number)
    ]
    return df, feature_cols


def _resolve_split_column(df: pd.DataFrame, split_name: str) -> str:
    exact = f"split_{split_name}"
    if exact in df.columns:
        return exact
    candidates = [col for col in df.columns if col.startswith("split_") and split_name in col]
    if not candidates:
        raise ValueError(
            f"No split column matching '{split_name}' found. "
            f"Available: {[c for c in df.columns if c.startswith('split_')]}"
        )
    return sorted(candidates)[0]


def _get_splits(df: pd.DataFrame, split_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        (df[split_col] == "train").to_numpy(),
        (df[split_col] == "val").to_numpy(),
        (df[split_col] == "test").to_numpy(),
    )


def _compute_metrics(y_true, y_pred, y_prob, batches: pd.Series | None = None) -> dict:
    labels = np.unique(y_true)
    matrix_labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }

    y_prob_arr = np.asarray(y_prob)
    if len(labels) < 2:
        result["macro_f1"] = float(result["accuracy"])
        result["balanced_accuracy"] = float(result["accuracy"])
        result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=matrix_labels).tolist()
        result["mcc"] = 0.0
        result["pr_auc"] = None
        result["auroc"] = None
        if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
            result["brier_score"] = float(brier_score_loss(y_true, y_prob_arr[:, 1]))
        result["note"] = "single class in test set"
    elif len(labels) == 2:
        result["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
        result["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        result["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=matrix_labels).tolist()
        if y_prob_arr.ndim == 2:
            y_prob_arr = y_prob_arr[:, 1]
        result["pr_auc"] = float(average_precision_score(y_true, y_prob_arr))
        result["auroc"] = float(roc_auc_score(y_true, y_prob_arr))
        result["brier_score"] = float(brier_score_loss(y_true, y_prob_arr))
    else:
        result["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
        result["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        result["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=matrix_labels).tolist()
        if y_prob_arr.ndim != 2:
            raise ValueError("Multiclass metrics require probability matrix of shape (N, C)")
        y_bin = label_binarize(y_true, classes=labels)
        result["pr_auc"] = float(average_precision_score(y_bin, y_prob_arr, average="macro"))
        result["auroc"] = float(roc_auc_score(y_bin, y_prob_arr, multi_class="ovr", average="macro"))

    if batches is not None:
        batch_metrics = {}
        for batch_name, idx in batches.groupby(batches).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            batch_metrics[str(batch_name)] = {
                "n": int(len(idx)),
                "accuracy": float(accuracy_score(y_true[idx], y_pred[idx])),
            }
        result["per_batch_accuracy"] = batch_metrics
    return result


def _train_xgboost(X_train, y_train, X_val, y_val, n_classes: int, seed: int):
    import xgboost as xgb

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": seed,
        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        "early_stopping_rounds": 20,
        "verbosity": 0,
    }
    if n_classes == 2:
        params["objective"] = "binary:logistic"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes

    model = xgb.XGBClassifier(**params)
    eval_set = [(X_val, y_val)] if len(y_val) else [(X_train, y_train)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


def _train_mlp(X_train, y_train, seed: int):
    y_train = np.asarray(y_train)
    labels, counts = np.unique(y_train, return_counts=True)
    use_early_stopping = len(X_train) >= 20 and (counts.min() if len(counts) else 0) >= 2
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=300,
        early_stopping=use_early_stopping,
        n_iter_no_change=15,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CP baseline model.")
    parser.add_argument("--model", required=True, choices=["xgboost", "mlp"])
    parser.add_argument("--task", required=True, choices=["m1", "m2"])
    parser.add_argument("--feature-set", required=True, choices=["profile", "image", "multimodal"])
    parser.add_argument("--split", required=True, choices=["random", "cold_compound", "scaffold", "batch_holdout"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-proxy-smoke", action="store_true")
    args = parser.parse_args(argv)

    meta_path = args.data_dir / "cp_ml_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if not meta.get("production_ready", True) and not args.allow_proxy_smoke:
            logger.error(
                "CP baseline training is blocked for plate_proxy exports. "
                "Re-run with --allow-proxy-smoke only for plumbing smoke validation."
            )
            return 1

    df, feature_cols = load_joined_dataset(args.data_dir, args.task, args.feature_set)
    if df.empty:
        logger.error("No rows found in CP export directory: %s", args.data_dir)
        return 1
    if not feature_cols:
        logger.error("No usable numeric %s features found in %s", args.feature_set, args.data_dir)
        return 1

    split_col = _resolve_split_column(df, args.split)
    if args.smoke_test:
        df = df.head(200).copy()

    valid = df["Y"].notna().copy()
    df = df[valid].copy()
    if df.empty:
        logger.error("No labeled rows available for task=%s", args.task)
        return 1

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y = df["Y"].to_numpy(dtype=int)

    train_mask, val_mask, test_mask = _get_splits(df, split_col)
    if not train_mask.any() or not test_mask.any():
        logger.error("Split %s does not contain both train and test rows", split_col)
        return 1

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_batches = df.loc[test_mask, "batch_name"].reset_index(drop=True)

    if len(np.unique(y_train)) < 2:
        logger.error("Training split has fewer than two classes for %s", split_col)
        return 1

    n_classes = len(np.unique(y))
    logger.info(
        "Training CP baseline model=%s task=%s features=%s split=%s rows(train=%d,val=%d,test=%d)",
        args.model, args.task, args.feature_set, split_col, len(y_train), len(y_val), len(y_test),
    )

    if args.model == "xgboost":
        model = _train_xgboost(X_train, y_train, X_val, y_val, n_classes=n_classes, seed=args.seed)
        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
    else:
        encoder = LabelEncoder()
        y_train_enc = encoder.fit_transform(y_train)
        model = _train_mlp(X_train, y_train_enc, seed=args.seed)
        y_pred_enc = model.predict(X_test)
        y_pred = encoder.inverse_transform(y_pred_enc)
        y_prob_enc = model.predict_proba(X_test)
        if len(encoder.classes_) == n_classes:
            y_prob = y_prob_enc
        else:
            y_prob = np.zeros((len(y_pred), n_classes), dtype=float)
            for idx, cls in enumerate(encoder.classes_):
                y_prob[:, int(cls)] = y_prob_enc[:, idx]

    metrics = _compute_metrics(y_test, y_pred, y_prob, batches=test_batches)

    run_name = f"{args.model}_{args.task}_{args.feature_set}_{args.split}_seed{args.seed}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "results.json", "w") as handle:
        json.dump(
            {
                "model": args.model,
                "task": args.task,
                "feature_set": args.feature_set,
                "split": split_col,
                "seed": args.seed,
                "n_features": len(feature_cols),
                **metrics,
            },
            handle,
            indent=2,
            default=_json_safe,
        )

    pred_df = pd.DataFrame(
        {
            "cp_result_id": df.loc[test_mask, "cp_result_id"].to_numpy(),
            "y_true": y_test,
            "y_pred": y_pred,
            "batch_name": test_batches.to_numpy(),
        }
    )
    pred_df.to_parquet(run_dir / "predictions.parquet", index=False)

    logger.info("Saved CP baseline results to %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
