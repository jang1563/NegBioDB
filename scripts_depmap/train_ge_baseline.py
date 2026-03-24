#!/usr/bin/env python
"""Train GE baseline models (XGBoost + MLP)."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train GE baseline models")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--task", type=str, default="m1", choices=["m1", "m2"])
    parser.add_argument("--split", type=str, default="random")
    parser.add_argument("--neg-source", type=str, default="negbiodb",
                        choices=["negbiodb", "uniform_random", "degree_matched"])
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "mlp"])
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(_PROJECT_ROOT / "results" / "ge"))
    # Data files for positives
    parser.add_argument("--gene-effect-file", type=str, default=None)
    parser.add_argument("--dependency-file", type=str, default=None)
    args = parser.parse_args()

    from negbiodb_depmap.depmap_db import get_connection
    from negbiodb_depmap.export import (
        build_ge_m1,
        build_ge_m2,
        export_ge_negatives,
        generate_uniform_random_negatives,
        load_essential_positives,
    )
    from negbiodb_depmap.ge_features import build_feature_matrix

    db_path = Path(args.db_path) if args.db_path else _PROJECT_ROOT / "data" / "negbiodb_depmap.db"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection(db_path)

    try:
        # Load negatives
        neg_query = """
        SELECT p.gene_id, p.cell_line_id, g.entrez_id, g.gene_symbol,
               c.model_id, p.gene_degree, p.cell_line_degree,
               p.mean_gene_effect, p.best_confidence
        FROM gene_cell_pairs p
        JOIN genes g ON p.gene_id = g.gene_id
        JOIN cell_lines c ON p.cell_line_id = c.cell_line_id
        """
        neg_df = pd.read_sql_query(neg_query, conn)

        # Load positives
        if args.gene_effect_file and args.dependency_file:
            pos_df = load_essential_positives(
                conn,
                Path(args.gene_effect_file),
                Path(args.dependency_file),
            )
        else:
            logger.warning("No gene-effect/dependency files provided. Using synthetic positives.")
            pos_df = pd.DataFrame(columns=neg_df.columns)

        if len(pos_df) == 0:
            logger.error("No positives loaded. Provide --gene-effect-file and --dependency-file.")
            sys.exit(1)

        # Build dataset
        if args.task == "m1":
            dataset = build_ge_m1(
                conn, pos_df, neg_df,
                balanced=args.balanced, seed=args.seed,
            )
        else:
            dataset = build_ge_m2(conn, pos_df, neg_df, seed=args.seed)

        # Build features
        X = build_feature_matrix(conn, dataset)
        y = dataset["label"].values

        # Split
        split_col = f"split_{args.split}_v1"
        if split_col not in dataset.columns:
            # Use random 70/10/20
            from negbiodb_depmap.llm_dataset import assign_splits
            dataset = assign_splits(dataset, seed=args.seed)
            split_col = "split"

        train_mask = dataset[split_col] == "train"
        val_mask = dataset[split_col] == "val"
        test_mask = dataset[split_col] == "test"

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        logger.info("Train: %d, Val: %d, Test: %d", len(y_train), len(y_val), len(y_test))

        # Train
        if args.model == "xgboost":
            from negbiodb_depmap.models.xgboost_ge import train_xgboost_ge, predict_xgboost_ge
            task_type = "binary" if args.task == "m1" else "multiclass"
            model = train_xgboost_ge(X_train, y_train, X_val, y_val, task=task_type, seed=args.seed)
            y_pred, y_prob = predict_xgboost_ge(model, X_test, task=task_type)
        else:
            from negbiodb_depmap.models.mlp_features import train_mlp_ge
            n_classes = 2 if args.task == "m1" else 3
            model, history = train_mlp_ge(X_train, y_train, X_val, y_val, n_classes=n_classes)
            import torch
            with torch.no_grad():
                logits = model(torch.tensor(X_test, dtype=torch.float32))
                y_prob = torch.softmax(logits, dim=1).numpy()
                y_pred = np.argmax(y_prob, axis=1)

        # Evaluate
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
        results = {
            "task": args.task,
            "model": args.model,
            "split": args.split,
            "neg_source": args.neg_source,
            "seed": args.seed,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
        }

        if args.task == "m1" and len(y_prob.shape) > 1:
            from sklearn.metrics import roc_auc_score
            try:
                results["auroc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
            except ValueError:
                results["auroc"] = None

        # Save
        run_name = f"{args.task}_{args.model}_{args.split}_{args.neg_source}_s{args.seed}"
        result_path = output_dir / f"{run_name}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Results: %s", json.dumps(results, indent=2))
        logger.info("Saved to %s", result_path)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
