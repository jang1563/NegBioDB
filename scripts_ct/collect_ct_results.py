#!/usr/bin/env python3
"""Collect CT baseline results into summary tables.

Reads all results.json files from results/ct_baselines/ and produces:
  - results/ct_table_m1.csv + .md  — M1 metrics (model × split × negative)
  - results/ct_table_m2.csv + .md  — M2 metrics (model × split)

Usage:
    python scripts_ct/collect_ct_results.py
    python scripts_ct/collect_ct_results.py --results-dir results/ct_baselines --out results/
    python scripts_ct/collect_ct_results.py --aggregate-seeds
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

# M1 metrics (in display order)
M1_METRICS = ["auroc", "auprc", "mcc", "log_auc", "accuracy", "f1"]
M1_METRIC_NAMES = {
    "auroc": "AUROC", "auprc": "AUPRC", "mcc": "MCC",
    "log_auc": "LogAUC", "accuracy": "Acc", "f1": "F1",
}

# M2 metrics (in display order)
M2_METRICS = ["macro_f1", "weighted_f1", "mcc", "accuracy"]
M2_METRIC_NAMES = {
    "macro_f1": "Macro-F1", "weighted_f1": "Wtd-F1",
    "mcc": "MCC", "accuracy": "Acc",
}

MODEL_ORDER = ["xgboost", "mlp", "gnn"]
SPLIT_ORDER = ["random", "cold_drug", "cold_condition", "temporal", "scaffold", "degree_balanced"]
NEG_ORDER = ["negbiodb", "uniform_random", "degree_matched"]


def load_results(results_dir: Path) -> pd.DataFrame:
    """Walk results_dir and load all results.json files."""
    rows = []
    for json_path in sorted(results_dir.glob("*/results.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", json_path, e)
            continue

        row: dict = {
            "model": data.get("model", "?"),
            "task": data.get("task", "?"),
            "split": data.get("split", "?"),
            "negative": data.get("negative", "?"),
            "dataset": data.get("dataset", "?"),
            "seed": data.get("seed", -1),
            "n_train": data.get("n_train", 0),
            "n_val": data.get("n_val", 0),
            "n_test": data.get("n_test", 0),
        }
        metrics = data.get("test_metrics", {})
        for m in M1_METRICS + M2_METRICS:
            row[m] = metrics.get(m, float("nan"))

        # M2 per-class accuracy
        per_class = metrics.get("per_class_accuracy", {})
        for cat in ["efficacy", "enrollment", "other", "strategic", "safety", "design", "regulatory", "pharmacokinetic"]:
            row[f"acc_{cat}"] = per_class.get(cat, float("nan"))

        rows.append(row)

    if not rows:
        logger.warning("No results.json files found in %s", results_dir)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info("Loaded %d result files.", len(df))
    return df


def build_m1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build M1 summary table."""
    m1 = df[df["task"] == "m1"].copy()
    if m1.empty:
        return pd.DataFrame()

    cols = ["model", "dataset", "split", "negative", "seed", "n_test"] + M1_METRICS
    return m1[cols].sort_values(["model", "dataset", "split", "negative", "seed"]).reset_index(drop=True)


def build_m2_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build M2 summary table."""
    m2 = df[df["task"] == "m2"].copy()
    if m2.empty:
        return pd.DataFrame()

    per_class_cols = [f"acc_{cat}" for cat in ["efficacy", "enrollment", "other", "strategic", "safety", "design", "regulatory", "pharmacokinetic"]]
    cols = ["model", "split", "seed", "n_test"] + M2_METRICS + per_class_cols
    available_cols = [c for c in cols if c in m2.columns]
    return m2[available_cols].sort_values(["model", "split", "seed"]).reset_index(drop=True)


def aggregate_over_seeds(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Aggregate metrics over seeds."""
    if df.empty:
        return pd.DataFrame()

    metrics = M1_METRICS if task == "m1" else M2_METRICS
    group_cols = ["model", "split", "negative"] if task == "m1" else ["model", "split"]
    if task == "m1":
        group_cols.append("dataset")

    agg_map: dict = {"seed": "nunique", "n_test": "mean"}
    for m in metrics:
        agg_map[m] = ["mean", "std"]

    grouped = df.groupby(group_cols, dropna=False).agg(agg_map)
    grouped.columns = [
        "n_seeds" if col == ("seed", "nunique")
        else "n_test_mean" if col == ("n_test", "mean")
        else f"{col[0]}_{col[1]}"
        for col in grouped.columns.to_flat_index()
    ]
    return grouped.reset_index()


def _fmt(val: float) -> str:
    return f"{val:.3f}" if np.isfinite(val) else "—"


def _fmt_ms(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "—"
    if np.isnan(std):
        return f"{mean:.3f}"
    return f"{mean:.3f}±{std:.3f}"


def format_m1_markdown(table: pd.DataFrame) -> str:
    """Format M1 table as Markdown."""
    lines = []
    headers = " | ".join(f"**{M1_METRIC_NAMES[m]}**" for m in M1_METRICS)
    lines.append(f"| **Model** | **Dataset** | **Split** | **Negative** | **Seed** | {headers} |")
    lines.append("|" + "---|" * (5 + len(M1_METRICS)))

    for _, row in table.iterrows():
        vals = " | ".join(_fmt(row[m]) for m in M1_METRICS)
        lines.append(f"| {row['model']} | {row['dataset']} | {row['split']} | {row['negative']} | {row['seed']} | {vals} |")
    return "\n".join(lines)


def format_m2_markdown(table: pd.DataFrame) -> str:
    """Format M2 table as Markdown."""
    lines = []
    headers = " | ".join(f"**{M2_METRIC_NAMES[m]}**" for m in M2_METRICS)
    lines.append(f"| **Model** | **Split** | **Seed** | {headers} |")
    lines.append("|" + "---|" * (3 + len(M2_METRICS)))

    for _, row in table.iterrows():
        vals = " | ".join(_fmt(row[m]) for m in M2_METRICS)
        lines.append(f"| {row['model']} | {row['split']} | {row['seed']} | {vals} |")
    return "\n".join(lines)


def summarize_exp_ct1(df: pd.DataFrame, out_dir: Path) -> None:
    """Exp CT-1 inflation analysis: compare negbiodb vs random/degree-matched.

    Shows AUROC delta for each model under random split, M1 balanced.
    """
    m1_random = df[(df["task"] == "m1") & (df["split"] == "random") & (df["dataset"] == "balanced")]
    if m1_random.empty or len(m1_random["negative"].unique()) < 2:
        logger.info("Not enough Exp CT-1 data for inflation analysis.")
        return

    lines = [
        "# Exp CT-1: Negative Source Inflation Analysis",
        "",
        "AUROC by model × negative source (M1, random split, balanced):",
        "",
        "| Model | NegBioDB | Uniform Random | Degree Matched | Delta (DM - NB) |",
        "|-------|----------|----------------|----------------|-----------------|",
    ]
    for model in MODEL_ORDER:
        sub = m1_random[m1_random["model"] == model]
        if sub.empty:
            continue
        nb = sub[sub["negative"] == "negbiodb"]["auroc"].mean()
        ur = sub[sub["negative"] == "uniform_random"]["auroc"].mean()
        dm = sub[sub["negative"] == "degree_matched"]["auroc"].mean()
        delta = dm - nb if np.isfinite(dm) and np.isfinite(nb) else float("nan")
        lines.append(
            f"| {model} | {_fmt(nb)} | {_fmt(ur)} | {_fmt(dm)} | {_fmt(delta)} |"
        )

    text = "\n".join(lines)
    (out_dir / "ct_exp_ct1_inflation.md").write_text(text)
    logger.info("Exp CT-1 inflation analysis saved → ct_exp_ct1_inflation.md")
    print("\n" + text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect CT baseline results.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "ct_baselines")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--aggregate-seeds", action="store_true")
    args = parser.parse_args(argv)

    df = load_results(args.results_dir)
    if df.empty:
        logger.error("No results found.")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    # M1 table
    m1_table = build_m1_table(df)
    if not m1_table.empty:
        m1_table.to_csv(args.out / "ct_table_m1.csv", index=False)
        (args.out / "ct_table_m1.md").write_text(format_m1_markdown(m1_table))
        logger.info("M1 table: %d rows → ct_table_m1.csv + .md", len(m1_table))

        print("\n" + "=" * 60)
        print("CT-M1 Results")
        print("=" * 60)
        print(m1_table[["model", "dataset", "split", "negative", "auroc", "auprc", "mcc"]].to_string(index=False))
    else:
        logger.info("No M1 results found.")

    # M2 table
    m2_table = build_m2_table(df)
    if not m2_table.empty:
        m2_table.to_csv(args.out / "ct_table_m2.csv", index=False)
        (args.out / "ct_table_m2.md").write_text(format_m2_markdown(m2_table))
        logger.info("M2 table: %d rows → ct_table_m2.csv + .md", len(m2_table))

        print("\n" + "=" * 60)
        print("CT-M2 Results")
        print("=" * 60)
        print(m2_table[["model", "split", "macro_f1", "weighted_f1", "mcc"]].to_string(index=False))
    else:
        logger.info("No M2 results found.")

    # Aggregated tables
    if args.aggregate_seeds:
        m1_df = df[df["task"] == "m1"]
        m2_df = df[df["task"] == "m2"]
        if not m1_df.empty:
            agg_m1 = aggregate_over_seeds(m1_df, "m1")
            agg_m1.to_csv(args.out / "ct_table_m1_aggregated.csv", index=False)
            logger.info("Aggregated M1 table saved.")
        if not m2_df.empty:
            agg_m2 = aggregate_over_seeds(m2_df, "m2")
            agg_m2.to_csv(args.out / "ct_table_m2_aggregated.csv", index=False)
            logger.info("Aggregated M2 table saved.")

    # Exp CT-1 inflation analysis (compare negbiodb vs random/degree-matched)
    summarize_exp_ct1(df, args.out)

    print("\n" + "=" * 60)
    logger.info("Done. %d total runs collected.", len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
