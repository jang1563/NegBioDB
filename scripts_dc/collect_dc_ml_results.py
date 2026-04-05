#!/usr/bin/env python3
"""Collect DC ML baseline results into summary tables.

Reads all results.json files from results/dc_baselines/ and produces:
  - results/dc_baselines/dc_table_m1.csv + .md  — M1 binary synergy metrics
  - results/dc_baselines/dc_table_m2.csv + .md  — M2 3-class synergy metrics

Usage:
    python scripts_dc/collect_dc_ml_results.py
    python scripts_dc/collect_dc_ml_results.py --results-dir results/dc_baselines --out results/
    python scripts_dc/collect_dc_ml_results.py --aggregate-seeds
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

# M1 metrics: binary synergy prediction (synergy vs. non-synergy)
M1_METRICS = ["auroc", "auprc", "mcc", "accuracy", "f1"]
M1_METRIC_NAMES = {
    "auroc": "AUROC", "auprc": "AUPRC", "mcc": "MCC",
    "accuracy": "Acc", "f1": "F1",
}

# M2 metrics: 3-class prediction (synergy / neutral / antagonism)
M2_METRICS = ["macro_auroc", "macro_f1", "weighted_f1", "mcc", "accuracy"]
M2_METRIC_NAMES = {
    "macro_auroc": "Macro-AUROC", "macro_f1": "Macro-F1",
    "weighted_f1": "Wtd-F1", "mcc": "MCC", "accuracy": "Acc",
}

MODEL_ORDER = ["xgboost", "mlp", "deepsynergy", "gnn"]
SPLIT_ORDER = [
    "random", "cold_compound", "cold_cell_line",
    "cold_both", "scaffold", "leave_one_tissue_out",
]


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
            "seed": data.get("seed", -1),
            "n_train": data.get("n_train", 0),
            "n_val": data.get("n_val", 0),
            "n_test": data.get("n_test", 0),
        }
        metrics = data.get("test_metrics", {})
        for m in M1_METRICS + M2_METRICS:
            row[m] = metrics.get(m, float("nan"))

        # M2 per-class accuracy (synergy / neutral / antagonism)
        per_class = metrics.get("per_class_accuracy", {})
        for cat in ("synergy", "neutral", "antagonism"):
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

    cols = ["model", "split", "seed", "n_test"] + M1_METRICS
    return m1[cols].sort_values(["model", "split", "seed"]).reset_index(drop=True)


def build_m2_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build M2 summary table."""
    m2 = df[df["task"] == "m2"].copy()
    if m2.empty:
        return pd.DataFrame()

    per_class_cols = [f"acc_{cat}" for cat in ("synergy", "neutral", "antagonism")]
    cols = ["model", "split", "seed", "n_test"] + M2_METRICS + per_class_cols
    available_cols = [c for c in cols if c in m2.columns]
    return m2[available_cols].sort_values(["model", "split", "seed"]).reset_index(drop=True)


def aggregate_over_seeds(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Aggregate metrics over seeds."""
    if df.empty:
        return pd.DataFrame()

    metrics = M1_METRICS if task == "m1" else M2_METRICS
    group_cols = ["model", "split"]

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


def format_m1_markdown(table: pd.DataFrame) -> str:
    """Format M1 table as Markdown."""
    lines = []
    headers = " | ".join(f"**{M1_METRIC_NAMES[m]}**" for m in M1_METRICS)
    lines.append(f"| **Model** | **Split** | **Seed** | {headers} |")
    lines.append("|" + "---|" * (3 + len(M1_METRICS)))

    for _, row in table.iterrows():
        vals = " | ".join(_fmt(row[m]) for m in M1_METRICS)
        lines.append(f"| {row['model']} | {row['split']} | {row['seed']} | {vals} |")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect DC ML baseline results.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "dc_baselines")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "dc_baselines")
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
        m1_table.to_csv(args.out / "dc_table_m1.csv", index=False)
        (args.out / "dc_table_m1.md").write_text(format_m1_markdown(m1_table))
        logger.info("M1 table: %d rows → dc_table_m1.csv + .md", len(m1_table))

        print("\n" + "=" * 60)
        print("DC-M1 Results (Binary Synergy)")
        print("=" * 60)
        display_cols = ["model", "split", "auroc", "auprc", "mcc"]
        avail = [c for c in display_cols if c in m1_table.columns]
        print(m1_table[avail].to_string(index=False))
    else:
        logger.info("No M1 results found.")

    # M2 table
    m2_table = build_m2_table(df)
    if not m2_table.empty:
        m2_table.to_csv(args.out / "dc_table_m2.csv", index=False)
        (args.out / "dc_table_m2.md").write_text(format_m2_markdown(m2_table))
        logger.info("M2 table: %d rows → dc_table_m2.csv + .md", len(m2_table))

        print("\n" + "=" * 60)
        print("DC-M2 Results (3-class Synergy)")
        print("=" * 60)
        display_cols = ["model", "split", "macro_auroc", "macro_f1", "weighted_f1", "mcc"]
        avail = [c for c in display_cols if c in m2_table.columns]
        print(m2_table[avail].to_string(index=False))
    else:
        logger.info("No M2 results found.")

    # Aggregated tables
    if args.aggregate_seeds:
        m1_df = df[df["task"] == "m1"]
        m2_df = df[df["task"] == "m2"]
        if not m1_df.empty:
            agg_m1 = aggregate_over_seeds(m1_df, "m1")
            agg_m1.to_csv(args.out / "dc_table_m1_aggregated.csv", index=False)
            logger.info("Aggregated M1 table saved.")
        if not m2_df.empty:
            agg_m2 = aggregate_over_seeds(m2_df, "m2")
            agg_m2.to_csv(args.out / "dc_table_m2_aggregated.csv", index=False)
            logger.info("Aggregated M2 table saved.")

    print("\n" + "=" * 60)
    logger.info("Done. %d total runs collected.", len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
