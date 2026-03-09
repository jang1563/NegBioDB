#!/usr/bin/env python3
"""Collect baseline results into a Table 1 summary.

Reads all results.json files from results/baselines/
and produces:
  - results/table1.csv   — full metrics table (model × split × negative)
  - results/table1.md    — Markdown formatted table (for paper draft)

Usage:
    uv run python scripts/collect_results.py
    uv run python scripts/collect_results.py --results-dir results/baselines --out results/
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

# Metrics to include in Table 1 (in display order)
TABLE_METRICS = ["log_auc", "auprc", "bedroc", "ef_1pct", "ef_5pct", "mcc", "auroc"]
TABLE_METRIC_NAMES = {
    "log_auc":  "LogAUC",
    "auprc":    "AUPRC",
    "bedroc":   "BEDROC",
    "ef_1pct":  "EF@1%",
    "ef_5pct":  "EF@5%",
    "mcc":      "MCC",
    "auroc":    "AUROC",
}

# Display order for models/splits
MODEL_ORDER = ["deepdta", "graphdta", "drugban"]
SPLIT_ORDER = ["random", "cold_compound", "cold_target", "ddb"]
NEG_ORDER   = ["negbiodb", "uniform_random", "degree_matched"]


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
            "model":    data.get("model", "?"),
            "split":    data.get("split", "?"),
            "negative": data.get("negative", "?"),
            "dataset":  data.get("dataset", "?"),
            "seed":     data.get("seed", -1),
            "n_train":  data.get("n_train", 0),
            "n_val":    data.get("n_val", 0),
            "n_test":   data.get("n_test", 0),
            "best_val_log_auc": data.get("best_val_log_auc", float("nan")),
        }
        metrics = data.get("test_metrics", {})
        for m in TABLE_METRICS:
            row[m] = metrics.get(m, float("nan"))
        rows.append(row)

    if not rows:
        logger.warning("No results.json files found in %s", results_dir)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info("Loaded %d result files.", len(df))
    return df


def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    """Build Table 1: model × split × negative source with all metrics."""
    # Order categories
    model_cat = pd.CategoricalDtype(MODEL_ORDER + [m for m in df["model"].unique() if m not in MODEL_ORDER], ordered=True)
    split_cat = pd.CategoricalDtype(SPLIT_ORDER + [s for s in df["split"].unique() if s not in SPLIT_ORDER], ordered=True)
    neg_cat   = pd.CategoricalDtype(NEG_ORDER   + [n for n in df["negative"].unique() if n not in NEG_ORDER], ordered=True)

    df = df.copy()
    df["model"]    = df["model"].astype(model_cat)
    df["split"]    = df["split"].astype(split_cat)
    df["negative"] = df["negative"].astype(neg_cat)

    cols = ["model", "split", "negative", "n_test"] + TABLE_METRICS
    return df[cols].sort_values(["model", "split", "negative"]).reset_index(drop=True)


def format_markdown(table: pd.DataFrame) -> str:
    """Format Table 1 as Markdown for the paper draft."""
    lines: list[str] = []

    # Header
    metric_headers = " | ".join(f"**{TABLE_METRIC_NAMES[m]}**" for m in TABLE_METRICS)
    lines.append(f"| **Model** | **Split** | **Negatives** | {metric_headers} |")
    lines.append("|" + "---|" * (3 + len(TABLE_METRICS)))

    for _, row in table.iterrows():
        metric_vals = " | ".join(
            f"{row[m]:.3f}" if not np.isnan(row[m]) else "—"
            for m in TABLE_METRICS
        )
        lines.append(f"| {row['model']} | {row['split']} | {row['negative']} | {metric_vals} |")

    return "\n".join(lines)


def summarize_exp1(df: pd.DataFrame) -> str:
    """Compute inflation percentages for the abstract (Exp 1 key result)."""
    if df.empty:
        return "No Exp 1 results available."

    lines = ["### Exp 1: NegBioDB vs. Random Negative Inflation"]
    for model in MODEL_ORDER:
        m_df = df[df["model"] == model]
        if m_df.empty:
            continue
        try:
            neg_val   = m_df[m_df["negative"] == "negbiodb"]  ["log_auc"].values[0]
            uni_val   = m_df[m_df["negative"] == "uniform_random"]["log_auc"].values[0]
            deg_val   = m_df[m_df["negative"] == "degree_matched"]["log_auc"].values[0]
            uni_inf   = 100 * (uni_val - neg_val) / max(abs(neg_val), 1e-9)
            deg_inf   = 100 * (deg_val - neg_val) / max(abs(neg_val), 1e-9)
            lines.append(
                f"  {model:10}: NegBioDB={neg_val:.3f}  uniform_random={uni_val:.3f}"
                f" (+{uni_inf:.1f}%)  degree_matched={deg_val:.3f} (+{deg_inf:.1f}%)"
            )
        except (IndexError, KeyError):
            lines.append(f"  {model:10}: incomplete")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect baseline results into Table 1.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "baselines")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    df = load_results(args.results_dir)
    if df.empty:
        logger.error("No results found. Run training jobs first.")
        return 1

    table = build_table1(df)

    # Save CSV
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "table1.csv"
    table.to_csv(csv_path, index=False)
    logger.info("Table 1 saved → %s", csv_path)

    # Save Markdown
    md_path = args.out / "table1.md"
    md_content = format_markdown(table)
    md_path.write_text(md_content)
    logger.info("Table 1 Markdown saved → %s", md_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Table 1 Summary")
    print("=" * 60)
    print(table[["model", "split", "negative", "log_auc", "auprc", "mcc"]].to_string(index=False))
    print()

    # Exp 1 inflation analysis
    exp1_df = df[df["split"] == "random"]
    print(summarize_exp1(exp1_df))
    print("=" * 60)

    logger.info("Done. %d completed runs.", len(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
