#!/usr/bin/env python3
"""Collect baseline results into a Table 1 summary.

Reads all results.json files from results/baselines/
and produces:
  - results/table1.csv   — full metrics table (model × split × negative)
  - results/table1.md    — Markdown formatted table (for paper draft)

Usage:
    uv run python scripts/collect_results.py
    uv run python scripts/collect_results.py --results-dir results/baselines --out results/
    uv run python scripts/collect_results.py --dataset balanced --seed 42
    uv run python scripts/collect_results.py --aggregate-seeds --dataset balanced
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
DEFAULT_DDB_REFERENCE = ROOT / "exports" / "negbiodb_m1_balanced_ddb.parquet"

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


def _canonical_run_name(model: str, dataset: str, split: str, negative: str, seed: int) -> str:
    return f"{model}_{dataset}_{split}_{negative}_seed{seed}"


def _run_artifact_mtime_ns(run_dir: Path, results_json: Path) -> int:
    """Return the newest training artifact timestamp for a run directory.

    Prefer checkpoint/log timestamps over results.json so eval-only rewrites do
    not accidentally make an old checkpoint look fresh.
    """
    training_artifacts = [
        run_dir / "best.pt",
        run_dir / "last.pt",
        run_dir / "training_log.csv",
    ]
    mtimes = [path.stat().st_mtime_ns for path in training_artifacts if path.exists()]
    if mtimes:
        return max(mtimes)
    return results_json.stat().st_mtime_ns


def _drop_stale_ddb_results(df: pd.DataFrame, ddb_reference: Path | None) -> pd.DataFrame:
    """Drop DDB runs trained before the current full-task DDB parquet existed."""
    if ddb_reference is None or df.empty or not ddb_reference.exists():
        return df

    ddb_mask = (df["dataset"] == "balanced") & (df["split"] == "ddb")
    if not ddb_mask.any():
        return df

    reference_mtime_ns = ddb_reference.stat().st_mtime_ns
    stale_mask = ddb_mask & (df["_run_mtime_ns"] < reference_mtime_ns)
    if not stale_mask.any():
        return df

    stale_paths = df.loc[stale_mask, "_results_path"].tolist()
    logger.warning(
        "Dropping %d stale DDB runs older than %s; retrain these models with the regenerated full-task DDB split.",
        int(stale_mask.sum()),
        ddb_reference,
    )
    for path in stale_paths[:3]:
        logger.warning("Stale DDB run excluded: %s", path)
    if len(stale_paths) > 3:
        logger.warning("... plus %d additional stale DDB runs.", len(stale_paths) - 3)

    return df.loc[~stale_mask].reset_index(drop=True)


def load_results(results_dir: Path, ddb_reference: Path | None = None) -> pd.DataFrame:
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
            "_run_name": data.get("run_name", ""),
            "_results_path": str(json_path),
            "_run_mtime_ns": _run_artifact_mtime_ns(json_path.parent, json_path),
        }
        metrics = data.get("test_metrics", {})
        for m in TABLE_METRICS:
            row[m] = metrics.get(m, float("nan"))
        seed_value = row["seed"] if row["seed"] is not None else -1
        row["_canonical_run_name"] = _canonical_run_name(
            row["model"],
            row["dataset"],
            row["split"],
            row["negative"],
            int(seed_value),
        )
        rows.append(row)

    if not rows:
        logger.warning("No results.json files found in %s", results_dir)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric_cols = ["seed", "n_train", "n_val", "n_test", "best_val_log_auc"] + TABLE_METRICS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _drop_stale_ddb_results(df, ddb_reference)
    if df.empty:
        logger.warning("All result rows were filtered out.")
        return df

    df["_is_canonical_run_name"] = df["_run_name"] == df["_canonical_run_name"]
    dedup_cols = ["model", "dataset", "seed", "split", "negative"]
    df = df.sort_values(
        dedup_cols + ["_is_canonical_run_name", "_run_mtime_ns", "_results_path"],
        ascending=[True, True, True, True, True, False, False, False],
    ).reset_index(drop=True)
    duplicate_mask = df.duplicated(subset=dedup_cols, keep="first")
    if duplicate_mask.any():
        logger.warning(
            "Dropping %d duplicate result rows with identical logical settings; keeping canonical/newest.",
            int(duplicate_mask.sum()),
        )
        df = df[~duplicate_mask].reset_index(drop=True)

    logger.info("Loaded %d result files.", len(df))
    return df.drop(columns=["_run_name", "_results_path", "_run_mtime_ns", "_canonical_run_name", "_is_canonical_run_name"])


def filter_results(
    df: pd.DataFrame,
    dataset: str | None = None,
    seeds: list[int] | None = None,
    models: list[str] | None = None,
    splits: list[str] | None = None,
    negatives: list[str] | None = None,
) -> pd.DataFrame:
    """Filter collected results to a controlled subset."""
    filtered = df.copy()
    if dataset is not None:
        filtered = filtered[filtered["dataset"] == dataset]
    if seeds:
        filtered = filtered[filtered["seed"].isin(seeds)]
    if models:
        filtered = filtered[filtered["model"].isin(models)]
    if splits:
        filtered = filtered[filtered["split"].isin(splits)]
    if negatives:
        filtered = filtered[filtered["negative"].isin(negatives)]
    return filtered.reset_index(drop=True)


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

    cols = ["model", "dataset", "seed", "split", "negative", "n_test"] + TABLE_METRICS
    return df[cols].sort_values(["model", "dataset", "seed", "split", "negative"]).reset_index(drop=True)


def aggregate_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics over seeds for identical experiment settings."""
    if df.empty:
        return pd.DataFrame()

    group_cols = ["model", "dataset", "split", "negative"]
    agg_map: dict[str, list[str] | str] = {
        "seed": "nunique",
        "n_test": "mean",
    }
    for metric in TABLE_METRICS:
        agg_map[metric] = ["mean", "std"]

    grouped = df.groupby(group_cols, dropna=False).agg(agg_map)
    grouped.columns = [
        "n_seeds" if col == ("seed", "nunique")
        else "n_test_mean" if col == ("n_test", "mean")
        else f"{col[0]}_{col[1]}"
        for col in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index()

    model_cat = pd.CategoricalDtype(MODEL_ORDER + [m for m in grouped["model"].unique() if m not in MODEL_ORDER], ordered=True)
    split_cat = pd.CategoricalDtype(SPLIT_ORDER + [s for s in grouped["split"].unique() if s not in SPLIT_ORDER], ordered=True)
    neg_cat = pd.CategoricalDtype(NEG_ORDER + [n for n in grouped["negative"].unique() if n not in NEG_ORDER], ordered=True)
    grouped["model"] = grouped["model"].astype(model_cat)
    grouped["split"] = grouped["split"].astype(split_cat)
    grouped["negative"] = grouped["negative"].astype(neg_cat)
    return grouped.sort_values(group_cols).reset_index(drop=True)


def format_markdown(table: pd.DataFrame) -> str:
    """Format Table 1 as Markdown for the paper draft."""
    lines: list[str] = []

    # Header
    metric_headers = " | ".join(f"**{TABLE_METRIC_NAMES[m]}**" for m in TABLE_METRICS)
    lines.append(f"| **Model** | **Dataset** | **Seed** | **Split** | **Negatives** | {metric_headers} |")
    lines.append("|" + "---|" * (5 + len(TABLE_METRICS)))

    for _, row in table.iterrows():
        metric_vals = " | ".join(
            f"{row[m]:.3f}" if not np.isnan(row[m]) else "—"
            for m in TABLE_METRICS
        )
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['seed']} | "
            f"{row['split']} | {row['negative']} | {metric_vals} |"
        )

    return "\n".join(lines)


def _fmt_mean_std(mean: float, std: float) -> str:
    """Format an aggregated metric as mean +/- std."""
    if np.isnan(mean):
        return "—"
    if np.isnan(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def format_aggregated_markdown(table: pd.DataFrame) -> str:
    """Format the seed-aggregated table as Markdown."""
    lines: list[str] = []

    metric_headers = " | ".join(f"**{TABLE_METRIC_NAMES[m]}**" for m in TABLE_METRICS)
    lines.append(f"| **Model** | **Dataset** | **Split** | **Negatives** | **Seeds** | {metric_headers} |")
    lines.append("|" + "---|" * (5 + len(TABLE_METRICS)))

    for _, row in table.iterrows():
        metric_vals = " | ".join(
            _fmt_mean_std(row[f"{m}_mean"], row.get(f"{m}_std", float("nan")))
            for m in TABLE_METRICS
        )
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['split']} | "
            f"{row['negative']} | {int(row['n_seeds'])} | {metric_vals} |"
        )

    return "\n".join(lines)


def summarize_exp1(df: pd.DataFrame) -> str:
    """Compute inflation percentages for the abstract (Exp 1 key result)."""
    if df.empty:
        return "No Exp 1 results available."

    rows = _compute_exp1_summary_rows(df)
    lines = ["### Exp 1: NegBioDB vs. Random Negative Inflation"]
    current_dataset = None
    for row in rows:
        if row["dataset"] != current_dataset:
            current_dataset = row["dataset"]
            lines.append(f"Dataset={current_dataset}")
        if row["status"] != "ok":
            lines.append(f"  {row['model']:10}: {row['status']}")
            continue

        lines.append(
            f"  {row['model']:10}: NegBioDB={row['negbiodb_mean']:.3f}  "
            f"uniform_random={row['uniform_random_mean']:.3f} (+{row['uniform_inflation']:.1f}%)  "
            f"degree_matched={row['degree_matched_mean']:.3f} (+{row['degree_inflation']:.1f}%)"
            f"  [n={row['n_seeds']}]"
        )

    return "\n".join(lines)


def _compute_exp1_summary_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    """Compute matched-seed Exp 1 summaries from raw result rows."""
    rows: list[dict[str, object]] = []
    for dataset in sorted(df["dataset"].dropna().unique()):
        dataset_df = df[df["dataset"] == dataset]
        for model in MODEL_ORDER:
            m_df = dataset_df[dataset_df["model"] == model]
            if m_df.empty:
                continue

            required = ["negbiodb", "uniform_random", "degree_matched"]
            seed_sets = []
            for negative in required:
                neg_df = m_df[m_df["negative"] == negative]
                if neg_df.empty:
                    rows.append({"dataset": dataset, "model": model, "status": "incomplete"})
                    seed_sets = []
                    break
                seed_sets.append(set(neg_df["seed"].tolist()))
            if not seed_sets:
                continue

            common_seeds = set.intersection(*seed_sets)
            if not common_seeds:
                rows.append({"dataset": dataset, "model": model, "status": "no matched seeds"})
                continue

            matched = m_df[m_df["seed"].isin(common_seeds)]
            stats = matched.groupby("negative", dropna=False)["log_auc"].agg(["mean", "std"])

            neg_val = float(stats.loc["negbiodb", "mean"])
            uni_val = float(stats.loc["uniform_random", "mean"])
            deg_val = float(stats.loc["degree_matched", "mean"])
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "status": "ok",
                    "n_seeds": len(common_seeds),
                    "negbiodb_mean": neg_val,
                    "negbiodb_std": float(stats.loc["negbiodb", "std"]),
                    "uniform_random_mean": uni_val,
                    "uniform_random_std": float(stats.loc["uniform_random", "std"]),
                    "degree_matched_mean": deg_val,
                    "degree_matched_std": float(stats.loc["degree_matched", "std"]),
                    "uniform_inflation": 100 * (uni_val - neg_val) / max(abs(neg_val), 1e-9),
                    "degree_inflation": 100 * (deg_val - neg_val) / max(abs(neg_val), 1e-9),
                }
            )

    return rows


def summarize_exp1_aggregated(df: pd.DataFrame) -> str:
    """Summarize Exp1 inflation for aggregated reporting using matched raw seeds."""
    if df.empty:
        return "No Exp 1 results available."

    rows = _compute_exp1_summary_rows(df)
    lines = ["### Exp 1: NegBioDB vs. Random Negative Inflation (Aggregated)"]
    current_dataset = None
    for row in rows:
        if row["dataset"] != current_dataset:
            current_dataset = row["dataset"]
            lines.append(f"Dataset={current_dataset}")
        if row["status"] != "ok":
            lines.append(f"  {row['model']:10}: {row['status']}")
            continue

        lines.append(
            f"  {row['model']:10}: "
            f"NegBioDB={_fmt_mean_std(row['negbiodb_mean'], row['negbiodb_std'])}  "
            f"uniform_random={_fmt_mean_std(row['uniform_random_mean'], row['uniform_random_std'])}"
            f" (+{row['uniform_inflation']:.1f}%)  "
            f"degree_matched={_fmt_mean_std(row['degree_matched_mean'], row['degree_matched_std'])}"
            f" (+{row['degree_inflation']:.1f}%)  [n={row['n_seeds']}]"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect baseline results into Table 1.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "baselines")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--dataset",
        choices=["balanced", "realistic"],
        help="Only include results for a single dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help="Only include runs for the given seed (repeatable)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=MODEL_ORDER,
        help="Only include runs for the given model (repeatable)",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        choices=SPLIT_ORDER,
        help="Only include runs for the given split (repeatable)",
    )
    parser.add_argument(
        "--negative",
        action="append",
        dest="negatives",
        choices=NEG_ORDER,
        help="Only include runs for the given negative source (repeatable)",
    )
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Also produce seed-aggregated tables (mean/std over seeds)",
    )
    parser.add_argument(
        "--allow-stale-ddb",
        action="store_true",
        help="Include DDB runs older than exports/negbiodb_m1_balanced_ddb.parquet.",
    )
    args = parser.parse_args(argv)

    ddb_reference = None
    if not args.allow_stale_ddb and DEFAULT_DDB_REFERENCE.exists():
        ddb_reference = DEFAULT_DDB_REFERENCE

    df = load_results(args.results_dir, ddb_reference=ddb_reference)
    if df.empty:
        logger.error("No results found. Run training jobs first.")
        return 1

    original_count = len(df)
    df = filter_results(
        df,
        dataset=args.dataset,
        seeds=args.seeds,
        models=args.models,
        splits=args.splits,
        negatives=args.negatives,
    )
    if df.empty:
        logger.error(
            "No results remain after filtering (dataset=%s, seeds=%s, models=%s, splits=%s, negatives=%s).",
            args.dataset,
            args.seeds,
            args.models,
            args.splits,
            args.negatives,
        )
        return 1
    if len(df) != original_count:
        logger.info(
            "Filtered results: %d -> %d rows (dataset=%s, seeds=%s, models=%s, splits=%s, negatives=%s)",
            original_count,
            len(df),
            args.dataset,
            args.seeds,
            args.models,
            args.splits,
            args.negatives,
        )

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

    agg_table = None
    if args.aggregate_seeds:
        agg_table = aggregate_over_seeds(df)
        agg_csv_path = args.out / "table1_aggregated.csv"
        agg_table.to_csv(agg_csv_path, index=False)
        logger.info("Aggregated Table 1 saved → %s", agg_csv_path)
        agg_md_path = args.out / "table1_aggregated.md"
        agg_md_path.write_text(format_aggregated_markdown(agg_table))
        logger.info("Aggregated Table 1 Markdown saved → %s", agg_md_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Table 1 Summary")
    print("=" * 60)
    print(
        table[["model", "dataset", "seed", "split", "negative", "log_auc", "auprc", "mcc"]]
        .to_string(index=False)
    )
    print()

    # Exp 1 inflation analysis
    exp1_df = df[df["split"] == "random"]
    print(summarize_exp1(exp1_df))
    if agg_table is not None:
        print()
        print("Aggregated Table 1 Summary")
        print("=" * 60)
        print(
            agg_table[["model", "dataset", "split", "negative", "n_seeds", "log_auc_mean", "log_auc_std", "auprc_mean", "mcc_mean"]]
            .to_string(index=False)
        )
        print()
        print(summarize_exp1_aggregated(exp1_df))
    print("=" * 60)

    logger.info("Done. %d completed runs.", len(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
