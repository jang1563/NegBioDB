#!/usr/bin/env python3
"""Collect and summarize LLM benchmark results into Table 2.

Reads results from results/llm/{task}_{model}_{config}_fs{set}/results.json
Generates Table 2: [Task × Model × Config × Metric], mean ± std across 3 few-shot sets.

Output:
  results/llm/table2.csv
  results/llm/table2.md
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "llm"

# Primary metrics per task
PRIMARY_METRICS = {
    "l1": ["accuracy", "macro_f1", "mcc"],
    "l2": ["schema_compliance", "entity_f1", "field_accuracy"],
    "l3": ["judge_accuracy", "judge_reasoning", "judge_completeness", "judge_specificity", "overall"],
    "l4": ["accuracy", "mcc", "evidence_citation_rate", "accuracy_pre_2023", "accuracy_post_2024", "contamination_gap"],
}


def load_all_results(results_dir: Path) -> list[dict]:
    """Load all results.json files."""
    results = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        # Skip judged directories (handled by L3 judge pipeline)
        if run_dir.name.endswith("_judged"):
            continue
        results_file = run_dir / "results.json"
        meta_file = run_dir / "run_meta.json"

        if not results_file.exists():
            continue

        # For L3, prefer judged results (judge pipeline adds overall score)
        judged_dir = results_dir / f"{run_dir.name}_judged"
        judged_results = judged_dir / "results.json"
        if run_dir.name.startswith("l3_") and judged_results.exists():
            results_file = judged_results

        with open(results_file) as f:
            metrics = json.load(f)
        meta = {}
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)

        # Parse run name: {task}_{model}_{config}_fs{set}
        name = run_dir.name
        parts = name.rsplit("_fs", 1)
        if len(parts) == 2:
            prefix = parts[0]
            fs_set = int(parts[1])
        else:
            prefix = name
            fs_set = 0

        # Parse prefix: {task}_{model}_{config}
        # task is always l1/l2/l3/l4
        task = prefix[:2]
        rest = prefix[3:]  # skip "l1_"
        # config is last part: zero-shot or 3-shot
        if rest.endswith("_zero-shot"):
            model = rest[:-10]
            config = "zero-shot"
        elif rest.endswith("_3-shot"):
            model = rest[:-7]
            config = "3-shot"
        else:
            model = rest
            config = meta.get("config", "unknown")

        results.append(
            {
                "run_name": name,
                "task": task,
                "model": model,
                "config": config,
                "fewshot_set": fs_set,
                "metrics": metrics,
                "meta": meta,
            }
        )

    return results


def aggregate_results(results: list[dict]) -> list[dict]:
    """Aggregate metrics across few-shot sets (mean ± std)."""
    # Group by (task, model, config)
    groups = defaultdict(list)
    for r in results:
        key = (r["task"], r["model"], r["config"])
        groups[key].append(r["metrics"])

    aggregated = []
    for (task, model, config), metric_list in sorted(groups.items()):
        # C-4: Zero-shot with deterministic models produces identical results.
        # Report N=1 to avoid misleading std=0 across fake replicates.
        effective_list = metric_list
        if config == "zero-shot" and len(metric_list) > 1:
            effective_list = [metric_list[0]]

        row = {
            "task": task,
            "model": model,
            "config": config,
            "n_runs": len(effective_list),
        }

        metrics = PRIMARY_METRICS.get(task, [])
        for metric in metrics:
            # Strip "judge_" prefix to look up raw key in results.json
            raw_key = metric.removeprefix("judge_")
            values = []
            for m in effective_list:
                # Backward compat: derive contamination_gap from pre/post
                if metric == "contamination_gap" and m.get("contamination_gap") is None:
                    pre = m.get("accuracy_pre_2023")
                    post = m.get("accuracy_post_2024")
                    val = round(pre - post, 4) if pre is not None and post is not None else None
                else:
                    val = m.get(raw_key)
                # Handle nested dicts (e.g., L3 overall.mean)
                if isinstance(val, dict):
                    val = val.get("mean")
                if val is not None and isinstance(val, (int, float)):
                    values.append(val)

            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = (
                    float(np.std(values, ddof=0)) if len(values) > 1 else 0.0
                )
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None

        aggregated.append(row)

    return aggregated


def format_table(aggregated: list[dict]) -> str:
    """Format as markdown table."""
    if not aggregated:
        return "No results found."

    # Determine all metric columns
    metric_cols = set()
    for row in aggregated:
        for key in row:
            if key.endswith("_mean"):
                metric_cols.add(key.replace("_mean", ""))
    metric_cols = sorted(metric_cols)

    # Header
    header = "| **Task** | **Model** | **Config** | **N** |"
    for m in metric_cols:
        header += f" **{m}** |"
    lines = [header]

    sep = "|" + "|".join(["---"] * (4 + len(metric_cols))) + "|"
    lines.append(sep)

    # Rows
    for row in aggregated:
        line = f"| {row['task']} | {row['model']} | {row['config']} | {row['n_runs']} |"
        for m in metric_cols:
            mean = row.get(f"{m}_mean")
            std = row.get(f"{m}_std")
            if mean is not None:
                if std and std > 0:
                    line += f" {mean:.3f}±{std:.3f} |"
                else:
                    line += f" {mean:.3f} |"
            else:
                line += " — |"
        lines.append(line)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect LLM results")
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR
    )
    args = parser.parse_args()

    print("Loading LLM results...")
    results = load_all_results(args.results_dir)
    print(f"  Found {len(results)} runs")

    if not results:
        print("No results found.")
        return

    # List runs
    for r in results:
        print(f"  {r['run_name']}: {r['task']} / {r['model']} / {r['config']}")

    print("\nAggregating across few-shot sets...")
    aggregated = aggregate_results(results)

    # Save CSV
    csv_path = args.results_dir / "table2.csv"
    with open(csv_path, "w") as f:
        # Header: union of all keys across all rows (preserving order)
        cols = []
        seen = set()
        for row in aggregated:
            for k in row:
                if k not in seen:
                    cols.append(k)
                    seen.add(k)
        f.write(",".join(cols) + "\n")
        for row in aggregated:
            f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")
    print(f"Saved CSV: {csv_path}")

    # Save Markdown
    md_path = args.results_dir / "table2.md"
    table_text = format_table(aggregated)
    with open(md_path, "w") as f:
        f.write(table_text)
    print(f"Saved Markdown: {md_path}")

    # Print table
    print(f"\n{table_text}")


if __name__ == "__main__":
    main()
