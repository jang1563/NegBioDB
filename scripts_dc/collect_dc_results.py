#!/usr/bin/env python3
"""Collect and aggregate DC benchmark results (LLM).

Reads all results from results/dc_llm/.

Usage:
  python scripts_dc/collect_dc_results.py
  python scripts_dc/collect_dc_results.py --llm-dir results/dc_llm
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = PROJECT_ROOT / "results" / "dc_llm"

PRIMARY_METRICS = {
    "dc-l1": ["accuracy", "macro_f1", "mcc"],
    "dc-l2": ["schema_compliance", "field_f1", "mechanism_f1"],
    "dc-l3": ["overall_mean"],
    "dc-l4": ["accuracy", "mcc"],
}


def load_all_llm_results(results_dir: Path) -> list[dict]:
    """Load all results.json files from DC LLM runs."""
    results = []
    if not results_dir.exists():
        return results

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.endswith("_judged"):
            continue

        results_file = run_dir / "results.json"
        meta_file = run_dir / "run_meta.json"

        if not results_file.exists():
            continue

        # For L3, prefer judged results
        judged_dir = results_dir / f"{run_dir.name}_judged"
        judged_results = judged_dir / "results.json"
        if run_dir.name.startswith("dc-l3_") and judged_results.exists():
            results_file = judged_results

        with open(results_file) as f:
            metrics = json.load(f)
        meta = {}
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)

        # Parse run name: {task}_{model}_{config}_fs{set}
        name = run_dir.name
        m_fs = re.match(r"(dc-l\d)_(.+)_(zero-shot|3-shot)_fs(\d+)$", name)
        m_no_fs = re.match(r"(dc-l\d)_(.+)_(zero-shot|3-shot)$", name)
        if m_fs:
            task, model, config, fs_set = (
                m_fs.group(1), m_fs.group(2), m_fs.group(3), int(m_fs.group(4))
            )
        elif m_no_fs:
            task, model, config = m_no_fs.group(1), m_no_fs.group(2), m_no_fs.group(3)
            fs_set = 0
        else:
            parts = name.rsplit("_fs", 1)
            prefix = parts[0] if len(parts) == 2 else name
            fs_set = int(parts[1]) if len(parts) == 2 else 0
            task = prefix.split("_", 1)[0]
            model = prefix[len(task) + 1:]
            config = meta.get("config", "unknown")

        # Count ERROR predictions
        error_rate = 0.0
        api_failure = False
        pred_path = run_dir / "predictions.jsonl"
        if pred_path.exists():
            error_count = 0
            total_preds = 0
            with open(pred_path) as pf:
                for line in pf:
                    total_preds += 1
                    try:
                        rec = json.loads(line)
                        if str(rec.get("prediction", "")).startswith("ERROR:"):
                            error_count += 1
                    except json.JSONDecodeError:
                        error_count += 1
            if total_preds > 0:
                error_rate = error_count / total_preds
                api_failure = error_rate > 0.5

        results.append({
            "run_name": name,
            "task": task,
            "model": model,
            "config": config,
            "fewshot_set": fs_set,
            "metrics": metrics,
            "meta": meta,
            "error_rate": error_rate,
            "api_failure": api_failure,
        })

    return results


def aggregate_results(results: list[dict]) -> list[dict]:
    """Aggregate metrics across few-shot sets (mean +/- std)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        key = (r["task"], r["model"], r["config"])
        groups[key].append(r)

    aggregated = []
    for (task, model, config), run_list in sorted(groups.items()):
        effective_list = run_list
        if config == "zero-shot" and len(run_list) > 1:
            effective_list = [run_list[0]]

        metric_list = [r["metrics"] for r in effective_list]
        error_rates = [r.get("error_rate", 0.0) for r in effective_list]
        any_api_failure = any(r.get("api_failure", False) for r in effective_list)

        row = {
            "task": task,
            "model": model,
            "config": config,
            "n_runs": len(effective_list),
            "error_rate_mean": float(np.mean(error_rates)),
            "api_failure": any_api_failure,
        }

        metrics = PRIMARY_METRICS.get(task, [])
        for metric in metrics:
            values = []
            for m in metric_list:
                val = m.get(metric)
                if isinstance(val, dict):
                    val = val.get("mean")
                if val is not None and isinstance(val, (int, float)):
                    values.append(val)

            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
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

    metric_cols = set()
    for row in aggregated:
        for key in row:
            if key.endswith("_mean") and key != "error_rate_mean":
                metric_cols.add(key.removesuffix("_mean"))
    metric_cols = sorted(metric_cols)

    header = "| **Task** | **Model** | **Config** | **N** |"
    for m in metric_cols:
        header += f" **{m}** |"
    lines = [header]
    lines.append("|" + "|".join(["---"] * (4 + len(metric_cols))) + "|")

    has_api_failure = False
    for row in aggregated:
        marker = ""
        if row.get("api_failure"):
            marker = " +"
            has_api_failure = True
        line = f"| {row['task']} | {row['model']}{marker} | {row['config']} | {row['n_runs']} |"
        for m in metric_cols:
            mean = row.get(f"{m}_mean")
            std = row.get(f"{m}_std")
            if mean is not None:
                if std and std > 0:
                    line += f" {mean:.3f}+/-{std:.3f} |"
                else:
                    line += f" {mean:.3f} |"
            else:
                line += " -- |"
        lines.append(line)

    if has_api_failure:
        lines.append("")
        lines.append("+ = >50% of predictions are API errors")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect DC benchmark results")
    parser.add_argument("--llm-dir", type=Path, default=LLM_DIR)
    args = parser.parse_args()

    # LLM results
    print(f"Loading DC LLM results from {args.llm_dir}...")
    results = load_all_llm_results(args.llm_dir)
    print(f"  Found {len(results)} runs")

    if not results:
        print("No LLM results found.")
        return

    for r in results:
        print(f"  {r['run_name']}: {r['task']} / {r['model']} / {r['config']}")

    print("\nAggregating...")
    aggregated = aggregate_results(results)

    # Save CSV
    csv_path = args.llm_dir / "dc_llm_summary.csv"
    with open(csv_path, "w") as f:
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
    md_path = args.llm_dir / "dc_llm_summary.md"
    table_text = format_table(aggregated)
    with open(md_path, "w") as f:
        f.write(table_text)
    print(f"Saved Markdown: {md_path}")
    print(f"\n{table_text}")

    print(f"\nDone. {len(results)} total runs collected.")


if __name__ == "__main__":
    main()
