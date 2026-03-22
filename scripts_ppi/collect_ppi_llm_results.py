#!/usr/bin/env python3
"""Collect PPI LLM benchmark results into summary tables.

Reads all results from results/ppi_llm/ and produces:
  - results/ppi_llm/ppi_llm_summary.csv + .md
  - 5 experimental analyses (PPI-LLM-1 through PPI-LLM-5)

Usage:
  python scripts_ppi/collect_ppi_llm_results.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "ppi_llm"

PRIMARY_METRICS = {
    "ppi-l1": ["accuracy", "macro_f1", "mcc"],
    "ppi-l2": ["schema_compliance", "entity_f1", "count_accuracy"],
    "ppi-l3": ["overall"],
    "ppi-l4": ["accuracy", "mcc", "evidence_citation_rate"],
}


def load_all_results(results_dir: Path) -> list[dict]:
    """Load all results.json files from PPI LLM runs."""
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
        if run_dir.name.startswith("ppi-l3_") and judged_results.exists():
            results_file = judged_results

        with open(results_file) as f:
            metrics = json.load(f)
        meta = {}
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)

        # Parse run name: {task}_{model}_{config}_fs{set}
        name = run_dir.name
        m_fs = re.match(r"(ppi-l\d)_(.+?)_(zero-shot|3-shot)_fs(\d+)$", name)
        m_no_fs = re.match(r"(ppi-l\d)_(.+?)_(zero-shot|3-shot)$", name)
        if m_fs:
            task, model, config, fs_set = (
                m_fs.group(1), m_fs.group(2), m_fs.group(3), int(m_fs.group(4))
            )
        elif m_no_fs:
            task, model, config = m_no_fs.group(1), m_no_fs.group(2), m_no_fs.group(3)
            fs_set = 0
        else:
            # Fallback: best-effort parsing from meta
            parts = name.rsplit("_fs", 1)
            prefix = parts[0] if len(parts) == 2 else name
            fs_set = int(parts[1]) if len(parts) == 2 else 0
            task = prefix.split("_", 1)[0]
            model = prefix[len(task) + 1:]
            config = meta.get("config", "unknown")

        results.append({
            "run_name": name,
            "task": task,
            "model": model,
            "config": config,
            "fewshot_set": fs_set,
            "metrics": metrics,
            "meta": meta,
        })

    return results


def aggregate_results(results: list[dict]) -> list[dict]:
    """Aggregate metrics across few-shot sets (mean +/- std)."""
    groups = defaultdict(list)
    for r in results:
        key = (r["task"], r["model"], r["config"])
        groups[key].append(r["metrics"])

    aggregated = []
    for (task, model, config), metric_list in sorted(groups.items()):
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
            values = []
            for m in effective_list:
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
            if key.endswith("_mean"):
                metric_cols.add(key.replace("_mean", ""))
    metric_cols = sorted(metric_cols)

    header = "| **Task** | **Model** | **Config** | **N** |"
    for m in metric_cols:
        header += f" **{m}** |"
    lines = [header]
    lines.append("|" + "|".join(["---"] * (4 + len(metric_cols))) + "|")

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


# ── Experimental Analyses ────────────────────────────────────────────────


def exp_ppi_llm_1(aggregated: list[dict]) -> str:
    """Exp PPI-LLM-1: Cross-level performance profile."""
    lines = ["# Exp PPI-LLM-1: Cross-Level Performance", ""]

    tasks = ["ppi-l1", "ppi-l2", "ppi-l3", "ppi-l4"]
    task_metric = {
        "ppi-l1": "accuracy", "ppi-l2": "entity_f1",
        "ppi-l3": "overall", "ppi-l4": "accuracy",
    }

    models = sorted(set(r["model"] for r in aggregated))
    header = "| Model | Config | " + " | ".join(tasks) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(tasks))) + "|"
    lines.extend([header, sep])

    for model in models:
        for config in ["zero-shot", "3-shot"]:
            row = f"| {model} | {config} |"
            for task in tasks:
                metric = task_metric[task]
                match = [
                    r for r in aggregated
                    if r["task"] == task and r["model"] == model and r["config"] == config
                ]
                if match:
                    val = match[0].get(f"{metric}_mean")
                    row += f" {val:.3f} |" if val is not None else " — |"
                else:
                    row += " — |"
            lines.append(row)

    return "\n".join(lines)


def exp_ppi_llm_2(results: list[dict]) -> str:
    """Exp PPI-LLM-2: Contamination analysis (PPI-L4 pre_2015 vs post_2020)."""
    lines = ["# Exp PPI-LLM-2: Contamination Analysis (PPI-L4)", ""]

    l4_runs = [r for r in results if r["task"] == "ppi-l4"]
    if not l4_runs:
        return "\n".join(lines + ["No PPI-L4 results found."])

    header = "| Model | Config | Acc pre_2015 | Acc post_2020 | Gap | Flag |"
    sep = "|---|---|---|---|---|---|"
    lines.extend([header, sep])

    for r in l4_runs:
        m = r["metrics"]
        pre = m.get("accuracy_pre_2015")
        post = m.get("accuracy_post_2020")
        gap = m.get("contamination_gap")
        flag = m.get("contamination_flag")
        pre_s = f"{pre:.3f}" if pre is not None else "—"
        post_s = f"{post:.3f}" if post is not None else "—"
        gap_s = f"{gap:.3f}" if gap is not None else "—"
        flag_s = "YES" if flag else "no" if flag is not None else "—"
        lines.append(f"| {r['model']} | {r['config']} | {pre_s} | {post_s} | {gap_s} | {flag_s} |")

    return "\n".join(lines)


def exp_ppi_llm_3(results: list[dict]) -> str:
    """Exp PPI-LLM-3: Difficulty gradient (PPI-L1 easy/medium/hard)."""
    lines = ["# Exp PPI-LLM-3: Difficulty Gradient (PPI-L1)", ""]

    l1_runs = [r for r in results if r["task"] == "ppi-l1"]
    if not l1_runs:
        return "\n".join(lines + ["No PPI-L1 results found."])

    header = "| Model | Config | Easy | Medium | Hard |"
    sep = "|---|---|---|---|---|"
    lines.extend([header, sep])

    for r in l1_runs:
        m = r["metrics"]
        per_diff = m.get("per_difficulty_accuracy", {})
        easy = per_diff.get("easy")
        med = per_diff.get("medium")
        hard = per_diff.get("hard")
        e_s = f"{easy:.3f}" if easy is not None else "—"
        m_s = f"{med:.3f}" if med is not None else "—"
        h_s = f"{hard:.3f}" if hard is not None else "—"
        lines.append(f"| {r['model']} | {r['config']} | {e_s} | {m_s} | {h_s} |")

    return "\n".join(lines)


def exp_ppi_llm_4(results: list[dict]) -> str:
    """Exp PPI-LLM-4: Per-class L1 accuracy (evidence type breakdown)."""
    lines = ["# Exp PPI-LLM-4: Per-Class L1 Accuracy", ""]
    lines.append("Categories: A=direct_experimental, B=systematic_screen (Y2H), "
                 "C=computational_inference, D=database_absence")
    lines.append("")

    l1_runs = [r for r in results if r["task"] == "ppi-l1"]
    if not l1_runs:
        return "\n".join(lines + ["No PPI-L1 results found."])

    # Map both letter codes and full names
    class_keys = [
        ("direct_experimental", "A (direct_exp)"),
        ("systematic_screen", "B (Y2H)"),
        ("computational_inference", "C (comp)"),
        ("database_absence", "D (db_abs)"),
    ]
    header = "| Model | Config | " + " | ".join(lbl for _, lbl in class_keys) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(class_keys)) + "|"
    lines.extend([header, sep])

    for r in l1_runs:
        m = r["metrics"]
        per_class = m.get("per_class_accuracy", {})
        vals = []
        for key, _ in class_keys:
            v = per_class.get(key)
            if v is None:
                # Try single-letter fallback
                letter = {"direct_experimental": "A", "systematic_screen": "B",
                          "computational_inference": "C", "database_absence": "D"}[key]
                v = per_class.get(letter)
            vals.append(f"{v:.3f}" if v is not None else "—")
        lines.append(f"| {r['model']} | {r['config']} | {' | '.join(vals)} |")

    return "\n".join(lines)


def exp_ppi_llm_5(results: list[dict]) -> str:
    """Exp PPI-LLM-5: L3 per-dimension judge scores."""
    lines = ["# Exp PPI-LLM-5: L3 Per-Dimension Judge Scores", ""]

    l3_runs = [r for r in results if r["task"] == "ppi-l3"]
    if not l3_runs:
        return "\n".join(lines + ["No PPI-L3 results found."])

    dims = ["biological_plausibility", "structural_reasoning",
            "mechanistic_completeness", "specificity"]
    header = "| Model | Config | " + " | ".join(d.replace("_", " ").title() for d in dims) + " | Overall |"
    sep = "|---|---|" + "|".join(["---"] * (len(dims) + 1)) + "|"
    lines.extend([header, sep])

    for r in l3_runs:
        m = r["metrics"]
        row = f"| {r['model']} | {r['config']} |"
        for dim in dims:
            dim_data = m.get(dim, {})
            mean = dim_data.get("mean") if isinstance(dim_data, dict) else None
            row += f" {mean:.2f} |" if mean is not None else " — |"
        overall = m.get("overall", {})
        overall_mean = overall.get("mean") if isinstance(overall, dict) else overall
        row += f" {overall_mean:.2f} |" if overall_mean is not None else " — |"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect PPI LLM results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    print("Loading PPI LLM results...")
    results = load_all_results(args.results_dir)
    print(f"  Found {len(results)} runs")

    if not results:
        print("No results found.")
        return

    for r in results:
        print(f"  {r['run_name']}: {r['task']} / {r['model']} / {r['config']}")

    print("\nAggregating...")
    aggregated = aggregate_results(results)

    # Save CSV
    csv_path = args.results_dir / "ppi_llm_summary.csv"
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
    md_path = args.results_dir / "ppi_llm_summary.md"
    table_text = format_table(aggregated)
    with open(md_path, "w") as f:
        f.write(table_text)
    print(f"Saved Markdown: {md_path}")
    print(f"\n{table_text}")

    # Experimental analyses
    exp_dir = args.results_dir
    for exp_num, exp_fn in [
        ("ppi_llm_exp1_cross_level", lambda: exp_ppi_llm_1(aggregated)),
        ("ppi_llm_exp2_contamination", lambda: exp_ppi_llm_2(results)),
        ("ppi_llm_exp3_difficulty", lambda: exp_ppi_llm_3(results)),
        ("ppi_llm_exp4_per_class", lambda: exp_ppi_llm_4(results)),
        ("ppi_llm_exp5_l3_dimensions", lambda: exp_ppi_llm_5(results)),
    ]:
        text = exp_fn()
        out = exp_dir / f"{exp_num}.md"
        with open(out, "w") as f:
            f.write(text)
        print(f"\n{text}")

    print(f"\nDone. {len(results)} total runs collected.")


if __name__ == "__main__":
    main()
