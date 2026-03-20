#!/usr/bin/env python3
"""Collect CT LLM benchmark results into summary tables.

Reads all results from results/ct_llm/ and produces:
  - results/ct_llm/ct_llm_summary.csv + .md
  - 4 experimental analyses (CT-LLM-1 through CT-LLM-4)

Usage:
  python scripts_ct/collect_ct_llm_results.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "ct_llm"

# Primary metrics per task
PRIMARY_METRICS = {
    "ct-l1": ["accuracy", "macro_f1", "mcc"],
    "ct-l2": ["schema_compliance", "category_accuracy", "field_f1_micro"],
    "ct-l3": ["overall"],
    "ct-l4": ["accuracy", "mcc", "evidence_citation_rate"],
}


def load_all_results(results_dir: Path) -> list[dict]:
    """Load all results.json files from CT LLM runs."""
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
        if run_dir.name.startswith("ct-l3_") and judged_results.exists():
            results_file = judged_results

        with open(results_file) as f:
            metrics = json.load(f)
        meta = {}
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)

        # Parse run name: {task}_{model}_{config}_fs{set}
        # CT task IDs are ct-l1 through ct-l4 (contain hyphens)
        name = run_dir.name
        parts = name.rsplit("_fs", 1)
        if len(parts) == 2:
            prefix = parts[0]
            fs_set = int(parts[1])
        else:
            prefix = name
            fs_set = 0

        # Parse prefix: ct-l1_model_config
        # Task is "ct-l1" etc. — use split("_", 1) for ct-l* prefix
        task = prefix.split("_", 1)[0]  # "ct-l1", "ct-l2", etc.
        rest = prefix[len(task) + 1:]  # skip "ct-l1_"

        if rest.endswith("_zero-shot"):
            model = rest[:-10]
            config = "zero-shot"
        elif rest.endswith("_3-shot"):
            model = rest[:-7]
            config = "3-shot"
        else:
            model = rest
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
    """Aggregate metrics across few-shot sets (mean ± std)."""
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


def exp_ct_llm_1(aggregated: list[dict]) -> str:
    """Exp CT-LLM-1: Cross-level performance profile."""
    lines = ["# Exp CT-LLM-1: Cross-Level Performance", ""]

    tasks = ["ct-l1", "ct-l2", "ct-l3", "ct-l4"]
    task_metric = {
        "ct-l1": "accuracy", "ct-l2": "category_accuracy",
        "ct-l3": "overall", "ct-l4": "accuracy",
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


def exp_ct_llm_2(results: list[dict]) -> str:
    """Exp CT-LLM-2: Contamination analysis (CT-L4 pre_2020 vs post_2023)."""
    lines = ["# Exp CT-LLM-2: Contamination Analysis (CT-L4)", ""]

    l4_runs = [r for r in results if r["task"] == "ct-l4"]
    if not l4_runs:
        return "\n".join(lines + ["No CT-L4 results found."])

    header = "| Model | Config | Acc pre_2020 | Acc post_2023 | Gap | Flag |"
    sep = "|---|---|---|---|---|---|"
    lines.extend([header, sep])

    for r in l4_runs:
        m = r["metrics"]
        pre = m.get("accuracy_pre_2020")
        post = m.get("accuracy_post_2023")
        gap = m.get("contamination_gap")
        flag = m.get("contamination_flag")
        pre_s = f"{pre:.3f}" if pre is not None else "—"
        post_s = f"{post:.3f}" if post is not None else "—"
        gap_s = f"{gap:.3f}" if gap is not None else "—"
        flag_s = "YES" if flag else "no" if flag is not None else "—"
        lines.append(f"| {r['model']} | {r['config']} | {pre_s} | {post_s} | {gap_s} | {flag_s} |")

    return "\n".join(lines)


def exp_ct_llm_3(results: list[dict]) -> str:
    """Exp CT-LLM-3: Difficulty gradient (CT-L1 easy/medium/hard)."""
    lines = ["# Exp CT-LLM-3: Difficulty Gradient (CT-L1)", ""]

    l1_runs = [r for r in results if r["task"] == "ct-l1"]
    if not l1_runs:
        return "\n".join(lines + ["No CT-L1 results found."])

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


def exp_ct_llm_4(aggregated: list[dict]) -> str:
    """Exp CT-LLM-4: Cross-domain comparison (CT vs DTI)."""
    lines = ["# Exp CT-LLM-4: Cross-Domain Comparison", ""]

    # Try to load DTI table2.csv
    dti_path = PROJECT_ROOT / "results" / "llm" / "table2.csv"
    if not dti_path.exists():
        return "\n".join(lines + ["DTI results not found (results/llm/table2.csv)."])

    import csv
    dti_rows = []
    with open(dti_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dti_rows.append(row)

    # Shared models (matched by sanitized model name).
    # Only models with identical sanitized names across domains will match.
    # Expected overlap (4 models): gpt-4o-mini, qwen32b, gemini-2.5-flash, llama70b.
    # CT-unique: claude-sonnet-4-6.
    ct_models = set(r["model"] for r in aggregated)
    dti_models = set(r.get("model", "") for r in dti_rows)
    shared = ct_models & dti_models

    if not shared:
        lines.append(f"No shared models found.")
        lines.append(f"CT models: {ct_models}")
        lines.append(f"DTI models: {dti_models}")
        return "\n".join(lines)

    lines.append(f"Shared models: {sorted(shared)}")
    lines.append("")
    lines.append("| Model | DTI-L1 Acc | CT-L1 Acc | DTI-L4 Acc | CT-L4 Acc |")
    lines.append("|---|---|---|---|---|")

    for model in sorted(shared):
        dti_l1 = next((r for r in dti_rows if r.get("model") == model and r.get("task") == "l1"), None)
        ct_l1 = next((r for r in aggregated if r["model"] == model and r["task"] == "ct-l1"), None)
        dti_l4 = next((r for r in dti_rows if r.get("model") == model and r.get("task") == "l4"), None)
        ct_l4 = next((r for r in aggregated if r["model"] == model and r["task"] == "ct-l4"), None)

        def _fmt(row, key):
            if row is None:
                return "—"
            val = row.get(key) or row.get(f"{key}_mean")
            if val is None or val == "":
                return "—"
            return f"{float(val):.3f}"

        lines.append(
            f"| {model} | {_fmt(dti_l1, 'accuracy')} | {_fmt(ct_l1, 'accuracy')} "
            f"| {_fmt(dti_l4, 'accuracy')} | {_fmt(ct_l4, 'accuracy')} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect CT LLM results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    print("Loading CT LLM results...")
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
    csv_path = args.results_dir / "ct_llm_summary.csv"
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
    md_path = args.results_dir / "ct_llm_summary.md"
    table_text = format_table(aggregated)
    with open(md_path, "w") as f:
        f.write(table_text)
    print(f"Saved Markdown: {md_path}")
    print(f"\n{table_text}")

    # Experimental analyses
    exp_dir = args.results_dir
    for exp_num, exp_fn in [
        ("ct_llm_exp1_cross_level", lambda: exp_ct_llm_1(aggregated)),
        ("ct_llm_exp2_contamination", lambda: exp_ct_llm_2(results)),
        ("ct_llm_exp3_difficulty", lambda: exp_ct_llm_3(results)),
        ("ct_llm_exp4_cross_domain", lambda: exp_ct_llm_4(aggregated)),
    ]:
        text = exp_fn()
        out = exp_dir / f"{exp_num}.md"
        with open(out, "w") as f:
            f.write(text)
        print(f"\n{text}")

    print(f"\nDone. {len(results)} total runs collected.")


if __name__ == "__main__":
    main()
