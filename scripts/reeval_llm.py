#!/usr/bin/env python3
"""Re-evaluate existing DTI LLM predictions with updated metrics.

Reads predictions.jsonl from run directories, recomputes metrics using
the latest evaluation code, and writes updated results.json (backing up
the original).

Usage:
    PYTHONPATH=src python scripts/reeval_llm.py --task l4
    PYTHONPATH=src python scripts/reeval_llm.py --run-dir results/llm/l4_gpt-4o-mini_3-shot_fs0
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "llm"
EXPORTS_DIR = PROJECT_ROOT / "exports" / "llm_benchmarks"

TASK_DATASET = {
    "l1": "l1_mcq.jsonl",
    "l2": "l2_candidates.jsonl",
    "l3": "l3_reasoning_pilot.jsonl",
    "l4": "l4_tested_untested.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def reeval_run(run_dir: Path, gold_records: dict[str, dict], task: str) -> bool:
    """Re-evaluate a single run directory. Returns True if successful."""
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        print(f"  SKIP {run_dir.name}: no predictions.jsonl")
        return False

    preds = load_jsonl(pred_path)
    if not preds:
        print(f"  SKIP {run_dir.name}: empty predictions")
        return False

    pred_texts = []
    gold_list = []
    for p in preds:
        qid = p.get("question_id", "")
        if qid not in gold_records:
            continue
        pred_texts.append(str(p.get("prediction", "")))
        gold_list.append(gold_records[qid])

    if not pred_texts:
        print(f"  SKIP {run_dir.name}: no matching question_ids")
        return False

    from negbiodb.llm_eval import compute_all_llm_metrics

    try:
        metrics = compute_all_llm_metrics(task, pred_texts, gold_list)
    except Exception as e:
        print(f"  ERROR {run_dir.name}: {e}")
        return False

    results_path = run_dir / "results.json"
    if results_path.exists():
        backup_path = run_dir / "results_original.json"
        if not backup_path.exists():
            shutil.copy2(results_path, backup_path)

    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  OK {run_dir.name}: {len(pred_texts)} predictions re-evaluated")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-evaluate DTI LLM predictions")
    parser.add_argument("--task", type=str, help="Task filter (e.g. l4)")
    parser.add_argument("--run-dir", type=Path, help="Single run directory")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--exports-dir", type=Path, default=EXPORTS_DIR)
    args = parser.parse_args(argv)

    if args.run_dir:
        run_dirs = [args.run_dir]
        task_match = re.match(r"(l\d)", args.run_dir.name)
        tasks = {task_match.group(1)} if task_match else set()
    else:
        run_dirs = sorted(
            d for d in args.results_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_") and not d.name.startswith("backup")
        )
        if args.task:
            run_dirs = [d for d in run_dirs if d.name.startswith(args.task + "_")]
            tasks = {args.task}
        else:
            tasks = set()
            for d in run_dirs:
                m = re.match(r"(l\d)", d.name)
                if m:
                    tasks.add(m.group(1))

    gold_by_task: dict[str, dict[str, dict]] = {}
    for task in tasks:
        dataset_file = args.exports_dir / TASK_DATASET.get(task, "")
        if not dataset_file.exists():
            print(f"WARNING: Gold dataset not found: {dataset_file}")
            continue
        records = load_jsonl(dataset_file)
        gold_by_task[task] = {
            r["question_id"]: r for r in records if r.get("split") in ("test", "val")
        }
        print(f"Loaded {len(gold_by_task[task])} gold records for {task}")

    n_ok = 0
    n_skip = 0
    for run_dir in run_dirs:
        task_match = re.match(r"(l\d)", run_dir.name)
        if not task_match:
            continue
        task = task_match.group(1)
        if task not in gold_by_task:
            n_skip += 1
            continue
        if reeval_run(run_dir, gold_by_task[task], task):
            n_ok += 1
        else:
            n_skip += 1

    print(f"\nDone: {n_ok} re-evaluated, {n_skip} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
