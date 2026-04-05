#!/usr/bin/env python3
"""DC LLM benchmark inference harness.

Mirrors scripts_vp/run_vp_llm_benchmark.py for the DC domain.
Reuses LLMClient from DTI. Uses DC-specific prompts and evaluation.

Usage:
  python scripts_dc/run_dc_llm_benchmark.py --task dc-l1 --model gemini-2.5-flash \
      --provider gemini --config zero-shot

  python scripts_dc/run_dc_llm_benchmark.py --task dc-l4 --model gpt-4o-mini \
      --provider openai --config 3-shot --fewshot-set 1

Output:
  results/dc_llm/{task}_{model}_{config}_fs{set}/
      predictions.jsonl, results.json, run_meta.json
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negbiodb.llm_client import LLMClient
from negbiodb_dc.llm_eval import compute_all_dc_llm_metrics
from negbiodb_dc.llm_prompts import format_dc_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "exports" / "dc_llm"
OUTPUT_BASE = PROJECT_ROOT / "results" / "dc_llm"

TASK_FILES = {
    "dc-l1": "dc_l1_dataset.jsonl",
    "dc-l2": "dc_l2_dataset.jsonl",
    "dc-l3": "dc_l3_dataset.jsonl",
    "dc-l4": "dc_l4_dataset.jsonl",
}

TASK_MAX_TOKENS = {
    "dc-l1": 256,
    "dc-l2": 1024,
    "dc-l3": 2048,
    "dc-l4": 256,
}


def load_dataset(task: str, data_dir: Path) -> list[dict]:
    """Load dataset JSONL file."""
    path = data_dir / TASK_FILES[task]
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def get_fewshot_examples(
    records: list[dict], fewshot_set: int, n_per_class: int = 3
) -> list[dict]:
    """Select few-shot examples from fewshot split."""
    import random

    fewshot = [r for r in records if r.get("split") == "fewshot"]
    if not fewshot:
        return []

    by_class = {}
    for r in fewshot:
        cls = r.get("gold_answer", "default")
        by_class.setdefault(cls, []).append(r)

    rng = random.Random(42 + fewshot_set)
    examples = []
    for cls, cls_records in by_class.items():
        pool = list(cls_records)
        rng.shuffle(pool)
        start = fewshot_set * n_per_class
        end = start + n_per_class
        examples.extend(pool[start:end])

    rng.shuffle(examples)
    return examples


def sanitize_model_name(model: str) -> str:
    """Convert model path/name to filesystem-safe string."""
    return model.split("/")[-1].replace(".", "-").lower()


def _json_safe(obj):
    """Convert numpy/pandas types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description="Run DC LLM benchmark")
    parser.add_argument("--task", required=True, choices=list(TASK_FILES.keys()))
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--provider", required=True, choices=["vllm", "gemini", "openai", "anthropic"]
    )
    parser.add_argument(
        "--config", default="zero-shot", choices=["zero-shot", "3-shot"]
    )
    parser.add_argument("--fewshot-set", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_BASE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    if args.max_tokens is None:
        args.max_tokens = TASK_MAX_TOKENS.get(args.task, 1024)

    model_name = sanitize_model_name(args.model)
    run_name = f"{args.task}_{model_name}_{args.config}_fs{args.fewshot_set}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== DC LLM Benchmark: {run_name} ===")
    print(f"Task: {args.task}, Model: {args.model} ({args.provider})")
    print(f"Config: {args.config}, fewshot_set={args.fewshot_set}")

    # Load data
    print("\nLoading dataset...")
    records = load_dataset(args.task, args.data_dir)
    test_records = [r for r in records if r.get("split") == "test"]
    print(f"  Total: {len(records)}, Test: {len(test_records)}")

    fewshot_examples = None
    if args.config == "3-shot":
        fewshot_examples = get_fewshot_examples(records, args.fewshot_set)
        print(f"  Few-shot examples: {len(fewshot_examples)}")

    # Init client
    print("\nInitializing LLM client...")
    client = LLMClient(
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Resume support — store predictions keyed by question_id to ensure
    # correct alignment when rebuilding the list in test_records order.
    pred_path = run_dir / "predictions.jsonl"
    completed: dict[str, str] = {}  # question_id -> prediction text

    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                rec = json.loads(line)
                completed[rec["question_id"]] = rec["prediction"]
        if completed:
            print(f"\nResuming: {len(completed)} predictions already complete")

    remaining = [
        (i, r) for i, r in enumerate(test_records)
        if r.get("question_id", f"Q{i}") not in completed
    ]

    print(f"\nRunning inference: {len(remaining)} remaining of {len(test_records)} total...")
    start_time = time.time()

    with open(pred_path, "a") as f:
        for j, (i, record) in enumerate(remaining):
            qid = record.get("question_id", f"Q{i}")
            system, user = format_dc_prompt(
                args.task, record, args.config, fewshot_examples
            )
            try:
                response = client.generate(user, system)
            except Exception as e:
                response = f"ERROR: {e}"
                print(f"  Error on example {i}: {e}")

            completed[qid] = response

            pred_record = {
                "question_id": qid,
                "prediction": response,
                "gold_answer": record.get("gold_answer"),
            }
            f.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
            f.flush()

            done = len(completed) - len(remaining) + j + 1 + (len(completed) - j - 1)
            done = j + 1 + (len(test_records) - len(remaining))
            if done % 50 == 0:
                elapsed = time.time() - start_time
                rate = (j + 1) / elapsed * 60
                print(f"  Progress: {done}/{len(test_records)} ({rate:.0f}/min)")

    elapsed = time.time() - start_time
    print(f"\nInference complete: {elapsed:.0f}s")

    # Rebuild predictions in test_records order for correct eval alignment
    predictions = [
        completed.get(r.get("question_id", f"Q{i}"), "")
        for i, r in enumerate(test_records)
    ]

    # Evaluate
    print("\nEvaluating...")
    metrics = compute_all_dc_llm_metrics(args.task, predictions, test_records)

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_safe)

    print("\n=== Results ===")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        elif isinstance(val, dict) and "mean" in val:
            print(f"  {key}: {val['mean']:.4f} +/- {val.get('std', 0):.4f}")
        else:
            print(f"  {key}: {val}")

    if args.task == "dc-l3":
        print(
            "\nNOTE: L3 results.json contains placeholder scores (n_parsed=0). "
            "Run scripts_dc/run_dc_l3_judge.py to obtain real judge scores."
        )

    # Save metadata
    meta = {
        "task": args.task,
        "model": args.model,
        "provider": args.provider,
        "config": args.config,
        "fewshot_set": args.fewshot_set,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "n_test": len(test_records),
        "n_predictions": len(predictions),
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
    }
    meta_path = run_dir / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults saved to {run_dir}/")


if __name__ == "__main__":
    main()
