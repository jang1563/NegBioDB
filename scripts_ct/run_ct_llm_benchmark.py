#!/usr/bin/env python3
"""CT LLM benchmark inference harness.

Mirrors scripts/run_llm_benchmark.py for the CT domain.
Reuses LLMClient from DTI. Uses CT-specific prompts and evaluation.

Usage:
  python scripts_ct/run_ct_llm_benchmark.py --task ct-l1 --model gemini-2.0-flash \
      --provider gemini --config zero-shot

  python scripts_ct/run_ct_llm_benchmark.py --task ct-l4 --model gpt-4o-mini \
      --provider openai --config 3-shot --fewshot-set 1 --api-key $OPENAI_API_KEY

Output:
  results/ct_llm/{task}_{model}_{config}_fs{set}/
      predictions.jsonl, results.json, run_meta.json
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negbiodb.llm_client import LLMClient
from negbiodb_ct.llm_eval import compute_all_ct_llm_metrics
from negbiodb_ct.llm_prompts import format_ct_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "exports" / "ct_llm"
OUTPUT_BASE = PROJECT_ROOT / "results" / "ct_llm"

TASK_FILES = {
    "ct-l1": "ct_l1_dataset.jsonl",
    "ct-l2": "ct_l2_dataset.jsonl",
    "ct-l3": "ct_l3_dataset.jsonl",
    "ct-l4": "ct_l4_dataset.jsonl",
}

# Max tokens per task
TASK_MAX_TOKENS = {
    "ct-l1": 256,
    "ct-l2": 1024,
    "ct-l3": 2048,
    "ct-l4": 256,
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
    """Select few-shot examples from fewshot split.

    3 independent sets (fewshot_set=0,1,2) for variance.
    """
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
    parser = argparse.ArgumentParser(description="Run CT LLM benchmark")
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

    # Default max tokens per task
    if args.max_tokens is None:
        args.max_tokens = TASK_MAX_TOKENS.get(args.task, 1024)

    model_name = sanitize_model_name(args.model)
    run_name = f"{args.task}_{model_name}_{args.config}_fs{args.fewshot_set}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CT LLM Benchmark: {run_name} ===")
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

    # Resume support
    pred_path = run_dir / "predictions.jsonl"
    predictions = []
    completed_ids = set()

    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                rec = json.loads(line)
                completed_ids.add(rec["question_id"])
                predictions.append(rec["prediction"])
        if completed_ids:
            print(f"\nResuming: {len(completed_ids)} predictions already complete")

    remaining = [
        (i, r) for i, r in enumerate(test_records)
        if r.get("question_id", f"Q{i}") not in completed_ids
    ]

    print(f"\nRunning inference: {len(remaining)} remaining of {len(test_records)} total...")
    start_time = time.time()

    with open(pred_path, "a") as f:
        for j, (i, record) in enumerate(remaining):
            system, user = format_ct_prompt(
                args.task, record, args.config, fewshot_examples
            )
            try:
                response = client.generate(user, system)
            except Exception as e:
                response = f"ERROR: {e}"
                print(f"  Error on example {i}: {e}")

            predictions.append(response)

            pred_record = {
                "question_id": record.get("question_id", f"Q{i}"),
                "prediction": response,
                "gold_answer": record.get("gold_answer"),
            }
            f.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
            f.flush()

            done = len(completed_ids) + j + 1
            if done % 50 == 0:
                elapsed = time.time() - start_time
                rate = (j + 1) / elapsed * 60
                print(f"  Progress: {done}/{len(test_records)} ({rate:.0f}/min)")

    elapsed = time.time() - start_time
    print(f"\nInference complete: {elapsed:.0f}s")

    # Evaluate
    print("\nEvaluating...")
    metrics = compute_all_ct_llm_metrics(args.task, predictions, test_records)

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_safe)

    print(f"\n=== Results ===")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        elif isinstance(val, dict) and "mean" in val:
            print(f"  {key}: {val['mean']:.4f} ± {val.get('std', 0):.4f}")
        else:
            print(f"  {key}: {val}")

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
