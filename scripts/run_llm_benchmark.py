#!/usr/bin/env python3
"""Main LLM benchmark inference harness.

Usage:
  python scripts/run_llm_benchmark.py --task l1 --model gemini-2.5-flash \
      --provider gemini --config zero-shot --fewshot-set 0

  python scripts/run_llm_benchmark.py --task l4 --model /path/to/Llama-70B \
      --provider vllm --config 3-shot --fewshot-set 1 --api-base http://localhost:8000/v1

Output structure:
  results/llm/{task}_{model}_{config}_fs{set}/
      predictions.jsonl   — raw LLM outputs
      results.json        — evaluation metrics
      run_meta.json       — model version, token count, timestamp
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from negbiodb.llm_client import LLMClient
from negbiodb.llm_eval import compute_all_llm_metrics
from negbiodb.llm_prompts import format_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "exports" / "llm_benchmarks"
OUTPUT_BASE = PROJECT_ROOT / "results" / "llm"

# Task -> dataset file mapping
TASK_FILES = {
    "l1": "l1_mcq.jsonl",
    "l2": "l2_gold.jsonl",
    "l3": "l3_reasoning_pilot.jsonl",
    "l4": "l4_tested_untested.jsonl",
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

    3 independent sets (fewshot_set=0,1,2) for variance reporting.
    Each set has n_per_class examples per class.
    """
    fewshot = [r for r in records if r.get("split") == "fewshot"]
    if not fewshot:
        return []

    # Group by class
    by_class = {}
    for r in fewshot:
        cls = r.get("class", r.get("correct_answer", "default"))
        by_class.setdefault(cls, []).append(r)

    # Select examples for this set
    import random
    rng = random.Random(42 + fewshot_set)
    examples = []
    for cls, cls_records in by_class.items():
        pool = list(cls_records)  # copy to avoid mutating input
        rng.shuffle(pool)
        # Offset by fewshot_set * n_per_class
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
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmark")
    parser.add_argument("--task", required=True, choices=["l1", "l2", "l3", "l4"])
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
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_BASE
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1, help="MCQ per call (L1 batching)")
    args = parser.parse_args()

    # ── Setup ──
    model_name = sanitize_model_name(args.model)
    run_name = f"{args.task}_{model_name}_{args.config}_fs{args.fewshot_set}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== LLM Benchmark: {run_name} ===")
    print(f"Task: {args.task}")
    print(f"Model: {args.model} ({args.provider})")
    print(f"Config: {args.config}, fewshot_set={args.fewshot_set}")

    # ── Load data ──
    print("\nLoading dataset...")
    records = load_dataset(args.task, args.data_dir)
    test_records = [r for r in records if r.get("split") == "test"]
    print(f"  Total: {len(records)}, Test: {len(test_records)}")

    # Get few-shot examples if needed
    fewshot_examples = None
    if args.config == "3-shot":
        fewshot_examples = get_fewshot_examples(records, args.fewshot_set)
        print(f"  Few-shot examples: {len(fewshot_examples)}")

    # ── Initialize client ──
    print("\nInitializing LLM client...")
    client = LLMClient(
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # ── Run inference (with resume support) ──
    pred_path = run_dir / "predictions.jsonl"
    predictions = []
    completed_ids = set()

    # Resume: load existing predictions if present
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
            system, user = format_prompt(
                args.task, record, args.config, fewshot_examples
            )
            try:
                response = client.generate(user, system)
            except Exception as e:
                response = f"ERROR: {e}"
                print(f"  Error on example {i}: {e}")

            predictions.append(response)

            # Write prediction immediately (crash recovery)
            pred_record = {
                "question_id": record.get("question_id", f"Q{i}"),
                "prediction": response,
                "gold_answer": record.get("correct_answer"),
            }
            f.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
            f.flush()

            done = len(completed_ids) + j + 1
            if done % 50 == 0:
                elapsed = time.time() - start_time
                rate = (j + 1) / elapsed * 60
                print(f"  Progress: {done}/{len(test_records)} ({rate:.0f}/min)")

    elapsed = time.time() - start_time
    print(f"\nInference complete: {elapsed:.0f}s ({len(test_records)/elapsed*60:.0f}/min)")

    # ── Evaluate ──
    print("\nEvaluating...")
    metrics = compute_all_llm_metrics(args.task, predictions, test_records)

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_safe)

    print(f"\n=== Results ===")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    # ── Save metadata ──
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
