#!/usr/bin/env python3
"""Run LLM-as-Judge evaluation for L3 reasoning predictions.

Reads L3 predictions from results/llm/l3_*/predictions.jsonl,
sends each response through a judge model with L3_JUDGE_PROMPT,
and computes aggregate scores.

Usage:
  python scripts/run_l3_judge.py --judge-provider gemini --judge-model gemini-2.5-flash
  python scripts/run_l3_judge.py --run-dir results/llm/l3_llama70b_3-shot_fs0 \
      --judge-provider openai --judge-model gpt-4o-mini

Output per run:
  results/llm/{run}_judged/
      judge_scores.jsonl   — per-item judge scores
      results.json         — aggregated L3 metrics
      judge_meta.json      — judge model info, timestamp
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from negbiodb.llm_client import LLMClient
from negbiodb.llm_eval import L3_JUDGE_PROMPT, evaluate_l3, parse_l3_judge_scores

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "llm"
DATA_DIR = PROJECT_ROOT / "exports" / "llm_benchmarks"
L3_GOLD_FILE = DATA_DIR / "l3_reasoning_pilot.jsonl"


def load_gold_records() -> list[dict]:
    """Load L3 gold records."""
    records = []
    with open(L3_GOLD_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_predictions(pred_path: Path) -> list[dict]:
    """Load predictions from JSONL."""
    preds = []
    with open(pred_path) as f:
        for line in f:
            preds.append(json.loads(line))
    return preds


def find_l3_runs(results_dir: Path) -> list[Path]:
    """Find all L3 run directories."""
    runs = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and d.name.startswith("l3_"):
            pred_file = d / "predictions.jsonl"
            if pred_file.exists():
                runs.append(d)
    return runs


def judge_run(
    run_dir: Path,
    gold_records: list[dict],
    client: LLMClient,
    judge_model: str,
) -> dict:
    """Judge all predictions in a run directory.

    Returns aggregated L3 metrics.
    """
    # Build gold lookup by question_id
    gold_by_id = {}
    for rec in gold_records:
        gold_by_id[rec["question_id"]] = rec

    # Load predictions
    pred_path = run_dir / "predictions.jsonl"
    predictions = load_predictions(pred_path)

    # Output dir
    judged_dir = run_dir.parent / f"{run_dir.name}_judged"
    judged_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load existing judge scores
    scores_path = judged_dir / "judge_scores.jsonl"
    completed = {}
    if scores_path.exists():
        with open(scores_path) as f:
            for line in f:
                rec = json.loads(line)
                completed[rec["question_id"]] = rec
        print(f"  Resume: {len(completed)} already judged")

    # Judge remaining predictions
    remaining = [p for p in predictions if p["question_id"] not in completed]
    print(f"  Judging {len(remaining)} remaining of {len(predictions)} total")

    start_time = time.time()
    with open(scores_path, "a") as f:
        for i, pred_rec in enumerate(remaining):
            qid = pred_rec["question_id"]
            gold = gold_by_id.get(qid)
            if gold is None:
                print(f"  Warning: no gold record for {qid}, skipping")
                continue

            # Format judge prompt
            prompt = L3_JUDGE_PROMPT.format(
                compound_name=gold.get("compound_name", "Unknown"),
                target_gene=gold.get("target_gene") or "Unknown",
                target_uniprot=gold.get("target_uniprot", "Unknown"),
                response=pred_rec["prediction"],
            )

            try:
                judge_response = client.generate(prompt)
            except Exception as e:
                print(f"  Error judging {qid}: {e}")
                judge_response = f"ERROR: {e}"

            # Parse scores
            scores = parse_l3_judge_scores(judge_response)

            score_rec = {
                "question_id": qid,
                "judge_response": judge_response,
                "scores": scores,
            }
            f.write(json.dumps(score_rec, ensure_ascii=False) + "\n")
            f.flush()
            completed[qid] = score_rec

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                print(f"  Progress: {i + 1}/{len(remaining)} ({rate:.1f}/min)")

    elapsed = time.time() - start_time
    print(f"  Judging complete: {elapsed:.0f}s")

    # Aggregate scores
    all_scores = []
    for pred_rec in predictions:
        qid = pred_rec["question_id"]
        if qid in completed and completed[qid].get("scores"):
            all_scores.append(completed[qid]["scores"])
        else:
            all_scores.append(None)

    metrics = evaluate_l3(all_scores)

    # Save results
    results_path = judged_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save metadata
    meta = {
        "judge_model": judge_model,
        "source_run": run_dir.name,
        "n_predictions": len(predictions),
        "n_judged": sum(1 for s in all_scores if s is not None),
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = judged_dir / "judge_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run L3 LLM-as-Judge evaluation")
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Single run directory to judge (default: all L3 runs)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
    )
    parser.add_argument(
        "--judge-provider", default="gemini",
        choices=["gemini", "openai", "vllm"],
    )
    parser.add_argument(
        "--judge-model", default="gemini-2.5-flash",
    )
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    # Load gold data
    print("Loading L3 gold records...")
    gold_records = load_gold_records()
    test_records = [r for r in gold_records if r.get("split") == "test"]
    print(f"  Total: {len(gold_records)}, Test: {len(test_records)}")

    # Initialize judge client
    print(f"\nInitializing judge: {args.judge_model} ({args.judge_provider})")
    client = LLMClient(
        provider=args.judge_provider,
        model=args.judge_model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Find runs to judge
    if args.run_dir:
        runs = [args.run_dir]
    else:
        runs = find_l3_runs(args.results_dir)
    print(f"  Found {len(runs)} L3 runs to judge")

    # Judge each run
    all_results = {}
    for run_dir in runs:
        print(f"\n=== Judging: {run_dir.name} ===")
        metrics = judge_run(run_dir, gold_records, client, args.judge_model)
        all_results[run_dir.name] = metrics

        # Print summary
        for dim in ["accuracy", "reasoning", "completeness", "specificity", "overall"]:
            if dim in metrics:
                m = metrics[dim]
                print(f"  {dim}: {m['mean']:.2f} ± {m['std']:.2f}")
        print(f"  Valid: {metrics.get('n_valid', 0)}/{metrics.get('n_total', 0)}")

    print(f"\n=== All {len(all_results)} runs judged ===")


if __name__ == "__main__":
    main()
