#!/usr/bin/env python3
"""Run second-judge validation for L3 reasoning evaluation.

Selects 3 representative L3 runs (best/median/worst), re-judges them
with a second judge model (GPT-4o-mini), and saves scores for
inter-rater agreement analysis.

Avoids circularity: GPT-4o-mini predictions are excluded from the sample
since GPT-4o-mini is used as the second judge.

Usage:
    python scripts/run_l3_judge_validation.py \
        --judge-provider openai --judge-model gpt-4o-mini

Output:
    results/llm/judge_validation/
        gpt4o_mini_scores.jsonl   — per-item second-judge scores
        gpt4o_mini_meta.json      — judge model info, timestamp
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from negbiodb.llm_client import LLMClient
from negbiodb.llm_eval import L3_JUDGE_PROMPT, parse_l3_judge_scores

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "llm"
DATA_DIR = PROJECT_ROOT / "exports" / "llm_benchmarks"
L3_GOLD_FILE = DATA_DIR / "l3_reasoning_pilot.jsonl"

# 3 representative runs (avoiding GPT-4o-mini predictions to prevent circularity)
REPRESENTATIVE_RUNS = [
    "l3_gemini-2-5-flash_3-shot_fs0",         # best overall (~4.64)
    "l3_qwen2-5-32b-instruct-awq_3-shot_fs0", # median overall (~3.68)
    "l3_mistral-7b-instruct-v0-3_3-shot_fs0",  # worst overall (~3.18)
]


def load_gold_records() -> dict[str, dict]:
    """Load L3 gold records indexed by question_id."""
    records = {}
    with open(L3_GOLD_FILE) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["question_id"]] = rec
    return records


def load_predictions(run_dir: Path) -> list[dict]:
    """Load predictions from JSONL."""
    preds = []
    pred_path = run_dir / "predictions.jsonl"
    with open(pred_path) as f:
        for line in f:
            preds.append(json.loads(line))
    return preds


def main():
    parser = argparse.ArgumentParser(description="Run second-judge L3 validation")
    parser.add_argument(
        "--judge-provider", default="openai",
        choices=["openai", "gemini", "vllm", "anthropic"],
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
    )
    args = parser.parse_args()

    # Output directory
    out_dir = args.results_dir / "judge_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load gold records
    print("Loading L3 gold records...")
    gold_by_id = load_gold_records()
    print(f"  Loaded {len(gold_by_id)} gold records")

    # Initialize second judge
    judge_name = args.judge_model.replace("/", "-")
    print(f"\nInitializing second judge: {args.judge_model} ({args.judge_provider})")
    client = LLMClient(
        provider=args.judge_provider,
        model=args.judge_model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Resume support
    scores_path = out_dir / f"{judge_name}_scores.jsonl"
    completed = {}
    if scores_path.exists():
        with open(scores_path) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["source_run"], rec["question_id"])
                if rec.get("scores") is not None:
                    completed[key] = rec
        print(f"  Resume: {len(completed)} already judged")

    # Collect all items to judge
    items = []
    for run_name in REPRESENTATIVE_RUNS:
        run_dir = args.results_dir / run_name
        if not run_dir.exists():
            print(f"  WARNING: Run not found: {run_name}, skipping")
            continue
        predictions = load_predictions(run_dir)
        for pred_rec in predictions:
            qid = pred_rec["question_id"]
            if (run_name, qid) not in completed:
                items.append((run_name, qid, pred_rec["prediction"]))

    print(f"\n  Total items to judge: {len(items)} "
          f"(from {len(REPRESENTATIVE_RUNS)} runs)")

    # Judge
    start_time = time.time()
    with open(scores_path, "a") as f:
        for i, (run_name, qid, prediction) in enumerate(items):
            gold = gold_by_id.get(qid)
            if gold is None:
                print(f"  Warning: no gold for {qid}, skipping")
                continue

            prompt = L3_JUDGE_PROMPT.format(
                compound_name=gold.get("compound_name", "Unknown"),
                target_gene=gold.get("target_gene") or "Unknown",
                target_uniprot=gold.get("target_uniprot", "Unknown"),
                response=prediction,
            )

            try:
                judge_response = client.generate(prompt)
            except Exception as e:
                print(f"  Error judging {run_name}/{qid}: {e}")
                judge_response = f"ERROR: {e}"

            scores = parse_l3_judge_scores(judge_response)

            score_rec = {
                "source_run": run_name,
                "question_id": qid,
                "judge_response": judge_response,
                "scores": scores,
            }
            f.write(json.dumps(score_rec, ensure_ascii=False) + "\n")
            f.flush()
            completed[(run_name, qid)] = score_rec

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                print(f"  Progress: {i + 1}/{len(items)} ({rate:.1f}/min)")

    elapsed = time.time() - start_time
    n_valid = sum(1 for v in completed.values() if v.get("scores") is not None)
    print(f"\nJudging complete: {elapsed:.0f}s, {n_valid}/{len(completed)} valid")

    # Save metadata
    meta = {
        "judge_model": args.judge_model,
        "judge_provider": args.judge_provider,
        "representative_runs": REPRESENTATIVE_RUNS,
        "n_total_items": len(completed),
        "n_valid_scores": n_valid,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_dir / f"{judge_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
