#!/usr/bin/env python3
"""Run LLM-as-Judge evaluation for DC-L3 reasoning predictions.

Reads DC-L3 predictions, sends through judge model, computes scores.

Usage:
  python scripts_dc/run_dc_l3_judge.py --judge-provider gemini --judge-model gemini-2.5-flash
  python scripts_dc/run_dc_l3_judge.py --run-dir results/dc_llm/dc-l3_qwen2-5-7b-instruct_3-shot_fs0

Output per run:
  results/dc_llm/{run}_judged/
      judge_scores.jsonl, results.json, judge_meta.json
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from negbiodb.llm_client import LLMClient
from negbiodb_dc.llm_eval import DC_L3_JUDGE_PROMPT, evaluate_dc_l3, parse_dc_l3_judge_scores

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "dc_llm"
DATA_DIR = PROJECT_ROOT / "exports" / "dc_llm"
L3_DATASET_FILE = DATA_DIR / "dc_l3_dataset.jsonl"


def load_gold_records() -> list[dict]:
    """Load DC-L3 gold records."""
    records = []
    with open(L3_DATASET_FILE) as f:
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
    """Find all DC-L3 run directories."""
    runs = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and d.name.startswith("dc-l3_"):
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
    """Judge all predictions in a run directory."""
    gold_by_id = {rec["question_id"]: rec for rec in gold_records if rec.get("split") == "test"}

    predictions = load_predictions(run_dir / "predictions.jsonl")

    judged_dir = run_dir.parent / f"{run_dir.name}_judged"
    judged_dir.mkdir(parents=True, exist_ok=True)

    # Resume support
    scores_path = judged_dir / "judge_scores.jsonl"
    completed = {}
    if scores_path.exists():
        with open(scores_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("scores") is not None:
                    completed[rec["question_id"]] = rec
        print(f"  Resume: {len(completed)} already judged")

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

            prompt = DC_L3_JUDGE_PROMPT.format(
                context_text=gold.get("context_text", ""),
                response_text=pred_rec["prediction"],
            )

            try:
                judge_response = client.generate(prompt)
            except Exception as e:
                print(f"  Error judging {qid}: {e}")
                judge_response = f"ERROR: {e}"

            scores = parse_dc_l3_judge_scores(judge_response)

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

    # Aggregate — DC-L3 evaluate_dc_l3 expects raw judge_response strings
    all_responses = []
    for pred_rec in predictions:
        qid = pred_rec["question_id"]
        if qid in completed:
            all_responses.append(completed[qid].get("judge_response", ""))
        else:
            all_responses.append("")

    metrics = evaluate_dc_l3(all_responses)

    with open(judged_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    meta = {
        "judge_model": judge_model,
        "source_run": run_dir.name,
        "n_predictions": len(predictions),
        "n_judged": sum(1 for r in all_responses if r and not r.startswith("ERROR:")),
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(judged_dir / "judge_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run DC-L3 LLM-as-Judge evaluation")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--judge-provider", default="gemini", choices=["gemini", "openai", "vllm"])
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    print("Loading DC-L3 gold records...")
    gold_records = load_gold_records()
    test_records = [r for r in gold_records if r.get("split") == "test"]
    print(f"  Total: {len(gold_records)}, Test: {len(test_records)}")

    print(f"\nInitializing judge: {args.judge_model} ({args.judge_provider})")
    client = LLMClient(
        provider=args.judge_provider,
        model=args.judge_model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.run_dir:
        runs = [args.run_dir]
    else:
        runs = find_l3_runs(args.results_dir)
    print(f"  Found {len(runs)} DC-L3 runs to judge")

    for run_dir in runs:
        print(f"\n=== Judging: {run_dir.name} ===")
        metrics = judge_run(run_dir, gold_records, client, args.judge_model)

        for dim in ["mechanistic_reasoning", "pathway_analysis",
                     "pharmacological_context", "therapeutic_relevance"]:
            mean = metrics.get(f"{dim}_mean", 0)
            std = metrics.get(f"{dim}_std", 0)
            print(f"  {dim}: {mean:.2f} +/- {std:.2f}")
        print(f"  Overall: {metrics.get('overall_mean', 0):.2f} +/- {metrics.get('overall_std', 0):.2f}")
        print(f"  Parsed: {metrics.get('n_parsed', 0)}/{metrics.get('n_total', 0)}")

    print(f"\n=== All {len(runs)} runs judged ===")


if __name__ == "__main__":
    main()
