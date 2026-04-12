#!/usr/bin/env python3
"""Run LLM evaluation for MD domain (L1-L4).

Evaluates a model against pre-built JSONL datasets and saves results.

Usage:
    # Zero-shot, Anthropic API
    python scripts_md/08_run_llm_eval.py --model claude-sonnet-4-6 --level l1

    # 3-shot, specific dataset file
    python scripts_md/08_run_llm_eval.py --model gpt-4o-mini --level l2 --fewshot-set 0

    # All levels
    python scripts_md/08_run_llm_eval.py --model gemini-2.5-flash --level all
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run MD LLM evaluation")
    parser.add_argument("--model", required=True,
                        help="Model identifier (claude-sonnet-4-6, gpt-4o-mini, etc.)")
    parser.add_argument("--level", default="all",
                        choices=["l1", "l2", "l3", "l4", "all"],
                        help="Benchmark level to evaluate")
    parser.add_argument("--fewshot-set", type=int, default=None,
                        choices=[0, 1, 2], help="Few-shot seed set (None = zero-shot)")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing md_l*.jsonl files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results JSON")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Limit records per level (for testing)")
    args = parser.parse_args()

    from negbiodb_md.llm_eval import (
        eval_l1, eval_l2, eval_l4, load_jsonl, save_results
    )
    from negbiodb_md.llm_prompts import (
        format_l1_prompt, format_l2_prompt, format_l3_prompt, format_l4_prompt,
        MD_SYSTEM_PROMPT,
    )
    from negbiodb.llm_client import call_llm  # shared LLM client

    input_dir = Path(args.input_dir) if args.input_dir else _PROJECT_ROOT / "exports" / "md_llm"
    output_dir = Path(args.output_dir) if args.output_dir else _PROJECT_ROOT / "results" / "md_llm"
    output_dir.mkdir(parents=True, exist_ok=True)

    shot_label = f"3shot_set{args.fewshot_set}" if args.fewshot_set is not None else "0shot"
    levels = ["l1", "l2", "l3", "l4"] if args.level == "all" else [args.level]

    for level in levels:
        dataset_path = input_dir / f"md_{level}.jsonl"
        if not dataset_path.exists():
            logger.warning("Dataset not found: %s (run 07_build_llm_datasets.py first)", dataset_path)
            continue

        records = load_jsonl(dataset_path)
        if args.max_records:
            records = records[:args.max_records]
        logger.info("Evaluating %s on MD-%s (%d records)", args.model, level.upper(), len(records))

        # Format prompts
        prompt_fn = {"l1": format_l1_prompt, "l2": format_l2_prompt,
                     "l3": format_l3_prompt, "l4": format_l4_prompt}[level]
        responses = []
        for rec in records:
            prompt = prompt_fn(rec)
            response = call_llm(
                model=args.model,
                system=MD_SYSTEM_PROMPT,
                user=prompt,
            )
            responses.append(response or "")

        # Compute metrics
        if level == "l1":
            metrics = eval_l1(records, responses)
        elif level == "l2":
            metrics = eval_l2(records, responses)
        elif level == "l4":
            metrics = eval_l4(records, responses)
        else:
            # L3: requires judge — save responses for separate judge pass
            metrics = {"n": len(records), "note": "L3 responses saved; run L3 judge separately"}

        result = {
            "model": args.model,
            "level": level,
            "shot_type": shot_label,
            "metrics": metrics,
            "n_records": len(records),
        }
        out_path = output_dir / f"md_{level}_{args.model.replace('/', '_')}_{shot_label}.json"
        save_results(result, out_path)

        # Save raw responses for L3 judge
        if level == "l3":
            resp_path = output_dir / f"md_l3_responses_{args.model.replace('/', '_')}_{shot_label}.jsonl"
            with open(resp_path, "w") as f:
                for rec, resp in zip(records, responses):
                    f.write(json.dumps({"record_id": rec["record_id"], "response": resp}) + "\n")

        print(f"MD-{level.upper()} ({args.model}, {shot_label}): {metrics}")


if __name__ == "__main__":
    main()
