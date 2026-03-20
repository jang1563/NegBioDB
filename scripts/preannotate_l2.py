#!/usr/bin/env python3
"""Pre-annotate L2 candidates with LLM extraction for human correction.

Reads l2_candidates.jsonl, runs each abstract through an LLM using the
L2 zero-shot prompt, and saves the pre-annotated results for human review.

Usage:
    python scripts/preannotate_l2.py --provider openai --model gpt-4o-mini
    python scripts/preannotate_l2.py --provider anthropic --model claude-sonnet-4-6

Output:
    exports/llm_benchmarks/l2_preannotated.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from negbiodb.llm_client import LLMClient
from negbiodb.llm_eval import parse_l2_response
from negbiodb.llm_prompts import SYSTEM_PROMPT, format_l2_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "exports" / "llm_benchmarks"
CANDIDATES_FILE = DATA_DIR / "l2_candidates.jsonl"
OUTPUT_FILE = DATA_DIR / "l2_preannotated.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Pre-annotate L2 candidates")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "gemini", "vllm", "anthropic"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    # Load candidates
    candidates = []
    with open(CANDIDATES_FILE) as f:
        for line in f:
            candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates")

    # Resume: check existing output
    completed = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                rec = json.loads(line)
                completed.add(rec["candidate_id"])
        print(f"  Resume: {len(completed)} already processed")

    remaining = [c for c in candidates if c["candidate_id"] not in completed]
    print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("All candidates already processed.")
        return

    # Initialize client
    print(f"\nInitializing: {args.model} ({args.provider})")
    client = LLMClient(
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Process each candidate
    start_time = time.time()
    with open(OUTPUT_FILE, "a") as out_f:
        for i, candidate in enumerate(remaining):
            system, user = format_l2_prompt(
                {"abstract_text": candidate["abstract_text"]},
                config="zero-shot",
            )

            try:
                response = client.generate(user, system)
            except Exception as e:
                print(f"  Error on {candidate['candidate_id']}: {e}")
                response = f"ERROR: {e}"

            # Parse LLM extraction
            extraction = parse_l2_response(response)

            output_rec = {
                **candidate,
                "llm_response": response,
                "llm_extraction": extraction,
                "preannotation_model": args.model,
            }
            out_f.write(json.dumps(output_rec, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                print(f"  Progress: {i + 1}/{len(remaining)} ({rate:.1f}/min)")

    elapsed = time.time() - start_time
    print(f"\nDone: {len(remaining)} processed in {elapsed:.0f}s")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
