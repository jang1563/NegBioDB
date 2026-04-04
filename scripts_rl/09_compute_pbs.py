#!/usr/bin/env python
"""Phase 4: Compute Publication Bias Score (PBS) before and after training.

Usage:
    python scripts_rl/09_compute_pbs.py \
        --before-dir results/negbiorl/phase4_eval/baseline/ \
        --after-dir results/negbiorl/phase4_eval/grpo/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from negbiorl.data_registry import ALL_DOMAINS, PROJECT_ROOT, get_gold_answer_field, load_jsonl, parse_l4_unified
from negbiorl.pbs_metric import compute_pbs, compute_pbs_delta


def main():
    parser = argparse.ArgumentParser(description="Compute PBS before/after training")
    parser.add_argument("--before-dir", type=Path, required=True)
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=ALL_DOMAINS)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "negbiorl" / "phase4_eval" / "pbs_analysis.json",
    )
    args = parser.parse_args()

    results = {}
    for domain in args.domains:
        before_path = args.before_dir / f"{domain}_l4_predictions.jsonl"
        after_path = args.after_dir / f"{domain}_l4_predictions.jsonl"

        if not before_path.exists() or not after_path.exists():
            print(f"  SKIP {domain}: missing predictions")
            continue

        before_preds = load_jsonl(before_path)
        after_preds = load_jsonl(after_path)

        gold_field = get_gold_answer_field(domain)

        def _parse_l4_predictions(preds: list[dict]) -> tuple[list, list]:
            """Parse raw L4 predictions through domain-specific parser."""
            parsed_preds, golds = [], []
            for p in preds:
                raw = p.get("prediction", "")
                answer, _ = parse_l4_unified(raw, domain) if raw else (None, None)
                parsed_preds.append(answer)
                golds.append(p.get(gold_field, p.get("gold_answer", "")))
            return parsed_preds, golds

        before_parsed, before_golds = _parse_l4_predictions(before_preds)
        after_parsed, after_golds = _parse_l4_predictions(after_preds)

        pbs_before = compute_pbs(before_parsed, before_golds)
        pbs_after = compute_pbs(after_parsed, after_golds)
        delta = compute_pbs_delta(pbs_before, pbs_after)

        results[domain] = {
            "before": pbs_before,
            "after": pbs_after,
            "delta": delta,
        }

        sign = "+" if delta["delta_pbs"] >= 0 else ""
        reduced = "REDUCED" if delta.get("bias_reduced", False) else "INCREASED"
        print(f"  {domain}: PBS {pbs_before['pbs']:.3f} → {pbs_after['pbs']:.3f} "
              f"({sign}{delta['delta_pbs']:.3f}) [{reduced}]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
