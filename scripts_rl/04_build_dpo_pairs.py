#!/usr/bin/env python
"""Phase 2: Build DPO pairs for the DPO-vs-GRPO ablation study.

Usage:
    python scripts_rl/04_build_dpo_pairs.py
    python scripts_rl/04_build_dpo_pairs.py --min-l3-gap 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from negbiorl.data_registry import TRAIN_DOMAINS, PROJECT_ROOT
from negbiorl.dpo_pairs import build_all_dpo_pairs, save_dpo_pairs


def main():
    parser = argparse.ArgumentParser(description="Build DPO training pairs")
    parser.add_argument("--domains", nargs="+", default=TRAIN_DOMAINS)
    parser.add_argument("--min-l3-gap", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "negbiorl" / "phase2_training_data",
    )
    args = parser.parse_args()

    pairs = build_all_dpo_pairs(
        domains=args.domains,
        min_l3_score_gap=args.min_l3_gap,
    )

    out_path = args.output_dir / "dpo_pairs.jsonl"
    count = save_dpo_pairs(pairs, out_path)
    print(f"Saved {count} DPO pairs to {out_path}")

    from collections import Counter
    dist = Counter(p["task"] for p in pairs)
    for task, c in sorted(dist.items()):
        print(f"  {task}: {c} pairs")


if __name__ == "__main__":
    main()
