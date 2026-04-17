#!/usr/bin/env python
"""Phase 2: Build GRPO dataset (prompts only) from NegBioDB exports.

Usage:
    python scripts_rl/03_build_grpo_dataset.py
    python scripts_rl/03_build_grpo_dataset.py --domains dti ct ppi ge --max-per 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

from negbiorl.data_registry import TRAIN_DOMAINS, PROJECT_ROOT
from negbiorl.sft_data import build_grpo_dataset, save_dataset


def main():
    parser = argparse.ArgumentParser(description="Build GRPO training dataset")
    parser.add_argument("--domains", nargs="+", default=TRAIN_DOMAINS)
    parser.add_argument("--tasks", nargs="+", default=["l1", "l4"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-per", type=int, default=None)
    parser.add_argument("--output-name", default="grpo_dataset.jsonl", help="Output filename")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "negbiorl" / "phase2_training_data",
    )
    args = parser.parse_args()

    print(f"Building GRPO dataset: domains={args.domains}, tasks={args.tasks}")
    records = build_grpo_dataset(
        domains=args.domains,
        tasks=args.tasks,
        split=args.split,
        max_per_domain_task=args.max_per,
    )

    out_path = args.output_dir / args.output_name
    count = save_dataset(records, out_path)
    print(f"Saved {count} GRPO records to {out_path}")

    from collections import Counter
    dist = Counter((r["domain"], r["task"]) for r in records)
    for (d, t), c in sorted(dist.items()):
        print(f"  {d}/{t}: {c}")


if __name__ == "__main__":
    main()
