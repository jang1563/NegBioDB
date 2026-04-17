#!/usr/bin/env python3
"""Build CP-L1 4-way MCQ dataset for the Cell Painting LLM benchmark."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "exports" / "cp_llm"

N_PER_CLASS = 125
FEWSHOT_PER_CLASS = 10
VAL_PER_CLASS = 15


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build CP-L1 MCQ dataset.")
    parser.add_argument("--db", "--db-path", dest="db", type=Path,
                        default=PROJECT_ROOT / "data" / "negbiodb_cp.db")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    parser.add_argument("--fewshot-per-class", type=int, default=FEWSHOT_PER_CLASS)
    parser.add_argument("--val-per-class", type=int, default=VAL_PER_CLASS)
    parser.add_argument(
        "--min-confidence",
        choices=["gold", "silver", "bronze", "copper"],
        default="bronze",
    )
    parser.add_argument("--allow-proxy-smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output_path = args.output or ((args.output_dir or OUTPUT_DIR) / "cp_l1_dataset.jsonl")

    from negbiodb_cp.llm_dataset import (
        OUTCOME_TO_L1_LETTER,
        apply_max_per_compound,
        assign_splits,
        construct_l1_context,
        difficulty_from_tier,
        load_cp_annotation_summary,
        load_cp_candidate_pool,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)
    df = load_cp_candidate_pool(
        args.db,
        min_confidence=args.min_confidence,
        allow_proxy_smoke=args.allow_proxy_smoke,
    )
    if df.empty:
        logger.error("No CP candidates found in %s", args.db)
        return 1

    df = apply_max_per_compound(df, max_per_compound=8, rng=rng)

    class_frames = []
    for label, letter in OUTCOME_TO_L1_LETTER.items():
        subset = df[df["outcome_label"] == label].copy()
        if subset.empty:
            logger.warning("Skipping empty CP-L1 class: %s", label)
            continue
        take = min(args.n_per_class, len(subset))
        if len(subset) > take:
            subset = subset.sample(n=take, random_state=args.seed).reset_index(drop=True)
        subset = assign_splits(
            subset,
            fewshot_size=min(args.fewshot_per_class, len(subset)),
            val_size=min(args.val_per_class, max(0, len(subset) - args.fewshot_per_class)),
            test_size=max(0, len(subset) - args.fewshot_per_class - args.val_per_class),
            seed=args.seed,
        )
        subset["gold_answer"] = letter
        class_frames.append(subset)

    if not class_frames:
        logger.error("No CP-L1 classes were available for export.")
        return 1

    combined = pd.concat(class_frames, ignore_index=True)
    records = []
    for i, (_, row) in enumerate(combined.iterrows()):
        records.append(
            {
                "question_id": f"CPL1-{i:04d}",
                "task": "cp-l1",
                "split": row["split"],
                "difficulty": difficulty_from_tier(row["confidence_tier"]),
                "context_text": construct_l1_context(row),
                "gold_answer": row["gold_answer"],
                "gold_category": row["outcome_label"],
                "metadata": {
                    "cp_result_id": int(row["cp_result_id"]),
                    "compound_name": row["compound_name"],
                    "batch_name": row["batch_name"],
                    "confidence_tier": row["confidence_tier"],
                },
            }
        )

    write_jsonl(records, output_path)
    write_dataset_metadata(
        output_path.parent,
        "cp-l1",
        {
            "n_total": len(records),
            "seed": args.seed,
            "min_confidence": args.min_confidence,
            "class_distribution": dict(combined["outcome_label"].value_counts()),
            "split_distribution": dict(combined["split"].value_counts()),
            **load_cp_annotation_summary(args.db),
        },
    )
    logger.info("CP-L1 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
