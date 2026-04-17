#!/usr/bin/env python3
"""Build DC-L1 MCQ dataset for LLM benchmark.

Generates 1,200 four-way MCQ records:
  A) Strongly synergistic (300) — ZIP > 10
  B) Weakly synergistic/Additive (300) — -5 ≤ ZIP ≤ 10
  C) Antagonistic (300) — -10 ≤ ZIP < -5
  D) Strongly antagonistic (300) — ZIP < -10

Split: fewshot 200 (50/class) + val 200 (50/class) + test 800 (200/class)

Usage:
    PYTHONPATH=src python scripts_dc/build_dc_l1_dataset.py
"""

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
OUTPUT_DIR = PROJECT_ROOT / "exports" / "dc_llm"

N_PER_CLASS = 300

CLASS_LABELS = {
    "A": "strongly_synergistic",
    "B": "synergistic_additive",
    "C": "antagonistic",
    "D": "strongly_antagonistic",
}


def classify_by_zip(median_zip: float) -> str | None:
    """Map median ZIP (or Bliss fallback) score to L1 class letter."""
    if median_zip is None or pd.isna(median_zip):
        return None
    median_zip = float(median_zip)
    if not np.isfinite(median_zip):
        return None
    if median_zip > 10:
        return "A"
    elif median_zip >= -5:
        return "B"
    elif median_zip >= -10:
        return "C"
    else:
        return "D"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build DC-L1 MCQ dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_dc.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "dc_l1_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_dc.llm_dataset import (
        apply_max_per_drug,
        assign_splits,
        construct_l1_context,
        load_dc_candidate_pool,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)

    # Load all pairs with ZIP scores
    from negbiodb_dc.dc_db import get_connection
    conn = get_connection(args.db)
    try:
        df = load_dc_candidate_pool(conn, min_confidence="bronze")
    finally:
        conn.close()

    # Classify by ZIP threshold (fall back to Bliss when ZIP is NULL)
    score_col = df["median_zip"].where(df["median_zip"].notna(), df["median_bliss"])
    df["gold_answer"] = score_col.apply(classify_by_zip)
    df = df.dropna(subset=["gold_answer"])
    df["gold_category"] = df["gold_answer"].map(CLASS_LABELS)

    logger.info("Class distribution: %s", df["gold_answer"].value_counts().to_dict())

    # Sample balanced classes
    class_dfs = []
    for letter in sorted(CLASS_LABELS.keys()):
        cls = df[df["gold_answer"] == letter]
        if len(cls) > N_PER_CLASS:
            cls = cls.sample(n=N_PER_CLASS, random_state=rng)
        class_dfs.append(cls)

    combined = pd.concat(class_dfs, ignore_index=True)
    combined = apply_max_per_drug(combined, max_per_drug=15, rng=rng)
    logger.info("Combined after max-per-drug: %d records", len(combined))

    # Assign splits (class-stratified)
    split_parts = []
    for letter in sorted(CLASS_LABELS.keys()):
        class_df = combined[combined["gold_answer"] == letter].copy()
        class_df = assign_splits(class_df, fewshot_size=50, val_size=50, seed=args.seed)
        split_parts.append(class_df)
    combined = pd.concat(split_parts, ignore_index=True)

    # Build JSONL records
    records = []
    for i, (_, row) in enumerate(combined.iterrows()):
        rec = {
            "question_id": f"DCL1-{i:04d}",
            "task": "dc-l1",
            "split": row["split"],
            "context_text": construct_l1_context(row),
            "gold_answer": row["gold_answer"],
            "gold_category": row["gold_category"],
            "metadata": {
                "pair_id": int(row["pair_id"]),
                "drug_a": row["drug_a_name"],
                "drug_b": row["drug_b_name"],
                "median_zip": float(row["median_zip"]) if pd.notna(row["median_zip"]) else None,
                "median_bliss": float(row["median_bliss"]) if pd.notna(row["median_bliss"]) else None,
            },
        }
        records.append(rec)

    write_jsonl(records, args.output)
    write_dataset_metadata(
        args.output.with_suffix(".meta.json"), "dc-l1",
        len(records), dict(combined["split"].value_counts()),
        seed=args.seed,
    )

    logger.info("DC-L1 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
