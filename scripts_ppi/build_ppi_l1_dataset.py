#!/usr/bin/env python3
"""Build PPI-L1 MCQ dataset for LLM benchmark.

Generates 1,200 four-way MCQ records across 4 evidence classes:
  A) Direct experimental (300) — IntAct gold/silver (co-IP, pulldown, etc.)
  B) Systematic screen   (300) — HuRI gold (Y2H screen)
  C) Computational inf.  (300) — huMAP silver (co-fractionation ML)
  D) Database absence    (300) — STRING bronze (zero combined score)

Difficulty: easy(40%), medium(35%), hard(25%)
Split: 240 fewshot (60/class) + 240 val (60/class) + 720 test (180/class)

Output: exports/ppi_llm/ppi_l1_dataset.jsonl

Usage:
    PYTHONPATH=src python scripts_ppi/build_ppi_l1_dataset.py
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
OUTPUT_DIR = PROJECT_ROOT / "exports" / "ppi_llm"

N_PER_CLASS = 300

# Difficulty proportions
FRAC_EASY = 0.40
FRAC_MEDIUM = 0.35
FRAC_HARD = 0.25

# Source → L1 gold answer mapping
SOURCE_CATEGORY = {
    "intact": "A",   # Direct experimental
    "huri": "B",     # Systematic screen
    "humap": "C",    # Computational inference
    "string": "D",   # Database score absence
}

SOURCE_LABEL = {
    "A": "direct_experimental",
    "B": "systematic_screen",
    "C": "computational_inference",
    "D": "database_absence",
}


def format_l1_context(row: pd.Series, difficulty: str) -> str:
    """Generate PPI-L1 context with evidence description."""
    from negbiodb_ppi.llm_dataset import construct_evidence_description

    gene1 = row.get("gene_symbol_1") or row.get("uniprot_1", "Protein_1")
    uniprot1 = row.get("uniprot_1", "")
    func1 = row.get("function_1", "")
    loc1 = row.get("location_1", "")

    gene2 = row.get("gene_symbol_2") or row.get("uniprot_2", "Protein_2")
    uniprot2 = row.get("uniprot_2", "")
    func2 = row.get("function_2", "")
    loc2 = row.get("location_2", "")

    lines = [
        f"Protein 1: {gene1} ({uniprot1})",
    ]
    if func1:
        lines.append(f"  Function: {func1[:200]}")
    if loc1:
        lines.append(f"  Location: {loc1}")

    lines.append(f"\nProtein 2: {gene2} ({uniprot2})")
    if func2:
        lines.append(f"  Function: {func2[:200]}")
    if loc2:
        lines.append(f"  Location: {loc2}")

    evidence = construct_evidence_description(row, difficulty=difficulty)
    lines.append(f"\nEvidence: {evidence}")

    return "\n".join(lines)


def sample_class(df: pd.DataFrame, n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Sample n records from df, with replacement if needed."""
    if len(df) >= n:
        return df.sample(n=n, random_state=rng, replace=False).reset_index(drop=True)
    else:
        logger.warning("Class has %d records, need %d. Sampling with replacement.", len(df), n)
        return df.sample(n=n, random_state=rng, replace=True).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build PPI-L1 MCQ dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_ppi.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "ppi_l1_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_ppi.llm_dataset import (
        apply_max_per_protein,
        assign_splits,
        load_ppi_candidate_pool,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)

    # Load candidates per source (limit large sources to 5x needed at SQL level)
    all_records = []
    for source, letter in SOURCE_CATEGORY.items():
        sql_limit = None if source == "intact" else N_PER_CLASS * 5
        df = load_ppi_candidate_pool(
            args.db, source_filter=f"= '{source}'", limit=sql_limit,
        )

        df = apply_max_per_protein(df, max_per_protein=10, rng=rng)
        df = sample_class(df, N_PER_CLASS, rng)
        df["gold_answer"] = letter
        df["gold_category"] = SOURCE_LABEL[letter]
        all_records.append(df)

    combined = pd.concat(all_records, ignore_index=True)
    logger.info("Combined: %d records across %d classes", len(combined), len(SOURCE_CATEGORY))

    # Assign difficulty
    n_total = len(combined)
    difficulties = (
        ["easy"] * int(n_total * FRAC_EASY)
        + ["medium"] * int(n_total * FRAC_MEDIUM)
        + ["hard"] * (n_total - int(n_total * FRAC_EASY) - int(n_total * FRAC_MEDIUM))
    )
    rng.shuffle(difficulties)
    combined["difficulty"] = difficulties[:len(combined)]

    # Assign splits (class-stratified)
    split_parts = []
    for letter in sorted(SOURCE_CATEGORY.values()):
        class_df = combined[combined["gold_answer"] == letter].copy()
        class_df = assign_splits(class_df, fewshot_size=60, val_size=60, test_size=180, seed=args.seed)
        split_parts.append(class_df)
    combined = pd.concat(split_parts, ignore_index=True)

    # Build JSONL records
    records = []
    for i, (_, row) in enumerate(combined.iterrows()):
        difficulty = row["difficulty"]
        context = format_l1_context(row, difficulty)
        rec = {
            "question_id": f"PPIL1-{i:04d}",
            "task": "ppi-l1",
            "split": row["split"],
            "difficulty": difficulty,
            "context_text": context,
            "gold_answer": row["gold_answer"],
            "gold_category": row["gold_category"],
            "metadata": {
                "source_db": row.get("source_db"),
                "confidence_tier": row.get("confidence_tier"),
                "result_id": int(row["result_id"]) if pd.notna(row.get("result_id")) else None,
                "gene_symbol_1": row.get("gene_symbol_1"),
                "gene_symbol_2": row.get("gene_symbol_2"),
                "detection_method": row.get("detection_method"),
            },
        }
        records.append(rec)

    write_jsonl(records, args.output)

    # Metadata
    stats = {
        "n_total": len(records),
        "n_per_class": {SOURCE_LABEL[k]: N_PER_CLASS for k in SOURCE_CATEGORY.values()},
        "difficulty_distribution": dict(combined["difficulty"].value_counts()),
        "split_distribution": dict(combined["split"].value_counts()),
        "seed": args.seed,
    }
    write_dataset_metadata(args.output.parent, "ppi-l1", stats)

    logger.info("PPI-L1 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
