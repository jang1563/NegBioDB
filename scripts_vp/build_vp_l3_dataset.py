#!/usr/bin/env python3
"""Build VP-L3 reasoning dataset for LLM benchmark.

Generates 200 benign variant records with rich context for reasoning evaluation.
Split: fewshot 20 + val 20 + test 160

Output: exports/vp_llm/vp_l3_dataset.jsonl

Usage:
    PYTHONPATH=src python scripts_vp/build_vp_l3_dataset.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "exports" / "vp_llm"

N_TOTAL = 200


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build VP-L3 reasoning dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_vp.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "vp_l3_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_vp.llm_dataset import (
        apply_max_per_gene,
        assign_splits,
        construct_l3_context,
        load_vp_candidate_pool,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)

    # Gold/silver tier benign variants with rich annotations
    df = load_vp_candidate_pool(
        args.db,
        tier_filter="IN ('gold', 'silver')",
        classification_filter="IN ('benign', 'likely_benign')",
        require_scores=True,
        limit=N_TOTAL * 5,
    )
    logger.info("Candidates: %d", len(df))

    df = apply_max_per_gene(df, max_per_gene=3, rng=rng)

    if len(df) > N_TOTAL:
        df = df.sample(n=N_TOTAL, random_state=rng).reset_index(drop=True)

    df = assign_splits(df, fewshot_size=20, val_size=20, test_size=160, seed=args.seed)

    # Build JSONL records
    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        context = construct_l3_context(row)
        rec = {
            "question_id": f"VPL3-{i:04d}",
            "task": "vp-l3",
            "split": row["split"],
            "context_text": context,
            "gold_answer": row.get("classification", "benign"),
            "gold_reasoning": None,  # To be filled by fewshot examples with human reasoning
            "metadata": {
                "gene_symbol": row.get("gene_symbol"),
                "variant_id": int(row["variant_id"]) if row.get("variant_id") is not None else None,
                "confidence_tier": row.get("confidence_tier"),
                "consequence_type": row.get("consequence_type"),
            },
        }
        records.append(rec)

    write_jsonl(records, args.output)
    write_dataset_metadata(args.output.parent, "vp-l3", {
        "n_total": len(records),
        "seed": args.seed,
        "split_distribution": dict(df["split"].value_counts()),
    })

    logger.info("VP-L3 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
