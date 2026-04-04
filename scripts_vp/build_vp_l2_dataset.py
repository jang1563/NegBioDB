#!/usr/bin/env python3
"""Build VP-L2 structured extraction dataset for LLM benchmark.

Generates 500 variant interpretation reports with gold extraction labels.
Split: fewshot 50 + val 50 + test 400

Output: exports/vp_llm/vp_l2_dataset.jsonl

Usage:
    PYTHONPATH=src python scripts_vp/build_vp_l2_dataset.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "exports" / "vp_llm"

N_TOTAL = 500


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build VP-L2 extraction dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_vp.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "vp_l2_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_vp.llm_dataset import (
        apply_max_per_gene,
        assign_splits,
        construct_l2_context,
        load_vp_candidate_pool,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)

    # Load candidates with ACMG criteria
    df = load_vp_candidate_pool(
        args.db,
        tier_filter="IN ('gold', 'silver', 'bronze')",
        require_scores=True,
        extra_where="nr.acmg_criteria IS NOT NULL AND nr.acmg_criteria != '[]'",
        limit=N_TOTAL * 3,
    )
    logger.info("Candidates with ACMG criteria: %d", len(df))

    # If not enough with criteria, also include without
    if len(df) < N_TOTAL:
        df_extra = load_vp_candidate_pool(
            args.db,
            tier_filter="IN ('gold', 'silver')",
            require_scores=True,
            limit=N_TOTAL * 2,
        )
        df = df._append(df_extra, ignore_index=True).drop_duplicates(subset=["result_id"])
        logger.info("After adding extra: %d", len(df))

    df = apply_max_per_gene(df, max_per_gene=5, rng=rng)

    if len(df) > N_TOTAL:
        df = df.sample(n=N_TOTAL, random_state=rng).reset_index(drop=True)

    df = assign_splits(df, fewshot_size=50, val_size=50, test_size=400, seed=args.seed)

    # Build JSONL records
    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        context = construct_l2_context(row)

        # Build gold extraction
        acmg_raw = row.get("acmg_criteria") or "[]"
        try:
            acmg_list = json.loads(acmg_raw) if isinstance(acmg_raw, str) else (acmg_raw or [])
        except (json.JSONDecodeError, TypeError):
            acmg_list = []

        gold_extraction = {
            "variants": [{
                "gene": row.get("gene_symbol", ""),
                "hgvs": row.get("hgvs_coding") or "",
                "classification": row.get("classification", "likely_benign"),
                "acmg_criteria_met": acmg_list,
                "population_frequency": float(row["gnomad_af_global"]) if row.get("gnomad_af_global") is not None else None,
                "condition": row.get("disease_name", "not specified"),
            }],
            "total_variants_discussed": 1,
            "classification_method": "ACMG/AMP",
        }

        rec = {
            "question_id": f"VPL2-{i:04d}",
            "task": "vp-l2",
            "split": row["split"],
            "context_text": context,
            "gold_extraction": gold_extraction,
            "metadata": {
                "gene_symbol": row.get("gene_symbol"),
                "variant_id": int(row["variant_id"]) if row.get("variant_id") is not None else None,
                "confidence_tier": row.get("confidence_tier"),
            },
        }
        records.append(rec)

    write_jsonl(records, args.output)
    write_dataset_metadata(args.output.parent, "vp-l2", {
        "n_total": len(records),
        "seed": args.seed,
        "split_distribution": dict(df["split"].value_counts()),
    })

    logger.info("VP-L2 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
