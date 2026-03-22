#!/usr/bin/env python3
"""Build PPI-L2 extraction dataset for LLM benchmark.

Generates 500 evidence extraction records using constructed evidence summaries
(fallback design — only 65 unique IntAct PMIDs, insufficient for PubMed extraction).

Each record contains a multi-pair evidence summary with 1-3 non-interacting pairs.
Gold standard derived from database fields.

Split: 50 fewshot + 50 val + 400 test

Output: exports/ppi_llm/ppi_l2_dataset.jsonl

Usage:
    PYTHONPATH=src python scripts_ppi/build_ppi_l2_dataset.py
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

N_TOTAL = 500
PAIRS_PER_RECORD = [1, 1, 2, 2, 3]  # Distribution: ~40% 1-pair, ~40% 2-pair, ~20% 3-pair


def construct_multi_pair_evidence(
    rows: list[pd.Series],
    include_positive: bool = False,
) -> tuple[str, dict]:
    """Build a multi-pair evidence summary and gold extraction.

    Returns (evidence_text, gold_extraction_dict).
    """
    from negbiodb_ppi.llm_dataset import DETECTION_METHOD_DESCRIPTIONS

    method = rows[0].get("detection_method", "experimental")
    source = rows[0].get("source_db", "unknown")
    method_desc = DETECTION_METHOD_DESCRIPTIONS.get(method, method) if method else "binding assay"

    # Build evidence text
    lines = []
    if source == "intact":
        lines.append(
            f"A study using {method_desc} tested multiple protein pairs for physical interaction."
        )
    elif source == "huri":
        lines.append(
            "A systematic yeast two-hybrid screen tested a panel of human protein "
            "pairs for binary interactions."
        )
    elif source == "humap":
        lines.append(
            "Computational analysis of co-fractionation proteomics data was used "
            "to predict protein complex membership for multiple protein pairs."
        )
    else:
        lines.append(
            "An integrated analysis across multiple evidence channels evaluated "
            "potential interactions between several protein pairs."
        )

    lines.append("")

    # Add pair-specific results
    gold_pairs = []
    for row in rows:
        gene1 = row.get("gene_symbol_1") or row.get("uniprot_1", "Protein")
        gene2 = row.get("gene_symbol_2") or row.get("uniprot_2", "Protein")
        lines.append(f"- {gene1} and {gene2}: no interaction detected")

        # Determine evidence strength
        tier = row.get("confidence_tier", "bronze")
        strength = {"gold": "strong", "silver": "moderate", "bronze": "weak"}.get(tier, "moderate")

        gold_pairs.append({
            "protein_1": gene1,
            "protein_2": gene2,
            "method": method_desc if method_desc else "experimental assay",
            "evidence_strength": strength,
        })

    if include_positive:
        lines.append("- Several other tested pairs showed positive interactions")

    evidence_text = "\n".join(lines)
    gold_extraction = {
        "non_interacting_pairs": gold_pairs,
        "total_negative_count": len(gold_pairs),
        "positive_interactions_mentioned": include_positive,
    }

    return evidence_text, gold_extraction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build PPI-L2 extraction dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_ppi.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "ppi_l2_dataset.jsonl")
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

    # Load from all sources (limit to 5000 — we only need ~1000 pair rows)
    df = load_ppi_candidate_pool(args.db, limit=5000)
    df = apply_max_per_protein(df, max_per_protein=5, rng=rng)

    # Shuffle
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    # Build records with varying pair counts
    records = []
    idx = 0
    record_count = 0

    while record_count < N_TOTAL and idx < len(df):
        n_pairs = PAIRS_PER_RECORD[record_count % len(PAIRS_PER_RECORD)]
        if idx + n_pairs > len(df):
            n_pairs = 1  # fallback to single pair

        pair_rows = [df.iloc[idx + j] for j in range(n_pairs)]
        idx += n_pairs

        # 30% chance of mentioning positive interactions
        include_positive = rng.random() < 0.30

        evidence_text, gold_extraction = construct_multi_pair_evidence(
            pair_rows, include_positive=include_positive,
        )

        rec = {
            "question_id": f"PPIL2-{record_count:04d}",
            "task": "ppi-l2",
            "split": "test",  # Will be overwritten by assign_splits
            "difficulty": "medium",
            "context_text": evidence_text,
            "gold_answer": gold_extraction.get("non_interacting_pairs", [{}])[0].get("method", ""),
            "gold_category": pair_rows[0].get("source_db", "unknown"),
            "gold_extraction": gold_extraction,
            "metadata": {
                "n_pairs": n_pairs,
                "source_db": pair_rows[0].get("source_db"),
                "result_ids": [int(r["result_id"]) for r in pair_rows if pd.notna(r.get("result_id"))],
                "include_positive": include_positive,
            },
        }
        records.append(rec)
        record_count += 1

    logger.info("Built %d L2 records from %d pair rows", len(records), idx)

    # Convert to df for split assignment
    records_df = pd.DataFrame({"idx": range(len(records))})
    records_df = assign_splits(records_df, fewshot_size=50, val_size=50, test_size=400, seed=args.seed)

    for i, (_, row) in enumerate(records_df.iterrows()):
        if i < len(records):
            records[i]["split"] = row["split"]

    write_jsonl(records, args.output)

    stats = {
        "n_total": len(records),
        "design": "constructed_evidence_fallback",
        "pair_distribution": dict(pd.Series([r["metadata"]["n_pairs"] for r in records]).value_counts()),
        "split_distribution": dict(pd.Series([r["split"] for r in records]).value_counts()),
        "seed": args.seed,
    }
    write_dataset_metadata(args.output.parent, "ppi-l2", stats)

    logger.info("PPI-L2 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
