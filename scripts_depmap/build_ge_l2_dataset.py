#!/usr/bin/env python3
"""Build GE-L2 extraction dataset for LLM benchmark.

Generates 500 evidence extraction records using constructed evidence descriptions
(DepMap publications don't reference individual gene-cell_line pairs in abstracts).

Each record contains a 1-3 gene essentiality finding per cell line.
Gold standard derived from database fields.

Split: 50 fewshot + 50 val + 400 test

Output: exports/ge_llm/ge_l2_dataset.jsonl

Usage:
    PYTHONPATH=src python scripts_depmap/build_ge_l2_dataset.py
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
OUTPUT_DIR = PROJECT_ROOT / "exports" / "ge_llm"

N_TOTAL = 500
GENES_PER_RECORD = [1, 1, 2, 2, 3]  # ~40% 1-gene, ~40% 2-gene, ~20% 3-gene


def construct_multi_gene_evidence(rows: list[pd.Series]) -> tuple[str, dict]:
    """Build a multi-gene evidence summary and gold extraction.

    Returns (evidence_text, gold_extraction_dict).
    """
    cell_line = rows[0].get("ccle_name") or rows[0].get("model_id", "UNKNOWN")
    lineage = rows[0].get("lineage", "unknown lineage")

    lines = [
        f"A genome-wide CRISPR-Cas9 knockout screen was performed in {cell_line} "
        f"({lineage}) as part of the Cancer Dependency Map (DepMap) project.",
        "",
        "Gene-level essentiality results (Chronos algorithm):",
    ]

    gold_genes = []
    for row in rows:
        gene = row.get("gene_symbol", "UNKNOWN")
        effect = row.get("gene_effect_score")
        dep_prob = row.get("dependency_probability")
        evidence_type = row.get("evidence_type", "crispr_nonessential")
        tier = row.get("confidence_tier", "bronze")

        score_str = f"gene effect = {effect:.3f}" if effect is not None and not pd.isna(effect) else "score unavailable"
        prob_str = f"dependency probability = {dep_prob:.3f}" if dep_prob is not None and not pd.isna(dep_prob) else ""
        parts = [score_str]
        if prob_str:
            parts.append(prob_str)

        lines.append(f"- {gene}: {', '.join(parts)} → classified as non-essential")

        gold_genes.append({
            "gene_name": gene,
            "cell_line": cell_line,
            "screen_method": "CRISPR" if "crispr" in evidence_type else "RNAi",
            "essentiality_status": "non-essential",
            "dependency_score": float(effect) if effect is not None and not pd.isna(effect) else None,
        })

    evidence_text = "\n".join(lines)
    gold_extraction = {
        "genes": gold_genes,
        "total_genes_mentioned": len(gold_genes),
        "cell_line": cell_line,
        "screen_type": "CRISPR",
    }

    return evidence_text, gold_extraction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build GE-L2 extraction dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_depmap.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "ge_l2_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_depmap.depmap_db import get_connection
    from negbiodb_depmap.llm_dataset import (
        apply_max_per_gene,
        assign_splits,
        load_ge_candidate_pool,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)

    conn = get_connection(args.db)
    try:
        df = load_ge_candidate_pool(conn, min_confidence="silver")
    finally:
        conn.close()

    df = apply_max_per_gene(df, max_per_gene=5, seed=args.seed)
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    # Build records with varying gene counts per record
    records = []
    idx = 0
    record_count = 0

    while record_count < N_TOTAL and idx < len(df):
        n_genes = GENES_PER_RECORD[record_count % len(GENES_PER_RECORD)]
        if idx + n_genes > len(df):
            n_genes = 1

        gene_rows = [df.iloc[idx + j] for j in range(n_genes)]
        idx += n_genes

        evidence_text, gold_extraction = construct_multi_gene_evidence(gene_rows)

        rec = {
            "question_id": f"GEL2-{record_count:04d}",
            "task": "ge-l2",
            "split": "test",
            "difficulty": "medium",
            "context_text": evidence_text,
            "gold_answer": gold_extraction["genes"][0].get("gene_name", ""),
            "gold_category": gene_rows[0].get("source_db", "depmap"),
            "gold_extraction": gold_extraction,
            "metadata": {
                "n_genes": n_genes,
                "source_db": gene_rows[0].get("source_db"),
                "gene_symbols": [r.get("gene_symbol") for r in gene_rows],
                "cell_line": gene_rows[0].get("ccle_name") or gene_rows[0].get("model_id"),
            },
        }
        records.append(rec)
        record_count += 1

    logger.info("Built %d L2 records from %d gene rows", len(records), idx)

    # Assign splits
    records_df = pd.DataFrame({"idx": range(len(records))})
    records_df = assign_splits(records_df, ratios={"train": 0.1, "val": 0.1, "test": 0.8})
    records_df["split"] = records_df["split"].replace({"train": "fewshot"})

    for i, (_, row) in enumerate(records_df.iterrows()):
        if i < len(records):
            records[i]["split"] = row["split"]

    write_jsonl(records, args.output)

    stats = {
        "n_total": len(records),
        "design": "constructed_evidence",
        "gene_distribution": dict(pd.Series([r["metadata"]["n_genes"] for r in records]).value_counts()),
        "split_distribution": dict(pd.Series([r["split"] for r in records]).value_counts()),
        "seed": args.seed,
    }
    write_dataset_metadata(args.output.parent, "ge-l2", stats)

    logger.info("GE-L2 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
