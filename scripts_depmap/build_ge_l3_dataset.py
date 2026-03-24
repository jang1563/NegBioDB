#!/usr/bin/env python3
"""Build GE-L3 reasoning dataset for LLM benchmark.

Generates 200 records for LLM-as-Judge reasoning evaluation.
Source: Gold/silver tier non-essential gene-cell_line pairs with gene descriptions.
Balance: ~50% reference non-essential, ~50% context non-essential (essential elsewhere).

Split: 20 fewshot + 20 val + 160 test

Output: exports/ge_llm/ge_l3_dataset.jsonl

Usage:
    PYTHONPATH=src python scripts_depmap/build_ge_l3_dataset.py
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

N_TOTAL = 200
N_REFERENCE_NE = 100  # Reference non-essential genes
N_CONTEXT_NE = 100    # Context non-essential (essential elsewhere)
MIN_DESC_LEN = 30


def _generate_gold_reasoning(row: pd.Series) -> str:
    """Generate template gold reasoning for fewshot examples."""
    gene = row.get("gene_symbol", "UNKNOWN")
    cell_line = row.get("ccle_name") or row.get("model_id", "UNKNOWN")
    lineage = row.get("lineage", "unknown lineage")
    desc = row.get("description") or "unknown function"
    is_rne = row.get("is_reference_nonessential", 0)
    degree = row.get("gene_degree")
    effect = row.get("gene_effect_score")

    parts = []
    parts.append(
        f"{gene} is described as: {desc[:200]}. "
        f"In {cell_line} ({lineage}), CRISPR knockout of {gene} shows no significant "
        f"effect on cell viability."
    )

    if is_rne:
        parts.append(
            f"{gene} is part of the reference non-essential gene set, indicating "
            f"it is dispensable for cell survival across multiple cell types and "
            f"screening contexts."
        )

    if degree and not pd.isna(degree) and degree > 100:
        parts.append(
            f"{gene} is non-essential in {int(degree)} cell lines in DepMap, "
            f"suggesting broad dispensability rather than lineage-specific tolerance."
        )

    if effect is not None and not pd.isna(effect) and effect > -0.2:
        parts.append(
            f"The gene effect score ({effect:.3f}) is close to zero, indicating "
            f"minimal fitness impact upon knockout."
        )

    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build GE-L3 reasoning dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_depmap.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "ge_l3_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_depmap.depmap_db import get_connection
    from negbiodb_depmap.llm_dataset import (
        apply_max_per_gene,
        assign_splits,
        construct_l3_context,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)

    conn = get_connection(args.db)
    try:
        # Load gold/silver tier with gene descriptions
        df = pd.read_sql_query("""
            SELECT
                nr.gene_id, nr.cell_line_id,
                g.gene_symbol, g.entrez_id, g.description,
                g.is_common_essential, g.is_reference_nonessential,
                c.model_id, c.ccle_name, c.lineage, c.primary_disease,
                nr.gene_effect_score, nr.dependency_probability,
                nr.evidence_type, nr.confidence_tier,
                p.gene_degree, p.cell_line_degree
            FROM ge_negative_results nr
            JOIN genes g ON nr.gene_id = g.gene_id
            JOIN cell_lines c ON nr.cell_line_id = c.cell_line_id
            LEFT JOIN gene_cell_pairs p ON nr.gene_id = p.gene_id
                AND nr.cell_line_id = p.cell_line_id
            WHERE nr.confidence_tier IN ('gold', 'silver')
              AND g.description IS NOT NULL
              AND LENGTH(g.description) >= ?
            ORDER BY RANDOM()
            LIMIT 2000
        """, conn, params=(MIN_DESC_LEN,))
    finally:
        conn.close()

    logger.info("Gold/silver with descriptions: %d records", len(df))

    df = apply_max_per_gene(df, max_per_gene=5, seed=args.seed)

    # Split into reference non-essential vs context non-essential
    ref_ne = df[df["is_reference_nonessential"] == 1].copy()
    ctx_ne = df[df["is_reference_nonessential"] == 0].copy()

    logger.info("Reference NE: %d, Context NE: %d", len(ref_ne), len(ctx_ne))

    selected = []
    n_ref = min(N_REFERENCE_NE, len(ref_ne))
    n_ctx = min(N_CONTEXT_NE, len(ctx_ne))

    if n_ref > 0:
        selected.append(ref_ne.sample(n=n_ref, random_state=rng))
    if n_ctx > 0:
        selected.append(ctx_ne.sample(n=n_ctx, random_state=rng))

    # Fill remaining
    n_remaining = N_TOTAL - n_ref - n_ctx
    if n_remaining > 0:
        remaining = df[~df.index.isin(pd.concat(selected).index)]
        if len(remaining) > 0:
            n_fill = min(n_remaining, len(remaining))
            selected.append(remaining.sample(n=n_fill, random_state=rng))

    combined = pd.concat(selected, ignore_index=True)
    logger.info("Selected %d records for L3", len(combined))

    # Assign splits
    combined = assign_splits(combined, ratios={"train": 0.1, "val": 0.1, "test": 0.8})
    combined["split"] = combined["split"].replace({"train": "fewshot"})

    # Build JSONL records
    records = []
    for i, (_, row) in enumerate(combined.iterrows()):
        context = construct_l3_context(row)
        ne_type = "reference" if row.get("is_reference_nonessential", 0) else "context"
        rec = {
            "question_id": f"GEL3-{i:04d}",
            "task": "ge-l3",
            "split": row["split"],
            "difficulty": "medium",
            "context_text": context,
            "gold_answer": row.get("evidence_type", ""),
            "gold_category": ne_type,
            "metadata": {
                "gene_symbol": row.get("gene_symbol"),
                "model_id": row.get("model_id"),
                "lineage": row.get("lineage"),
                "confidence_tier": row.get("confidence_tier"),
                "is_reference_nonessential": int(row.get("is_reference_nonessential", 0)),
                "gene_degree": int(row["gene_degree"]) if pd.notna(row.get("gene_degree")) else None,
            },
        }
        if row["split"] == "fewshot":
            rec["gold_reasoning"] = _generate_gold_reasoning(row)
        records.append(rec)

    write_jsonl(records, args.output)

    stats = {
        "n_total": len(records),
        "ne_type_distribution": {
            "reference": n_ref,
            "context": n_ctx,
        },
        "split_distribution": dict(combined["split"].value_counts()),
        "seed": args.seed,
    }
    write_dataset_metadata(args.output.parent, "ge-l3", stats)

    logger.info("GE-L3 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
