#!/usr/bin/env python3
"""Build GE-L4 tested/untested dataset for LLM benchmark.

Generates 500 gene-cell_line records:
  250 tested (125 from older DepMap release, 125 from recent release)
  250 untested (125 trick: well-known gene + tested cell line, 125 obvious: obscure gene)

Temporal contamination design: pairs from older releases (e.g., 22Q2) are likely
in LLM training data. Pairs only in recent releases (e.g., 25Q3) are likely novel.

Output: exports/ge_llm/ge_l4_dataset.jsonl

Usage:
    PYTHONPATH=src python scripts_depmap/build_ge_l4_dataset.py
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

N_TESTED_OLD = 125
N_TESTED_NEW = 125
N_UNTESTED_TRICK = 125
N_UNTESTED_OBVIOUS = 125


def load_tested_pairs(conn, n_old: int, n_new: int, rng: np.random.RandomState) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load tested pairs, split by gene degree as temporal proxy.

    High-degree genes (non-essential in many cell lines) were likely in earlier releases.
    Low-degree genes (fewer cell lines) were likely added in recent releases.
    """
    df = pd.read_sql_query("""
        SELECT
            g.gene_id, g.gene_symbol, g.entrez_id,
            g.is_common_essential, g.is_reference_nonessential,
            c.cell_line_id, c.model_id, c.ccle_name, c.lineage, c.primary_disease,
            nr.gene_effect_score, nr.dependency_probability,
            nr.confidence_tier, nr.evidence_type,
            p.gene_degree, p.cell_line_degree
        FROM ge_negative_results nr
        JOIN genes g ON nr.gene_id = g.gene_id
        JOIN cell_lines c ON nr.cell_line_id = c.cell_line_id
        LEFT JOIN gene_cell_pairs p ON nr.gene_id = p.gene_id
            AND nr.cell_line_id = p.cell_line_id
        WHERE nr.confidence_tier IN ('gold', 'silver')
        ORDER BY RANDOM()
        LIMIT 5000
    """, conn)

    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Use gene_degree as temporal proxy
    median_degree = df["gene_degree"].median()
    old_candidates = df[df["gene_degree"].fillna(0) >= median_degree]
    new_candidates = df[df["gene_degree"].fillna(0) < median_degree]

    n_old_actual = min(n_old, len(old_candidates))
    n_new_actual = min(n_new, len(new_candidates))

    old_df = old_candidates.sample(n=n_old_actual, random_state=rng) if n_old_actual > 0 else pd.DataFrame()
    new_df = new_candidates.sample(n=n_new_actual, random_state=rng) if n_new_actual > 0 else pd.DataFrame()

    logger.info("Tested: old=%d (high degree), new=%d (low degree)", n_old_actual, n_new_actual)
    return old_df, new_df


def generate_untested_trick(
    conn,
    known_pairs: set[tuple[int, int]],
    n: int,
    rng: np.random.RandomState,
) -> list[dict]:
    """Generate trick untested pairs: well-studied genes in tested cell lines.

    These are genes and cell lines that exist in DB but were NOT paired together.
    """
    # Well-studied genes (high degree)
    genes = pd.read_sql_query("""
        SELECT g.gene_id, g.gene_symbol, g.entrez_id,
               g.is_common_essential, g.is_reference_nonessential
        FROM genes g
        JOIN gene_cell_pairs p ON g.gene_id = p.gene_id
        WHERE p.gene_degree > 500
        GROUP BY g.gene_id
        ORDER BY RANDOM() LIMIT 200
    """, conn)

    cell_lines = pd.read_sql_query("""
        SELECT cell_line_id, model_id, ccle_name, lineage, primary_disease
        FROM cell_lines
        ORDER BY RANDOM() LIMIT 100
    """, conn)

    pairs = []
    attempts = 0
    max_attempts = n * 50

    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        gi = rng.randint(len(genes))
        ci = rng.randint(len(cell_lines))
        gid = int(genes.iloc[gi]["gene_id"])
        clid = int(cell_lines.iloc[ci]["cell_line_id"])

        if (gid, clid) in known_pairs:
            continue

        rec = {**genes.iloc[gi].to_dict(), **cell_lines.iloc[ci].to_dict()}
        rec["gene_effect_score"] = None
        rec["dependency_probability"] = None
        rec["untested_type"] = "trick"
        pairs.append(rec)
        known_pairs.add((gid, clid))

    logger.info("Generated %d trick pairs in %d attempts", len(pairs), attempts)
    return pairs


def generate_untested_obvious(
    conn,
    known_pairs: set[tuple[int, int]],
    n: int,
    rng: np.random.RandomState,
) -> list[dict]:
    """Generate obvious untested pairs: obscure genes not in DepMap screens."""
    # Genes with no negative results (not screened)
    genes = pd.read_sql_query("""
        SELECT g.gene_id, g.gene_symbol, g.entrez_id,
               g.is_common_essential, g.is_reference_nonessential
        FROM genes g
        LEFT JOIN ge_negative_results nr ON g.gene_id = nr.gene_id
        WHERE nr.result_id IS NULL
        ORDER BY RANDOM() LIMIT 500
    """, conn)

    if len(genes) == 0:
        # Fallback: genes with very low degree
        genes = pd.read_sql_query("""
            SELECT g.gene_id, g.gene_symbol, g.entrez_id,
                   g.is_common_essential, g.is_reference_nonessential
            FROM genes g
            JOIN gene_cell_pairs p ON g.gene_id = p.gene_id
            WHERE p.gene_degree <= 5
            GROUP BY g.gene_id
            ORDER BY RANDOM() LIMIT 500
        """, conn)

    cell_lines = pd.read_sql_query("""
        SELECT cell_line_id, model_id, ccle_name, lineage, primary_disease
        FROM cell_lines
        ORDER BY RANDOM() LIMIT 100
    """, conn)

    pairs = []
    attempts = 0
    max_attempts = n * 50

    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        gi = rng.randint(len(genes))
        ci = rng.randint(len(cell_lines))
        gid = int(genes.iloc[gi]["gene_id"])
        clid = int(cell_lines.iloc[ci]["cell_line_id"])

        if (gid, clid) in known_pairs:
            continue

        rec = {**genes.iloc[gi].to_dict(), **cell_lines.iloc[ci].to_dict()}
        rec["gene_effect_score"] = None
        rec["dependency_probability"] = None
        rec["untested_type"] = "obvious"
        pairs.append(rec)
        known_pairs.add((gid, clid))

    logger.info("Generated %d obvious pairs in %d attempts", len(pairs), attempts)
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build GE-L4 tested/untested dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_depmap.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "ge_l4_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_depmap.depmap_db import get_connection
    from negbiodb_depmap.llm_dataset import (
        construct_l4_context,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)

    conn = get_connection(args.db)
    try:
        # Tested pairs
        tested_old, tested_new = load_tested_pairs(conn, N_TESTED_OLD, N_TESTED_NEW, rng)

        # Build known pairs set
        known_pairs = set()
        rows = conn.execute("SELECT gene_id, cell_line_id FROM ge_negative_results").fetchall()
        for r in rows:
            known_pairs.add((r[0], r[1]))

        # Untested pairs
        trick_pairs = generate_untested_trick(conn, known_pairs, N_UNTESTED_TRICK, rng)
        obvious_pairs = generate_untested_obvious(conn, known_pairs, N_UNTESTED_OBVIOUS, rng)
    finally:
        conn.close()

    # Build records
    records = []

    # Tested old (high degree, likely in LLM training data)
    for _, row in tested_old.iterrows():
        context = construct_l4_context(row.to_dict())
        records.append({
            "question_id": f"GEL4-{len(records):04d}",
            "task": "ge-l4",
            "split": "test",
            "difficulty": "medium",
            "context_text": context,
            "gold_answer": "tested",
            "gold_category": "tested",
            "temporal_group": "old_release",
            "metadata": {
                "gene_symbol": row.get("gene_symbol"),
                "model_id": row.get("model_id"),
                "lineage": row.get("lineage"),
                "gene_degree": int(row["gene_degree"]) if pd.notna(row.get("gene_degree")) else None,
            },
        })

    # Tested new (low degree, likely novel)
    for _, row in tested_new.iterrows():
        context = construct_l4_context(row.to_dict())
        records.append({
            "question_id": f"GEL4-{len(records):04d}",
            "task": "ge-l4",
            "split": "test",
            "difficulty": "medium",
            "context_text": context,
            "gold_answer": "tested",
            "gold_category": "tested",
            "temporal_group": "new_release",
            "metadata": {
                "gene_symbol": row.get("gene_symbol"),
                "model_id": row.get("model_id"),
                "lineage": row.get("lineage"),
                "gene_degree": int(row["gene_degree"]) if pd.notna(row.get("gene_degree")) else None,
            },
        })

    # Untested trick
    for pair in trick_pairs:
        context = construct_l4_context(pair)
        records.append({
            "question_id": f"GEL4-{len(records):04d}",
            "task": "ge-l4",
            "split": "test",
            "difficulty": "hard",
            "context_text": context,
            "gold_answer": "untested",
            "gold_category": "untested",
            "temporal_group": None,
            "metadata": {
                "untested_type": "trick",
                "gene_symbol": pair.get("gene_symbol"),
                "model_id": pair.get("model_id"),
                "lineage": pair.get("lineage"),
            },
        })

    # Untested obvious
    for pair in obvious_pairs:
        context = construct_l4_context(pair)
        records.append({
            "question_id": f"GEL4-{len(records):04d}",
            "task": "ge-l4",
            "split": "test",
            "difficulty": "easy",
            "context_text": context,
            "gold_answer": "untested",
            "gold_category": "untested",
            "temporal_group": None,
            "metadata": {
                "untested_type": "obvious",
                "gene_symbol": pair.get("gene_symbol"),
                "model_id": pair.get("model_id"),
                "lineage": pair.get("lineage"),
            },
        })

    # Assign splits (class-balanced)
    rng.shuffle(records)
    tested_records = [r for r in records if r["gold_answer"] == "tested"]
    untested_records = [r for r in records if r["gold_answer"] == "untested"]

    for subset in [tested_records, untested_records]:
        rng.shuffle(subset)
        for i, rec in enumerate(subset):
            if i < 25:
                rec["split"] = "fewshot"
            elif i < 50:
                rec["split"] = "val"
            else:
                rec["split"] = "test"

    records = tested_records + untested_records
    rng.shuffle(records)

    for i, rec in enumerate(records):
        rec["question_id"] = f"GEL4-{i:04d}"

    write_jsonl(records, args.output)

    stats = {
        "n_total": len(records),
        "n_tested": len(tested_records),
        "n_untested": len(untested_records),
        "n_tested_old": len(tested_old),
        "n_tested_new": len(tested_new),
        "n_untested_trick": len(trick_pairs),
        "n_untested_obvious": len(obvious_pairs),
        "split_distribution": {
            s: sum(1 for r in records if r["split"] == s)
            for s in ["fewshot", "val", "test"]
        },
        "seed": args.seed,
    }
    write_dataset_metadata(args.output.parent, "ge-l4", stats)

    logger.info("GE-L4 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
