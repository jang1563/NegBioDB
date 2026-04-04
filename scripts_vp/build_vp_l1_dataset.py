#!/usr/bin/env python3
"""Build VP-L1 MCQ dataset for LLM benchmark.

Generates 1,200 four-way MCQ records across 4 classification classes:
  A) Pathogenic        (300) — variant causes disease
  B) Likely benign     (300) — evidence suggests not pathogenic
  C) VUS               (300) — uncertain significance
  D) Benign            (300) — strong evidence variant does not cause disease

Difficulty: easy(40%), medium(35%), hard(25%)
Split: fewshot 200 (50/class) + val 200 (50/class) + test 800 (200/class)

Output: exports/vp_llm/vp_l1_dataset.jsonl

Usage:
    PYTHONPATH=src python scripts_vp/build_vp_l1_dataset.py
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
OUTPUT_DIR = PROJECT_ROOT / "exports" / "vp_llm"

N_PER_CLASS = 300

CLASS_LABELS = {
    "A": "pathogenic",
    "B": "likely_benign",
    "C": "uncertain_significance",
    "D": "benign",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build VP-L1 MCQ dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_vp.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "vp_l1_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_vp.llm_dataset import (
        apply_max_per_gene,
        assign_splits,
        construct_l1_context,
        load_vp_candidate_pool,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)

    # Class D: Benign (gold/silver tier, strong evidence)
    class_d = load_vp_candidate_pool(
        args.db,
        tier_filter="IN ('gold', 'silver')",
        classification_filter="= 'benign'",
        require_scores=True,
        limit=N_PER_CLASS * 3,
    )
    if len(class_d) > N_PER_CLASS:
        class_d = class_d.sample(n=N_PER_CLASS, random_state=rng).reset_index(drop=True)
    class_d["gold_answer"] = "D"
    class_d["gold_category"] = CLASS_LABELS["D"]

    # Class B: Likely benign (bronze tier or likely_benign classification)
    class_b = load_vp_candidate_pool(
        args.db,
        classification_filter="= 'likely_benign'",
        require_scores=True,
        limit=N_PER_CLASS * 3,
    )
    if len(class_b) > N_PER_CLASS:
        class_b = class_b.sample(n=N_PER_CLASS, random_state=rng).reset_index(drop=True)
    class_b["gold_answer"] = "B"
    class_b["gold_category"] = CLASS_LABELS["B"]

    # Class A: Pathogenic — need separate query for pathogenic variants
    from negbiodb_vp.vp_db import get_connection
    conn = get_connection(args.db)
    try:
        # Pathogenic variants stored as positives (for ML export)
        # Query variants with pathogenic classification from vp_negative_results
        # (or from a separate positive source if available)
        class_a_df = pd.read_sql_query("""
            SELECT v.variant_id, v.chromosome, v.position, v.ref_allele, v.alt_allele,
                   v.variant_type, v.consequence_type, v.hgvs_coding, v.hgvs_protein,
                   v.gnomad_af_global, v.cadd_phred, v.revel_score,
                   v.alphamissense_score, v.alphamissense_class,
                   v.phylop_score, v.gerp_score, v.sift_score, v.polyphen2_score,
                   g.gene_id, g.gene_symbol, g.pli_score, g.loeuf_score,
                   g.missense_z, g.clingen_validity, g.gene_moi,
                   d.disease_id, d.canonical_name AS disease_name,
                   d.inheritance_pattern
            FROM variants v
            LEFT JOIN genes g ON v.gene_id = g.gene_id
            LEFT JOIN vp_negative_results nr ON v.variant_id = nr.variant_id
            LEFT JOIN diseases d ON nr.disease_id = d.disease_id
            WHERE v.cadd_phred IS NOT NULL
            AND v.cadd_phred > 20
            AND (v.revel_score IS NULL OR v.revel_score > 0.5)
            ORDER BY RANDOM() LIMIT ?
        """, conn, params=[N_PER_CLASS * 3])

        # Class C: VUS — variants with intermediate scores
        class_c_df = pd.read_sql_query("""
            SELECT v.variant_id, v.chromosome, v.position, v.ref_allele, v.alt_allele,
                   v.variant_type, v.consequence_type, v.hgvs_coding, v.hgvs_protein,
                   v.gnomad_af_global, v.cadd_phred, v.revel_score,
                   v.alphamissense_score, v.alphamissense_class,
                   v.phylop_score, v.gerp_score, v.sift_score, v.polyphen2_score,
                   g.gene_id, g.gene_symbol, g.pli_score, g.loeuf_score,
                   g.missense_z, g.clingen_validity, g.gene_moi,
                   d.disease_id, d.canonical_name AS disease_name,
                   d.inheritance_pattern
            FROM variants v
            LEFT JOIN genes g ON v.gene_id = g.gene_id
            LEFT JOIN vp_negative_results nr ON v.variant_id = nr.variant_id
            LEFT JOIN diseases d ON nr.disease_id = d.disease_id
            WHERE v.cadd_phred IS NOT NULL
            AND v.cadd_phred BETWEEN 10 AND 20
            ORDER BY RANDOM() LIMIT ?
        """, conn, params=[N_PER_CLASS * 3])
    finally:
        conn.close()

    if len(class_a_df) > N_PER_CLASS:
        class_a_df = class_a_df.sample(n=N_PER_CLASS, random_state=rng).reset_index(drop=True)
    class_a_df["gold_answer"] = "A"
    class_a_df["gold_category"] = CLASS_LABELS["A"]

    if len(class_c_df) > N_PER_CLASS:
        class_c_df = class_c_df.sample(n=N_PER_CLASS, random_state=rng).reset_index(drop=True)
    class_c_df["gold_answer"] = "C"
    class_c_df["gold_category"] = CLASS_LABELS["C"]

    combined = pd.concat([class_a_df, class_b, class_c_df, class_d], ignore_index=True)
    combined = apply_max_per_gene(combined, max_per_gene=10, rng=rng)
    logger.info("Combined: %d records", len(combined))

    # Assign difficulty
    n_total = len(combined)
    difficulties = (
        ["easy"] * int(n_total * 0.40)
        + ["medium"] * int(n_total * 0.35)
        + ["hard"] * (n_total - int(n_total * 0.40) - int(n_total * 0.35))
    )
    rng.shuffle(difficulties)
    combined["difficulty"] = difficulties[:len(combined)]

    # Assign splits (class-stratified)
    n_classes = 4
    fewshot_per_class = 50
    val_per_class = 50
    test_per_class = 200

    split_parts = []
    for letter in sorted(CLASS_LABELS.keys()):
        class_df = combined[combined["gold_answer"] == letter].copy()
        class_df = assign_splits(
            class_df, fewshot_per_class, val_per_class, test_per_class, seed=args.seed,
        )
        split_parts.append(class_df)
    combined = pd.concat(split_parts, ignore_index=True)

    # Build JSONL records
    records = []
    for i, (_, row) in enumerate(combined.iterrows()):
        difficulty = row["difficulty"]
        context = construct_l1_context(row)
        rec = {
            "question_id": f"VPL1-{i:04d}",
            "task": "vp-l1",
            "split": row["split"],
            "difficulty": difficulty,
            "context_text": context,
            "gold_answer": row["gold_answer"],
            "gold_category": row["gold_category"],
            "metadata": {
                "gene_symbol": row.get("gene_symbol"),
                "variant_id": int(row["variant_id"]) if pd.notna(row.get("variant_id")) else None,
                "consequence_type": row.get("consequence_type"),
            },
        }
        records.append(rec)

    write_jsonl(records, args.output)
    write_dataset_metadata(args.output.parent, "vp-l1", {
        "n_total": len(records),
        "seed": args.seed,
        "split_distribution": dict(combined["split"].value_counts()),
    })

    logger.info("VP-L1 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
