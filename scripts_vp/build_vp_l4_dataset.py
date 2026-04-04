#!/usr/bin/env python3
"""Build VP-L4 tested/untested discrimination dataset for LLM benchmark.

Generates 475 variant-disease pairs:
  - Tested pre-2020: 125 pairs (ClinVar submission before 2020)
  - Tested post-2023: 125 pairs (ClinVar submission after 2023)
  - Untested trick: 100 pairs (well-known gene + unrelated disease)
  - Untested rare: 125 pairs (rare variant not in ClinVar)

Split: fewshot 50 + val 50 + test 375

Output: exports/vp_llm/vp_l4_dataset.jsonl

Usage:
    PYTHONPATH=src python scripts_vp/build_vp_l4_dataset.py
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build VP-L4 discrimination dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_vp.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "vp_l4_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_vp.llm_dataset import (
        assign_splits,
        construct_l4_context,
        write_dataset_metadata,
        write_jsonl,
    )
    from negbiodb_vp.vp_db import get_connection

    rng = np.random.RandomState(args.seed)
    conn = get_connection(args.db)

    try:
        # Tested pre-2020: ClinVar submissions before 2020
        pre_2020 = pd.read_sql_query("""
            SELECT v.variant_id, v.chromosome, v.position, v.ref_allele, v.alt_allele,
                   v.consequence_type, v.hgvs_coding, v.hgvs_protein,
                   g.gene_symbol, d.canonical_name AS disease_name,
                   d.disease_id, nr.submission_year
            FROM vp_negative_results nr
            JOIN variants v ON nr.variant_id = v.variant_id
            LEFT JOIN genes g ON v.gene_id = g.gene_id
            LEFT JOIN diseases d ON nr.disease_id = d.disease_id
            WHERE nr.submission_year < 2020
            AND nr.confidence_tier IN ('gold', 'silver')
            ORDER BY RANDOM() LIMIT 400
        """, conn)
        if len(pre_2020) > 125:
            pre_2020 = pre_2020.sample(n=125, random_state=rng).reset_index(drop=True)
        pre_2020["gold_answer"] = "tested"
        pre_2020["temporal_group"] = "pre_2020"
        logger.info("Pre-2020 tested: %d", len(pre_2020))

        # Tested post-2023: ClinVar submissions after 2023
        post_2023 = pd.read_sql_query("""
            SELECT v.variant_id, v.chromosome, v.position, v.ref_allele, v.alt_allele,
                   v.consequence_type, v.hgvs_coding, v.hgvs_protein,
                   g.gene_symbol, d.canonical_name AS disease_name,
                   d.disease_id, nr.submission_year
            FROM vp_negative_results nr
            JOIN variants v ON nr.variant_id = v.variant_id
            LEFT JOIN genes g ON v.gene_id = g.gene_id
            LEFT JOIN diseases d ON nr.disease_id = d.disease_id
            WHERE nr.submission_year >= 2023
            AND nr.confidence_tier IN ('gold', 'silver', 'bronze')
            ORDER BY RANDOM() LIMIT 400
        """, conn)
        if len(post_2023) > 125:
            post_2023 = post_2023.sample(n=125, random_state=rng).reset_index(drop=True)
        post_2023["gold_answer"] = "tested"
        post_2023["temporal_group"] = "post_2023"
        logger.info("Post-2023 tested: %d", len(post_2023))

        # Untested trick: well-known gene + unrelated disease
        genes = pd.read_sql_query("""
            SELECT gene_id, gene_symbol FROM genes
            WHERE gene_symbol IN ('BRCA1', 'BRCA2', 'TP53', 'CFTR', 'LDLR',
                                   'MLH1', 'MSH2', 'PKD1', 'RB1', 'VHL')
        """, conn)
        diseases = pd.read_sql_query("""
            SELECT disease_id, canonical_name AS disease_name FROM diseases
            ORDER BY RANDOM() LIMIT 500
        """, conn)

        # Get tested pairs to exclude
        tested_pairs = set()
        rows = conn.execute(
            "SELECT variant_id, disease_id FROM variant_disease_pairs"
        ).fetchall()
        for r in rows:
            tested_pairs.add((r[0], r[1]))

        trick_records = []
        for _ in range(100 * 20):
            if len(trick_records) >= 100:
                break
            gi = rng.randint(len(genes))
            di = rng.randint(len(diseases))
            gene_row = genes.iloc[gi]
            disease_row = diseases.iloc[di]

            # Create a fake variant for this gene
            trick_records.append({
                "variant_id": None,
                "chromosome": "?",
                "position": rng.randint(1_000_000, 200_000_000),
                "ref_allele": rng.choice(["A", "C", "G", "T"]),
                "alt_allele": rng.choice(["A", "C", "G", "T"]),
                "consequence_type": "missense",
                "hgvs_coding": None,
                "hgvs_protein": None,
                "gene_symbol": gene_row["gene_symbol"],
                "disease_name": disease_row["disease_name"],
                "disease_id": int(disease_row["disease_id"]),
                "gold_answer": "untested",
                "temporal_group": "untested_trick",
            })

        untested_trick = pd.DataFrame(trick_records[:100])
        logger.info("Untested trick: %d", len(untested_trick))

        # Untested rare: variants not in ClinVar
        all_genes = pd.read_sql_query(
            "SELECT gene_id, gene_symbol FROM genes ORDER BY RANDOM() LIMIT 200", conn,
        )
        rare_records = []
        for _ in range(125 * 20):
            if len(rare_records) >= 125:
                break
            gi = rng.randint(len(all_genes))
            di = rng.randint(len(diseases))
            gene_row = all_genes.iloc[gi]
            disease_row = diseases.iloc[di]

            rare_records.append({
                "variant_id": None,
                "chromosome": str(rng.randint(1, 23)),
                "position": rng.randint(1_000_000, 200_000_000),
                "ref_allele": rng.choice(["A", "C", "G", "T"]),
                "alt_allele": rng.choice(["A", "C", "G", "T"]),
                "consequence_type": rng.choice(["missense", "synonymous", "intronic"]),
                "hgvs_coding": None,
                "hgvs_protein": None,
                "gene_symbol": gene_row["gene_symbol"],
                "disease_name": disease_row["disease_name"],
                "disease_id": int(disease_row["disease_id"]),
                "gold_answer": "untested",
                "temporal_group": "untested_rare",
            })

        untested_rare = pd.DataFrame(rare_records[:125])
        logger.info("Untested rare: %d", len(untested_rare))
    finally:
        conn.close()

    combined = pd.concat([pre_2020, post_2023, untested_trick, untested_rare], ignore_index=True)
    combined = assign_splits(combined, fewshot_size=50, val_size=50, test_size=375, seed=args.seed)
    logger.info("Combined: %d records", len(combined))

    # Build JSONL records
    records = []
    for i, (_, row) in enumerate(combined.iterrows()):
        context = construct_l4_context(row)
        rec = {
            "question_id": f"VPL4-{i:04d}",
            "task": "vp-l4",
            "split": row["split"],
            "context_text": context,
            "gold_answer": row["gold_answer"],
            "temporal_group": row["temporal_group"],
            "metadata": {
                "gene_symbol": row.get("gene_symbol"),
                "disease_name": row.get("disease_name"),
                "temporal_group": row["temporal_group"],
            },
        }
        records.append(rec)

    write_jsonl(records, args.output)
    write_dataset_metadata(args.output.parent, "vp-l4", {
        "n_total": len(records),
        "seed": args.seed,
        "temporal_distribution": dict(combined["temporal_group"].value_counts()),
        "split_distribution": dict(combined["split"].value_counts()),
    })

    logger.info("VP-L4 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
