#!/usr/bin/env python3
"""Build PPI-L4 tested/untested dataset for LLM benchmark.

Generates 500 protein pair records:
  250 tested (125 pre-2015, 125 post-2020)
  250 untested (125 trick, 125 obvious)

Tested: IntAct gold tier pairs with publication year from ppi_publication_abstracts.
Untested trick: Same compartment but not in any interaction DB.
Untested obvious: Different compartments, unrelated functions.

Output: exports/ppi_llm/ppi_l4_dataset.jsonl

Usage:
    PYTHONPATH=src python scripts_ppi/build_ppi_l4_dataset.py
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

N_TESTED_PRE = 125
N_TESTED_POST = 125
N_UNTESTED_TRICK = 125
N_UNTESTED_OBVIOUS = 125


def load_tested_pairs(db_path: Path) -> pd.DataFrame:
    """Load tested pairs from IntAct with publication years."""
    from negbiodb_ppi.ppi_db import get_connection

    conn = get_connection(db_path)
    try:
        # IntAct records only (not HuRI — single 2020 paper)
        df = pd.read_sql_query("""
            SELECT
                nr.result_id, nr.source_db, nr.confidence_tier,
                nr.detection_method, e.pubmed_id,
                p1.protein_id AS protein_id_1, p1.uniprot_accession AS uniprot_1,
                p1.gene_symbol AS gene_symbol_1,
                p1.subcellular_location AS location_1,
                p1.function_description AS function_1,
                p2.protein_id AS protein_id_2, p2.uniprot_accession AS uniprot_2,
                p2.gene_symbol AS gene_symbol_2,
                p2.subcellular_location AS location_2,
                p2.function_description AS function_2,
                pa.publication_year
            FROM ppi_negative_results nr
            JOIN proteins p1 ON nr.protein1_id = p1.protein_id
            JOIN proteins p2 ON nr.protein2_id = p2.protein_id
            LEFT JOIN ppi_experiments e ON nr.experiment_id = e.experiment_id
            LEFT JOIN ppi_publication_abstracts pa ON e.pubmed_id = pa.pmid
            WHERE nr.source_db = 'intact'
              AND p1.gene_symbol IS NOT NULL
              AND p2.gene_symbol IS NOT NULL
        """, conn)
    finally:
        conn.close()

    logger.info("IntAct tested pairs: %d (with pub year: %d)",
                len(df), df["publication_year"].notna().sum())
    return df


def load_all_known_pairs(db_path: Path) -> set[tuple[str, str]]:
    """Load all known interaction pairs (positive + negative) for verification."""
    from negbiodb_ppi.ppi_db import get_connection

    conn = get_connection(db_path)
    pairs = set()
    try:
        # Negative results
        rows = conn.execute("""
            SELECT p1.gene_symbol, p2.gene_symbol
            FROM ppi_negative_results nr
            JOIN proteins p1 ON nr.protein1_id = p1.protein_id
            JOIN proteins p2 ON nr.protein2_id = p2.protein_id
            WHERE p1.gene_symbol IS NOT NULL AND p2.gene_symbol IS NOT NULL
        """).fetchall()
        for r in rows:
            pairs.add((min(r[0], r[1]), max(r[0], r[1])))

        # Positive pairs
        rows = conn.execute("""
            SELECT p1.gene_symbol, p2.gene_symbol
            FROM protein_protein_pairs pp
            JOIN proteins p1 ON pp.protein1_id = p1.protein_id
            JOIN proteins p2 ON pp.protein2_id = p2.protein_id
            WHERE p1.gene_symbol IS NOT NULL AND p2.gene_symbol IS NOT NULL
        """).fetchall()
        for r in rows:
            pairs.add((min(r[0], r[1]), max(r[0], r[1])))
    finally:
        conn.close()

    logger.info("Known pairs (positive + negative): %d", len(pairs))
    return pairs


def load_proteins_with_annotations(db_path: Path) -> pd.DataFrame:
    """Load proteins with gene_symbol and location for untested pair generation."""
    from negbiodb_ppi.ppi_db import get_connection

    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query("""
            SELECT protein_id, uniprot_accession, gene_symbol,
                   subcellular_location, function_description
            FROM proteins
            WHERE gene_symbol IS NOT NULL
              AND subcellular_location IS NOT NULL
        """, conn)
    finally:
        conn.close()
    return df


def _extract_compartment(location: str | None) -> str:
    """Extract primary compartment from subcellular location string."""
    if not location:
        return "unknown"
    loc = location.lower()
    if "nucleus" in loc or "nuclear" in loc:
        return "nucleus"
    if "extracellular" in loc or "secreted" in loc:
        return "extracellular"
    if "mitochondri" in loc:
        return "mitochondria"
    if "endoplasmic" in loc:
        return "er"
    if "golgi" in loc:
        return "golgi"
    if "plasma membrane" in loc or "cell membrane" in loc:
        return "plasma_membrane"
    if "cytoplasm" in loc or "cytosol" in loc:
        return "cytoplasm"
    if "membrane" in loc:
        return "membrane"
    return "other"


def generate_untested_trick(
    proteins: pd.DataFrame,
    known_pairs: set[tuple[str, str]],
    n: int,
    rng: np.random.RandomState,
) -> list[dict]:
    """Generate untested trick pairs: same compartment, not in any DB."""
    proteins = proteins.copy()
    proteins["compartment"] = proteins["subcellular_location"].apply(_extract_compartment)

    # Group by compartment
    compartment_groups = {
        c: grp for c, grp in proteins.groupby("compartment")
        if len(grp) >= 10 and c not in ("unknown", "other")
    }

    pairs = []
    attempts = 0
    max_attempts = n * 50

    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        # Pick a random compartment
        comp = rng.choice(list(compartment_groups.keys()))
        grp = compartment_groups[comp]
        if len(grp) < 2:
            continue

        idx = rng.choice(len(grp), size=2, replace=False)
        p1 = grp.iloc[idx[0]]
        p2 = grp.iloc[idx[1]]

        gene1 = p1["gene_symbol"]
        gene2 = p2["gene_symbol"]
        pair_key = (min(gene1, gene2), max(gene1, gene2))

        if pair_key in known_pairs:
            continue

        pairs.append({
            "gene_symbol_1": gene1,
            "gene_symbol_2": gene2,
            "uniprot_1": p1["uniprot_accession"],
            "uniprot_2": p2["uniprot_accession"],
            "protein_name_1": gene1,
            "protein_name_2": gene2,
            "location_1": p1["subcellular_location"],
            "location_2": p2["subcellular_location"],
            "function_1": p1.get("function_description"),
            "function_2": p2.get("function_description"),
            "untested_type": "trick",
            "compartment": comp,
        })
        known_pairs.add(pair_key)  # prevent duplicates

    logger.info("Generated %d trick pairs in %d attempts", len(pairs), attempts)
    return pairs


def generate_untested_obvious(
    proteins: pd.DataFrame,
    known_pairs: set[tuple[str, str]],
    n: int,
    rng: np.random.RandomState,
) -> list[dict]:
    """Generate untested obvious pairs: different compartments, unrelated functions."""
    proteins = proteins.copy()
    proteins["compartment"] = proteins["subcellular_location"].apply(_extract_compartment)

    # Define "distant" compartment pairs
    distant_pairs_list = [
        ("nucleus", "extracellular"),
        ("nucleus", "plasma_membrane"),
        ("mitochondria", "extracellular"),
        ("er", "nucleus"),
        ("golgi", "extracellular"),
        ("cytoplasm", "extracellular"),
    ]

    compartment_groups = {
        c: grp for c, grp in proteins.groupby("compartment")
        if len(grp) >= 5
    }

    pairs = []
    attempts = 0
    max_attempts = n * 50

    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        c1, c2 = distant_pairs_list[rng.randint(len(distant_pairs_list))]

        if c1 not in compartment_groups or c2 not in compartment_groups:
            continue

        grp1 = compartment_groups[c1]
        grp2 = compartment_groups[c2]

        p1 = grp1.iloc[rng.randint(len(grp1))]
        p2 = grp2.iloc[rng.randint(len(grp2))]

        gene1 = p1["gene_symbol"]
        gene2 = p2["gene_symbol"]
        pair_key = (min(gene1, gene2), max(gene1, gene2))

        if pair_key in known_pairs:
            continue

        pairs.append({
            "gene_symbol_1": gene1,
            "gene_symbol_2": gene2,
            "uniprot_1": p1["uniprot_accession"],
            "uniprot_2": p2["uniprot_accession"],
            "protein_name_1": gene1,
            "protein_name_2": gene2,
            "location_1": p1["subcellular_location"],
            "location_2": p2["subcellular_location"],
            "function_1": p1.get("function_description"),
            "function_2": p2.get("function_description"),
            "untested_type": "obvious",
            "compartment_1": c1,
            "compartment_2": c2,
        })
        known_pairs.add(pair_key)

    logger.info("Generated %d obvious pairs in %d attempts", len(pairs), attempts)
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build PPI-L4 tested/untested dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_ppi.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "ppi_l4_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_ppi.llm_dataset import (
        assign_splits,
        construct_l4_context,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)

    # --- Tested pairs ---
    tested_df = load_tested_pairs(args.db)

    # Temporal split
    pre_2015 = tested_df[tested_df["publication_year"].fillna(0) <= 2015]
    post_2020 = tested_df[tested_df["publication_year"].fillna(0) >= 2020]

    # If not enough post-2020 IntAct, supplement with HuRI (2020 paper)
    n_pre = min(N_TESTED_PRE, len(pre_2015))
    n_post = min(N_TESTED_POST, len(post_2020))

    tested_pre = pre_2015.sample(n=n_pre, random_state=rng) if n_pre > 0 else pd.DataFrame()
    tested_post = post_2020.sample(n=n_post, random_state=rng) if n_post > 0 else pd.DataFrame()

    # If post_2020 insufficient, add HuRI pairs
    if n_post < N_TESTED_POST:
        from negbiodb_ppi.ppi_db import get_connection
        conn = get_connection(args.db)
        huri_df = pd.read_sql_query("""
            SELECT
                nr.result_id, nr.source_db, nr.confidence_tier,
                nr.detection_method,
                p1.protein_id AS protein_id_1, p1.uniprot_accession AS uniprot_1,
                p1.gene_symbol AS gene_symbol_1,
                p1.subcellular_location AS location_1,
                p1.function_description AS function_1,
                p2.protein_id AS protein_id_2, p2.uniprot_accession AS uniprot_2,
                p2.gene_symbol AS gene_symbol_2,
                p2.subcellular_location AS location_2,
                p2.function_description AS function_2
            FROM ppi_negative_results nr
            JOIN proteins p1 ON nr.protein1_id = p1.protein_id
            JOIN proteins p2 ON nr.protein2_id = p2.protein_id
            WHERE nr.source_db = 'huri'
              AND p1.gene_symbol IS NOT NULL
              AND p2.gene_symbol IS NOT NULL
            ORDER BY RANDOM() LIMIT ?
        """, conn, params=(N_TESTED_POST - n_post,))
        conn.close()
        huri_df["publication_year"] = 2020
        tested_post = pd.concat([tested_post, huri_df], ignore_index=True)
        n_post = len(tested_post)

    logger.info("Tested: pre_2015=%d, post_2020=%d", n_pre, n_post)

    # --- Untested pairs ---
    known_pairs = load_all_known_pairs(args.db)
    proteins = load_proteins_with_annotations(args.db)

    trick_pairs = generate_untested_trick(proteins, known_pairs, N_UNTESTED_TRICK, rng)
    obvious_pairs = generate_untested_obvious(proteins, known_pairs, N_UNTESTED_OBVIOUS, rng)

    # --- Build records ---
    records = []

    # Tested pre-2015
    for i, (_, row) in enumerate(tested_pre.iterrows()):
        row_dict = row.to_dict()
        row_dict["protein_name_1"] = row.get("gene_symbol_1", "")
        row_dict["protein_name_2"] = row.get("gene_symbol_2", "")
        context = construct_l4_context(row_dict)
        records.append({
            "question_id": f"PPIL4-{len(records):04d}",
            "task": "ppi-l4",
            "split": "test",
            "difficulty": "medium",
            "context_text": context,
            "gold_answer": "tested",
            "gold_category": "tested",
            "temporal_group": "pre_2015",
            "metadata": {
                "source_db": row.get("source_db"),
                "publication_year": int(row["publication_year"]) if pd.notna(row.get("publication_year")) else None,
                "gene_symbol_1": row.get("gene_symbol_1"),
                "gene_symbol_2": row.get("gene_symbol_2"),
                "detection_method": row.get("detection_method"),
                "result_id": int(row["result_id"]) if pd.notna(row.get("result_id")) else None,
            },
        })

    # Tested post-2020
    for _, row in tested_post.iterrows():
        row_dict = row.to_dict()
        row_dict["protein_name_1"] = row.get("gene_symbol_1", "")
        row_dict["protein_name_2"] = row.get("gene_symbol_2", "")
        context = construct_l4_context(row_dict)
        records.append({
            "question_id": f"PPIL4-{len(records):04d}",
            "task": "ppi-l4",
            "split": "test",
            "difficulty": "medium",
            "context_text": context,
            "gold_answer": "tested",
            "gold_category": "tested",
            "temporal_group": "post_2020",
            "metadata": {
                "source_db": row.get("source_db"),
                "publication_year": int(row["publication_year"]) if pd.notna(row.get("publication_year")) else None,
                "gene_symbol_1": row.get("gene_symbol_1"),
                "gene_symbol_2": row.get("gene_symbol_2"),
                "detection_method": row.get("detection_method"),
                "result_id": int(row["result_id"]) if pd.notna(row.get("result_id")) else None,
            },
        })

    # Untested trick
    for pair in trick_pairs:
        context = construct_l4_context(pair)
        records.append({
            "question_id": f"PPIL4-{len(records):04d}",
            "task": "ppi-l4",
            "split": "test",
            "difficulty": "hard",
            "context_text": context,
            "gold_answer": "untested",
            "gold_category": "untested",
            "temporal_group": None,
            "metadata": {
                "untested_type": "trick",
                "gene_symbol_1": pair["gene_symbol_1"],
                "gene_symbol_2": pair["gene_symbol_2"],
                "compartment": pair.get("compartment"),
            },
        })

    # Untested obvious
    for pair in obvious_pairs:
        context = construct_l4_context(pair)
        records.append({
            "question_id": f"PPIL4-{len(records):04d}",
            "task": "ppi-l4",
            "split": "test",
            "difficulty": "easy",
            "context_text": context,
            "gold_answer": "untested",
            "gold_category": "untested",
            "temporal_group": None,
            "metadata": {
                "untested_type": "obvious",
                "gene_symbol_1": pair["gene_symbol_1"],
                "gene_symbol_2": pair["gene_symbol_2"],
                "compartment_1": pair.get("compartment_1"),
                "compartment_2": pair.get("compartment_2"),
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

    # Re-index question IDs after shuffle
    for i, rec in enumerate(records):
        rec["question_id"] = f"PPIL4-{i:04d}"

    write_jsonl(records, args.output)

    stats = {
        "n_total": len(records),
        "n_tested": len(tested_records),
        "n_untested": len(untested_records),
        "n_tested_pre_2015": n_pre,
        "n_tested_post_2020": n_post,
        "n_untested_trick": len(trick_pairs),
        "n_untested_obvious": len(obvious_pairs),
        "split_distribution": {
            s: sum(1 for r in records if r["split"] == s)
            for s in ["fewshot", "val", "test"]
        },
        "seed": args.seed,
    }
    write_dataset_metadata(args.output.parent, "ppi-l4", stats)

    logger.info("PPI-L4 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
