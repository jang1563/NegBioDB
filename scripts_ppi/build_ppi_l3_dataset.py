#!/usr/bin/env python3
"""Build PPI-L3 reasoning dataset for LLM benchmark.

Generates 200 records for LLM-as-Judge reasoning evaluation.
Source: Gold tier (IntAct + HuRI) with rich protein annotations.
Both proteins must have function_description.
Balance: ~50% same-compartment, ~50% different-compartment pairs.

Split: 20 fewshot + 20 val + 160 test

Output: exports/ppi_llm/ppi_l3_dataset.jsonl

Usage:
    PYTHONPATH=src python scripts_ppi/build_ppi_l3_dataset.py
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

N_TOTAL = 200
N_SAME_COMPARTMENT = 100
N_DIFF_COMPARTMENT = 100
MIN_FUNC_LEN = 50  # Minimum function_description length


def _generate_gold_reasoning(row: pd.Series) -> str:
    """Generate template gold reasoning from protein annotations for fewshot examples."""
    gene1 = row.get("gene_symbol_1") or row.get("uniprot_1", "Protein_1")
    gene2 = row.get("gene_symbol_2") or row.get("uniprot_2", "Protein_2")
    func1 = (row.get("function_1") or "unknown function")[:200]
    func2 = (row.get("function_2") or "unknown function")[:200]
    loc1 = row.get("location_1") or ""
    loc2 = row.get("location_2") or ""
    source = row.get("source_db", "")
    method = row.get("detection_method", "")

    parts = []
    # Biological plausibility
    parts.append(
        f"{gene1} is described as: {func1}. "
        f"{gene2} is described as: {func2}. "
        f"These distinct biological roles suggest limited functional overlap "
        f"requiring direct physical association."
    )
    # Structural/localization reasoning
    if loc1 and loc2:
        if loc1.lower() != loc2.lower():
            parts.append(
                f"{gene1} localizes to {loc1}, while {gene2} localizes to {loc2}. "
                f"Different subcellular compartments reduce the probability of direct interaction."
            )
        else:
            parts.append(
                f"Although both proteins are found in {loc1}, co-localization alone "
                f"does not imply physical interaction."
            )
    # Evidence basis
    if source == "intact" and method:
        from negbiodb_ppi.llm_dataset import DETECTION_METHOD_DESCRIPTIONS

        method_desc = DETECTION_METHOD_DESCRIPTIONS.get(method, method)
        parts.append(
            f"A {method_desc} experiment directly tested for binding between "
            f"{gene1} and {gene2} and found no detectable interaction."
        )
    elif source == "huri":
        parts.append(
            f"Systematic yeast two-hybrid screening tested this pair across "
            f"multiple replicates and found no positive interaction signal."
        )
    else:
        parts.append(
            "Experimental evidence does not support a physical interaction between "
            f"{gene1} and {gene2}."
        )

    return " ".join(parts)


def _same_compartment(loc1: str | None, loc2: str | None) -> bool | None:
    """Check if two proteins share a subcellular compartment."""
    if not loc1 or not loc2:
        return None
    # Extract primary compartment keywords
    compartments = [
        "nucleus", "cytoplasm", "membrane", "mitochondri",
        "endoplasmic reticulum", "golgi", "extracellular",
        "cytosol", "nuclear", "plasma membrane",
    ]
    locs1 = {c for c in compartments if c in loc1.lower()}
    locs2 = {c for c in compartments if c in loc2.lower()}
    if not locs1 or not locs2:
        return None
    return bool(locs1 & locs2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build PPI-L3 reasoning dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_ppi.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "ppi_l3_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_ppi.llm_dataset import (
        apply_max_per_protein,
        assign_splits,
        construct_l3_context,
        load_ppi_candidate_pool,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)

    # Load gold-tier with annotations required (limit for speed — 2000 is plenty for 200)
    df = load_ppi_candidate_pool(
        args.db,
        tier_filter="IN ('gold', 'silver')",
        require_annotations=True,
        limit=2000,
    )
    logger.info("Gold/silver with annotations: %d records", len(df))

    # Filter: both proteins must have substantial function descriptions
    mask = (
        df["function_1"].str.len().fillna(0) >= MIN_FUNC_LEN
    ) & (
        df["function_2"].str.len().fillna(0) >= MIN_FUNC_LEN
    )
    df = df[mask].copy()
    logger.info("After function length filter (>=%d chars): %d records", MIN_FUNC_LEN, len(df))

    df = apply_max_per_protein(df, max_per_protein=5, rng=rng)

    # Classify compartment relationship
    df["same_compartment"] = df.apply(
        lambda r: _same_compartment(r.get("location_1"), r.get("location_2")),
        axis=1,
    )

    same = df[df["same_compartment"] == True].copy()  # noqa: E712
    diff = df[df["same_compartment"] == False].copy()  # noqa: E712
    unknown = df[df["same_compartment"].isna()].copy()

    logger.info(
        "Compartment: same=%d, different=%d, unknown=%d",
        len(same), len(diff), len(unknown),
    )

    # Sample balanced sets
    selected = []
    n_same = min(N_SAME_COMPARTMENT, len(same))
    n_diff = min(N_DIFF_COMPARTMENT, len(diff))

    if n_same > 0:
        selected.append(same.sample(n=n_same, random_state=rng))
    if n_diff > 0:
        selected.append(diff.sample(n=n_diff, random_state=rng))

    # Fill remaining from unknown
    n_remaining = N_TOTAL - n_same - n_diff
    if n_remaining > 0 and len(unknown) > 0:
        n_fill = min(n_remaining, len(unknown))
        selected.append(unknown.sample(n=n_fill, random_state=rng))

    combined = pd.concat(selected, ignore_index=True)
    logger.info("Selected %d records for L3", len(combined))

    # Assign splits
    combined = assign_splits(combined, fewshot_size=20, val_size=20, test_size=160, seed=args.seed)

    # Build JSONL records
    records = []
    for i, (_, row) in enumerate(combined.iterrows()):
        context = construct_l3_context(row)
        compartment_type = "same" if row.get("same_compartment") == True else (  # noqa: E712
            "different" if row.get("same_compartment") == False else "unknown"  # noqa: E712
        )
        rec = {
            "question_id": f"PPIL3-{i:04d}",
            "task": "ppi-l3",
            "split": row["split"],
            "difficulty": "medium",
            "context_text": context,
            "gold_answer": row.get("source_db", ""),
            "gold_category": compartment_type,
            "metadata": {
                "source_db": row.get("source_db"),
                "confidence_tier": row.get("confidence_tier"),
                "result_id": int(row["result_id"]) if pd.notna(row.get("result_id")) else None,
                "gene_symbol_1": row.get("gene_symbol_1"),
                "gene_symbol_2": row.get("gene_symbol_2"),
                "detection_method": row.get("detection_method"),
                "compartment_type": compartment_type,
                "uniprot_1": row.get("uniprot_1"),
                "uniprot_2": row.get("uniprot_2"),
            },
        }
        # Fewshot records need gold_reasoning for 3-shot L3 prompts
        if row["split"] == "fewshot":
            rec["gold_reasoning"] = _generate_gold_reasoning(row)
        records.append(rec)

    write_jsonl(records, args.output)

    stats = {
        "n_total": len(records),
        "compartment_distribution": dict(combined["same_compartment"].value_counts(dropna=False)),
        "split_distribution": dict(combined["split"].value_counts()),
        "source_distribution": dict(combined["source_db"].value_counts()),
        "seed": args.seed,
    }
    write_dataset_metadata(args.output.parent, "ppi-l3", stats)

    logger.info("PPI-L3 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
