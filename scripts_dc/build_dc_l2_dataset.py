#!/usr/bin/env python3
"""Build DC-L2 mechanism extraction dataset for LLM benchmark.

Generates 500 records for structured extraction of drug combination
interaction details (interaction_type, mechanism, shared_targets, pathways).

Split: fewshot 30 + val 70 + test 400

Usage:
    PYTHONPATH=src python scripts_dc/build_dc_l2_dataset.py
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
OUTPUT_DIR = PROJECT_ROOT / "exports" / "dc_llm"

# Mechanism categories based on target overlap and class
_MECHANISM_MAP = {
    ("antagonistic", True): "competitive_binding",
    ("antagonistic", False): "opposing_pathways",
    ("synergistic", True): "target_redundancy",
    ("synergistic", False): "pathway_crosstalk",
    ("additive", True): "target_redundancy",
    ("additive", False): "unknown",
    ("context_dependent", True): "pharmacokinetic_interaction",
    ("context_dependent", False): "unknown",
}


def _infer_mechanism(row) -> str:
    """Infer mechanism from consensus class and target overlap."""
    has_shared = (row.get("num_shared_targets") or 0) > 0
    cls = row.get("consensus_class", "unknown")
    return _MECHANISM_MAP.get((cls, has_shared), "unknown")


def _infer_interaction_type(consensus_class: str) -> str:
    """Map consensus class to interaction type."""
    mapping = {
        "synergistic": "synergistic",
        "antagonistic": "antagonistic",
        "additive": "additive",
        "context_dependent": "additive",
    }
    return mapping.get(consensus_class, "additive")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build DC-L2 extraction dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_dc.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "dc_l2_dataset.jsonl")
    parser.add_argument("--n-records", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_dc.llm_dataset import (
        apply_max_per_drug,
        assign_splits,
        construct_l2_context,
        load_dc_candidate_pool,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)

    from negbiodb_dc.dc_db import get_connection
    conn = get_connection(args.db)
    try:
        df = load_dc_candidate_pool(conn, min_confidence="silver")
    finally:
        conn.close()

    # Require target info for meaningful extraction
    df = df[df["drug_a_targets"].notna() & df["drug_b_targets"].notna()]

    if len(df) > args.n_records * 2:
        df = df.sample(n=args.n_records * 2, random_state=rng)

    df = apply_max_per_drug(df, max_per_drug=10, rng=rng)

    if len(df) > args.n_records:
        df = df.sample(n=args.n_records, random_state=rng).reset_index(drop=True)

    df = assign_splits(df, fewshot_size=30, val_size=70, seed=args.seed)

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        # Build gold extraction
        shared = []
        if row.get("drug_a_targets") and row.get("drug_b_targets"):
            targets_a = set(row["drug_a_targets"].split(";"))
            targets_b = set(row["drug_b_targets"].split(";"))
            shared = sorted(targets_a & targets_b)

        gold_extraction = {
            "drug_a": {"name": row["drug_a_name"],
                       "primary_targets": [t for t in (row.get("drug_a_targets") or "").split(";") if t]},
            "drug_b": {"name": row["drug_b_name"],
                       "primary_targets": [t for t in (row.get("drug_b_targets") or "").split(";") if t]},
            "shared_targets": shared,
            "interaction_type": _infer_interaction_type(row.get("consensus_class", "")),
            "mechanism_of_interaction": _infer_mechanism(row),
            "affected_pathways": [],  # Cannot be auto-inferred; LLM must generate
        }

        rec = {
            "question_id": f"DCL2-{i:04d}",
            "task": "dc-l2",
            "split": row["split"],
            "context_text": construct_l2_context(row),
            "gold_extraction": gold_extraction,
            "metadata": {
                "pair_id": int(row["pair_id"]),
                "drug_a": row["drug_a_name"],
                "drug_b": row["drug_b_name"],
                "consensus_class": row.get("consensus_class"),
            },
        }
        records.append(rec)

    write_jsonl(records, args.output)
    write_dataset_metadata(
        args.output.with_suffix(".meta.json"), "dc-l2",
        len(records), dict(df["split"].value_counts()),
        seed=args.seed,
    )

    logger.info("DC-L2 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
