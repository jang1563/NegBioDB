#!/usr/bin/env python3
"""Build DC-L4 tested/untested discrimination dataset for LLM benchmark.

Generates 475 records across 4 temporal groups:
  - classic_combos (125): Well-known in ALMANAC + DrugComb — gold_answer=tested
  - recent_combos (100): Recent DrugComb entries — gold_answer=tested
  - untested_plausible (125): Same therapeutic area but not in DB — gold_answer=untested
  - untested_rare (125): Different therapeutic areas, not in DB — gold_answer=untested

Split: fewshot 40 (10/group) + test 435

Usage:
    PYTHONPATH=src python scripts_dc/build_dc_l4_dataset.py
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
OUTPUT_DIR = PROJECT_ROOT / "exports" / "dc_llm"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build DC-L4 tested/untested dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_dc.db")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "dc_l4_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_dc.llm_dataset import (
        assign_splits,
        construct_l4_context,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)

    from negbiodb_dc.dc_db import get_connection
    conn = get_connection(args.db)
    try:
        # Classic combos: gold tier, multiple sources (well-known)
        classic = pd.read_sql_query("""
            SELECT ddp.*, ca.drug_name AS drug_a_name, cb.drug_name AS drug_b_name,
                   ca.known_targets AS drug_a_targets, cb.known_targets AS drug_b_targets
            FROM drug_drug_pairs ddp
            JOIN compounds ca ON ddp.compound_a_id = ca.compound_id
            JOIN compounds cb ON ddp.compound_b_id = cb.compound_id
            WHERE ddp.best_confidence = 'gold' AND ddp.num_sources >= 2
            ORDER BY RANDOM() LIMIT 250
        """, conn)

        # Recent combos: silver/bronze tier (newer data)
        recent = pd.read_sql_query("""
            SELECT ddp.*, ca.drug_name AS drug_a_name, cb.drug_name AS drug_b_name,
                   ca.known_targets AS drug_a_targets, cb.known_targets AS drug_b_targets
            FROM drug_drug_pairs ddp
            JOIN compounds ca ON ddp.compound_a_id = ca.compound_id
            JOIN compounds cb ON ddp.compound_b_id = cb.compound_id
            WHERE ddp.best_confidence IN ('silver', 'bronze') AND ddp.num_sources = 1
            ORDER BY RANDOM() LIMIT 200
        """, conn)

        # All compounds for untested generation
        all_compounds = pd.read_sql_query("""
            SELECT compound_id, drug_name, known_targets, atc_code
            FROM compounds WHERE drug_name IS NOT NULL
        """, conn)

        # Existing pairs for exclusion
        existing_pairs = set()
        cursor = conn.execute("SELECT compound_a_id, compound_b_id FROM drug_drug_pairs")
        for row in cursor:
            existing_pairs.add((row[0], row[1]))
    finally:
        conn.close()

    # Sample tested records
    if len(classic) > 125:
        classic = classic.sample(n=125, random_state=rng)
    classic["gold_answer"] = "tested"
    classic["temporal_group"] = "classic_combos"

    if len(recent) > 100:
        recent = recent.sample(n=100, random_state=rng)
    recent["gold_answer"] = "tested"
    recent["temporal_group"] = "recent_combos"

    # Generate untested pairs
    compounds = all_compounds.to_dict("records")
    untested_plausible = []
    untested_rare = []

    for _ in range(5000):
        if len(untested_plausible) >= 125 and len(untested_rare) >= 125:
            break
        idx = rng.choice(len(compounds), size=2, replace=False)
        ca, cb = compounds[idx[0]], compounds[idx[1]]
        pair_key = tuple(sorted([ca["compound_id"], cb["compound_id"]]))
        if pair_key in existing_pairs:
            continue

        # Check ATC codes for therapeutic area overlap
        atc_a = (ca.get("atc_code") or "")[:3]
        atc_b = (cb.get("atc_code") or "")[:3]
        same_area = atc_a and atc_b and atc_a == atc_b

        row = {
            "drug_a_name": ca["drug_name"],
            "drug_b_name": cb["drug_name"],
            "drug_a_targets": ca.get("known_targets"),
            "drug_b_targets": cb.get("known_targets"),
            "gold_answer": "untested",
        }

        if same_area and len(untested_plausible) < 125:
            row["temporal_group"] = "untested_plausible"
            untested_plausible.append(row)
        elif not same_area and len(untested_rare) < 125:
            row["temporal_group"] = "untested_rare"
            untested_rare.append(row)

    untested_p_df = pd.DataFrame(untested_plausible)
    untested_r_df = pd.DataFrame(untested_rare)

    combined = pd.concat(
        [classic, recent, untested_p_df, untested_r_df], ignore_index=True
    )
    combined = assign_splits(combined, fewshot_size=40, val_size=0, seed=args.seed)

    records = []
    for i, (_, row) in enumerate(combined.iterrows()):
        rec = {
            "question_id": f"DCL4-{i:04d}",
            "task": "dc-l4",
            "split": row["split"],
            "context_text": construct_l4_context(row),
            "gold_answer": row["gold_answer"],
            "temporal_group": row.get("temporal_group"),
            "metadata": {
                "drug_a": row["drug_a_name"],
                "drug_b": row["drug_b_name"],
            },
        }
        records.append(rec)

    write_jsonl(records, args.output)
    write_dataset_metadata(
        args.output.with_suffix(".meta.json"), "dc-l4",
        len(records), dict(combined["split"].value_counts()),
        seed=args.seed,
    )

    logger.info("DC-L4 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
