#!/usr/bin/env python3
"""Build CT-L4 tested/untested dataset for LLM benchmark.

Generates 500 drug-condition pairs:
  250 tested (125 pre-2020, 125 post-2023)
  250 untested (125 trick, 125 obvious)

Output: exports/ct_llm/ct_l4_dataset.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from negbiodb_ct.llm_dataset import is_code_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "exports" / "ct_llm"

N_TESTED_PRE = 125
N_TESTED_POST = 125
N_UNTESTED_TRICK = 125
N_UNTESTED_OBVIOUS = 125

def _is_recognizable_drug(name: str, has_chembl: bool) -> bool:
    """Check if drug name is recognizable (not just a code name)."""
    if not name:
        return False
    name = name.strip()
    if is_code_name(name):
        return False
    # Must have ≥2 words OR be a known drug (chembl_id resolved)
    if has_chembl:
        return True
    return len(name.split()) >= 2 or len(name) >= 8


def load_all_tested_pairs(db_path: Path) -> set[tuple[int, int]]:
    """Load all (intervention_id, condition_id) that share a trial."""
    from negbiodb_ct.ct_db import get_connection

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT DISTINCT ti.intervention_id, tc.condition_id
            FROM trial_interventions ti
            JOIN trial_conditions tc ON ti.trial_id = tc.trial_id"""
        ).fetchall()
        return set(rows)
    finally:
        conn.close()


def load_icp_pairs(db_path: Path) -> set[tuple[int, int]]:
    """Load all (intervention_id, condition_id) from intervention_condition_pairs."""
    from negbiodb_ct.ct_db import get_connection

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT intervention_id, condition_id FROM intervention_condition_pairs"
        ).fetchall()
        return set(rows)
    finally:
        conn.close()


def select_tested_pairs(db_path: Path, seed: int) -> list[dict]:
    """Select tested pairs with pre_2020/post_2023 temporal split."""
    from negbiodb_ct.ct_db import get_connection
    from negbiodb_ct.llm_dataset import infer_therapeutic_area

    conn = get_connection(db_path)
    rng = np.random.RandomState(seed)

    sql = """
    SELECT DISTINCT
        i.intervention_id, i.intervention_name, i.molecular_type,
        i.chembl_id, i.canonical_smiles,
        c.condition_id, c.condition_name,
        MIN(CASE
            WHEN ct.completion_date IS NOT NULL
                 AND LENGTH(ct.completion_date) >= 4
            THEN CAST(SUBSTR(ct.completion_date, 1, 4) AS INTEGER)
            ELSE NULL
        END) AS earliest_year
    FROM trial_failure_results tfr
    JOIN interventions i ON tfr.intervention_id = i.intervention_id
    JOIN conditions c ON tfr.condition_id = c.condition_id
    LEFT JOIN clinical_trials ct ON tfr.trial_id = ct.trial_id
    WHERE i.intervention_type IN ('drug', 'biologic', 'combination')
      AND LOWER(i.intervention_name) NOT LIKE '%placebo%'
    GROUP BY i.intervention_id, c.condition_id
    HAVING earliest_year IS NOT NULL
       AND earliest_year BETWEEN 1990 AND 2030
    """
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()

    logger.info("Tested pair candidates: %d", len(df))

    # Filter recognizable drug names (Python regex, not SQL)
    df["recognizable"] = df.apply(
        lambda r: _is_recognizable_drug(
            r["intervention_name"], pd.notna(r["chembl_id"])
        ),
        axis=1,
    )
    df = df[df["recognizable"]].copy()
    logger.info("After name filter: %d", len(df))

    # Split by temporal group
    pre_2020 = df[df["earliest_year"] < 2020].copy()
    post_2023 = df[df["earliest_year"] >= 2023].copy()
    logger.info("Pre-2020: %d, Post-2023: %d", len(pre_2020), len(post_2023))

    results = []
    for temporal_df, n_target, group_name in [
        (pre_2020, N_TESTED_PRE, "pre_2020"),
        (post_2023, N_TESTED_POST, "post_2023"),
    ]:
        # Dedup by (intervention_id, condition_id)
        temporal_df = temporal_df.drop_duplicates(
            subset=["intervention_id", "condition_id"], keep="first"
        )
        n = min(n_target, len(temporal_df))
        sampled = temporal_df.sample(n, random_state=rng.randint(0, 2**31))

        for _, row in sampled.iterrows():
            condition = row.get("condition_name", "")
            ta = infer_therapeutic_area(condition)
            results.append({
                "question_id": None,
                "task": "CT-L4",
                "gold_answer": "tested",
                "gold_category": None,
                "difficulty": None,
                "temporal_group": group_name,
                "context_text": _format_l4_context(row, ta),
                "metadata": {
                    "intervention_id": int(row["intervention_id"]),
                    "condition_id": int(row["condition_id"]),
                    "intervention_name": row["intervention_name"],
                    "condition_name": condition,
                    "molecular_type": row.get("molecular_type"),
                    "therapeutic_area": ta,
                    "earliest_year": int(row["earliest_year"]),
                },
            })

    logger.info("Tested pairs selected: %d", len(results))
    return results


def select_untested_pairs(
    db_path: Path,
    tested_trial_pairs: set[tuple[int, int]],
    icp_pairs: set[tuple[int, int]],
    seed: int,
) -> list[dict]:
    """Select untested pairs: trick + obvious."""
    from negbiodb_ct.ct_db import get_connection
    from negbiodb_ct.llm_dataset import infer_therapeutic_area

    conn = get_connection(db_path)
    rng = np.random.RandomState(seed + 100)

    # Load all interventions and conditions
    interventions = pd.read_sql_query(
        """SELECT intervention_id, intervention_name, molecular_type,
                  chembl_id, canonical_smiles, intervention_type
           FROM interventions
           WHERE intervention_type IN ('drug', 'biologic', 'combination')
             AND LOWER(intervention_name) NOT LIKE '%placebo%'""",
        conn,
    )
    conditions = pd.read_sql_query(
        "SELECT condition_id, condition_name FROM conditions", conn
    )
    conn.close()

    # Filter recognizable drugs
    interventions["recognizable"] = interventions.apply(
        lambda r: _is_recognizable_drug(
            r["intervention_name"], pd.notna(r["chembl_id"])
        ),
        axis=1,
    )
    named_drugs = interventions[interventions["recognizable"]].copy()

    # Compute intervention degree for "high-degree" selection
    drug_ids_in_icp = {}
    for iid, cid in icp_pairs:
        drug_ids_in_icp[iid] = drug_ids_in_icp.get(iid, 0) + 1

    named_drugs["degree"] = named_drugs["intervention_id"].map(drug_ids_in_icp).fillna(0)
    high_degree_drugs = named_drugs.nlargest(500, "degree")

    # Compute condition therapeutic areas
    conditions["ta"] = conditions["condition_name"].apply(
        lambda x: infer_therapeutic_area(x) if pd.notna(x) else "other"
    )

    # Build per-drug tested conditions
    drug_tested_conditions: dict[int, set[int]] = {}
    for iid, cid in icp_pairs:
        drug_tested_conditions.setdefault(iid, set()).add(cid)

    # --- Trick pairs: drug × plausible-but-untested condition ---
    trick_pairs = []
    drug_indices = rng.permutation(len(high_degree_drugs))

    for idx in drug_indices:
        if len(trick_pairs) >= N_UNTESTED_TRICK:
            break
        drug_row = high_degree_drugs.iloc[idx]
        iid = drug_row["intervention_id"]
        tested_conds = drug_tested_conditions.get(iid, set())

        if not tested_conds:
            continue

        # Find conditions with same therapeutic area that this drug HASN'T been tested for
        tested_cond_rows = conditions[conditions["condition_id"].isin(tested_conds)]
        if tested_cond_rows.empty:
            continue
        drug_ta = tested_cond_rows["ta"].mode()
        if drug_ta.empty:
            continue
        primary_ta = drug_ta.iloc[0]

        # Candidate untested conditions in same TA
        same_ta = conditions[
            (conditions["ta"] == primary_ta)
            & (~conditions["condition_id"].isin(tested_conds))
        ]
        if same_ta.empty:
            continue

        # Pick a random untested condition
        cand = same_ta.sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
        cid = cand["condition_id"]

        # Verify truly untested
        if (iid, cid) in tested_trial_pairs or (iid, cid) in icp_pairs:
            continue

        trick_pairs.append({
            "question_id": None,
            "task": "CT-L4",
            "gold_answer": "untested",
            "gold_category": None,
            "difficulty": None,
            "temporal_group": None,
            "untested_type": "trick",
            "context_text": _format_l4_context_untested(drug_row, cand),
            "metadata": {
                "intervention_id": int(iid),
                "condition_id": int(cid),
                "intervention_name": drug_row["intervention_name"],
                "condition_name": cand["condition_name"],
                "molecular_type": drug_row.get("molecular_type"),
                "therapeutic_area": cand["ta"],
            },
        })

    logger.info("Trick untested pairs: %d", len(trick_pairs))

    # --- Obvious pairs: drug × clearly unrelated condition ---
    obvious_pairs = []
    # Mismatch mapping: oncology drugs × cardiology conditions, etc.
    mismatch_pairs = [
        ("oncology", "cardiology"),
        ("cardiology", "oncology"),
        ("neurology", "infectious"),
        ("infectious", "neurology"),
        ("metabolic", "autoimmune"),
        ("autoimmune", "metabolic"),
        ("psychiatry", "respiratory"),
        ("respiratory", "psychiatry"),
    ]

    n_mismatch = len(mismatch_pairs)
    for mi, (drug_ta, cond_ta) in enumerate(mismatch_pairs):
        if len(obvious_pairs) >= N_UNTESTED_OBVIOUS:
            break
        # Drugs tested mostly in drug_ta
        ta_drugs = []
        for _, drug_row in named_drugs.iterrows():
            iid = drug_row["intervention_id"]
            tested_conds = drug_tested_conditions.get(iid, set())
            if not tested_conds:
                continue
            tested_rows = conditions[conditions["condition_id"].isin(tested_conds)]
            if tested_rows.empty:
                continue
            if (tested_rows["ta"] == drug_ta).mean() >= 0.5:
                ta_drugs.append(drug_row)
            if len(ta_drugs) >= 50:
                break

        ta_conditions = conditions[conditions["ta"] == cond_ta]
        if not ta_drugs or ta_conditions.empty:
            continue

        remaining_slots = N_UNTESTED_OBVIOUS - len(obvious_pairs)
        remaining_pairs = max(1, n_mismatch - mi)
        n_per = max(1, remaining_slots // remaining_pairs)
        rng.shuffle(ta_drugs)

        for drug_row in ta_drugs[:n_per * 2]:
            if len(obvious_pairs) >= N_UNTESTED_OBVIOUS:
                break
            iid = drug_row["intervention_id"]
            tested_conds = drug_tested_conditions.get(iid, set())

            cand_pool = ta_conditions[~ta_conditions["condition_id"].isin(tested_conds)]
            if cand_pool.empty:
                continue
            cand = cand_pool.sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
            cid = cand["condition_id"]

            if (iid, cid) in tested_trial_pairs or (iid, cid) in icp_pairs:
                continue

            obvious_pairs.append({
                "question_id": None,
                "task": "CT-L4",
                "gold_answer": "untested",
                "gold_category": None,
                "difficulty": None,
                "temporal_group": None,
                "untested_type": "obvious",
                "context_text": _format_l4_context_untested(drug_row, cand),
                "metadata": {
                    "intervention_id": int(iid),
                    "condition_id": int(cid),
                    "intervention_name": drug_row["intervention_name"],
                    "condition_name": cand["condition_name"],
                    "molecular_type": drug_row.get("molecular_type"),
                    "therapeutic_area": cand["ta"] if "ta" in cand.index else infer_therapeutic_area(cand["condition_name"]),
                },
            })

    logger.info("Obvious untested pairs: %d", len(obvious_pairs))
    return trick_pairs + obvious_pairs


def _format_l4_context(row: pd.Series, therapeutic_area: str) -> str:
    """Format L4 context for tested pairs (minimal info)."""
    lines = [f"Drug: {row.get('intervention_name', 'Unknown')}"]
    mol_type = row.get("molecular_type")
    if mol_type:
        lines.append(f"Drug type: {mol_type}")
    lines.append(f"Condition: {row.get('condition_name', 'Unknown')}")
    lines.append(f"Therapeutic area: {therapeutic_area}")
    return "\n".join(lines)


def _format_l4_context_untested(drug_row, cond_row) -> str:
    """Format L4 context for untested pairs."""
    from negbiodb_ct.llm_dataset import infer_therapeutic_area

    lines = [f"Drug: {drug_row.get('intervention_name', 'Unknown')}"]
    mol_type = drug_row.get("molecular_type")
    if mol_type:
        lines.append(f"Drug type: {mol_type}")
    cond_name = cond_row.get("condition_name", "Unknown") if hasattr(cond_row, "get") else cond_row["condition_name"]
    lines.append(f"Condition: {cond_name}")
    ta = cond_row.get("ta") if hasattr(cond_row, "get") and "ta" in cond_row.index else infer_therapeutic_area(cond_name)
    lines.append(f"Therapeutic area: {ta}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from negbiodb_ct.llm_dataset import assign_splits, write_dataset_metadata, write_jsonl

    parser = argparse.ArgumentParser(description="Build CT-L4 tested/untested dataset")
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_ct.db")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.db_path.exists():
        logger.error("CT database not found: %s", args.db_path)
        return 1

    # Load verification sets
    tested_trial_pairs = load_all_tested_pairs(args.db_path)
    icp_pairs = load_icp_pairs(args.db_path)
    logger.info("Tested trial pairs: %d, ICP pairs: %d", len(tested_trial_pairs), len(icp_pairs))

    # Select tested and untested
    tested = select_tested_pairs(args.db_path, args.seed)
    untested = select_untested_pairs(
        args.db_path, tested_trial_pairs, icp_pairs, args.seed
    )

    all_records = tested + untested
    logger.info("Total records: %d (tested=%d, untested=%d)", len(all_records), len(tested), len(untested))

    if not all_records:
        logger.error("No records generated!")
        return 1

    # Verify: no untested pair in tested sets
    n_leaks = 0
    for rec in all_records:
        if rec["gold_answer"] == "untested":
            iid = rec["metadata"]["intervention_id"]
            cid = rec["metadata"]["condition_id"]
            if (iid, cid) in tested_trial_pairs or (iid, cid) in icp_pairs:
                n_leaks += 1
    logger.info("Verification: %d leaked untested pairs (should be 0)", n_leaks)

    # Split: class-balanced (25/class fewshot, 25/class val, rest test)
    df = pd.DataFrame(all_records)
    tested_df = df[df["gold_answer"] == "tested"].copy()
    untested_df = df[df["gold_answer"] == "untested"].copy()

    tested_split = assign_splits(tested_df, 25, 25, len(tested_df) - 50, args.seed)
    untested_split = assign_splits(untested_df, 25, 25, len(untested_df) - 50, args.seed)
    final_df = pd.concat([tested_split, untested_split], ignore_index=True)

    output_records = []
    for i, (_, row) in enumerate(final_df.iterrows()):
        rec = row.to_dict()
        rec["question_id"] = f"CTL4-{i:04d}"
        output_records.append(rec)

    output_path = args.output_dir / "ct_l4_dataset.jsonl"
    write_jsonl(output_records, output_path)

    from collections import Counter
    splits = Counter(r["split"] for r in output_records)
    classes = Counter(r["gold_answer"] for r in output_records)
    temporal = Counter(r.get("temporal_group") for r in output_records if r.get("temporal_group"))

    logger.info("=== CT-L4 Dataset Summary ===")
    logger.info("Total: %d", len(output_records))
    logger.info("Classes: %s", dict(classes))
    logger.info("Temporal: %s", dict(temporal))
    logger.info("Splits: %s", dict(splits))

    write_dataset_metadata(args.output_dir, "ct-l4", {
        "total": len(output_records),
        "classes": dict(classes),
        "temporal": dict(temporal),
        "splits": dict(splits),
        "n_leaks": n_leaks,
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
