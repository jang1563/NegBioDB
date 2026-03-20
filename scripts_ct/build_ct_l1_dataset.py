#!/usr/bin/env python3
"""Build CT-L1 MCQ dataset for LLM benchmark.

Generates 1,500 five-way MCQ records across 5 classes:
  A) Safety    (300) — drug toxicity, adverse events, safety signals
  B) Efficacy  (300) — failed to demonstrate therapeutic benefit
  C) Enrollment(300) — failed to recruit sufficient participants
  D) Strategic (300) — business/strategic/portfolio discontinuation
  E) Other     (300) — design, regulatory, PK, or other reasons

Difficulty: gold=easy(40%), silver=medium(35%), bronze=hard(25%)
Split: 300 fewshot (60/class) + 300 val (60/class) + 900 test (180/class)

Output: exports/ct_llm/ct_l1_dataset.jsonl
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
OUTPUT_DIR = PROJECT_ROOT / "exports" / "ct_llm"

# Class sizes
N_PER_CLASS = 300

# 8-way → 5-way MCQ mapping (from canonical source in llm_prompts)
from negbiodb_ct.llm_prompts import CATEGORY_TO_MCQ

MCQ_TO_LABEL = {
    "A": "safety",
    "B": "efficacy",
    "C": "enrollment",
    "D": "strategic",
    "E": "other",
}

# Difficulty by confidence tier
TIER_TO_DIFFICULTY = {
    "gold": "easy",
    "silver": "medium",
    "bronze": "hard",
}

# Difficulty proportions (target)
FRAC_EASY = 0.40
FRAC_MEDIUM = 0.35
FRAC_HARD = 0.25


def format_l1_context(row: pd.Series) -> str:
    """Generate CT-L1 MCQ context from a trial failure record.

    Gold/Silver: full quantitative context
    Bronze: text-only context (drug, condition, phase, why_stopped)
    """
    lines = [f"Drug: {row.get('intervention_name', 'Unknown')}"]

    # Add molecular type for biologics
    mol_type = row.get("molecular_type")
    if mol_type and mol_type != "small_molecule":
        lines.append(f"Drug type: {mol_type}")

    lines.append(f"Condition: {row.get('condition_name', 'Unknown')}")

    phase = row.get("trial_phase") or row.get("highest_phase_reached")
    if phase:
        lines.append(f"Phase: {phase}")

    tier = row.get("confidence_tier", "bronze")

    if tier in ("gold", "silver"):
        # Full quantitative context
        design = row.get("control_type")
        if design:
            lines.append(f"Design: {design}")
        blinding = row.get("blinding")
        if blinding and pd.notna(blinding):
            lines.append(f"Blinding: {blinding}")
        enrollment = row.get("enrollment_actual")
        if enrollment and pd.notna(enrollment):
            lines.append(f"Enrollment: {int(enrollment)}")

        endpoint_met = row.get("primary_endpoint_met")
        if endpoint_met and pd.notna(endpoint_met):
            lines.append(f"Primary endpoint met: {endpoint_met}")
        p_val = row.get("p_value_primary")
        if p_val and pd.notna(p_val):
            lines.append(f"p-value: {p_val}")
        effect = row.get("effect_size")
        if effect and pd.notna(effect):
            etype = row.get("effect_size_type", "")
            lines.append(f"Effect size ({etype}): {effect}" if etype else f"Effect size: {effect}")
        ci_lo, ci_hi = row.get("ci_lower"), row.get("ci_upper")
        if ci_lo and pd.notna(ci_lo) and ci_hi and pd.notna(ci_hi):
            lines.append(f"95% CI: [{ci_lo}, {ci_hi}]")

        saes = row.get("serious_adverse_events")
        if saes and pd.notna(saes):
            lines.append(f"Serious adverse events: {saes}")

        arm = row.get("arm_description")
        if arm and pd.notna(arm):
            lines.append(f"Arm: {arm}")

        interp = row.get("result_interpretation")
        if interp and pd.notna(interp):
            lines.append(f"Interpretation: {interp}")
    else:
        # Bronze: text-only context
        enrollment = row.get("enrollment_actual")
        if enrollment and pd.notna(enrollment):
            lines.append(f"Enrollment: {int(enrollment)}")
        why = row.get("why_stopped")
        if why and pd.notna(why):
            lines.append(f'Termination reason: "{why}"')
        detail = row.get("failure_detail")
        if detail and pd.notna(detail):
            lines.append(f"Detail: {detail}")

    return "\n".join(lines)


def build_l1_dataset(db_path: Path, seed: int) -> list[dict]:
    """Build CT-L1 dataset from DB."""
    from negbiodb_ct.llm_dataset import (
        apply_max_per_drug,
        infer_therapeutic_area,
        load_candidate_pool,
    )

    rng = np.random.RandomState(seed)

    # Load non-copper candidates
    pool = load_candidate_pool(db_path, tier_filter="!= 'copper'")
    pool["mcq_letter"] = pool["failure_category"].map(CATEGORY_TO_MCQ)
    pool = pool[pool["mcq_letter"].notna()].copy()
    logger.info("Pool after MCQ mapping: %d records", len(pool))

    # Apply max-per-drug cap
    pool = apply_max_per_drug(pool, rng=rng)

    # Assign difficulty by tier
    pool["difficulty"] = pool["confidence_tier"].map(TIER_TO_DIFFICULTY)
    pool.loc[pool["difficulty"].isna(), "difficulty"] = "hard"

    # Sample per class
    all_records = []
    for mcq_letter in "ABCDE":
        class_pool = pool[pool["mcq_letter"] == mcq_letter].copy()

        if len(class_pool) == 0:
            logger.warning("No records for class %s!", mcq_letter)
            continue

        # Stratify by difficulty
        n_target = min(N_PER_CLASS, len(class_pool))
        n_easy = int(n_target * FRAC_EASY)
        n_medium = int(n_target * FRAC_MEDIUM)
        n_hard = n_target - n_easy - n_medium

        sampled_parts = []
        for diff, count in [("easy", n_easy), ("medium", n_medium), ("hard", n_hard)]:
            diff_pool = class_pool[class_pool["difficulty"] == diff]
            actual = min(count, len(diff_pool))
            if actual > 0:
                sampled_parts.append(
                    diff_pool.sample(actual, random_state=rng.randint(0, 2**31))
                )
            shortfall = count - actual
            if shortfall > 0:
                logger.info(
                    "Class %s, diff %s: shortfall %d (available %d)",
                    mcq_letter, diff, shortfall, len(diff_pool),
                )

        if not sampled_parts:
            # Fallback: sample from entire class pool
            n_sample = min(n_target, len(class_pool))
            sampled = class_pool.sample(n_sample, random_state=rng.randint(0, 2**31))
        else:
            sampled = pd.concat(sampled_parts, ignore_index=True)
            # Fill shortfall from remaining pool
            remaining = n_target - len(sampled)
            if remaining > 0:
                used_ids = set(sampled["result_id"])
                leftover = class_pool[~class_pool["result_id"].isin(used_ids)]
                extra = leftover.sample(
                    min(remaining, len(leftover)),
                    random_state=rng.randint(0, 2**31),
                )
                sampled = pd.concat([sampled, extra], ignore_index=True)

        logger.info(
            "Class %s: sampled %d (target %d)",
            mcq_letter, len(sampled), N_PER_CLASS,
        )

        for _, row in sampled.iterrows():
            context_text = format_l1_context(row)
            therapeutic_area = infer_therapeutic_area(row.get("condition_name", ""))
            all_records.append({
                "question_id": None,  # Assigned after splitting
                "task": "CT-L1",
                "gold_answer": mcq_letter,
                "gold_category": row["failure_category"],
                "difficulty": row.get("difficulty", "medium"),
                "context_text": context_text,
                "metadata": {
                    "result_id": int(row["result_id"]),
                    "source_trial_id": row.get("source_trial_id"),
                    "intervention_name": row.get("intervention_name"),
                    "condition_name": row.get("condition_name"),
                    "confidence_tier": row.get("confidence_tier"),
                    "therapeutic_area": therapeutic_area,
                },
            })

    logger.info("Total records: %d", len(all_records))
    return all_records


def main(argv: list[str] | None = None) -> int:
    from negbiodb_ct.llm_dataset import (
        assign_splits,
        write_dataset_metadata,
        write_jsonl,
    )

    parser = argparse.ArgumentParser(description="Build CT-L1 MCQ dataset")
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_ct.db")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.db_path.exists():
        logger.error("CT database not found: %s", args.db_path)
        return 1

    records = build_l1_dataset(args.db_path, args.seed)
    if not records:
        logger.error("No records generated!")
        return 1

    # Convert to DataFrame for splitting
    df = pd.DataFrame(records)
    df["mcq_letter"] = df["gold_answer"]

    # Class-balanced split: 60/class fewshot, 60/class val, rest test
    split_records = []
    for letter in "ABCDE":
        class_df = df[df["mcq_letter"] == letter].copy()
        class_df = assign_splits(
            class_df,
            fewshot_size=60,
            val_size=60,
            test_size=len(class_df) - 120,
            seed=args.seed,
        )
        split_records.append(class_df)

    final_df = pd.concat(split_records, ignore_index=True)

    # Assign question IDs
    output_records = []
    for i, (_, row) in enumerate(final_df.iterrows()):
        rec = row.to_dict()
        rec["question_id"] = f"CTL1-{i:04d}"
        # Remove temporary column
        rec.pop("mcq_letter", None)
        output_records.append(rec)

    # Write
    output_path = args.output_dir / "ct_l1_dataset.jsonl"
    write_jsonl(output_records, output_path)

    # Stats
    from collections import Counter
    splits = Counter(r["split"] for r in output_records)
    classes = Counter(r["gold_answer"] for r in output_records)
    diffs = Counter(r["difficulty"] for r in output_records)

    logger.info("=== CT-L1 Dataset Summary ===")
    logger.info("Total: %d", len(output_records))
    logger.info("Classes: %s", dict(classes))
    logger.info("Difficulty: %s", dict(diffs))
    logger.info("Splits: %s", dict(splits))

    write_dataset_metadata(args.output_dir, "ct-l1", {
        "total": len(output_records),
        "classes": dict(classes),
        "difficulty": dict(diffs),
        "splits": dict(splits),
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
