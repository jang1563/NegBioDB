#!/usr/bin/env python3
"""Build CT-L2 structured extraction dataset for LLM benchmark.

Generates 500 records from bronze tier (why_stopped required).
7-way category proportional, capped at 40% efficacy.
Phase 1 gold: failure_category only.

Output: exports/ct_llm/ct_l2_dataset.jsonl
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

N_TOTAL = 500
MAX_EFFICACY_FRAC = 0.40


def format_l2_context(row: pd.Series) -> str:
    """Generate CT-L2 extraction context (bronze tier).

    Drug + condition + phase + why_stopped (quoted).
    """
    lines = [
        f"Drug: {row.get('intervention_name', 'Unknown')}",
        f"Condition: {row.get('condition_name', 'Unknown')}",
    ]
    phase = row.get("trial_phase") or row.get("highest_phase_reached")
    if phase:
        lines.append(f"Phase: {phase}")
    why = row.get("why_stopped", "")
    if why and pd.notna(why):
        lines.append(f'Termination text: "{why}"')
    return "\n".join(lines)


def classify_difficulty(row: pd.Series) -> str:
    """Difficulty by why_stopped text complexity."""
    why = str(row.get("why_stopped", "") or "")
    cat = str(row.get("failure_category", "")).lower()

    # Easy: explicit category keyword present, reasonable length
    easy_keywords = {
        "efficacy": ["futility", "lack of efficacy", "no benefit", "failed to demonstrate"],
        "safety": ["adverse", "toxicity", "safety", "hepatotox", "cardiotox"],
        "enrollment": ["enrollment", "accrual", "recruitment", "enroll"],
        "strategic": ["business", "strategic", "portfolio", "commercial"],
    }
    for kw_cat, kws in easy_keywords.items():
        if cat == kw_cat and any(kw in why.lower() for kw in kws):
            if len(why) >= 50:
                return "easy"

    # Hard: very short or vague
    if len(why) < 20:
        return "hard"

    return "medium"


def build_l2_dataset(db_path: Path, seed: int) -> list[dict]:
    """Build CT-L2 dataset from DB."""
    from negbiodb_ct.llm_dataset import (
        apply_max_per_drug,
        infer_therapeutic_area,
        load_candidate_pool,
    )

    rng = np.random.RandomState(seed)

    # Load bronze-only with non-empty why_stopped
    pool = load_candidate_pool(
        db_path,
        tier_filter="= 'bronze'",
        extra_where="ct.why_stopped IS NOT NULL AND ct.why_stopped != ''",
    )
    logger.info("Bronze pool with why_stopped: %d records", len(pool))

    if len(pool) == 0:
        logger.error("No bronze records with why_stopped found!")
        return []

    # Apply max-per-drug
    pool = apply_max_per_drug(pool, rng=rng)

    # Template dedup: drop exact why_stopped duplicates
    before = len(pool)
    pool = pool.drop_duplicates(subset=["why_stopped"], keep="first")
    logger.info("After why_stopped dedup: %d (dropped %d)", len(pool), before - len(pool))

    # Assign difficulty
    pool["difficulty"] = pool.apply(classify_difficulty, axis=1)

    # Category proportional sampling, capped at 40% efficacy
    cat_counts = pool["failure_category"].value_counts()
    logger.info("Category distribution:\n%s", cat_counts.to_string())

    # Compute target per category
    total_target = min(N_TOTAL, len(pool))
    cat_targets = {}
    for cat, count in cat_counts.items():
        frac = count / len(pool)
        target = int(total_target * frac)
        if cat == "efficacy":
            target = min(target, int(total_target * MAX_EFFICACY_FRAC))
        cat_targets[cat] = target

    # Redistribute excess from efficacy cap
    allocated = sum(cat_targets.values())
    if allocated < total_target:
        deficit = total_target - allocated
        non_efficacy = [c for c in cat_targets if c != "efficacy"]
        for c in non_efficacy:
            add = min(deficit, len(pool[pool["failure_category"] == c]) - cat_targets[c])
            if add > 0:
                cat_targets[c] += add
                deficit -= add
            if deficit <= 0:
                break

    # Sample
    sampled_parts = []
    for cat, target in cat_targets.items():
        cat_pool = pool[pool["failure_category"] == cat]
        n = min(target, len(cat_pool))
        if n > 0:
            sampled_parts.append(
                cat_pool.sample(n, random_state=rng.randint(0, 2**31))
            )
            logger.info("Category %s: sampled %d (target %d)", cat, n, target)

    sampled = pd.concat(sampled_parts, ignore_index=True)
    logger.info("Total sampled: %d", len(sampled))

    # Build records
    records = []
    for _, row in sampled.iterrows():
        context_text = format_l2_context(row)
        # gold_extraction: Phase 1 only has failure_category as gold truth;
        # other fields are null placeholders for Phase 2 manual annotation.
        gold_extraction = {
            "failure_category": row["failure_category"],
            "failure_subcategory": None,
            "affected_system": None,
            "severity_indicator": None,
            "quantitative_evidence": None,
            "decision_maker": None,
            "patient_impact": None,
        }
        records.append({
            "question_id": None,
            "task": "CT-L2",
            "gold_answer": row["failure_category"],
            "gold_extraction": gold_extraction,
            "gold_category": row["failure_category"],
            "difficulty": row["difficulty"],
            "context_text": context_text,
            "metadata": {
                "result_id": int(row["result_id"]),
                "source_trial_id": row.get("source_trial_id"),
                "intervention_name": row.get("intervention_name"),
                "condition_name": row.get("condition_name"),
                "confidence_tier": "bronze",
                "why_stopped": row.get("why_stopped"),
                "therapeutic_area": infer_therapeutic_area(
                    row.get("condition_name", "")
                ),
            },
        })

    return records


def main(argv: list[str] | None = None) -> int:
    from negbiodb_ct.llm_dataset import (
        assign_splits,
        write_dataset_metadata,
        write_jsonl,
    )

    parser = argparse.ArgumentParser(description="Build CT-L2 extraction dataset")
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_ct.db")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.db_path.exists():
        logger.error("CT database not found: %s", args.db_path)
        return 1

    records = build_l2_dataset(args.db_path, args.seed)
    if not records:
        logger.error("No records generated!")
        return 1

    df = pd.DataFrame(records)
    df = assign_splits(df, fewshot_size=50, val_size=50, test_size=400, seed=args.seed)

    output_records = []
    for i, (_, row) in enumerate(df.iterrows()):
        rec = row.to_dict()
        rec["question_id"] = f"CTL2-{i:04d}"
        output_records.append(rec)

    output_path = args.output_dir / "ct_l2_dataset.jsonl"
    write_jsonl(output_records, output_path)

    from collections import Counter
    splits = Counter(r["split"] for r in output_records)
    cats = Counter(r["gold_category"] for r in output_records)
    diffs = Counter(r["difficulty"] for r in output_records)

    logger.info("=== CT-L2 Dataset Summary ===")
    logger.info("Total: %d", len(output_records))
    logger.info("Categories: %s", dict(cats))
    logger.info("Difficulty: %s", dict(diffs))
    logger.info("Splits: %s", dict(splits))

    write_dataset_metadata(args.output_dir, "ct-l2", {
        "total": len(output_records),
        "categories": dict(cats),
        "difficulty": dict(diffs),
        "splits": dict(splits),
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
