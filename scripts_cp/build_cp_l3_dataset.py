#!/usr/bin/env python3
"""Build CP-L3 evidence-grounded explanation dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "exports" / "cp_llm"

N_TOTAL = 400


def _gold_reasoning(row) -> str:
    from negbiodb_cp.etl_jump import (
        DEFAULT_DMSO_DISTANCE_THRESHOLD,
        STRONG_DISTANCE_MULTIPLIER,
        SILVER_REPRO_THRESHOLD,
        TOXIC_VIABILITY_THRESHOLD,
    )
    label = row.get("outcome_label", "unknown")
    dist = float(row["dmso_distance_mean"])
    repro = float(row["replicate_reproducibility"])
    viability = float(row["viability_ratio"])
    inactive_thresh = DEFAULT_DMSO_DISTANCE_THRESHOLD
    strong_thresh = inactive_thresh * STRONG_DISTANCE_MULTIPLIER

    if label == "inactive":
        proximity = "well below" if dist < inactive_thresh * 0.5 else "below"
        return (
            f"The DMSO distance ({dist:.3f}) is {proximity} the inactive threshold "
            f"({inactive_thresh}), indicating minimal morphological change. "
            f"Reproducibility is {repro:.3f} and viability ratio {viability:.3f}. "
            "These metrics support a morphologically non-responsive call."
        )
    if label == "strong_phenotype":
        return (
            f"The DMSO distance ({dist:.3f}) exceeds the strong phenotype threshold "
            f"({strong_thresh}), with reproducibility {repro:.3f} "
            f"(above {SILVER_REPRO_THRESHOLD} cutoff) and viability ratio {viability:.3f} "
            f"(above {TOXIC_VIABILITY_THRESHOLD} toxicity cutoff). "
            "This indicates a genuine, reproducible morphological effect."
        )
    if label == "toxic_or_artifact":
        return (
            f"The viability ratio ({viability:.3f}) falls below the toxicity threshold "
            f"({TOXIC_VIABILITY_THRESHOLD}), indicating cell death or artifact. "
            f"The DMSO distance is {dist:.3f} and reproducibility {repro:.3f}, "
            "but the low viability overrides these as the dominant signal."
        )
    # weak_phenotype
    return (
        f"The DMSO distance ({dist:.3f}) is above the inactive threshold ({inactive_thresh}) "
        f"but below the strong phenotype threshold ({strong_thresh}), or the "
        f"reproducibility ({repro:.3f}) is below {SILVER_REPRO_THRESHOLD}. "
        f"Viability ratio is {viability:.3f}. This indicates a modest, less confident phenotypic shift."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build CP-L3 reasoning dataset.")
    parser.add_argument("--db", "--db-path", dest="db", type=Path,
                        default=PROJECT_ROOT / "data" / "negbiodb_cp.db")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-records", type=int, default=N_TOTAL)
    parser.add_argument("--fewshot-size", type=int, default=40)
    parser.add_argument("--val-size", type=int, default=40)
    parser.add_argument(
        "--min-confidence",
        choices=["gold", "silver", "bronze", "copper"],
        default="bronze",
    )
    parser.add_argument("--allow-proxy-smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output_path = args.output or ((args.output_dir or OUTPUT_DIR) / "cp_l3_dataset.jsonl")

    from negbiodb_cp.llm_dataset import (
        apply_max_per_compound,
        assign_splits,
        construct_l3_context,
        difficulty_from_tier,
        load_cp_annotation_summary,
        load_cp_candidate_pool,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)
    df = load_cp_candidate_pool(
        args.db,
        min_confidence=args.min_confidence,
        allow_proxy_smoke=args.allow_proxy_smoke,
    )
    if df.empty:
        logger.error("No CP candidates found in %s", args.db)
        return 1

    df = apply_max_per_compound(df, max_per_compound=6, rng=rng)
    if len(df) > args.n_records:
        df = df.sample(n=args.n_records, random_state=args.seed).reset_index(drop=True)
    df = assign_splits(
        df,
        fewshot_size=min(args.fewshot_size, len(df)),
        val_size=min(args.val_size, max(0, len(df) - args.fewshot_size)),
        test_size=max(0, len(df) - args.fewshot_size - args.val_size),
        seed=args.seed,
    )

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        records.append(
            {
                "question_id": f"CPL3-{i:04d}",
                "task": "cp-l3",
                "split": row["split"],
                "difficulty": difficulty_from_tier(row["confidence_tier"]),
                "context_text": construct_l3_context(row),
                "gold_answer": row["outcome_label"],
                "gold_category": row["outcome_label"],
                "metadata": {
                    "cp_result_id": int(row["cp_result_id"]),
                    "confidence_tier": row["confidence_tier"],
                    "has_orthogonal_evidence": int(row["has_orthogonal_evidence"]),
                    "gold_reasoning": _gold_reasoning(row),
                },
            }
        )

    write_jsonl(records, output_path)
    write_dataset_metadata(
        output_path.parent,
        "cp-l3",
        {
            "n_total": len(records),
            "seed": args.seed,
            "min_confidence": args.min_confidence,
            "split_distribution": dict(df["split"].value_counts()),
            **load_cp_annotation_summary(args.db),
        },
    )
    logger.info("CP-L3 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
