#!/usr/bin/env python3
"""Build CP-L2 structured extraction dataset for the Cell Painting LLM benchmark."""

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

N_TOTAL = 500


def _build_gold_extraction(row) -> dict:
    return {
        "compound_identifier": row.get("compound_name") or row.get("inchikey") or "",
        "dose": float(row["dose"]) if row.get("dose") is not None else None,
        "dose_unit": row.get("dose_unit", "uM"),
        "cell_line": row.get("cell_line_name", ""),
        "batch_id": row.get("batch_name", ""),
        "dmso_distance_summary": f"{float(row['dmso_distance_mean']):.3f}",
        "reproducibility_summary": f"{float(row['replicate_reproducibility']):.3f}",
        "qc_summary": f"viability_ratio={float(row['viability_ratio']):.3f}; "
        f"n_valid={int(row['num_valid_observations'])}",
        "outcome_label": row.get("outcome_label", ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build CP-L2 extraction dataset.")
    parser.add_argument("--db", "--db-path", dest="db", type=Path,
                        default=PROJECT_ROOT / "data" / "negbiodb_cp.db")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-records", type=int, default=N_TOTAL)
    parser.add_argument("--fewshot-size", type=int, default=50)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument(
        "--min-confidence",
        choices=["gold", "silver", "bronze", "copper"],
        default="bronze",
    )
    parser.add_argument("--allow-proxy-smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output_path = args.output or ((args.output_dir or OUTPUT_DIR) / "cp_l2_dataset.jsonl")

    from negbiodb_cp.llm_dataset import (
        apply_max_per_compound,
        assign_splits,
        construct_l2_context,
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
        gold_extraction = _build_gold_extraction(row)
        records.append(
            {
                "question_id": f"CPL2-{i:04d}",
                "task": "cp-l2",
                "split": row["split"],
                "difficulty": difficulty_from_tier(row["confidence_tier"]),
                "context_text": construct_l2_context(row),
                "gold_answer": row["outcome_label"],
                "gold_category": row["outcome_label"],
                "metadata": {
                    "cp_result_id": int(row["cp_result_id"]),
                    "confidence_tier": row["confidence_tier"],
                    "gold_extraction": gold_extraction,
                },
            }
        )

    write_jsonl(records, output_path)
    write_dataset_metadata(
        output_path.parent,
        "cp-l2",
        {
            "n_total": len(records),
            "seed": args.seed,
            "min_confidence": args.min_confidence,
            "split_distribution": dict(df["split"].value_counts()),
            **load_cp_annotation_summary(args.db),
        },
    )
    logger.info("CP-L2 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
