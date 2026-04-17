#!/usr/bin/env python3
"""Build CP-L4 tested/untested tuple discrimination dataset."""

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
OUTPUT_DIR = PROJECT_ROOT / "exports" / "cp_llm"

N_PER_LABEL = 250
FEWSHOT_PER_LABEL = 25
VAL_PER_LABEL = 25
STANDARD_UNTESTED_DOSES = [0.1, 0.3, 1.0, 3.0, 10.0]


def _tested_tuple_frame(df: pd.DataFrame) -> pd.DataFrame:
    order_cols = ["compound_id", "cell_line_id", "dose", "dose_unit", "timepoint_h", "cp_result_id"]
    dedup_cols = ["compound_id", "cell_line_id", "dose", "dose_unit", "timepoint_h"]
    tested = (
        df.sort_values(order_cols)
        .drop_duplicates(subset=dedup_cols, keep="first")
        .reset_index(drop=True)
        .copy()
    )
    tested["gold_answer"] = "tested"
    tested["gold_category"] = "tested"
    return tested


def _generate_untested_rows(tested: pd.DataFrame, target_n: int, rng: np.random.RandomState) -> pd.DataFrame:
    existing = {
        (
            int(row["compound_id"]),
            int(row["cell_line_id"]),
            float(row["dose"]) if row["dose"] is not None else 0.0,
            row["dose_unit"] or "uM",
            float(row["timepoint_h"]) if row["timepoint_h"] is not None else 48.0,
        )
        for row in tested.to_dict(orient="records")
    }
    compounds = tested[["compound_id", "compound_name", "inchikey"]].drop_duplicates().to_dict(orient="records")
    cell_lines = tested[["cell_line_id", "cell_line_name"]].drop_duplicates().to_dict(orient="records")
    dose_units = tested["dose_unit"].dropna().unique().tolist() or ["uM"]
    timepoints = sorted(float(v) for v in tested["timepoint_h"].dropna().unique().tolist()) or [48.0]
    _non_null_doses = sorted(float(v) for v in tested["dose"].dropna().unique().tolist())
    candidate_doses: list[float | None]
    if not _non_null_doses:
        candidate_doses = [None]
    elif len(_non_null_doses) == 1:
        existing_set = {round(v, 6) for v in _non_null_doses}
        extras = [dose for dose in STANDARD_UNTESTED_DOSES if round(dose, 6) not in existing_set]
        candidate_doses = _non_null_doses + extras
    else:
        candidate_doses = _non_null_doses

    # Build per-compound tested set to enable cross-compound pairing
    compound_tested = {}
    for row in tested.to_dict(orient="records"):
        cid = int(row["compound_id"])
        if cid not in compound_tested:
            compound_tested[cid] = set()
        compound_tested[cid].add((
            int(row["cell_line_id"]),
            float(row["dose"]) if row["dose"] is not None else 0.0,
            row["dose_unit"] or "uM",
            float(row["timepoint_h"]) if row["timepoint_h"] is not None else 48.0,
        ))

    rows = []
    seen = set()

    for _ in range(target_n * 60):
        if len(rows) >= target_n:
            break
        # Pick a compound and pair it with a cell_line/dose from a DIFFERENT
        # compound to create a realistic but untested combination.
        compound = compounds[int(rng.randint(len(compounds)))]
        cell_line = cell_lines[int(rng.randint(len(cell_lines)))]
        dose = candidate_doses[int(rng.randint(len(candidate_doses)))]
        dose_unit = dose_units[int(rng.randint(len(dose_units)))]
        timepoint_h = timepoints[int(rng.randint(len(timepoints)))]
        key = (
            int(compound["compound_id"]),
            int(cell_line["cell_line_id"]),
            float(dose) if dose is not None else 0.0,
            dose_unit,
            float(timepoint_h),
        )
        if key in existing or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "compound_id": int(compound["compound_id"]),
                "compound_name": compound.get("compound_name"),
                "inchikey": compound.get("inchikey"),
                "cell_line_id": int(cell_line["cell_line_id"]),
                "cell_line_name": cell_line.get("cell_line_name"),
                "dose": float(dose) if dose is not None else None,
                "dose_unit": dose_unit,
                "timepoint_h": float(timepoint_h),
                "gold_answer": "untested",
                "gold_category": "untested",
                "confidence_tier": "copper",
                "batch_name": None,
                "cp_result_id": None,
            }
        )

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build CP-L4 tested/untested dataset.")
    parser.add_argument("--db", "--db-path", dest="db", type=Path,
                        default=PROJECT_ROOT / "data" / "negbiodb_cp.db")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-per-label", "--n-per-class", dest="n_per_label", type=int, default=N_PER_LABEL)
    parser.add_argument("--fewshot-per-class", dest="fewshot_per_label", type=int, default=FEWSHOT_PER_LABEL)
    parser.add_argument("--val-per-class", dest="val_per_label", type=int, default=VAL_PER_LABEL)
    parser.add_argument(
        "--min-confidence",
        choices=["gold", "silver", "bronze", "copper"],
        default="bronze",
    )
    parser.add_argument("--allow-proxy-smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output_path = args.output or ((args.output_dir or OUTPUT_DIR) / "cp_l4_dataset.jsonl")

    from negbiodb_cp.llm_dataset import (
        assign_splits,
        construct_l4_context,
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

    tested = _tested_tuple_frame(df)
    if len(tested) > args.n_per_label:
        tested = tested.sample(n=args.n_per_label, random_state=args.seed).reset_index(drop=True)
    untested = _generate_untested_rows(tested, args.n_per_label, rng)
    if untested.empty:
        logger.error("Unable to synthesize any untested CP tuples.")
        return 1

    label_frames = []
    for answer, subset in [("tested", tested), ("untested", untested)]:
        subset = subset.copy()
        subset = assign_splits(
            subset,
            fewshot_size=min(args.fewshot_per_label, len(subset)),
            val_size=min(args.val_per_label, max(0, len(subset) - args.fewshot_per_label)),
            test_size=max(0, len(subset) - args.fewshot_per_label - args.val_per_label),
            seed=args.seed,
        )
        subset["gold_answer"] = answer
        subset["gold_category"] = answer
        label_frames.append(subset)

    combined = pd.concat(label_frames, ignore_index=True)
    records = []
    for i, (_, row) in enumerate(combined.iterrows()):
        records.append(
            {
                "question_id": f"CPL4-{i:04d}",
                "task": "cp-l4",
                "split": row["split"],
                "difficulty": (
                    difficulty_from_tier(row["confidence_tier"])
                    if row["gold_answer"] == "tested" else "hard"
                ),
                "context_text": construct_l4_context(row),
                "gold_answer": row["gold_answer"],
                "gold_category": row["gold_category"],
                "metadata": {
                    "cp_result_id": (
                        int(row["cp_result_id"]) if pd.notna(row.get("cp_result_id")) else None
                    ),
                    "compound_id": int(row["compound_id"]),
                    "cell_line_id": int(row["cell_line_id"]),
                    "dose": float(row["dose"]) if row["dose"] is not None else None,
                    "dose_unit": row["dose_unit"] or "uM",
                    "timepoint_h": float(row["timepoint_h"]) if row["timepoint_h"] is not None else 48.0,
                },
            }
        )

    write_jsonl(records, output_path)
    write_dataset_metadata(
        output_path.parent,
        "cp-l4",
        {
            "n_total": len(records),
            "seed": args.seed,
            "min_confidence": args.min_confidence,
            "label_distribution": dict(combined["gold_answer"].value_counts()),
            "split_distribution": dict(combined["split"].value_counts()),
            **load_cp_annotation_summary(args.db),
        },
    )
    logger.info("CP-L4 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
