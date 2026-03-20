#!/usr/bin/env python3
"""Export CT ML benchmark datasets (pairs, M1, M2) with split columns.

Usage:
  PYTHONPATH=src python scripts_ct/export_ct_ml_dataset.py [options]

Options:
  --db-path PATH       CT database path (default: data/negbiodb_ct.db)
  --cto-parquet PATH   CTO outcomes parquet (default: data/ct/cto/cto_outcomes.parquet)
  --output-dir PATH    Output directory (default: exports/ct)
  --seed INT           Random seed (default: 42)
  --splits-only        Print split summary without writing files
  --skip-m1            Skip M1 binary dataset (e.g., when CTO not available)
  --skip-m2            Skip M2 result-level dataset
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_ct_ml")

# Add src to path if needed
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from negbiodb_ct.ct_db import DEFAULT_CT_DB_PATH
from negbiodb_ct.ct_export import (
    apply_all_ct_splits,
    apply_ct_m2_splits,
    build_ct_m1_dataset,
    export_ct_failure_dataset,
    export_ct_m2_dataset,
    generate_ct_leakage_report,
    load_ct_m2_data,
    load_ct_pairs_df,
    load_cto_success_pairs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Export CT ML benchmark datasets with split columns."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_CT_DB_PATH,
        help="Path to CT database",
    )
    parser.add_argument(
        "--cto-parquet",
        type=Path,
        default=Path("data/ct/cto/cto_outcomes.parquet"),
        help="Path to CTO outcomes parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/ct"),
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Print split summary only, no file output",
    )
    parser.add_argument(
        "--skip-m1",
        action="store_true",
        help="Skip M1 binary dataset",
    )
    parser.add_argument(
        "--skip-m2",
        action="store_true",
        help="Skip M2 result-level dataset",
    )
    args = parser.parse_args()

    t0 = time.time()

    # Step 1: Load all pairs
    logger.info("Step 1/7: Loading all CT pairs…")
    pairs_df = load_ct_pairs_df(args.db_path)
    logger.info("  → %d pairs loaded", len(pairs_df))

    # Step 2: Apply 6 splits
    logger.info("Step 2/7: Applying 6 split strategies…")
    pairs_with_splits = apply_all_ct_splits(pairs_df, args.seed)

    # Print split summary
    for col in [c for c in pairs_with_splits.columns if c.startswith("split_")]:
        counts = pairs_with_splits[col].value_counts(dropna=False)
        logger.info("  %s: %s", col, dict(counts))

    if args.splits_only:
        logger.info("--splits-only mode: no files written.")
        return

    # Step 3: Export failure pairs dataset
    logger.info("Step 3/7: Exporting failure pairs dataset…")
    result = export_ct_failure_dataset(args.db_path, args.output_dir, args.seed)
    logger.info("  → %d rows → %s", result["total_rows"], result["parquet_path"])

    # Step 4-5: M1 binary dataset
    if not args.skip_m1:
        logger.info("Step 4/7: Loading CTO success pairs…")
        if not args.cto_parquet.exists():
            logger.warning("CTO parquet not found: %s — skipping M1", args.cto_parquet)
        else:
            success_df, conflict_keys = load_cto_success_pairs(
                args.cto_parquet, args.db_path
            )
            logger.info(
                "  → %d clean success pairs, %d conflicts",
                len(success_df),
                len(conflict_keys),
            )

            logger.info("Step 5/7: Building M1 datasets (balanced + realistic + smiles_only)…")
            silver_gold_df = load_ct_pairs_df(
                args.db_path, min_confidence="silver"
            )
            logger.info("  → %d silver+gold failure pairs", len(silver_gold_df))

            m1_results = build_ct_m1_dataset(
                silver_gold_df,
                success_df,
                conflict_keys,
                args.output_dir,
                args.seed,
            )
            for variant, info in m1_results.items():
                logger.info(
                    "  M1 %s: %d rows (%d pos, %d neg) → %s",
                    variant,
                    info["total"],
                    info["n_pos"],
                    info["n_neg"],
                    info["path"],
                )
    else:
        logger.info("Steps 4-5/7: Skipped (--skip-m1)")

    # Step 6: M2 result-level dataset
    if not args.skip_m2:
        logger.info("Step 6/7: Loading and splitting M2 data…")
        m2_df = load_ct_m2_data(args.db_path)
        m2_with_splits = apply_ct_m2_splits(m2_df, args.seed)
        m2_result = export_ct_m2_dataset(m2_with_splits, args.output_dir)
        logger.info(
            "  → %d results → %s",
            m2_result["total_rows"],
            m2_result["parquet_path"],
        )
    else:
        logger.info("Step 6/7: Skipped (--skip-m2)")

    # Step 7: Leakage report
    logger.info("Step 7/7: Generating leakage report…")
    report_path = args.output_dir / "ct_leakage_report.json"
    cto_path = args.cto_parquet if args.cto_parquet.exists() and not args.skip_m1 else None
    report = generate_ct_leakage_report(
        args.db_path,
        cto_path=cto_path,
        output_path=report_path,
        seed=args.seed,
    )

    # Print key integrity results
    cold = report.get("cold_split_integrity", {})
    for split_name, info in cold.items():
        leaks = info.get("leaks", -1)
        status = "PASS" if leaks == 0 else "FAIL"
        logger.info("  %s leakage check: %s (leaks=%d)", split_name, status, leaks)

    m1_check = report.get("m1_conflict_free", {})
    if m1_check:
        status = "PASS" if m1_check.get("verified") else "FAIL"
        logger.info(
            "  M1 conflict-free: %s (overlapping=%d)",
            status,
            m1_check.get("overlapping_pairs", -1),
        )

    elapsed = time.time() - t0
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
