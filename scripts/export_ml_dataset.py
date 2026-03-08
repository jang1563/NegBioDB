"""Export ML benchmark datasets from NegBioDB.

Orchestrates the full export pipeline:
  1. Generate DB-level splits (random, cold_compound, cold_target, etc.)
  2. Export negative dataset (Parquet + splits CSV)
  3. Extract ChEMBL positives + merge M1 balanced/realistic
  4. Generate random negative controls for Exp 1
  5. Generate leakage report

Usage:
    uv run python scripts/export_ml_dataset.py
    uv run python scripts/export_ml_dataset.py --splits-only
    uv run python scripts/export_ml_dataset.py --random-negatives
    uv run python scripts/export_ml_dataset.py --skip-positives

Prerequisites:
    - Database populated: all ETL scripts run
    - ChEMBL SQLite: data/chembl/chembl_36.db
"""

import argparse
import logging
import time
from pathlib import Path

from negbiodb.db import DEFAULT_DB_PATH, connect
from negbiodb.export import (
    export_negative_dataset,
    extract_chembl_positives,
    generate_cold_compound_split,
    generate_cold_target_split,
    generate_degree_balanced_split,
    generate_degree_matched_negatives,
    generate_leakage_report,
    generate_random_split,
    generate_scaffold_split,
    generate_temporal_split,
    generate_uniform_random_negatives,
    merge_positive_negative,
)

logger = logging.getLogger(__name__)


def _generate_all_splits(db_path: Path, seed: int) -> None:
    """Generate all 6 DB-level split strategies."""
    split_fns = [
        ("random_v1", generate_random_split),
        ("cold_compound_v1", generate_cold_compound_split),
        ("cold_target_v1", generate_cold_target_split),
        ("temporal_v1", generate_temporal_split),
        ("scaffold_v1", generate_scaffold_split),
        ("degree_balanced_v1", generate_degree_balanced_split),
    ]
    with connect(db_path) as conn:
        for name, fn in split_fns:
            t0 = time.time()
            logger.info("Generating split: %s", name)
            fn(conn, seed=seed)
            conn.commit()
            logger.info("  %s done (%.1f min)", name, (time.time() - t0) / 60)


def main():
    parser = argparse.ArgumentParser(
        description="Export NegBioDB ML benchmark datasets"
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH,
        help="Path to NegBioDB SQLite database",
    )
    parser.add_argument(
        "--chembl-db", type=Path, default=Path("data/chembl/chembl_36.db"),
        help="Path to ChEMBL SQLite database",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("exports"),
        help="Output directory for exported files",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splits and sampling",
    )
    parser.add_argument(
        "--splits-only", action="store_true",
        help="Only generate DB-level splits, skip export",
    )
    parser.add_argument(
        "--skip-positives", action="store_true",
        help="Skip ChEMBL positive extraction and M1 merge",
    )
    parser.add_argument(
        "--random-negatives", action="store_true",
        help="Generate random negative controls for Exp 1",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    t_total = time.time()

    # Step 1: Generate splits
    logger.info("=== Step 1: Generating DB-level splits ===")
    _generate_all_splits(args.db, args.seed)

    if args.splits_only:
        logger.info("Done (splits-only mode).")
        return

    # Step 2: Export negative dataset
    logger.info("=== Step 2: Exporting negative dataset ===")
    t0 = time.time()
    export_result = export_negative_dataset(args.db, args.output_dir)
    logger.info(
        "Exported %d pairs (%.1f min)",
        export_result["total_rows"],
        (time.time() - t0) / 60,
    )

    # Step 3: M1 merge (positives + negatives)
    if not args.skip_positives:
        logger.info("=== Step 3: Extracting ChEMBL positives + M1 merge ===")

        # 3a: Extract positives
        t0 = time.time()
        positives = extract_chembl_positives(args.chembl_db, args.db)
        logger.info(
            "Extracted %d positives (%.1f min)",
            len(positives), (time.time() - t0) / 60,
        )

        # 3b: Merge
        t0 = time.time()
        m1_result = merge_positive_negative(
            positives, args.db, args.output_dir, seed=args.seed,
        )
        for variant in ("balanced", "realistic"):
            info = m1_result[variant]
            logger.info(
                "M1 %s: %d pos + %d neg = %d total → %s",
                variant, info["n_pos"], info["n_neg"],
                info["total"], Path(info["path"]).name,
            )
        logger.info("M1 merge done (%.1f min)", (time.time() - t0) / 60)

        # Step 4: Random negatives for Exp 1
        if args.random_negatives:
            logger.info("=== Step 4: Random negative controls (Exp 1) ===")

            t0 = time.time()
            n_neg = m1_result["balanced"]["n_neg"]
            logger.info("Target: %d random negatives (matching M1 balanced)", n_neg)

            uniform_result = generate_uniform_random_negatives(
                args.db, positives, n_samples=n_neg,
                output_dir=args.output_dir, seed=args.seed,
            )
            logger.info(
                "Uniform random: %d total → %s (%.1f min)",
                uniform_result["total"], Path(uniform_result["path"]).name,
                (time.time() - t0) / 60,
            )

            t0 = time.time()
            degree_result = generate_degree_matched_negatives(
                args.db, positives, n_samples=n_neg,
                output_dir=args.output_dir, seed=args.seed,
            )
            logger.info(
                "Degree-matched: %d total → %s (%.1f min)",
                degree_result["total"], Path(degree_result["path"]).name,
                (time.time() - t0) / 60,
            )

    # Step 5: Leakage report
    logger.info("=== Step 5: Generating leakage report ===")
    report = generate_leakage_report(args.db, args.output_dir)
    logger.info(
        "Report: %d compounds, %d targets, %d pairs → %s",
        report["db_summary"]["compounds"],
        report["db_summary"]["targets"],
        report["db_summary"]["pairs"],
        Path(report["report_path"]).name,
    )

    logger.info(
        "=== All done (%.1f min total) ===",
        (time.time() - t_total) / 60,
    )


if __name__ == "__main__":
    main()
