#!/usr/bin/env python3
"""Export PPI ML benchmark datasets from NegBioDB.

Orchestrates the full PPI export pipeline:
  1. Generate DB-level splits (random, cold_protein, cold_both, degree_balanced)
  2. Export negative dataset → negbiodb_ppi_pairs.parquet

Usage:
    PYTHONPATH=src python scripts_ppi/export_ppi_ml_dataset.py
    PYTHONPATH=src python scripts_ppi/export_ppi_ml_dataset.py --splits-only
    PYTHONPATH=src python scripts_ppi/export_ppi_ml_dataset.py --db data/negbiodb_ppi.db

Prerequisites:
    - Database populated: all PPI ETL scripts run
    - Protein sequences fetched: scripts_ppi/fetch_sequences.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


def _generate_all_splits(db_path: Path, seed: int) -> None:
    """Generate all 4 DB-level split strategies."""
    from negbiodb_ppi.ppi_db import get_connection
    from negbiodb_ppi.export import (
        generate_random_split,
        generate_cold_protein_split,
        generate_cold_both_partition,
        generate_degree_balanced_split,
    )

    conn = get_connection(db_path)
    try:
        splits = [
            ("random", generate_random_split),
            ("cold_protein", generate_cold_protein_split),
            ("cold_both (Metis)", generate_cold_both_partition),
            ("degree_balanced", generate_degree_balanced_split),
        ]
        for name, fn in splits:
            t0 = time.time()
            logger.info("Generating split: %s", name)
            result = fn(conn, seed=seed)
            elapsed = time.time() - t0
            counts = result.get("counts", {})
            logger.info(
                "  %s done (%.1fs) — train=%d, val=%d, test=%d",
                name, elapsed,
                counts.get("train", 0),
                counts.get("val", 0),
                counts.get("test", 0),
            )
            if "excluded" in result:
                logger.info("  excluded (cross-partition): %d", result["excluded"])
        conn.commit()
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export PPI ML benchmark datasets"
    )
    parser.add_argument(
        "--db", type=Path,
        default=ROOT / "data" / "negbiodb_ppi.db",
        help="Path to negbiodb_ppi.db",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "exports" / "ppi",
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
        "--exclude-source", type=str, default=None,
        help="Exclude negatives from a specific source (e.g., 'huri')",
    )
    args = parser.parse_args(argv)

    if not args.db.exists():
        logger.error("Database not found: %s", args.db)
        return 1

    t_total = time.time()

    # Step 1: Generate DB-level splits
    logger.info("=== Step 1: Generating DB-level splits ===")
    _generate_all_splits(args.db, args.seed)

    if args.splits_only:
        logger.info("Done (splits-only mode, %.1f min).", (time.time() - t_total) / 60)
        return 0

    # Step 2: Export negatives to Parquet
    logger.info("=== Step 2: Exporting negative dataset ===")
    from negbiodb_ppi.export import export_negative_dataset

    t0 = time.time()
    result = export_negative_dataset(
        args.db, args.output_dir,
        exclude_source=args.exclude_source,
    )
    logger.info(
        "Exported %d pairs → %s (%.1f min)",
        result["total_rows"],
        result["parquet_path"],
        (time.time() - t0) / 60,
    )

    # Verify sequences
    import pandas as pd
    df = pd.read_parquet(result["parquet_path"], columns=["sequence_1", "sequence_2"])
    has_seq = df["sequence_1"].notna() & df["sequence_2"].notna()
    logger.info(
        "Sequences: %d/%d pairs have both sequences (%.1f%%)",
        has_seq.sum(), len(df), 100 * has_seq.sum() / max(len(df), 1),
    )

    logger.info(
        "=== All done (%.1f min total) ===",
        (time.time() - t_total) / 60,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
