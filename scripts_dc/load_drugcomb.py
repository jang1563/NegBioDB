#!/usr/bin/env python3
"""Load DrugComb synergy data into DC database.

Usage:
    python scripts_dc/load_drugcomb.py [--db-path data/negbiodb_dc.db] \
        [--data-dir data/dc/drugcomb] [--batch-size 5000] \
        [--synergy-parquet data/dc/synergy_scores.parquet]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from negbiodb_dc.dc_db import DEFAULT_DC_DB_PATH, run_dc_migrations
from negbiodb_dc.etl_drugcomb import run_drugcomb_etl
from negbiodb_dc.synergy_compute import SynergyScores


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="Load DrugComb data into DC database")
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DC_DB_PATH,
        help="Path to DC database",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=_PROJECT_ROOT / "data" / "dc" / "drugcomb",
        help="Directory containing DrugComb files",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--synergy-parquet", type=Path, default=None,
        help="Path to pre-computed synergy_scores.parquet from compute_synergy_scores.py",
    )
    args = parser.parse_args()

    # Ensure migrations are applied
    run_dc_migrations(args.db_path)

    # Load pre-computed synergy scores if provided
    synergy_scores = None
    if args.synergy_parquet is not None:
        if not args.synergy_parquet.exists():
            print(f"ERROR: synergy parquet not found: {args.synergy_parquet}", file=sys.stderr)
            return 1
        scores_df = pd.read_parquet(args.synergy_parquet)
        synergy_scores = {
            row.block_id: SynergyScores(
                zip_score=row.zip_score,
                bliss_score=row.bliss_score,
                loewe_score=row.loewe_score,
                hsa_score=row.hsa_score,
            )
            for row in scores_df.itertuples()
        }
        logging.getLogger(__name__).info(
            "Loaded %d pre-computed synergy scores", len(synergy_scores)
        )

    stats = run_drugcomb_etl(args.db_path, args.data_dir, args.batch_size, synergy_scores)
    print(f"DrugComb ETL complete: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
