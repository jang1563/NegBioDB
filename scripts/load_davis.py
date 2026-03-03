"""Load DAVIS kinase binding dataset into NegBioDB.

Usage:
    uv run python scripts/load_davis.py [--skip-api]

Prerequisites:
    - Database created: make db
    - DAVIS data downloaded: make download-davis
"""

import argparse
import logging
from pathlib import Path

from negbiodb.db import DEFAULT_DB_PATH
from negbiodb.etl_davis import run_davis_etl


def main():
    parser = argparse.ArgumentParser(description="Load DAVIS into NegBioDB")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to NegBioDB SQLite database",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip UniProt API mapping, use cached mapping only",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("=== DAVIS ETL ===")
    stats = run_davis_etl(args.db_path, skip_api=args.skip_api)

    print(f"\n=== DAVIS ETL Summary ===")
    print(f"Compounds: {stats['compounds_inserted']} standardized")
    print(f"Targets:   {stats['targets_inserted']} mapped, "
          f"{stats['targets_unmapped']} unmapped (skipped)")
    print(f"Results:   {stats['results_loaded']} negative results loaded")
    print(f"Skipped:   {stats['results_skipped_unmapped']} unmapped, "
          f"{stats['results_skipped_active']} active, "
          f"{stats['results_skipped_borderline']} borderline")
    print(f"Pairs:     {stats['pairs_created']} compound-target pairs")
    print("\nDAVIS ETL complete.")


if __name__ == "__main__":
    main()
