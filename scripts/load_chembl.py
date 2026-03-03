"""Load ChEMBL inactive DTI data into NegBioDB.

Usage:
    uv run python scripts/load_chembl.py [--chembl-db PATH]

Prerequisites:
    - Database created: make db
    - ChEMBL data downloaded: make download-chembl
"""

import argparse
import logging
from pathlib import Path

from negbiodb.db import DEFAULT_DB_PATH
from negbiodb.etl_chembl import run_chembl_etl


def main():
    parser = argparse.ArgumentParser(description="Load ChEMBL inactives into NegBioDB")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to NegBioDB SQLite database",
    )
    parser.add_argument(
        "--chembl-db",
        type=Path,
        default=None,
        help="Path to ChEMBL SQLite database (auto-detected if not specified)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("=== ChEMBL ETL ===")
    stats = run_chembl_etl(args.db_path, chembl_db_path=args.chembl_db)

    print(f"\n=== ChEMBL ETL Summary ===")
    print(f"Extracted:  {stats['records_extracted']} records from ChEMBL")
    print(f"Compounds:  {stats['compounds_standardized']} standardized, "
          f"{stats['compounds_failed_rdkit']} failed RDKit")
    print(f"Targets:    {stats['targets_prepared']} prepared")
    print(f"Results:    {stats['results_inserted']} inserted, "
          f"{stats['results_skipped']} skipped")
    print(f"Pairs:      {stats['pairs_total']} total compound-target pairs (all sources)")
    print("\nChEMBL ETL complete.")


if __name__ == "__main__":
    main()
