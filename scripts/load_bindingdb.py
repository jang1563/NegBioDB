"""Load BindingDB inactive DTI data into NegBioDB.

Usage:
    uv run python scripts/load_bindingdb.py
"""

import argparse
import logging
from pathlib import Path

from negbiodb.db import DEFAULT_DB_PATH
from negbiodb.etl_bindingdb import run_bindingdb_etl


def main():
    parser = argparse.ArgumentParser(description="Load BindingDB inactives into NegBioDB")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to NegBioDB SQLite database",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Chunk size for BindingDB TSV",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=== BindingDB ETL ===")
    stats = run_bindingdb_etl(args.db_path, chunksize=args.chunksize)

    print("\n=== BindingDB ETL Summary ===")
    print(f"Rows read:      {stats['rows_read']}")
    print(f"Filtered:       {stats['rows_filtered_inactive']}")
    print(f"Skipped:        {stats['rows_skipped']}")
    print(f"Attempted ins:  {stats['rows_attempted_insert']}")
    print(f"Inserted:       {stats['results_inserted']}")
    print(f"Pairs total:    {stats['pairs_total']}")
    print("\nBindingDB ETL complete.")


if __name__ == "__main__":
    main()
