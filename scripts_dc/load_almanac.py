#!/usr/bin/env python3
"""Load NCI-ALMANAC synergy data into DC database.

Usage:
    python scripts_dc/load_almanac.py [--db-path data/negbiodb_dc.db] \
        [--data-dir data/dc/almanac] [--batch-size 5000]
"""

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from negbiodb_dc.dc_db import DEFAULT_DC_DB_PATH, run_dc_migrations
from negbiodb_dc.etl_almanac import run_almanac_etl


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="Load NCI-ALMANAC data into DC database")
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DC_DB_PATH,
        help="Path to DC database",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=_PROJECT_ROOT / "data" / "dc" / "almanac",
        help="Directory containing NCI-ALMANAC files",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    run_dc_migrations(args.db_path)

    stats = run_almanac_etl(args.db_path, args.data_dir, args.batch_size)
    print(f"NCI-ALMANAC ETL complete: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
