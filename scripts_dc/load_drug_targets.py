#!/usr/bin/env python3
"""Load drug-target mappings from DGIdb into DC database.

Usage:
    python scripts_dc/load_drug_targets.py [--db-path data/negbiodb_dc.db] \
        [--data-dir data/dc/drug_targets]
"""

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from negbiodb_dc.dc_db import DEFAULT_DC_DB_PATH, run_dc_migrations
from negbiodb_dc.etl_drug_targets import run_drug_targets_etl


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Load drug-target mappings into DC database"
    )
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DC_DB_PATH,
        help="Path to DC database",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=_PROJECT_ROOT / "data" / "dc" / "drug_targets",
        help="Directory containing DGIdb files",
    )
    args = parser.parse_args()

    run_dc_migrations(args.db_path)

    stats = run_drug_targets_etl(args.db_path, args.data_dir)
    print(f"Drug-target ETL complete: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
