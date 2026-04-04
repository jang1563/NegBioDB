#!/usr/bin/env python3
"""Load ClinGen gene-disease validity data into VP database.

Usage:
    PYTHONPATH=src python scripts_vp/load_clingen.py --csv data/vp/clingen/gene_validity.csv
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Load ClinGen data into VP database")
    parser.add_argument("--db-path", type=Path, default=_PROJECT_ROOT / "data" / "negbiodb_vp.db")
    parser.add_argument("--csv", type=Path, required=True, help="ClinGen gene-disease validity CSV")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from negbiodb_vp.vp_db import get_connection, run_vp_migrations
    run_vp_migrations(args.db_path)

    from negbiodb_vp.etl_clingen import load_clingen_validity
    conn = get_connection(args.db_path)
    try:
        stats = load_clingen_validity(conn, args.csv)
        print("\n=== ClinGen Results ===")
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v:,}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
