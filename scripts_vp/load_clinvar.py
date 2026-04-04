#!/usr/bin/env python3
"""Load ClinVar data into VP database.

Runs the full ClinVar ETL: parse variant_summary + submission_summary,
insert genes/variants/diseases/submissions/negative_results, then
refresh aggregated pairs.

Usage:
    PYTHONPATH=src python scripts_vp/load_clinvar.py [--db-path data/negbiodb_vp.db]
                                                     [--data-dir data/vp/clinvar]
                                                     [--batch-size 5000]
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Load ClinVar data into VP database")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=_PROJECT_ROOT / "data" / "negbiodb_vp.db",
        help="VP database path",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "vp" / "clinvar",
        help="ClinVar data directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5000, help="Commit every N inserts"
    )
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # Ensure DB exists with schema
    from negbiodb_vp.vp_db import run_vp_migrations

    run_vp_migrations(args.db_path)
    print(f"Database ready: {args.db_path}")

    # Run ETL
    from negbiodb_vp.etl_clinvar import run_clinvar_etl

    stats = run_clinvar_etl(args.db_path, args.data_dir, args.batch_size)

    print("\n=== ClinVar ETL Results ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v:,}")

    # Refresh aggregated pairs
    from negbiodb_vp.vp_db import get_connection, refresh_all_vp_pairs

    conn = get_connection(args.db_path)
    try:
        pair_count = refresh_all_vp_pairs(conn)
        conn.commit()
        print(f"\n  Aggregated pairs: {pair_count:,}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
