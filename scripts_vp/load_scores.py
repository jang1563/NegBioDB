#!/usr/bin/env python3
"""Load pre-extracted computational scores into VP database.

Usage:
    PYTHONPATH=src python scripts_vp/load_scores.py --scores data/vp/scores/merged_scores.tsv
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Load computational scores into VP database")
    parser.add_argument("--db-path", type=Path, default=_PROJECT_ROOT / "data" / "negbiodb_vp.db")
    parser.add_argument("--scores", type=Path, required=True, help="Merged scores TSV")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from negbiodb_vp.vp_db import get_connection, run_vp_migrations
    run_vp_migrations(args.db_path)

    from negbiodb_vp.etl_scores import annotate_scores
    conn = get_connection(args.db_path)
    try:
        stats = annotate_scores(conn, args.scores, args.batch_size)
        print("\n=== Score Annotation Results ===")
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v:,}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
