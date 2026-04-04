#!/usr/bin/env python3
"""Export VP variant loci to a TSV for HPC score extraction."""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Export VP score targets")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=_PROJECT_ROOT / "data" / "negbiodb_vp.db",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "data" / "vp" / "scores" / "vp_score_targets.tsv",
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from negbiodb_vp.score_extract import export_score_targets
    from negbiodb_vp.vp_db import get_connection, run_vp_migrations

    run_vp_migrations(args.db_path)
    conn = get_connection(args.db_path)
    try:
        stats = export_score_targets(conn, args.output)
        print("\n=== VP Score Target Export Results ===")
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v:,}")
        print(f"  output: {args.output}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
