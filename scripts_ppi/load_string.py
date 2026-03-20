#!/usr/bin/env python
"""Load STRING v12.0 zero-score negatives into PPI database (Bronze tier)."""

import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Load STRING zero-score negatives into PPI DB"
    )
    parser.add_argument(
        "--db-path", type=str, default=None, help="PPI database path"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="STRING data directory"
    )
    parser.add_argument(
        "--min-degree", type=int, default=5,
        help="Minimum STRING degree for well-studied proteins (default: 5)",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=500_000,
        help="Maximum negative pairs (default: 500000)",
    )
    args = parser.parse_args()

    from negbiodb_ppi.etl_string import run_string_etl

    stats = run_string_etl(
        db_path=args.db_path,
        data_dir=args.data_dir,
        min_degree=args.min_degree,
        max_pairs=args.max_pairs,
    )

    print("\nSTRING ETL complete:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
