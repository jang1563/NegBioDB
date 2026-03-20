#!/usr/bin/env python
"""Load IntAct negative interactions into PPI database (Gold tier)."""

import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Load IntAct negatives into PPI DB"
    )
    parser.add_argument(
        "--db-path", type=str, default=None, help="PPI database path"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing intact_negative.txt",
    )
    parser.add_argument(
        "--filename", type=str, default="intact_negative.txt",
        help="IntAct negative file name (default: intact_negative.txt)",
    )
    args = parser.parse_args()

    from negbiodb_ppi.etl_intact import run_intact_etl

    stats = run_intact_etl(
        db_path=args.db_path,
        data_dir=args.data_dir,
        filename=args.filename,
    )

    print("\nIntAct ETL complete:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
