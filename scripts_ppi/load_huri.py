#!/usr/bin/env python
"""Load HuRI Y2H-derived negative pairs into PPI database (Gold tier)."""

import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Load HuRI Y2H-derived negatives into PPI DB"
    )
    parser.add_argument(
        "--db-path", type=str, default=None, help="PPI database path"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="HuRI data directory"
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Max negative pairs (default: no limit for Gold)",
    )
    parser.add_argument(
        "--ppi-file", type=str, default=None,
        help="HuRI PPI filename (default: HI-union.tsv)",
    )
    parser.add_argument(
        "--orfeome-file", type=str, default=None,
        help="ORFeome gene list filename",
    )
    args = parser.parse_args()

    from negbiodb_ppi.etl_huri import run_huri_etl

    kwargs = {
        "db_path": args.db_path,
        "data_dir": args.data_dir,
        "max_pairs": args.max_pairs,
    }
    if args.ppi_file is not None:
        kwargs["ppi_file"] = args.ppi_file
    if args.orfeome_file is not None:
        kwargs["orfeome_file"] = args.orfeome_file
    stats = run_huri_etl(**kwargs)

    print("\nHuRI ETL complete:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
