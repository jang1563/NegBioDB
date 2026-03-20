#!/usr/bin/env python
"""Load hu.MAP 3.0 negative pairs into PPI database (Silver tier)."""

import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Load hu.MAP 3.0 negatives into PPI DB"
    )
    parser.add_argument(
        "--db-path", type=str, default=None, help="PPI database path"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="hu.MAP data directory"
    )
    parser.add_argument(
        "--neg-files", type=str, nargs="+", default=None,
        help="Negative pair filenames (default: from config)",
    )
    args = parser.parse_args()

    from negbiodb.download import load_config
    from negbiodb_ppi.etl_humap import run_humap_etl

    neg_files = args.neg_files
    if neg_files is None:
        cfg = load_config()
        humap_cfg = cfg["ppi_domain"]["downloads"]["humap"]
        neg_files = [humap_cfg["neg_train"], humap_cfg["neg_test"]]

    stats = run_humap_etl(
        db_path=args.db_path,
        data_dir=args.data_dir,
        neg_files=neg_files,
    )

    print("\nhu.MAP ETL complete:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
