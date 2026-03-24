#!/usr/bin/env python
"""Load DEMETER2 RNAi data into GE database."""

import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _PROJECT_ROOT / "data" / "depmap_raw"


def main():
    parser = argparse.ArgumentParser(description="Load DEMETER2 RNAi data into GE DB")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=str(_DEFAULT_DATA))
    parser.add_argument("--release", type=str, default="DEMETER2_v6")
    args = parser.parse_args()

    from negbiodb_depmap.etl_rnai import load_demeter2

    data_dir = Path(args.data_dir)
    db_path = Path(args.db_path) if args.db_path else _PROJECT_ROOT / "data" / "negbiodb_depmap.db"

    stats = load_demeter2(
        db_path=db_path,
        rnai_file=data_dir / "D2_combined_gene_dep_scores.csv",
        depmap_release=args.release,
    )

    print("\nDEMETER2 RNAi ETL complete:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
