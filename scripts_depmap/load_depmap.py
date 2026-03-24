#!/usr/bin/env python
"""Load DepMap CRISPR data into GE database."""

import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _PROJECT_ROOT / "data" / "depmap_raw"


def main():
    parser = argparse.ArgumentParser(description="Load DepMap CRISPR data into GE DB")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=str(_DEFAULT_DATA))
    parser.add_argument("--release", type=str, default="25Q3")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    from negbiodb_depmap.etl_depmap import load_depmap_crispr

    data_dir = Path(args.data_dir)
    db_path = Path(args.db_path) if args.db_path else None

    stats = load_depmap_crispr(
        db_path=db_path or _PROJECT_ROOT / "data" / "negbiodb_depmap.db",
        gene_effect_file=data_dir / "CRISPRGeneEffect.csv",
        dependency_file=data_dir / "CRISPRGeneDependency.csv",
        model_file=data_dir / "Model.csv",
        essential_file=data_dir / "AchillesCommonEssentialControls.csv",
        nonessential_file=data_dir / "AchillesNonessentialControls.csv",
        depmap_release=args.release,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
    )

    print("\nDepMap CRISPR ETL complete:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
