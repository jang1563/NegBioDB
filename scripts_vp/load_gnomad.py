#!/usr/bin/env python3
"""Load gnomAD data into VP database.

Usage:
    PYTHONPATH=src python scripts_vp/load_gnomad.py --constraint data/vp/gnomad/gnomad.v4.1.constraint_metrics.tsv
    PYTHONPATH=src python scripts_vp/load_gnomad.py --frequencies data/vp/gnomad/variant_frequencies.tsv
    PYTHONPATH=src python scripts_vp/load_gnomad.py --copper data/vp/gnomad/copper_variants.tsv
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Load gnomAD data into VP database")
    parser.add_argument("--db-path", type=Path, default=_PROJECT_ROOT / "data" / "negbiodb_vp.db")
    parser.add_argument("--constraint", type=Path, help="Gene constraint metrics TSV")
    parser.add_argument("--frequencies", type=Path, help="Variant frequency TSV (pre-extracted)")
    parser.add_argument("--copper", type=Path, help="Copper-tier common variant TSV (pre-filtered)")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from negbiodb_vp.vp_db import get_connection, run_vp_migrations
    run_vp_migrations(args.db_path)
    conn = get_connection(args.db_path)

    try:
        if args.constraint:
            from negbiodb_vp.etl_gnomad import load_gene_constraints
            stats = load_gene_constraints(conn, args.constraint)
            print("\n=== Gene Constraint Results ===")
            for k, v in sorted(stats.items()):
                print(f"  {k}: {v:,}")

        if args.frequencies:
            from negbiodb_vp.etl_gnomad import annotate_variant_frequencies
            stats = annotate_variant_frequencies(conn, args.frequencies, args.batch_size)
            print("\n=== Frequency Annotation Results ===")
            for k, v in sorted(stats.items()):
                print(f"  {k}: {v:,}")

        if args.copper:
            from negbiodb_vp.etl_gnomad import generate_copper_tier
            stats = generate_copper_tier(conn, args.copper, batch_size=args.batch_size)
            print("\n=== Copper Tier Results ===")
            for k, v in sorted(stats.items()):
                print(f"  {k}: {v:,}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
