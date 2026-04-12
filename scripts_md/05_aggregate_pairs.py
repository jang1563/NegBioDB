#!/usr/bin/env python3
"""Aggregate md_biomarker_results into md_metabolite_disease_pairs.

Run after all ingest/standardize scripts are complete.

Usage:
    python scripts_md/05_aggregate_pairs.py [--db PATH] [--upgrade-gold]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Aggregate MD pairs")
    parser.add_argument("--db", type=str, default=None, help="Path to MD database")
    parser.add_argument("--upgrade-gold", action="store_true",
                        help="Upgrade replicated pairs to gold tier")
    args = parser.parse_args()

    from negbiodb_md.md_db import get_md_connection
    from negbiodb_md.etl_aggregate import aggregate_pairs, upgrade_gold_tier, compute_statistics

    conn = get_md_connection(args.db)

    print("Aggregating pairs...")
    n_pairs = aggregate_pairs(conn)
    print(f"Created {n_pairs} metabolite-disease pairs")

    if args.upgrade_gold:
        n_upgraded = upgrade_gold_tier(conn)
        print(f"Upgraded {n_upgraded} pairs to gold tier")

    stats = compute_statistics(conn)
    print("\nMD Database Statistics:")
    print(json.dumps(stats, indent=2))

    conn.close()


if __name__ == "__main__":
    main()
