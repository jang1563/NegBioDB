#!/usr/bin/env python3
"""Post-ingest standardization: fill NULL metabolite classes via ClassyFire.

Run after all ingest scripts to enrich metabolite_class columns.

Usage:
    python scripts_md/04_standardize.py [--db PATH]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Enrich metabolite classes via ClassyFire")
    parser.add_argument("--db", type=str, default=None, help="Path to MD database")
    args = parser.parse_args()

    from negbiodb_md.md_db import get_md_connection
    from negbiodb_md.etl_standardize import Standardizer

    conn = get_md_connection(args.db)
    standardizer = Standardizer()

    n_null = conn.execute(
        "SELECT COUNT(*) FROM md_metabolites WHERE metabolite_class IS NULL AND inchikey IS NOT NULL"
    ).fetchone()[0]
    print(f"Metabolites with NULL class + known InChIKey: {n_null}")

    if n_null > 0:
        updated = standardizer.enrich_metabolite_classes(conn)
        print(f"ClassyFire: {updated} metabolites enriched")
    else:
        print("All metabolites already have class assignments")

    conn.close()


if __name__ == "__main__":
    main()
