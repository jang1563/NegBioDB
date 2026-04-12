#!/usr/bin/env python3
"""Export MD ML dataset to Parquet with split columns.

Usage:
    python scripts_md/06_export_ml.py [--db PATH] [--output-dir PATH] [--seed N]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Export MD ML dataset")
    parser.add_argument("--db", type=str, default=None, help="Path to MD database")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    from negbiodb_md.md_db import get_md_connection
    from negbiodb_md.md_export import export_ml_dataset

    conn = get_md_connection(args.db)

    n_results = conn.execute("SELECT COUNT(*) FROM md_biomarker_results").fetchone()[0]
    n_pairs = conn.execute("SELECT COUNT(*) FROM md_metabolite_disease_pairs").fetchone()[0]
    print(f"MD database: {n_results} results, {n_pairs} pairs")

    if n_pairs == 0:
        print("No pairs found. Run 05_aggregate_pairs.py first.")
        conn.close()
        sys.exit(1)

    out_path = export_ml_dataset(conn, args.output_dir, seed=args.seed)
    print(f"ML dataset exported: {out_path}")
    conn.close()


if __name__ == "__main__":
    main()
