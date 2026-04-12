#!/usr/bin/env python3
"""Build LLM benchmark datasets (L1-L4) for MD domain.

Usage:
    python scripts_md/07_build_llm_datasets.py [--db PATH] [--output-dir PATH] [--seed N]
    python scripts_md/07_build_llm_datasets.py --levels l1 l2  # build only specific levels
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Build MD LLM datasets")
    parser.add_argument("--db", type=str, default=None, help="Path to MD database")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--levels", nargs="+", choices=["l1", "l2", "l3", "l4"],
                        default=["l1", "l2", "l3", "l4"], help="Levels to build")
    args = parser.parse_args()

    from negbiodb_md.md_db import get_md_connection
    from negbiodb_md.llm_dataset import (
        build_l1_dataset, build_l2_dataset, build_l3_dataset, build_l4_dataset
    )

    conn = get_md_connection(args.db)

    n_neg = conn.execute(
        "SELECT COUNT(*) FROM md_biomarker_results WHERE is_significant = 0"
    ).fetchone()[0]
    n_pos = conn.execute(
        "SELECT COUNT(*) FROM md_biomarker_results WHERE is_significant = 1"
    ).fetchone()[0]
    print(f"MD database: {n_neg} negatives, {n_pos} positives")

    if n_neg == 0:
        print("No negative results found. Run ingest scripts first.")
        conn.close()
        sys.exit(1)

    builders = {
        "l1": build_l1_dataset,
        "l2": build_l2_dataset,
        "l3": build_l3_dataset,
        "l4": build_l4_dataset,
    }

    for level in args.levels:
        print(f"Building MD-{level.upper()}...")
        out = builders[level](conn, seed=args.seed, output_dir=args.output_dir)
        print(f"  → {out}")

    conn.close()


if __name__ == "__main__":
    main()
