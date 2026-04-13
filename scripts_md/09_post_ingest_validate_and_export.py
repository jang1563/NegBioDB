#!/usr/bin/env python3
"""Validate MD ingest outputs and export ML/LLM datasets in one step.

Usage:
    python scripts_md/09_post_ingest_validate_and_export.py
    python scripts_md/09_post_ingest_validate_and_export.py --validate-only
    python scripts_md/09_post_ingest_validate_and_export.py --levels l1 l2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _format_counts(rows: list[tuple[object, int]]) -> str:
    if not rows:
        return "(none)"
    return ", ".join(f"{key}={count}" for key, count in rows)


def _count_jsonl(path: Path) -> int:
    with open(path) as fh:
        return sum(1 for line in fh if line.strip())


def _validate_ingest(conn) -> tuple[list[tuple[int, int]], list[tuple[str, int]], int]:
    sig_counts = conn.execute(
        """SELECT is_significant, COUNT(*)
           FROM md_biomarker_results
           GROUP BY is_significant
           ORDER BY is_significant"""
    ).fetchall()
    tier_counts = conn.execute(
        """SELECT tier, COUNT(*)
           FROM md_biomarker_results
           WHERE is_significant = 0
           GROUP BY tier
           ORDER BY CASE tier
               WHEN 'gold' THEN 1
               WHEN 'silver' THEN 2
               WHEN 'bronze' THEN 3
               WHEN 'copper' THEN 4
               ELSE 5
           END"""
    ).fetchall()
    pair_count = conn.execute(
        "SELECT COUNT(*) FROM md_metabolite_disease_pairs"
    ).fetchone()[0]

    sig_map = {int(k): int(v) for k, v in sig_counts}
    if sig_map.get(0, 0) == 0 or sig_map.get(1, 0) == 0:
        raise RuntimeError(
            "MD ingest validation failed: expected both is_significant=0 and "
            "is_significant=1 rows in md_biomarker_results."
        )
    if int(pair_count) == 0:
        raise RuntimeError(
            "MD ingest validation failed: md_metabolite_disease_pairs is empty."
        )
    return sig_counts, tier_counts, int(pair_count)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate MD ingest outputs and export ML/LLM datasets"
    )
    parser.add_argument("--db", type=str, default=None, help="Path to MD database")
    parser.add_argument(
        "--ml-output-dir", type=str, default=None, help="Output directory for MD ML export"
    )
    parser.add_argument(
        "--llm-output-dir",
        type=str,
        default=None,
        help="Output directory for MD LLM JSONL exports",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=["l1", "l2", "l3", "l4"],
        default=["l1", "l2", "l3", "l4"],
        help="LLM levels to build",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run DB sanity checks only; skip exports",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip MD ML parquet export",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip MD LLM JSONL exports",
    )
    args = parser.parse_args()

    from negbiodb_md.md_db import get_md_connection
    from negbiodb_md.md_export import export_ml_dataset
    from negbiodb_md.llm_dataset import (
        build_l1_dataset,
        build_l2_dataset,
        build_l3_dataset,
        build_l4_dataset,
    )

    conn = get_md_connection(args.db)
    try:
        study_count = conn.execute("SELECT COUNT(*) FROM md_studies").fetchone()[0]
        metabolite_count = conn.execute("SELECT COUNT(*) FROM md_metabolites").fetchone()[0]
        disease_count = conn.execute("SELECT COUNT(*) FROM md_diseases").fetchone()[0]
        sig_counts, tier_counts, pair_count = _validate_ingest(conn)

        print("=== MD ingest validation ===")
        print(f"Studies: {study_count}")
        print(f"Metabolites: {metabolite_count}")
        print(f"Diseases: {disease_count}")
        print(f"Results by significance: {_format_counts(sig_counts)}")
        print(f"Negative tiers: {_format_counts(tier_counts)}")
        print(f"Pairs: {pair_count}")

        if args.validate_only:
            print("Validation complete. Exports skipped (--validate-only).")
            return 0

        if not args.skip_ml:
            ml_path = export_ml_dataset(conn, args.ml_output_dir, seed=args.seed)
            print(f"ML export: {ml_path}")

        if not args.skip_llm:
            builders = {
                "l1": build_l1_dataset,
                "l2": build_l2_dataset,
                "l3": build_l3_dataset,
                "l4": build_l4_dataset,
            }
            print("=== Building MD LLM datasets ===")
            for level in args.levels:
                out_path = builders[level](
                    conn,
                    seed=args.seed,
                    output_dir=args.llm_output_dir,
                )
                record_count = _count_jsonl(out_path)
                print(f"{level.upper()}: {out_path} ({record_count} records)")

        print("MD post-ingest workflow complete.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
