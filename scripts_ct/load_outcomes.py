"""Run outcome enrichment on NegBioDB-CT database.

Enriches existing trial_failure_results with quantitative outcomes
from AACT outcome_analyses and Shi & Du 2024 datasets.

Prerequisites:
  - AACT data loaded + failure classification done
  - Shi & Du downloads complete

Usage:
    python scripts_ct/load_outcomes.py [--db DB_PATH] [--data-dir DIR]
"""

import argparse
import logging
from pathlib import Path

from negbiodb_ct.etl_outcomes import run_outcome_enrichment
from negbiodb_ct.ct_db import DEFAULT_CT_DB_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Enrich failure results with quantitative outcomes"
    )
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_CT_DB_PATH),
        help="Path to CT database (default: data/negbiodb_ct.db)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="AACT data directory (default: from config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    data_dir = Path(args.data_dir) if args.data_dir else None

    print("=== Outcome Enrichment Pipeline ===")
    stats = run_outcome_enrichment(
        db_path=Path(args.db),
        data_dir=data_dir,
    )

    print("\n=== Results ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
