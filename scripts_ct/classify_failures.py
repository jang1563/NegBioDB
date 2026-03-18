"""Run failure classification pipeline on NegBioDB-CT database.

Three-tier detection:
  Tier 1: Terminated trials + NLP on why_stopped → bronze
  Tier 2: Completed trials + p > 0.05 → silver/gold
  Tier 3: CTO binary failure labels → copper

Prerequisites:
  - AACT data loaded (scripts_ct/load_aact.py)
  - Open Targets + CTO downloads (scripts_ct/download_*.py)

Usage:
    python scripts_ct/classify_failures.py [--db DB_PATH] [--data-dir DIR]
"""

import argparse
import logging
from pathlib import Path

from negbiodb_ct.etl_classify import run_classification_pipeline
from negbiodb_ct.ct_db import DEFAULT_CT_DB_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Run failure classification on CT database"
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

    print("=== Failure Classification Pipeline ===")
    stats = run_classification_pipeline(
        db_path=Path(args.db),
        data_dir=data_dir,
    )

    print("\n=== Results ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
