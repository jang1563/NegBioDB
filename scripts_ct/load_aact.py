"""Load AACT data into NegBioDB-CT.

Usage:
    python scripts_ct/load_aact.py [--db-path PATH] [--data-dir PATH]

Prerequisites:
    - Database created: migrations applied automatically
    - AACT data downloaded: python scripts_ct/download_aact.py
"""

import argparse
import logging
from pathlib import Path

from negbiodb_ct.ct_db import DEFAULT_CT_DB_PATH
from negbiodb_ct.etl_aact import run_aact_etl


def main():
    parser = argparse.ArgumentParser(description="Load AACT into NegBioDB-CT")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_CT_DB_PATH,
                        help="Path to CT SQLite database")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing AACT .txt files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("=== AACT ETL ===")
    stats = run_aact_etl(args.db_path, data_dir=args.data_dir)

    print(f"\n=== AACT ETL Summary ===")
    print(f"Studies read:             {stats['studies_read']}")
    print(f"Interventions:            {stats['interventions_inserted']}")
    print(f"Conditions:               {stats['conditions_inserted']}")
    print(f"Trials:                   {stats['trials_inserted']}")
    print(f"Trial-Intervention links: {stats['trial_interventions_linked']}")
    print(f"Trial-Condition links:    {stats['trial_conditions_linked']}")
    print(f"Publications linked:      {stats['publications_linked']}")
    print("\nAACT ETL complete.")


if __name__ == "__main__":
    main()
