"""Run drug name resolution cascade on NegBioDB-CT database.

4-step cascade:
  Step 1: ChEMBL molecule_synonyms exact match
  Step 2: PubChem REST API name lookup
  Step 3: Fuzzy match (Jaro-Winkler > 0.90)
  Step 4: Manual override CSV

Prerequisites:
  - AACT data loaded (scripts_ct/load_aact.py)
  - ChEMBL SQLite downloaded (data/chembl/)

Usage:
    python scripts_ct/resolve_drugs.py [--db DB_PATH] [--skip-pubchem] [--skip-fuzzy]
"""

import argparse
import logging
from pathlib import Path

from negbiodb_ct.drug_resolver import run_drug_resolution
from negbiodb_ct.ct_db import DEFAULT_CT_DB_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Resolve drug intervention names to ChEMBL IDs"
    )
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_CT_DB_PATH),
        help="Path to CT database (default: data/negbiodb_ct.db)",
    )
    parser.add_argument(
        "--chembl-db", type=str, default=None,
        help="Path to ChEMBL SQLite (default: auto-detect from config)",
    )
    parser.add_argument(
        "--skip-pubchem", action="store_true",
        help="Skip PubChem API step (Step 2)",
    )
    parser.add_argument(
        "--skip-fuzzy", action="store_true",
        help="Skip fuzzy matching step (Step 3)",
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

    chembl_path = Path(args.chembl_db) if args.chembl_db else None

    print("=== Drug Name Resolution ===")
    stats = run_drug_resolution(
        db_path=Path(args.db),
        chembl_db_path=chembl_path,
        skip_pubchem=args.skip_pubchem,
        skip_fuzzy=args.skip_fuzzy,
    )

    print("\n=== Results ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
