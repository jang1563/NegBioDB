#!/usr/bin/env python
"""Load PRISM drug sensitivity data into GE database (bridge tables)."""

import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _PROJECT_ROOT / "data" / "depmap_raw"


def main():
    parser = argparse.ArgumentParser(description="Load PRISM data into GE DB")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=str(_DEFAULT_DATA))
    parser.add_argument("--compound-meta", type=str, default=None, help="Repurposing Hub sample sheet")
    parser.add_argument("--primary-file", type=str, default=None)
    parser.add_argument("--secondary-file", type=str, default=None)
    parser.add_argument("--release", type=str, default="PRISM_24Q2")
    args = parser.parse_args()

    from negbiodb_depmap.etl_prism import load_prism_compounds, load_prism_sensitivity

    db_path = Path(args.db_path) if args.db_path else _PROJECT_ROOT / "data" / "negbiodb_depmap.db"

    # Load compounds
    if args.compound_meta:
        stats = load_prism_compounds(db_path, Path(args.compound_meta))
        print("\nPRISM compound loading:")
        for k, v in stats.items():
            print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

    # Load sensitivity
    primary = Path(args.primary_file) if args.primary_file else None
    secondary = Path(args.secondary_file) if args.secondary_file else None
    if primary or secondary:
        stats = load_prism_sensitivity(
            db_path, primary_file=primary, secondary_file=secondary,
            depmap_release=args.release,
        )
        print("\nPRISM sensitivity loading:")
        for k, v in stats.items():
            print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
