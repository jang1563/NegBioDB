#!/usr/bin/env python
"""Export GE ML datasets with splits and pair aggregation."""

import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Export GE ML datasets")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(_PROJECT_ROOT / "exports" / "ge"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from negbiodb_depmap.depmap_db import get_connection, refresh_all_ge_pairs
    from negbiodb_depmap.export import (
        export_ge_negatives,
        generate_cold_both_split,
        generate_cold_cell_line_split,
        generate_cold_gene_split,
        generate_degree_balanced_split,
        generate_random_split,
    )

    db_path = Path(args.db_path) if args.db_path else _PROJECT_ROOT / "data" / "negbiodb_depmap.db"
    output_dir = Path(args.output_dir)

    conn = get_connection(db_path)
    try:
        # Refresh pair aggregation
        print("Refreshing pair aggregation...")
        count = refresh_all_ge_pairs(conn)
        conn.commit()
        print(f"  {count:,} gene-cell_line pairs")

        # Generate splits
        print("\nGenerating splits...")
        for name, func in [
            ("random", generate_random_split),
            ("cold_gene", generate_cold_gene_split),
            ("cold_cell_line", generate_cold_cell_line_split),
            ("cold_both", generate_cold_both_split),
            ("degree_balanced", generate_degree_balanced_split),
        ]:
            result = func(conn, seed=args.seed)
            print(f"  {name}: {result['counts']}")

        # Export
        out_path = output_dir / "negbiodb_ge_pairs.parquet"
        n = export_ge_negatives(conn, out_path)
        print(f"\nExported {n:,} pairs to {out_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
