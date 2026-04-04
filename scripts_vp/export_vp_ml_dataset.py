#!/usr/bin/env python3
"""Export VP ML dataset with 6 split strategies.

Generates parquet files with tabular features and split assignments.

Usage:
    PYTHONPATH=src python scripts_vp/export_vp_ml_dataset.py
    PYTHONPATH=src python scripts_vp/export_vp_ml_dataset.py --db data/negbiodb_vp.db --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export VP ML dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_vp.db")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "exports" / "vp_ml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_vp.export import export_vp_dataset, generate_all_splits
    from negbiodb_vp.vp_db import get_connection

    conn = get_connection(args.db)
    try:
        logger.info("Generating all 6 split strategies...")
        generate_all_splits(conn, seed=args.seed)

        logger.info("Exporting to parquet...")
        n_rows = export_vp_dataset(conn, args.output_dir)
        logger.info("Exported %d rows to %s", n_rows, args.output_dir)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
