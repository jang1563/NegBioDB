#!/usr/bin/env python3
"""Export DC domain ML dataset to parquet with all splits.

Usage:
    PYTHONPATH=src python scripts_dc/export_dc_ml_dataset.py
    PYTHONPATH=src python scripts_dc/export_dc_ml_dataset.py --db data/negbiodb_dc.db --seed 42
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
    parser = argparse.ArgumentParser(description="Export DC ML dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_dc.db")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "exports" / "dc_ml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.db.exists():
        logger.error("Database not found: %s", args.db)
        return 1

    from negbiodb_dc.dc_db import get_connection
    from negbiodb_dc.export import export_dc_dataset, generate_all_splits

    conn = get_connection(args.db)
    try:
        logger.info("Generating all 6 splits with seed=%d ...", args.seed)
        splits = generate_all_splits(conn, seed=args.seed)
        for name, sid in splits.items():
            logger.info("  %s → split_id=%d", name, sid)

        output_path = args.output_dir / "negbiodb_dc_pairs.parquet"
        n = export_dc_dataset(conn, output_path)
        logger.info("Exported %d pairs to %s", n, output_path)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
