#!/usr/bin/env python3
"""Export VP ML dataset with 6 split strategies.

Generates two parquet files:
  - negbiodb_vp_m1_balanced.parquet  : gold/silver (Y=1) balanced 1:1 with bronze/copper (Y=0)
  - negbiodb_vp_m1_realistic.parquet : all pairs at natural class ratio

Y label:
  Y=1  ->  gold or silver confidence tier (high-evidence confirmed benign)
  Y=0  ->  bronze or copper tier (lower-evidence benign)

Usage:
    PYTHONPATH=src python scripts_vp/export_vp_ml_dataset.py
    PYTHONPATH=src python scripts_vp/export_vp_ml_dataset.py --db data/negbiodb_vp.db --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output_dir / "_tmp_all_pairs.parquet"

    conn = get_connection(args.db)
    try:
        logger.info("Generating all 6 split strategies...")
        generate_all_splits(conn, seed=args.seed)

        logger.info("Exporting base dataset to temp file...")
        n_rows = export_vp_dataset(conn, tmp_path)
        logger.info("Exported %d rows", n_rows)
    finally:
        conn.close()

    logger.info("Loading base dataset and computing Y labels...")
    df = pd.read_parquet(tmp_path)
    tmp_path.unlink()

    # Y=1: gold/silver (high-confidence confirmed benign negative results)
    # Y=0: bronze/copper (lower-confidence)
    df["Y"] = df["confidence_tier"].isin({"gold", "silver"}).astype(int)

    n_pos = int((df["Y"] == 1).sum())
    n_neg = int((df["Y"] == 0).sum())
    logger.info("Y=1 (gold/silver): %d, Y=0 (bronze/copper): %d", n_pos, n_neg)

    rng = np.random.default_rng(args.seed)

    # m1_realistic: natural ratio, all rows
    realistic_path = args.output_dir / "negbiodb_vp_m1_realistic.parquet"
    df.to_parquet(realistic_path, index=False)
    logger.info("Exported %d rows to %s", len(df), realistic_path)

    # m1_balanced: subsample Y=0 to match Y=1 count
    pos_df = df[df["Y"] == 1]
    neg_df = df[df["Y"] == 0]
    neg_sampled = neg_df.sample(n=n_pos, random_state=int(rng.integers(1 << 31)))
    balanced_df = pd.concat([pos_df, neg_sampled], ignore_index=True).sample(
        frac=1, random_state=int(rng.integers(1 << 31))
    )
    balanced_path = args.output_dir / "negbiodb_vp_m1_balanced.parquet"
    balanced_df.to_parquet(balanced_path, index=False)
    logger.info("Exported %d rows to %s", len(balanced_df), balanced_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
