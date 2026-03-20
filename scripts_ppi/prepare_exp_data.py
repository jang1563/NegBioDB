#!/usr/bin/env python3
"""Prepare PPI ML benchmark datasets from exported negatives + HuRI positives.

Produces (in exports/ppi/):
  - ppi_m1_balanced.parquet       — 1:1 pos/neg (NegBioDB curated negatives)
  - ppi_m1_realistic.parquet      — 1:10 pos/neg (NegBioDB curated negatives)
  - ppi_m1_uniform_random.parquet — 1:1 pos/neg (Exp 1 control A)
  - ppi_m1_degree_matched.parquet — 1:1 pos/neg (Exp 1 control B)
  - ppi_m1_balanced_ddb.parquet   — M1 balanced + split_degree_balanced (Exp 4)

Prerequisite:
  - exports/ppi/negbiodb_ppi_pairs.parquet  (from export step)
  - data/ppi/huri/HI-union.tsv
  - data/ppi/huri/ensg_to_uniprot.tsv
  - data/negbiodb_ppi.db

Usage:
    PYTHONPATH=src python scripts_ppi/prepare_exp_data.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare PPI ML benchmark datasets")
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "exports" / "ppi",
        help="Directory for input/output parquets",
    )
    parser.add_argument(
        "--db", type=Path, default=ROOT / "data" / "negbiodb_ppi.db",
        help="Path to negbiodb_ppi.db",
    )
    parser.add_argument(
        "--huri-dir", type=Path, default=ROOT / "data" / "ppi" / "huri",
        help="Directory containing HI-union.tsv and ensg_to_uniprot.tsv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-exp1", action="store_true")
    parser.add_argument("--skip-exp4", action="store_true")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    db_path: Path = args.db
    huri_dir: Path = args.huri_dir

    # Verify inputs
    neg_parquet = data_dir / "negbiodb_ppi_pairs.parquet"
    huri_file = huri_dir / "HI-union.tsv"
    ensg_file = huri_dir / "ensg_to_uniprot.tsv"

    for p in [neg_parquet, huri_file, ensg_file, db_path]:
        if not p.exists():
            logger.error("Required file missing: %s", p)
            return 1

    data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load negatives + positives
    # ------------------------------------------------------------------
    from negbiodb_ppi.export import (
        load_huri_positives_df,
        resolve_conflicts,
        build_m1_balanced,
        build_m1_realistic,
        add_degree_balanced_split,
        apply_ppi_m1_splits,
        generate_uniform_random_negatives,
        generate_degree_matched_negatives,
        control_pairs_to_df,
    )

    logger.info("Loading negatives from %s", neg_parquet.name)
    neg_df = pd.read_parquet(neg_parquet)
    logger.info("Negatives: %d rows", len(neg_df))

    # Filter to rows with sequences
    has_seq = neg_df["sequence_1"].notna() & neg_df["sequence_2"].notna()
    neg_df = neg_df[has_seq].reset_index(drop=True)
    logger.info("Negatives with sequences: %d rows", len(neg_df))

    logger.info("Loading HuRI positives...")
    pos_df = load_huri_positives_df(
        ppi_path=huri_file,
        ensg_mapping_path=ensg_file,
        db_path=db_path,
    )
    logger.info("Positives: %d rows", len(pos_df))

    # Conflict resolution
    neg_df, pos_df, n_conflicts = resolve_conflicts(neg_df, pos_df)
    logger.info("After conflict resolution: %d neg, %d pos, %d conflicts removed",
                len(neg_df), len(pos_df), n_conflicts)

    # Positive pair set for exclusion
    positive_pairs = set(zip(pos_df["uniprot_id_1"], pos_df["uniprot_id_2"]))

    # ------------------------------------------------------------------
    # M1 datasets (NegBioDB curated negatives)
    # ------------------------------------------------------------------
    logger.info("Building M1 balanced (1:1)...")
    m1_bal = build_m1_balanced(neg_df, pos_df, seed=args.seed)
    m1_bal_path = data_dir / "ppi_m1_balanced.parquet"
    m1_bal.to_parquet(m1_bal_path, index=False)
    logger.info("Saved %d rows → %s", len(m1_bal), m1_bal_path.name)

    logger.info("Building M1 realistic (1:10)...")
    m1_real = build_m1_realistic(neg_df, pos_df, ratio=10, seed=args.seed)
    m1_real_path = data_dir / "ppi_m1_realistic.parquet"
    m1_real.to_parquet(m1_real_path, index=False)
    logger.info("Saved %d rows → %s", len(m1_real), m1_real_path.name)

    # ------------------------------------------------------------------
    # Exp 1: Control negatives
    # ------------------------------------------------------------------
    if not args.skip_exp1:
        n_pos = len(pos_df)

        logger.info("Generating %d uniform random negatives (Exp 1A)...", n_pos)
        uniform_pairs = generate_uniform_random_negatives(
            db_path, positive_pairs, n_samples=n_pos, seed=args.seed
        )
        uniform_neg = control_pairs_to_df(uniform_pairs, db_path)
        m1_uniform = pd.concat([pos_df, uniform_neg], ignore_index=True)
        m1_uniform = apply_ppi_m1_splits(m1_uniform, seed=args.seed)
        uniform_path = data_dir / "ppi_m1_uniform_random.parquet"
        m1_uniform.to_parquet(uniform_path, index=False)
        logger.info("Saved %d rows → %s", len(m1_uniform), uniform_path.name)

        logger.info("Generating %d degree-matched negatives (Exp 1B)...", n_pos)
        deg_pairs = generate_degree_matched_negatives(
            db_path, positive_pairs, n_samples=n_pos, seed=args.seed
        )
        deg_neg = control_pairs_to_df(deg_pairs, db_path)
        m1_deg = pd.concat([pos_df, deg_neg], ignore_index=True)
        m1_deg = apply_ppi_m1_splits(m1_deg, seed=args.seed)
        deg_path = data_dir / "ppi_m1_degree_matched.parquet"
        m1_deg.to_parquet(deg_path, index=False)
        logger.info("Saved %d rows → %s", len(m1_deg), deg_path.name)
    else:
        logger.info("Skipping Exp 1 (--skip-exp1)")

    # ------------------------------------------------------------------
    # Exp 4: DDB split
    # ------------------------------------------------------------------
    if not args.skip_exp4:
        logger.info("Building M1 balanced + DDB split (Exp 4)...")
        m1_ddb = add_degree_balanced_split(
            pd.read_parquet(m1_bal_path), seed=args.seed
        )
        ddb_path = data_dir / "ppi_m1_balanced_ddb.parquet"
        m1_ddb.to_parquet(ddb_path, index=False)
        logger.info("Saved %d rows → %s", len(m1_ddb), ddb_path.name)
    else:
        logger.info("Skipping Exp 4 (--skip-exp4)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=== Summary ===")
    for f in sorted(data_dir.glob("ppi_m1_*.parquet")):
        df = pd.read_parquet(f, columns=["Y"])
        n_pos = (df["Y"] == 1).sum()
        n_neg = (df["Y"] == 0).sum()
        logger.info("  %s: %d rows (pos=%d, neg=%d)", f.name, len(df), n_pos, n_neg)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
