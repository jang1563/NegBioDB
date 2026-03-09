#!/usr/bin/env python3
"""Prepare additional M1 datasets for Exp 1 and Exp 4.

Exp 1 (NegBioDB vs. random negatives):
  - negbiodb_m1_uniform_random.parquet   — untested pairs, uniform sampling
  - negbiodb_m1_degree_matched.parquet   — untested pairs, degree-distribution matched

Exp 4 (Node degree bias — DDB split):
  - negbiodb_m1_balanced_ddb.parquet     — M1 balanced + split_degree_balanced column

Usage:
    uv run python scripts/prepare_exp_data.py
    uv run python scripts/prepare_exp_data.py --data-dir exports/ --db data/negbiodb.db

Prerequisite:
    - exports/negbiodb_m1_balanced.parquet (from export_ml_dataset.py)
    - exports/chembl_positives_pchembl6.parquet
    - exports/negbiodb_dti_pairs.parquet
    - data/negbiodb.db
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

# Root of the project
ROOT = Path(__file__).parent.parent


def prepare_exp1_uniform(
    db_path: Path, positives: pd.DataFrame, output_dir: Path, seed: int = 42
) -> Path:
    """Generate M1 dataset with uniform random negatives (Exp 1 control A)."""
    from negbiodb.export import generate_uniform_random_negatives

    logger.info("Generating uniform random negatives (Exp 1 control A)...")
    result = generate_uniform_random_negatives(
        negbiodb_path=db_path,
        positives=positives,
        n_samples=len(positives),   # 1:1 balanced
        output_dir=output_dir,
        seed=seed,
    )
    out_path = Path(result["path"])
    logger.info(
        "Saved %d rows to %s (pos=%d, neg=%d)",
        result["total"],
        out_path.name,
        result["n_pos"],
        result["n_neg"],
    )
    return out_path


def prepare_exp1_degree_matched(
    db_path: Path, positives: pd.DataFrame, output_dir: Path, seed: int = 42
) -> Path:
    """Generate M1 dataset with degree-matched random negatives (Exp 1 control B)."""
    from negbiodb.export import generate_degree_matched_negatives

    logger.info("Generating degree-matched random negatives (Exp 1 control B)...")
    result = generate_degree_matched_negatives(
        negbiodb_path=db_path,
        positives=positives,
        n_samples=len(positives),   # 1:1 balanced
        output_dir=output_dir,
        seed=seed,
    )
    out_path = Path(result["path"])
    logger.info(
        "Saved %d rows to %s (pos=%d, neg=%d)",
        result["total"],
        out_path.name,
        result["n_pos"],
        result["n_neg"],
    )
    return out_path


def prepare_exp4_ddb(
    m1_path: Path, pairs_path: Path, output_dir: Path
) -> Path:
    """Add split_degree_balanced column to M1 balanced for Exp 4 (DDB split).

    Strategy:
    - Negatives (Y=0): join with negbiodb_dti_pairs.parquet on (inchikey, uniprot_id)
      to get their pre-computed split_degree_balanced assignment.
    - Positives (Y=1): ChEMBL actives have no degree_balanced assignment;
      use split_random as a proxy (same compounds, same random split logic).
    """
    logger.info("Building M1 balanced + DDB split (Exp 4)...")

    m1 = pd.read_parquet(m1_path)
    logger.info("M1 balanced loaded: %d rows", len(m1))

    # Load only the columns we need from the large pairs file
    pairs_cols = ["inchikey", "uniprot_id", "split_degree_balanced"]
    pairs = pd.read_parquet(pairs_path, columns=pairs_cols)
    logger.info("Pairs loaded: %d rows", len(pairs))

    m1_neg = m1[m1["Y"] == 0].copy()
    m1_pos = m1[m1["Y"] == 1].copy()

    # Join negatives with DDB split
    m1_neg = m1_neg.merge(pairs, on=["inchikey", "uniprot_id"], how="left")
    n_null = m1_neg["split_degree_balanced"].isna().sum()
    if n_null > 0:
        logger.warning(
            "%d negatives (%.1f%%) have no DDB split assignment; "
            "falling back to split_random.",
            n_null,
            100 * n_null / len(m1_neg),
        )
        m1_neg["split_degree_balanced"] = m1_neg["split_degree_balanced"].fillna(
            m1_neg["split_random"]
        )

    # Positives: use split_random as DDB split proxy
    m1_pos["split_degree_balanced"] = m1_pos["split_random"]

    m1_ddb = pd.concat([m1_neg, m1_pos], ignore_index=True)
    logger.info(
        "DDB split distribution (negatives): %s",
        m1_neg["split_degree_balanced"].value_counts().to_dict(),
    )

    out_path = output_dir / "negbiodb_m1_balanced_ddb.parquet"
    m1_ddb.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(m1_ddb), out_path.name)
    return out_path


def verify_schema(path: Path, reference_path: Path) -> None:
    """Check that output schema matches the reference M1 file."""
    import pyarrow.parquet as pq
    ref_cols = set(pq.read_schema(reference_path).names)
    out_cols = set(pq.read_schema(path).names)
    extra = out_cols - ref_cols
    missing = ref_cols - out_cols
    if extra or missing:
        logger.warning(
            "%s schema differs from reference — extra: %s, missing: %s",
            path.name,
            extra,
            missing,
        )
    else:
        logger.info("%s schema matches reference M1 ✓", path.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Exp 1 and Exp 4 datasets")
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "exports",
        help="Directory containing M1 parquets (default: exports/)"
    )
    parser.add_argument(
        "--db", type=Path, default=ROOT / "data" / "negbiodb.db",
        help="Path to negbiodb.db SQLite file"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for negative sampling (default: 42)"
    )
    parser.add_argument(
        "--skip-exp1", action="store_true",
        help="Skip Exp 1 random negative generation (slow, ~30 min)"
    )
    parser.add_argument(
        "--skip-exp4", action="store_true",
        help="Skip Exp 4 DDB split preparation"
    )
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    db_path: Path = args.db

    # Verify inputs exist
    m1_balanced = data_dir / "negbiodb_m1_balanced.parquet"
    positives_path = data_dir / "chembl_positives_pchembl6.parquet"
    pairs_path = data_dir / "negbiodb_dti_pairs.parquet"

    for p in [m1_balanced, positives_path, pairs_path]:
        if not p.exists():
            logger.error("Required file missing: %s", p)
            return 1

    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return 1

    logger.info("Using data_dir=%s, db=%s, seed=%d", data_dir, db_path, args.seed)

    positives = pd.read_parquet(positives_path)
    logger.info("Loaded %d ChEMBL positives", len(positives))

    # --- Exp 1: random negatives ----------------------------------------
    if not args.skip_exp1:
        out_uniform = prepare_exp1_uniform(db_path, positives, data_dir, args.seed)
        verify_schema(out_uniform, m1_balanced)

        out_deg = prepare_exp1_degree_matched(db_path, positives, data_dir, args.seed)
        verify_schema(out_deg, m1_balanced)
    else:
        logger.info("Skipping Exp 1 (--skip-exp1)")

    # --- Exp 4: DDB split -----------------------------------------------
    if not args.skip_exp4:
        out_ddb = prepare_exp4_ddb(m1_balanced, pairs_path, data_dir)
    else:
        logger.info("Skipping Exp 4 (--skip-exp4)")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
