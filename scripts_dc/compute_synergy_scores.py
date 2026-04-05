#!/usr/bin/env python3
"""Batch compute synergy scores using R SynergyFinder via rpy2.

Reads DrugComb raw dose-response data and computes ZIP, Bliss, Loewe,
and HSA synergy scores per experiment block.

Designed for HPC execution (see slurm/run_dc_compute_synergy.slurm).

Usage:
    python scripts_dc/compute_synergy_scores.py \
        --csv data/dc/drugcomb/drugcomb_data_v1.4.csv \
        --output data/dc/synergy_scores.parquet \
        [--use-python-fallback] [--limit 100]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from negbiodb_dc.synergy_compute import (
    SynergyScores,
    compute_synergy,
    is_r_available,
)

logger = logging.getLogger(__name__)


def compute_block_synergy(block_df: pd.DataFrame, use_r: bool) -> SynergyScores:
    """Compute synergy scores for a single experiment block.

    Args:
        block_df: DataFrame for one block_id with columns:
            drug_row, drug_col, conc_r, conc_c, inhibition
        use_r: Whether to use R SynergyFinder.

    Returns:
        SynergyScores.
    """
    # Identify column names
    drug_row_col = next(
        (c for c in ("drug_row", "drug_a") if c in block_df.columns), None
    )
    drug_col_col = next(
        (c for c in ("drug_col", "drug_b") if c in block_df.columns), None
    )
    conc_r_col = next(
        (c for c in ("conc_r", "conc_row") if c in block_df.columns), None
    )
    conc_c_col = next(
        (c for c in ("conc_c", "conc_col") if c in block_df.columns), None
    )
    response_col = next(
        (c for c in ("inhibition", "response", "percentgrowth") if c in block_df.columns),
        None,
    )

    if not all([drug_row_col, drug_col_col, conc_r_col, conc_c_col, response_col]):
        return SynergyScores()

    drug_row = str(block_df[drug_row_col].iloc[0])
    drug_col = str(block_df[drug_col_col].iloc[0])

    # Build dose-response matrix
    conc_rows = sorted(block_df[conc_r_col].unique())
    conc_cols = sorted(block_df[conc_c_col].unique())

    if len(conc_rows) < 2 or len(conc_cols) < 2:
        return SynergyScores()

    # Pivot to matrix
    pivot = block_df.pivot_table(
        values=response_col,
        index=conc_r_col,
        columns=conc_c_col,
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=conc_rows, columns=conc_cols)

    if pivot.isnull().any().any():
        pivot = pivot.fillna(pivot.mean().mean())

    response_matrix = pivot.values.tolist()

    return compute_synergy(
        drug_row, drug_col,
        [float(c) for c in conc_rows],
        [float(c) for c in conc_cols],
        response_matrix,
        use_r=use_r,
    )


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Batch compute synergy scores from DrugComb dose-response data"
    )
    parser.add_argument(
        "--csv", type=Path, required=True,
        help="Path to drugcomb_data_v1.4.csv",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output parquet file for synergy scores",
    )
    parser.add_argument(
        "--use-python-fallback", action="store_true",
        help="Use pure Python Bliss fallback instead of R SynergyFinder",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N blocks (for testing)",
    )
    args = parser.parse_args()

    use_r = not args.use_python_fallback
    if use_r and not is_r_available():
        logger.warning("R/SynergyFinder not available, using Python Bliss fallback")
        use_r = False

    logger.info("Computing synergy scores (use_r=%s)", use_r)

    # Read full CSV at once — avoids block-boundary splits across chunks.
    # The 2 GB DrugComb file fits in HPC RAM (32 GB node).
    logger.info("Loading CSV: %s", args.csv)
    df = pd.read_csv(args.csv, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]

    block_col = next((c for c in ("block_id", "blockid") if c in df.columns), None)
    if not block_col:
        logger.error(
            "No block_id column found in CSV. Available columns: %s",
            sorted(df.columns.tolist()),
        )
        return 1

    logger.info("Found %d unique blocks", df[block_col].nunique())

    results = []
    blocks_processed = 0

    for block_id, block_df in df.groupby(block_col):
        if args.limit and blocks_processed >= args.limit:
            break

        try:
            scores = compute_block_synergy(block_df, use_r)
            drug_row_col = next(
                (c for c in ("drug_row", "drug_a") if c in block_df.columns), None
            )
            drug_col_col = next(
                (c for c in ("drug_col", "drug_b") if c in block_df.columns), None
            )
            cl_col = next(
                (c for c in ("cell_line_name", "cell_line") if c in block_df.columns),
                None,
            )

            results.append({
                "block_id": block_id,
                "drug_row": str(block_df[drug_row_col].iloc[0]) if drug_row_col else None,
                "drug_col": str(block_df[drug_col_col].iloc[0]) if drug_col_col else None,
                "cell_line": str(block_df[cl_col].iloc[0]) if cl_col else None,
                "zip_score": scores.zip_score,
                "bliss_score": scores.bliss_score,
                "loewe_score": scores.loewe_score,
                "hsa_score": scores.hsa_score,
            })
            blocks_processed += 1

            if blocks_processed % 1000 == 0:
                logger.info("Processed %d blocks", blocks_processed)
        except Exception as e:
            logger.warning("Block %s failed: %s", block_id, e)

    if not results:
        logger.error("No blocks processed successfully — output not written")
        return 1

    # Save results
    df_out = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    logger.info(
        "Saved %d synergy scores to %s", len(df_out), args.output
    )

    # Print summary
    for col in ("zip_score", "bliss_score", "loewe_score", "hsa_score"):
        non_null = df_out[col].notna().sum()
        if non_null > 0:
            logger.info(
                "%s: mean=%.2f, median=%.2f, non-null=%d/%d",
                col, df_out[col].mean(), df_out[col].median(),
                non_null, len(df_out),
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
