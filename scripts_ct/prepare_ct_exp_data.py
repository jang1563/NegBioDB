#!/usr/bin/env python3
"""Prepare Exp CT-1 control datasets: random and degree-matched negatives.

Generates M1 parquets with alternative negative sources (parallels DTI Exp 1):
  - negbiodb_ct_m1_uniform_random.parquet   — random drug-condition pairs
  - negbiodb_ct_m1_degree_matched.parquet   — degree-distribution matched pairs

Usage:
    python scripts_ct/prepare_ct_exp_data.py \\
        --db-path data/negbiodb_ct.db \\
        --cto-parquet data/ct/cto/cto_outcomes.parquet \\
        --output-dir exports/ct/ \\
        --seed 42

Prerequisite:
    - Populated CT database (data/negbiodb_ct.db)
    - CTO parquet (for success pairs)
    - exports/ct/negbiodb_ct_m1_balanced.parquet (reference schema)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


def _load_all_entity_pairs(db_path: Path) -> set[tuple[int, int]]:
    """Load all (intervention_id, condition_id) from intervention_condition_pairs."""
    from negbiodb_ct.ct_db import get_connection

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT intervention_id, condition_id FROM intervention_condition_pairs"
        ).fetchall()
        return set(rows)
    finally:
        conn.close()


def _load_entity_degrees(db_path: Path) -> tuple[dict[int, int], dict[int, int]]:
    """Load intervention and condition degree distributions."""
    from negbiodb_ct.ct_db import get_connection

    conn = get_connection(db_path)
    try:
        i_rows = conn.execute(
            "SELECT DISTINCT intervention_id, intervention_degree "
            "FROM intervention_condition_pairs"
        ).fetchall()
        c_rows = conn.execute(
            "SELECT DISTINCT condition_id, condition_degree "
            "FROM intervention_condition_pairs"
        ).fetchall()
        i_deg = {r[0]: r[1] for r in i_rows}
        c_deg = {r[0]: r[1] for r in c_rows}
        return i_deg, c_deg
    finally:
        conn.close()


def _load_interventions_info(db_path: Path) -> pd.DataFrame:
    """Load intervention metadata for feature joining."""
    from negbiodb_ct.ct_db import get_connection

    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(
            """SELECT intervention_id, canonical_smiles AS smiles,
                      inchikey, inchikey_connectivity, chembl_id,
                      molecular_type, intervention_type
               FROM interventions""",
            conn,
        )
        return df
    finally:
        conn.close()


def _load_conditions_info(db_path: Path) -> pd.DataFrame:
    """Load condition metadata for feature joining."""
    from negbiodb_ct.ct_db import get_connection

    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT condition_id, condition_name, mesh_id FROM conditions",
            conn,
        )
        return df
    finally:
        conn.close()


def generate_uniform_random_negatives(
    db_path: Path,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Sample random (intervention, condition) pairs not in DB as negatives."""
    rng = np.random.RandomState(seed)
    existing = _load_all_entity_pairs(db_path)
    i_deg, c_deg = _load_entity_degrees(db_path)

    all_interventions = list(i_deg.keys())
    all_conditions = list(c_deg.keys())

    logger.info(
        "Sampling %d uniform random negatives from %d interventions × %d conditions",
        n_samples, len(all_interventions), len(all_conditions),
    )

    negatives: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = n_samples * 20
    while len(negatives) < n_samples and attempts < max_attempts:
        i_id = all_interventions[rng.randint(0, len(all_interventions))]
        c_id = all_conditions[rng.randint(0, len(all_conditions))]
        pair = (i_id, c_id)
        if pair not in existing and pair not in seen:
            negatives.append(pair)
            seen.add(pair)
        attempts += 1

    logger.info("Sampled %d uniform random negatives (%d attempts)", len(negatives), attempts)
    return pd.DataFrame(negatives, columns=["intervention_id", "condition_id"])


def generate_degree_matched_negatives(
    db_path: Path,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Sample degree-matched random negatives (weighted by degree product)."""
    rng = np.random.RandomState(seed)
    existing = _load_all_entity_pairs(db_path)
    i_deg, c_deg = _load_entity_degrees(db_path)

    all_interventions = list(i_deg.keys())
    all_conditions = list(c_deg.keys())

    # Weight by degree
    i_weights = np.array([i_deg[i] for i in all_interventions], dtype=np.float64)
    i_weights /= i_weights.sum()
    c_weights = np.array([c_deg[c] for c in all_conditions], dtype=np.float64)
    c_weights /= c_weights.sum()

    logger.info("Sampling %d degree-matched negatives", n_samples)

    negatives: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = n_samples * 20
    while len(negatives) < n_samples and attempts < max_attempts:
        i_idx = rng.choice(len(all_interventions), p=i_weights)
        c_idx = rng.choice(len(all_conditions), p=c_weights)
        i_id = all_interventions[i_idx]
        c_id = all_conditions[c_idx]
        pair = (i_id, c_id)
        if pair not in existing and pair not in seen:
            negatives.append(pair)
            seen.add(pair)
        attempts += 1

    logger.info("Sampled %d degree-matched negatives (%d attempts)", len(negatives), attempts)
    return pd.DataFrame(negatives, columns=["intervention_id", "condition_id"])


def _enrich_and_build_m1(
    success_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    db_path: Path,
    seed: int,
    output_path: Path,
) -> None:
    """Join metadata, add labels, compute splits, and save M1 parquet."""
    from negbiodb_ct.ct_export import (
        generate_ct_cold_condition_split,
        generate_ct_cold_drug_split,
        generate_ct_random_split,
    )

    interventions = _load_interventions_info(db_path)
    conditions = _load_conditions_info(db_path)

    # Enrich negatives with metadata
    neg_enriched = neg_df.merge(interventions, on="intervention_id", how="left")
    neg_enriched = neg_enriched.merge(conditions, on="condition_id", how="left")
    neg_enriched["Y"] = 0

    # Enrich successes
    if "smiles" not in success_df.columns:
        success_enriched = success_df.merge(interventions, on="intervention_id", how="left")
        success_enriched = success_enriched.merge(conditions, on="condition_id", how="left")
    else:
        success_enriched = success_df.copy()
    success_enriched["Y"] = 1

    # Combine, then add degree data for both sides uniformly
    combined = pd.concat([success_enriched, neg_enriched], ignore_index=True)
    i_deg, c_deg = _load_entity_degrees(db_path)
    i_deg_df = pd.DataFrame(list(i_deg.items()), columns=["intervention_id", "intervention_degree"])
    c_deg_df = pd.DataFrame(list(c_deg.items()), columns=["condition_id", "condition_degree"])
    # Drop any pre-existing degree columns to avoid suffixes
    for col in ["intervention_degree", "condition_degree"]:
        if col in combined.columns:
            combined = combined.drop(columns=[col])
    combined = combined.merge(i_deg_df, on="intervention_id", how="left")
    combined = combined.merge(c_deg_df, on="condition_id", how="left")
    combined["pair_id"] = range(len(combined))

    # Compute 3 splits (seed offsets match apply_ct_m1_splits convention)
    for split_name, split_fn, seed_offset in [
        ("random", generate_ct_random_split, 0),
        ("cold_drug", generate_ct_cold_drug_split, 1),
        ("cold_condition", generate_ct_cold_condition_split, 2),
    ]:
        fold_map = split_fn(combined, seed=seed + seed_offset)
        combined[f"split_{split_name}"] = combined["pair_id"].map(fold_map)

    combined.to_parquet(output_path, index=False)
    logger.info(
        "Saved %d rows to %s (pos=%d, neg=%d)",
        len(combined), output_path.name,
        (combined["Y"] == 1).sum(), (combined["Y"] == 0).sum(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare CT Exp CT-1 datasets")
    parser.add_argument("--db-path", type=Path, default=ROOT / "data" / "negbiodb_ct.db")
    parser.add_argument("--cto-parquet", type=Path, default=ROOT / "data" / "ct" / "cto" / "cto_outcomes.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "exports" / "ct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.db_path.exists():
        logger.error("CT database not found: %s", args.db_path)
        return 1

    # Load CTO success pairs
    from negbiodb_ct.ct_export import load_cto_success_pairs

    success_df, conflicts = load_cto_success_pairs(args.cto_parquet, args.db_path)
    logger.info("Success pairs: %d (conflicts removed: %d)", len(success_df), len(conflicts))

    n_neg = len(success_df)  # 1:1 balanced
    if n_neg == 0:
        logger.error("No success pairs found. Check CTO parquet and DB.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Uniform random
    logger.info("=== Generating uniform random negatives ===")
    uniform_neg = generate_uniform_random_negatives(args.db_path, n_neg, args.seed)
    _enrich_and_build_m1(
        success_df, uniform_neg, args.db_path, args.seed,
        args.output_dir / "negbiodb_ct_m1_uniform_random.parquet",
    )

    # Degree-matched
    logger.info("=== Generating degree-matched negatives ===")
    deg_neg = generate_degree_matched_negatives(args.db_path, n_neg, args.seed + 1)
    _enrich_and_build_m1(
        success_df, deg_neg, args.db_path, args.seed,
        args.output_dir / "negbiodb_ct_m1_degree_matched.parquet",
    )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
