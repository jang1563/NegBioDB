"""ML dataset export pipeline for NegBioDB-CT (Clinical Trial Failure domain).

Two ML tasks:
  CT-M1: Drug-Condition Failure Prediction (binary, pair-level)
  CT-M2: Failure Category Classification (7/8-way, result-level, non-copper)

Six split strategies (all in-memory, no DB tables needed):
  random, cold_drug, cold_condition, temporal, scaffold, degree_balanced
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from negbiodb_ct.ct_db import DEFAULT_CT_DB_PATH, get_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CT_SPLIT_STRATEGIES = [
    "random",
    "cold_drug",
    "cold_condition",
    "temporal",
    "scaffold",
    "degree_balanced",
]

# research/14 Section 4.2 temporal cutoffs:
# val = 2018-2019 (pre-COVID), test = 2020+ (includes COVID spike for Exp CT-3)
CT_TEMPORAL_TRAIN_CUTOFF = 2018  # exclusive upper: <=2017 → train
CT_TEMPORAL_VAL_CUTOFF = 2020  # exclusive upper: 2018-2019 → val, 2020+ → test

_DEFAULT_RATIOS: dict[str, float] = {"train": 0.7, "val": 0.1, "test": 0.2}

CATEGORY_TO_INT: dict[str, int] = {
    "efficacy": 0,
    "enrollment": 1,
    "other": 2,
    "strategic": 3,
    "safety": 4,
    "design": 5,
    "regulatory": 6,
    "pharmacokinetic": 7,
}

# Confidence tier ordering for min_confidence filtering
_TIER_RANK = {"gold": 1, "silver": 2, "bronze": 3, "copper": 4}

# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------


def load_ct_pairs_df(
    db_path: str | Path = DEFAULT_CT_DB_PATH,
    *,
    smiles_only: bool = False,
    min_confidence: str | None = None,
) -> pd.DataFrame:
    """Load intervention_condition_pairs with enrichment from related tables.

    Parameters
    ----------
    db_path : path to CT database
    smiles_only : if True, only return pairs where SMILES is available
    min_confidence : minimum confidence tier filter.
        None → all pairs, "silver" → silver+gold, "gold" → gold only.

    Returns
    -------
    DataFrame with 20 columns (split columns added separately).
    """
    conn = get_connection(db_path)
    try:
        # Pre-compute earliest completion year per (intervention, condition)
        conn.execute("DROP TABLE IF EXISTS _pair_years")
        conn.execute(
            """CREATE TEMP TABLE _pair_years AS
            SELECT tfr.intervention_id, tfr.condition_id,
                   MIN(CAST(SUBSTR(ct.completion_date, 1, 4) AS INTEGER))
                       AS earliest_completion_year
            FROM trial_failure_results tfr
            JOIN clinical_trials ct ON tfr.trial_id = ct.trial_id
            WHERE ct.completion_date IS NOT NULL
              AND CAST(SUBSTR(ct.completion_date, 1, 4) AS INTEGER)
                  BETWEEN 1990 AND 2030
            GROUP BY tfr.intervention_id, tfr.condition_id"""
        )

        # Pre-compute target counts per intervention
        conn.execute("DROP TABLE IF EXISTS _target_counts")
        conn.execute(
            """CREATE TEMP TABLE _target_counts AS
            SELECT intervention_id,
                   COUNT(DISTINCT uniprot_accession) AS target_count
            FROM intervention_targets
            GROUP BY intervention_id"""
        )

        # Build WHERE clause
        where_parts: list[str] = []
        if smiles_only:
            where_parts.append("i.canonical_smiles IS NOT NULL")
        if min_confidence is not None:
            rank = _TIER_RANK.get(min_confidence)
            if rank is None:
                raise ValueError(f"Unknown confidence tier: {min_confidence}")
            allowed = [t for t, r in _TIER_RANK.items() if r <= rank]
            placeholders = ", ".join(f"'{t}'" for t in allowed)
            where_parts.append(
                f"icp.best_confidence IN ({placeholders})"
            )
        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)

        sql = f"""
        SELECT
            icp.pair_id,
            icp.intervention_id,
            icp.condition_id,
            i.canonical_smiles          AS smiles,
            i.inchikey,
            i.inchikey_connectivity,
            i.chembl_id,
            i.molecular_type,
            i.intervention_type,
            c.condition_name,
            c.mesh_id,
            icp.best_confidence         AS confidence_tier,
            icp.primary_failure_category,
            icp.num_trials,
            icp.num_sources,
            icp.highest_phase_reached,
            icp.intervention_degree,
            icp.condition_degree,
            COALESCE(tc.target_count, 0) AS target_count,
            py.earliest_completion_year
        FROM intervention_condition_pairs icp
        JOIN interventions i ON icp.intervention_id = i.intervention_id
        JOIN conditions c ON icp.condition_id = c.condition_id
        LEFT JOIN _pair_years py
            ON icp.intervention_id = py.intervention_id
            AND icp.condition_id = py.condition_id
        LEFT JOIN _target_counts tc
            ON icp.intervention_id = tc.intervention_id
        {where_clause}
        ORDER BY icp.pair_id
        """

        df = pd.read_sql_query(sql, conn)

        # Cleanup temp tables
        conn.execute("DROP TABLE IF EXISTS _pair_years")
        conn.execute("DROP TABLE IF EXISTS _target_counts")

        logger.info(
            "Loaded %d CT pairs (smiles_only=%s, min_confidence=%s)",
            len(df),
            smiles_only,
            min_confidence,
        )
        return df
    finally:
        conn.close()


def load_ct_m2_data(
    db_path: str | Path = DEFAULT_CT_DB_PATH,
) -> pd.DataFrame:
    """Load trial_failure_results for CT-M2 classification (non-copper).

    Returns DataFrame with result-level data including trial features.
    Adds failure_category_int column.
    """
    conn = get_connection(db_path)
    try:
        sql = """
        SELECT
            tfr.result_id,
            tfr.intervention_id,
            tfr.condition_id,
            tfr.trial_id,
            tfr.failure_category,
            i.canonical_smiles          AS smiles,
            i.inchikey,
            i.inchikey_connectivity,
            i.chembl_id,
            i.molecular_type,
            c.condition_name,
            c.mesh_id,
            ct.trial_phase,
            ct.randomized,
            ct.blinding,
            ct.control_type,
            ct.enrollment_actual,
            ct.sponsor_type,
            tfr.highest_phase_reached,
            tfr.p_value_primary,
            tfr.effect_size,
            tfr.primary_endpoint_met,
            tfr.result_interpretation,
            tfr.confidence_tier,
            icp.intervention_degree,
            icp.condition_degree,
            CAST(SUBSTR(ct.completion_date, 1, 4) AS INTEGER) AS completion_year
        FROM trial_failure_results tfr
        JOIN interventions i ON tfr.intervention_id = i.intervention_id
        JOIN conditions c ON tfr.condition_id = c.condition_id
        LEFT JOIN clinical_trials ct ON tfr.trial_id = ct.trial_id
        LEFT JOIN intervention_condition_pairs icp
            ON tfr.intervention_id = icp.intervention_id
            AND tfr.condition_id = icp.condition_id
        WHERE tfr.confidence_tier != 'copper'
        ORDER BY tfr.result_id
        """
        df = pd.read_sql_query(sql, conn)

        # Add integer category column
        unknown = set(df["failure_category"].dropna().unique()) - set(
            CATEGORY_TO_INT.keys()
        )
        if unknown:
            raise ValueError(
                f"Unknown failure categories not in CATEGORY_TO_INT: {unknown}"
            )
        df["failure_category_int"] = df["failure_category"].map(CATEGORY_TO_INT)

        logger.info("Loaded %d CT-M2 results (non-copper)", len(df))
        return df
    finally:
        conn.close()


def load_cto_success_pairs(
    cto_path: str | Path,
    db_path: str | Path = DEFAULT_CT_DB_PATH,
) -> tuple[pd.DataFrame, set[tuple[int, int]]]:
    """Extract CTO success pairs as CT-M1 positive class.

    Returns
    -------
    (success_df, conflict_pair_keys)
        success_df: DataFrame of clean success pairs (Y=1)
        conflict_pair_keys: set of (intervention_id, condition_id) tuples
            that appear in both success and failure sets
    """
    cto_path = Path(cto_path)
    if not cto_path.exists():
        logger.warning("CTO parquet not found: %s", cto_path)
        empty = pd.DataFrame(
            columns=[
                "intervention_id",
                "condition_id",
                "smiles",
                "inchikey",
                "inchikey_connectivity",
                "chembl_id",
                "molecular_type",
                "intervention_type",
                "condition_name",
                "mesh_id",
            ]
        )
        return empty, set()

    # Step 1: Load CTO success NCT IDs
    cto = pd.read_parquet(cto_path)
    success_ncts = cto.loc[cto["labels"] == 1.0, "nct_id"].tolist()
    logger.info("CTO success trials: %d", len(success_ncts))

    if not success_ncts:
        empty = pd.DataFrame(
            columns=[
                "intervention_id",
                "condition_id",
                "smiles",
                "inchikey",
                "inchikey_connectivity",
                "chembl_id",
                "molecular_type",
                "intervention_type",
                "condition_name",
                "mesh_id",
            ]
        )
        return empty, set()

    conn = get_connection(db_path)
    try:
        # Step 2-3: Match NCT IDs → expand to (intervention, condition) pairs
        placeholders = ", ".join("?" * len(success_ncts))
        sql = f"""
        SELECT DISTINCT
            ti.intervention_id,
            tc.condition_id,
            i.canonical_smiles      AS smiles,
            i.inchikey,
            i.inchikey_connectivity,
            i.chembl_id,
            i.molecular_type,
            i.intervention_type,
            c.condition_name,
            c.mesh_id
        FROM clinical_trials ct
        JOIN trial_interventions ti ON ct.trial_id = ti.trial_id
        JOIN trial_conditions tc ON ct.trial_id = tc.trial_id
        JOIN interventions i ON ti.intervention_id = i.intervention_id
        JOIN conditions c ON tc.condition_id = c.condition_id
        WHERE ct.source_trial_id IN ({placeholders})
          AND ct.source_db = 'clinicaltrials_gov'
        """
        expanded = pd.read_sql_query(sql, conn, params=success_ncts)
        logger.info("CTO expanded to %d (intervention, condition) pairs", len(expanded))

        # Step 4: Find conflict pairs
        failure_keys = set(
            conn.execute(
                "SELECT intervention_id, condition_id "
                "FROM intervention_condition_pairs"
            ).fetchall()
        )
        expanded_keys = set(
            zip(expanded["intervention_id"], expanded["condition_id"])
        )
        conflict_pair_keys = expanded_keys & failure_keys
        logger.info("Conflict pairs (in both success+failure): %d", len(conflict_pair_keys))

        # Step 5: Remove conflicts from success set
        if conflict_pair_keys:
            mask = ~pd.Series(
                list(zip(expanded["intervention_id"], expanded["condition_id"]))
            ).isin(conflict_pair_keys)
            success_df = expanded[mask.values].reset_index(drop=True)
        else:
            success_df = expanded.reset_index(drop=True)

        logger.info("Clean CTO success pairs: %d", len(success_df))
        return success_df, conflict_pair_keys
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Split Functions — all return dict[pair_id_or_result_id, fold_str]
# ---------------------------------------------------------------------------


def _assign_by_entity_groups(
    ids: np.ndarray,
    entity_keys: np.ndarray,
    seed: int,
    ratios: dict[str, float],
) -> dict[int, str]:
    """Generic cold-split: group ids by entity_keys, assign folds to entities.

    All ids sharing the same entity_key get the same fold.
    """
    # Build entity → [ids] mapping
    from collections import defaultdict

    entity_to_ids: dict[Any, list[int]] = defaultdict(list)
    for id_val, ek in zip(ids, entity_keys):
        entity_to_ids[ek].append(id_val)

    unique_entities = sorted(entity_to_ids.keys(), key=str)
    n = len(unique_entities)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["val"])

    fold_map: dict[int, str] = {}
    for i, idx in enumerate(perm):
        entity = unique_entities[idx]
        if i < train_end:
            fold = "train"
        elif i < val_end:
            fold = "val"
        else:
            fold = "test"
        for id_val in entity_to_ids[entity]:
            fold_map[id_val] = fold

    return fold_map


def generate_ct_random_split(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict[int, str]:
    """Random 70/10/20 split across all items."""
    ratios = ratios or _DEFAULT_RATIOS
    ids = df["pair_id"].values if "pair_id" in df.columns else df["result_id"].values
    n = len(ids)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["val"])

    fold_map: dict[int, str] = {}
    for i, idx in enumerate(perm):
        if i < train_end:
            fold_map[int(ids[idx])] = "train"
        elif i < val_end:
            fold_map[int(ids[idx])] = "val"
        else:
            fold_map[int(ids[idx])] = "test"
    return fold_map


def generate_ct_cold_drug_split(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict[int, str]:
    """Cold-drug split: group by inchikey_connectivity or intervention_id."""
    ratios = ratios or _DEFAULT_RATIOS
    id_col = "pair_id" if "pair_id" in df.columns else "result_id"
    ids = df[id_col].values

    # Build entity keys: inchikey_connectivity for SMILES, iid_ prefix for non-SMILES
    has_ik = df["inchikey_connectivity"].notna()
    entity_keys = np.where(
        has_ik,
        df["inchikey_connectivity"].values,
        "iid_" + df["intervention_id"].astype(str).values,
    )
    return _assign_by_entity_groups(ids, entity_keys, seed, ratios)


def generate_ct_cold_condition_split(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict[int, str]:
    """Cold-condition split: group by mesh_id or condition_id."""
    ratios = ratios or _DEFAULT_RATIOS
    id_col = "pair_id" if "pair_id" in df.columns else "result_id"
    ids = df[id_col].values

    has_mesh = df["mesh_id"].notna()
    entity_keys = np.where(
        has_mesh,
        df["mesh_id"].values,
        "cid_" + df["condition_id"].astype(str).values,
    )
    return _assign_by_entity_groups(ids, entity_keys, seed, ratios)


def generate_ct_temporal_split(
    df: pd.DataFrame,
    train_cutoff: int = CT_TEMPORAL_TRAIN_CUTOFF,
    val_cutoff: int = CT_TEMPORAL_VAL_CUTOFF,
) -> dict[int, str]:
    """Temporal split based on earliest_completion_year or completion_year.

    <=2017 → train, 2018-2019 → val, 2020+ → test.
    NULL → train (conservative).
    """
    id_col = "pair_id" if "pair_id" in df.columns else "result_id"
    year_col = (
        "earliest_completion_year"
        if "earliest_completion_year" in df.columns
        else "completion_year"
    )

    ids = df[id_col].values
    years = df[year_col].values

    folds = np.full(len(df), "train", dtype=object)
    not_null = pd.notna(years)
    folds[not_null & (years >= train_cutoff) & (years < val_cutoff)] = "val"
    folds[not_null & (years >= val_cutoff)] = "test"

    return {int(id_val): fold for id_val, fold in zip(ids, folds)}


def generate_ct_scaffold_split(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict[int, str | None]:
    """Scaffold split via Murcko frameworks.

    Non-SMILES pairs → None (NULL). Only SMILES-having pairs participate.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

    ratios = ratios or _DEFAULT_RATIOS
    id_col = "pair_id" if "pair_id" in df.columns else "result_id"

    # Separate SMILES and non-SMILES
    has_smiles = df["smiles"].notna()
    smiles_df = df[has_smiles]
    no_smiles_df = df[~has_smiles]

    # Assign NULL to non-SMILES items
    fold_map: dict[int, str | None] = {
        int(v): None for v in no_smiles_df[id_col].values
    }

    if len(smiles_df) == 0:
        return fold_map

    # Compute scaffolds: group by inchikey_connectivity to avoid duplicates
    ik_col = "inchikey_connectivity"
    ik_to_scaffold: dict[str, str] = {}
    for ik, smi in zip(smiles_df[ik_col], smiles_df["smiles"]):
        if pd.isna(ik) or ik in ik_to_scaffold:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            ik_to_scaffold[ik] = "NONE"
        else:
            try:
                scaf = GetScaffoldForMol(mol)
                ik_to_scaffold[ik] = Chem.MolToSmiles(scaf) if scaf else "NONE"
            except Exception:
                ik_to_scaffold[ik] = "NONE"

    # For rows without inchikey_connectivity, use intervention_id as key
    has_ik = smiles_df[ik_col].notna()
    entities = np.where(
        has_ik,
        smiles_df[ik_col].values,
        "iid_" + smiles_df["intervention_id"].astype(str).values,
    )
    row_to_entity: dict[int, str] = dict(
        zip(smiles_df[id_col].astype(int), entities)
    )

    # Build scaffold → [entity_keys] mapping
    from collections import defaultdict

    scaffold_to_entities: dict[str, set[str]] = defaultdict(set)
    entity_to_ids: dict[str, list[int]] = defaultdict(list)
    for row_id, entity in row_to_entity.items():
        scaffold = ik_to_scaffold.get(entity, "NONE")
        scaffold_to_entities[scaffold].add(entity)
        entity_to_ids[entity].append(row_id)

    # Count pairs per scaffold for greedy assignment
    scaffold_sizes = []
    for scaf, entities in scaffold_to_entities.items():
        n_pairs = sum(len(entity_to_ids[e]) for e in entities)
        scaffold_sizes.append((n_pairs, scaf))
    scaffold_sizes.sort(reverse=True)

    # Group by size, shuffle within same-size groups
    rng = np.random.RandomState(seed)
    sorted_scaffolds = []
    i = 0
    while i < len(scaffold_sizes):
        j = i
        while j < len(scaffold_sizes) and scaffold_sizes[j][0] == scaffold_sizes[i][0]:
            j += 1
        group = [s[1] for s in scaffold_sizes[i:j]]
        rng.shuffle(group)
        sorted_scaffolds.extend(group)
        i = j

    # Greedy fill: train first, then val, then test
    total_smiles = len(smiles_df)
    target_train = int(total_smiles * ratios["train"])
    target_val = target_train + int(total_smiles * ratios["val"])

    running = 0
    for scaf in sorted_scaffolds:
        if running < target_train:
            fold = "train"
        elif running < target_val:
            fold = "val"
        else:
            fold = "test"
        for entity in scaffold_to_entities[scaf]:
            for row_id in entity_to_ids[entity]:
                fold_map[row_id] = fold
                running += 1

    return fold_map


def generate_ct_degree_balanced_split(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    n_bins: int = 10,
) -> dict[int, str]:
    """Degree-balanced split using log-scale binning."""
    ratios = ratios or _DEFAULT_RATIOS
    id_col = "pair_id" if "pair_id" in df.columns else "result_id"

    ids = df[id_col].values
    i_deg = np.maximum(df["intervention_degree"].fillna(1).values, 1).astype(float)
    c_deg = np.maximum(df["condition_degree"].fillna(1).values, 1).astype(float)

    # Log-scale bin edges
    i_bins = np.logspace(
        np.log10(i_deg.min()), np.log10(i_deg.max() + 1), n_bins + 1
    )
    c_bins = np.logspace(
        np.log10(c_deg.min()), np.log10(c_deg.max() + 1), n_bins + 1
    )

    i_bin_idx = np.clip(np.digitize(i_deg, i_bins) - 1, 0, n_bins - 1)
    c_bin_idx = np.clip(np.digitize(c_deg, c_bins) - 1, 0, n_bins - 1)
    bin_keys = i_bin_idx * n_bins + c_bin_idx

    # Stratified split within each bin
    rng = np.random.RandomState(seed)
    fold_map: dict[int, str] = {}

    for bin_val in np.unique(bin_keys):
        mask = bin_keys == bin_val
        bin_ids = ids[mask]
        n = len(bin_ids)
        perm = rng.permutation(n)

        train_end = int(n * ratios["train"])
        val_end = train_end + int(n * ratios["val"])

        for i, idx in enumerate(perm):
            if i < train_end:
                fold_map[int(bin_ids[idx])] = "train"
            elif i < val_end:
                fold_map[int(bin_ids[idx])] = "val"
            else:
                fold_map[int(bin_ids[idx])] = "test"

    return fold_map


# ---------------------------------------------------------------------------
# Split Application Functions
# ---------------------------------------------------------------------------


def apply_all_ct_splits(
    pairs_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply all 6 CT split strategies to pairs_df. Returns a copy."""
    df = pairs_df.copy()

    splits = {
        "split_random": generate_ct_random_split(df, seed),
        "split_cold_drug": generate_ct_cold_drug_split(df, seed),
        "split_cold_condition": generate_ct_cold_condition_split(df, seed),
        "split_temporal": generate_ct_temporal_split(df),
        "split_scaffold": generate_ct_scaffold_split(df, seed),
        "split_degree_balanced": generate_ct_degree_balanced_split(df, seed),
    }

    id_col = "pair_id" if "pair_id" in df.columns else "result_id"
    for col_name, fold_map in splits.items():
        df[col_name] = df[id_col].map(fold_map)

    return df


def apply_ct_m1_splits(
    df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply 3 M1-relevant splits. Returns a copy.

    Uses intervention_id/condition_id for cold grouping on the full merged set.
    """
    result = df.copy()

    # For M1, we use a synthetic row index as ID
    result = result.reset_index(drop=True)
    result["_m1_id"] = result.index

    # Random split
    n = len(result)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    ratios = _DEFAULT_RATIOS
    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["val"])
    random_map: dict[int, str] = {}
    for i, idx in enumerate(perm):
        if i < train_end:
            random_map[idx] = "train"
        elif i < val_end:
            random_map[idx] = "val"
        else:
            random_map[idx] = "test"
    result["split_random"] = result["_m1_id"].map(random_map)

    # Cold drug split
    has_ik = result["inchikey_connectivity"].notna()
    entity_keys = np.where(
        has_ik,
        result["inchikey_connectivity"].values,
        "iid_" + result["intervention_id"].astype(str).values,
    )
    cold_drug_map = _assign_by_entity_groups(
        result["_m1_id"].values, entity_keys, seed + 1, ratios
    )
    result["split_cold_drug"] = result["_m1_id"].map(cold_drug_map)

    # Cold condition split
    has_mesh = result["mesh_id"].notna()
    cond_keys = np.where(
        has_mesh,
        result["mesh_id"].values,
        "cid_" + result["condition_id"].astype(str).values,
    )
    cold_cond_map = _assign_by_entity_groups(
        result["_m1_id"].values, cond_keys, seed + 2, ratios
    )
    result["split_cold_condition"] = result["_m1_id"].map(cold_cond_map)

    result = result.drop(columns=["_m1_id"])
    return result


def apply_ct_m2_splits(
    m2_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply 6 splits to M2 result-level data. Returns a copy.

    Scaffold split: non-SMILES results → NULL.
    """
    df = m2_df.copy()

    splits = {
        "split_random": generate_ct_random_split(df, seed),
        "split_cold_drug": generate_ct_cold_drug_split(df, seed),
        "split_cold_condition": generate_ct_cold_condition_split(df, seed),
        "split_temporal": generate_ct_temporal_split(df),
        "split_scaffold": generate_ct_scaffold_split(df, seed),
        "split_degree_balanced": generate_ct_degree_balanced_split(df, seed),
    }

    for col_name, fold_map in splits.items():
        df[col_name] = df["result_id"].map(fold_map)

    return df


# ---------------------------------------------------------------------------
# CT-M1 Dataset Builder
# ---------------------------------------------------------------------------


def build_ct_m1_dataset(
    pairs_df: pd.DataFrame,
    success_df: pd.DataFrame,
    conflict_keys: set[tuple[int, int]],
    output_dir: str | Path,
    seed: int = 42,
) -> dict:
    """Build CT-M1 binary dataset: failure (Y=0) vs CTO success (Y=1).

    Parameters
    ----------
    pairs_df : silver+gold failure pairs (from load_ct_pairs_df with
        min_confidence='silver')
    success_df : clean CTO success pairs (from load_cto_success_pairs)
    conflict_keys : set of (intervention_id, condition_id) conflict pairs
    output_dir : output directory
    seed : random seed

    Returns dict with keys: balanced, realistic, smiles_only
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Remove conflict pairs from failure side
    if conflict_keys:
        mask = ~pd.Series(
            list(zip(pairs_df["intervention_id"], pairs_df["condition_id"]))
        ).isin(conflict_keys)
        clean_failures = pairs_df[mask.values].copy()
    else:
        clean_failures = pairs_df.copy()
    clean_failures["Y"] = 0
    logger.info("Clean failure pairs (silver+gold, conflict-free): %d", len(clean_failures))

    # Step 2: Add Y=1 to success
    success = success_df.copy()
    success["Y"] = 1

    # Step 3: Merge full set and apply splits
    # pd.concat fills missing columns (e.g. confidence_tier for Y=1 rows) with NaN
    merged = pd.concat(
        [clean_failures, success],
        ignore_index=True,
    )
    merged = apply_ct_m1_splits(merged, seed)

    n_pos = int((merged["Y"] == 1).sum())
    n_neg = int((merged["Y"] == 0).sum())
    logger.info("M1 merged: %d pos + %d neg = %d total", n_pos, n_neg, len(merged))

    results: dict[str, dict] = {}
    rng = np.random.RandomState(seed)

    # Step 4: Balanced variant
    if n_pos > 0 and n_neg > 0:
        n_sample = min(n_pos, n_neg)
        neg_idx = merged[merged["Y"] == 0].index
        pos_idx = merged[merged["Y"] == 1].index
        sampled_neg = rng.choice(neg_idx, size=n_sample, replace=False)
        sampled_pos = rng.choice(pos_idx, size=n_sample, replace=False)
        balanced = merged.loc[np.concatenate([sampled_pos, sampled_neg])].copy()
        balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
        path_b = output_dir / "negbiodb_ct_m1_balanced.parquet"
        balanced.to_parquet(path_b, compression="zstd", index=False)
        results["balanced"] = {
            "path": str(path_b),
            "n_pos": n_sample,
            "n_neg": n_sample,
            "total": 2 * n_sample,
        }
        logger.info("M1 balanced: %d rows → %s", 2 * n_sample, path_b)

    # Step 5: Realistic variant (requires both classes)
    if n_pos > 0 and n_neg > 0:
        path_r = output_dir / "negbiodb_ct_m1_realistic.parquet"
        realistic = merged.sample(frac=1, random_state=seed).reset_index(drop=True)
        realistic.to_parquet(path_r, compression="zstd", index=False)
        results["realistic"] = {
            "path": str(path_r),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "total": len(realistic),
        }
        logger.info("M1 realistic: %d rows → %s", len(realistic), path_r)
    else:
        logger.warning("M1 realistic skipped: n_pos=%d, n_neg=%d", n_pos, n_neg)

    # Step 6: SMILES-only variant
    smiles_merged = merged[merged["smiles"].notna()].copy()
    n_spos = int((smiles_merged["Y"] == 1).sum())
    n_sneg = int((smiles_merged["Y"] == 0).sum())
    if n_spos > 0 and n_sneg > 0:
        n_sample = min(n_spos, n_sneg)
        sneg_idx = smiles_merged[smiles_merged["Y"] == 0].index
        spos_idx = smiles_merged[smiles_merged["Y"] == 1].index
        sampled_sneg = rng.choice(sneg_idx, size=n_sample, replace=False)
        sampled_spos = rng.choice(spos_idx, size=n_sample, replace=False)
        smiles_bal = smiles_merged.loc[
            np.concatenate([sampled_spos, sampled_sneg])
        ].copy()
        smiles_bal = smiles_bal.sample(frac=1, random_state=seed).reset_index(
            drop=True
        )
        path_s = output_dir / "negbiodb_ct_m1_smiles_only.parquet"
        smiles_bal.to_parquet(path_s, compression="zstd", index=False)
        results["smiles_only"] = {
            "path": str(path_s),
            "n_pos": n_sample,
            "n_neg": n_sample,
            "total": 2 * n_sample,
        }
        logger.info("M1 smiles_only: %d rows → %s", 2 * n_sample, path_s)

    return results


# ---------------------------------------------------------------------------
# Export Functions
# ---------------------------------------------------------------------------


def export_ct_failure_dataset(
    db_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> dict:
    """Export all CT failure pairs with 6 split columns.

    Produces:
      - negbiodb_ct_pairs.parquet (full dataset, all tiers, no Y column)
      - negbiodb_ct_splits.csv (lightweight: IDs + split columns)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_df = load_ct_pairs_df(db_path)
    pairs_df = apply_all_ct_splits(pairs_df, seed)

    # Write parquet
    parquet_path = output_dir / "negbiodb_ct_pairs.parquet"
    pairs_df.to_parquet(parquet_path, compression="zstd", index=False)
    logger.info("Exported %d pairs → %s", len(pairs_df), parquet_path)

    # Write lightweight splits CSV (truncated SMILES for quick ID)
    csv_df = pairs_df.copy()
    csv_df["smiles_short"] = csv_df["smiles"].str[:14]
    csv_cols = [
        "pair_id",
        "intervention_id",
        "condition_id",
        "smiles_short",
        "chembl_id",
        "mesh_id",
    ] + [c for c in pairs_df.columns if c.startswith("split_")]
    csv_path = output_dir / "negbiodb_ct_splits.csv"
    csv_df[csv_cols].to_csv(csv_path, index=False)

    return {
        "total_rows": len(pairs_df),
        "parquet_path": str(parquet_path),
        "splits_csv_path": str(csv_path),
    }


def export_ct_m2_dataset(
    m2_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """Export CT-M2 result-level dataset with split columns."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "negbiodb_ct_m2.parquet"
    m2_df.to_parquet(path, compression="zstd", index=False)
    logger.info("Exported %d M2 results → %s", len(m2_df), path)

    return {"total_rows": len(m2_df), "parquet_path": str(path)}


# ---------------------------------------------------------------------------
# Leakage Report
# ---------------------------------------------------------------------------


def generate_ct_leakage_report(
    db_path: str | Path,
    cto_path: str | Path | None = None,
    output_path: str | Path | None = None,
    seed: int = 42,
) -> dict:
    """Generate CT domain integrity and leakage report."""
    report: dict[str, Any] = {}

    # 1. DB summary
    conn = get_connection(db_path)
    try:
        summary = {}
        for table in [
            "clinical_trials",
            "trial_failure_results",
            "interventions",
            "conditions",
            "intervention_condition_pairs",
        ]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            summary[table] = count

        tier_dist = dict(
            conn.execute(
                "SELECT best_confidence, COUNT(*) "
                "FROM intervention_condition_pairs GROUP BY best_confidence"
            ).fetchall()
        )
        summary["tier_distribution"] = tier_dist
        report["db_summary"] = summary
    finally:
        conn.close()

    # 2. Cold split integrity
    pairs_df = load_ct_pairs_df(db_path)
    pairs_with_splits = apply_all_ct_splits(pairs_df, seed)

    cold_integrity: dict[str, dict] = {}
    for split_col, entity_col in [
        ("split_cold_drug", "inchikey_connectivity"),
        ("split_cold_condition", "mesh_id"),
    ]:
        train_entities = set(
            pairs_with_splits.loc[
                pairs_with_splits[split_col] == "train", entity_col
            ].dropna()
        )
        test_entities = set(
            pairs_with_splits.loc[
                pairs_with_splits[split_col] == "test", entity_col
            ].dropna()
        )
        leaks = train_entities & test_entities
        cold_integrity[split_col] = {"leaks": len(leaks)}
    report["cold_split_integrity"] = cold_integrity

    # 3. Split fold counts
    split_counts: dict[str, dict] = {}
    for col in [c for c in pairs_with_splits.columns if c.startswith("split_")]:
        counts = pairs_with_splits[col].value_counts(dropna=False).to_dict()
        split_counts[col] = {str(k): v for k, v in counts.items()}
    report["split_fold_counts"] = split_counts

    # 3b. Tier distribution per fold
    tier_per_fold: dict[str, dict] = {}
    for col in [c for c in pairs_with_splits.columns if c.startswith("split_")]:
        cross = pd.crosstab(
            pairs_with_splits["confidence_tier"],
            pairs_with_splits[col],
        )
        tier_per_fold[col] = cross.to_dict()
    report["tier_distribution_per_fold"] = tier_per_fold

    # 4. SMILES coverage per fold
    smiles_cov: dict[str, dict] = {}
    has_smiles = pairs_with_splits["smiles"].notna()
    for col in [c for c in pairs_with_splits.columns if c.startswith("split_")]:
        cov = {}
        for fold in ["train", "val", "test"]:
            fold_mask = pairs_with_splits[col] == fold
            n_fold = fold_mask.sum()
            n_smiles = (fold_mask & has_smiles).sum()
            cov[fold] = {
                "total": int(n_fold),
                "smiles": int(n_smiles),
                "pct": round(100 * n_smiles / max(n_fold, 1), 1),
            }
        smiles_cov[col] = cov
    report["smiles_coverage_per_fold"] = smiles_cov

    # 5. CTO conflict stats + M1 conflict-free verification
    if cto_path:
        success_df, conflict_keys = load_cto_success_pairs(cto_path, db_path)
        report["cto_conflicts"] = {"n_conflict_pairs": len(conflict_keys)}

        # M1 conflict-free verification: ensure no (intervention_id, condition_id)
        # appears in both Y=0 and Y=1 after conflict removal
        silver_gold = load_ct_pairs_df(db_path, min_confidence="silver")
        if conflict_keys:
            sg_mask = ~pd.Series(
                list(zip(silver_gold["intervention_id"], silver_gold["condition_id"]))
            ).isin(conflict_keys)
            clean_failures = silver_gold[sg_mask.values]
        else:
            clean_failures = silver_gold
        fail_keys = set(
            zip(clean_failures["intervention_id"], clean_failures["condition_id"])
        )
        success_keys = set(
            zip(success_df["intervention_id"], success_df["condition_id"])
        )
        m1_leaks = fail_keys & success_keys
        report["m1_conflict_free"] = {
            "clean_failures": len(clean_failures),
            "clean_success": len(success_df),
            "overlapping_pairs": len(m1_leaks),
            "verified": len(m1_leaks) == 0,
        }

    # 6. M2 failure_category × split cross-table
    try:
        m2_df = load_ct_m2_data(db_path)
        m2_with_splits = apply_ct_m2_splits(m2_df, seed)
        m2_cross: dict[str, dict] = {}
        for split_col in [c for c in m2_with_splits.columns if c.startswith("split_")]:
            cross = pd.crosstab(
                m2_with_splits["failure_category"],
                m2_with_splits[split_col],
            )
            m2_cross[split_col] = cross.to_dict()
        report["m2_category_by_split"] = m2_cross
    except Exception as e:
        report["m2_category_by_split"] = {"error": str(e)}

    # 7. therapeutic_area coverage warning
    conn = get_connection(db_path)
    try:
        ta_count = conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE therapeutic_area IS NOT NULL"
        ).fetchone()[0]
        total_cond = conn.execute("SELECT COUNT(*) FROM conditions").fetchone()[0]
        report["therapeutic_area_coverage"] = {
            "populated": ta_count,
            "total": total_cond,
            "pct": round(100 * ta_count / max(total_cond, 1), 1),
            "warning": "0% coverage — CT-6 should use mesh_id + degree instead"
            if ta_count == 0
            else None,
        }
    finally:
        conn.close()

    # Write JSON if output path given
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Leakage report → %s", output_path)

    return report
