"""ML dataset export pipeline for NegBioDB."""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

from negbiodb.db import connect, create_database, refresh_all_pairs
from negbiodb.download import load_config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Split helpers
# ------------------------------------------------------------------

def _register_split(
    conn: sqlite3.Connection,
    name: str,
    strategy: str,
    seed: int | None,
    ratios: dict[str, float],
) -> int:
    """Insert a split definition and return its split_id."""
    conn.execute(
        """INSERT OR IGNORE INTO split_definitions
        (split_name, split_strategy, random_seed,
         train_ratio, val_ratio, test_ratio)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (name, strategy, seed,
         ratios["train"], ratios["val"], ratios["test"]),
    )
    row = conn.execute(
        "SELECT split_id FROM split_definitions WHERE split_name = ?",
        (name,),
    ).fetchone()
    return int(row[0])


def _assign_folds_by_group(
    conn: sqlite3.Connection,
    split_id: int,
    group_col: str,
    seed: int,
    ratios: dict[str, float],
) -> dict[str, int]:
    """Assign folds by grouping on a column (cold-compound or cold-target).

    All pairs sharing the same group_col value get the same fold.
    Returns dict with fold counts.
    """
    groups = [
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT {group_col} FROM compound_target_pairs"
            f" ORDER BY {group_col}"
        ).fetchall()
    ]

    rng = np.random.RandomState(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    group_to_fold: dict[int, str] = {}
    for i, gid in enumerate(groups):
        if i < n_train:
            group_to_fold[gid] = "train"
        elif i < n_train + n_val:
            group_to_fold[gid] = "val"
        else:
            group_to_fold[gid] = "test"

    # Write via temp table + JOIN for performance
    conn.execute(
        f"CREATE TEMP TABLE _group_folds ({group_col} INTEGER PRIMARY KEY, fold TEXT)"
    )
    conn.executemany(
        f"INSERT INTO _group_folds ({group_col}, fold) VALUES (?, ?)",
        group_to_fold.items(),
    )
    conn.execute(
        f"""INSERT INTO split_assignments (pair_id, split_id, fold)
        SELECT ctp.pair_id, ?, gf.fold
        FROM compound_target_pairs ctp
        JOIN _group_folds gf ON ctp.{group_col} = gf.{group_col}""",
        (split_id,),
    )
    conn.execute("DROP TABLE _group_folds")

    counts: dict[str, int] = {}
    for fold, cnt in conn.execute(
        "SELECT fold, COUNT(*) FROM split_assignments WHERE split_id = ? GROUP BY fold",
        (split_id,),
    ).fetchall():
        counts[fold] = cnt

    return counts


# ------------------------------------------------------------------
# Must-have splits
# ------------------------------------------------------------------

BATCH_SIZE = 500_000


def generate_random_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict:
    """Generate a random 70/10/20 split across all pairs."""
    if ratios is None:
        ratios = {"train": 0.7, "val": 0.1, "test": 0.2}

    split_id = _register_split(conn, "random_v1", "random", seed, ratios)

    pair_ids = np.array(
        [r[0] for r in conn.execute(
            "SELECT pair_id FROM compound_target_pairs ORDER BY pair_id"
        ).fetchall()],
        dtype=np.int64,
    )
    n = len(pair_ids)
    logger.info("Random split: %d pairs", n)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    fold_labels = np.empty(n, dtype="U5")
    fold_labels[indices[:n_train]] = "train"
    fold_labels[indices[n_train:n_train + n_val]] = "val"
    fold_labels[indices[n_train + n_val:]] = "test"

    # Batch insert
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = [
            (int(pair_ids[i]), split_id, fold_labels[i])
            for i in range(start, end)
        ]
        conn.executemany(
            "INSERT INTO split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
            batch,
        )
    conn.commit()

    counts = {}
    for fold, cnt in conn.execute(
        "SELECT fold, COUNT(*) FROM split_assignments WHERE split_id = ? GROUP BY fold",
        (split_id,),
    ).fetchall():
        counts[fold] = cnt

    logger.info("Random split done: %s", counts)
    return {"split_id": split_id, "counts": counts}


def generate_cold_compound_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict:
    """Generate cold-compound split: test compounds unseen in train."""
    if ratios is None:
        ratios = {"train": 0.7, "val": 0.1, "test": 0.2}

    split_id = _register_split(
        conn, "cold_compound_v1", "cold_compound", seed, ratios
    )
    counts = _assign_folds_by_group(conn, split_id, "compound_id", seed, ratios)
    conn.commit()
    logger.info("Cold-compound split done: %s", counts)
    return {"split_id": split_id, "counts": counts}


def generate_cold_target_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict:
    """Generate cold-target split: test targets unseen in train."""
    if ratios is None:
        ratios = {"train": 0.7, "val": 0.1, "test": 0.2}

    split_id = _register_split(
        conn, "cold_target_v1", "cold_target", seed, ratios
    )
    counts = _assign_folds_by_group(conn, split_id, "target_id", seed, ratios)
    conn.commit()
    logger.info("Cold-target split done: %s", counts)
    return {"split_id": split_id, "counts": counts}


# ------------------------------------------------------------------
# Should-have splits
# ------------------------------------------------------------------

def generate_temporal_split(
    conn: sqlite3.Connection,
    train_cutoff: int = 2020,
    val_cutoff: int = 2023,
) -> dict:
    """Generate temporal split based on earliest_year.

    Pairs with earliest_year < train_cutoff → train,
    train_cutoff <= earliest_year < val_cutoff → val,
    earliest_year >= val_cutoff → test.
    Pairs with NULL earliest_year → train (conservative).
    """
    ratios = {"train": 0.0, "val": 0.0, "test": 0.0}  # not ratio-based
    split_id = _register_split(
        conn, "temporal_v1", "temporal", None, ratios
    )

    conn.execute(
        """INSERT INTO split_assignments (pair_id, split_id, fold)
        SELECT pair_id, ?,
            CASE
                WHEN earliest_year IS NULL OR earliest_year < ? THEN 'train'
                WHEN earliest_year < ? THEN 'val'
                ELSE 'test'
            END
        FROM compound_target_pairs""",
        (split_id, train_cutoff, val_cutoff),
    )
    conn.commit()

    counts: dict[str, int] = {}
    for fold, cnt in conn.execute(
        "SELECT fold, COUNT(*) FROM split_assignments WHERE split_id = ? GROUP BY fold",
        (split_id,),
    ).fetchall():
        counts[fold] = cnt

    total = sum(counts.values())
    for fold in ("train", "val", "test"):
        pct = counts.get(fold, 0) / total * 100 if total else 0
        logger.info("Temporal %s: %d (%.1f%%)", fold, counts.get(fold, 0), pct)
    if counts.get("test", 0) / total < 0.05 and total > 0:
        logger.warning(
            "Temporal test set is very small (%.1f%%). "
            "Consider adjusting cutoff years.",
            counts.get("test", 0) / total * 100,
        )

    return {"split_id": split_id, "counts": counts}


SCAFFOLD_BATCH = 100_000


def _compute_scaffolds(
    conn: sqlite3.Connection,
) -> dict[str, list[int]]:
    """Compute Murcko scaffolds for all compounds, return scaffold→[compound_ids].

    Uses generic scaffolds (all side chains removed, all atoms→carbon).
    Compounds that fail RDKit parsing get scaffold='NONE'.
    """
    scaffold_to_compounds: dict[str, list[int]] = defaultdict(list)

    rows = conn.execute(
        "SELECT compound_id, canonical_smiles FROM compounds ORDER BY compound_id"
    ).fetchall()

    for compound_id, smiles in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scaffold_to_compounds["NONE"].append(compound_id)
            continue
        try:
            core = GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(core)
            if not scaffold_smi:
                scaffold_smi = "NONE"
        except Exception:
            scaffold_smi = "NONE"
        scaffold_to_compounds[scaffold_smi].append(compound_id)

    logger.info(
        "Scaffold computation: %d compounds → %d unique scaffolds",
        len(rows),
        len(scaffold_to_compounds),
    )
    return dict(scaffold_to_compounds)


def generate_scaffold_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict:
    """Generate scaffold split using Murcko scaffolds.

    Groups compounds by scaffold, then assigns groups to folds using
    a greedy size-based approach (largest scaffolds first → train).
    All pairs for a compound inherit its scaffold's fold.
    """
    if ratios is None:
        ratios = {"train": 0.7, "val": 0.1, "test": 0.2}

    split_id = _register_split(
        conn, "scaffold_v1", "scaffold", seed, ratios
    )

    # Get scaffold → compound_ids mapping
    scaffold_to_compounds = _compute_scaffolds(conn)

    # Sort scaffold groups by size (largest first) for greedy assignment
    # Tie-break by scaffold SMILES for determinism
    sorted_scaffolds = sorted(
        scaffold_to_compounds.items(),
        key=lambda x: (-len(x[1]), x[0]),
    )

    # Count total pairs per compound for size-aware assignment
    compound_pair_counts: dict[int, int] = {}
    for cid, cnt in conn.execute(
        "SELECT compound_id, COUNT(*) FROM compound_target_pairs GROUP BY compound_id"
    ).fetchall():
        compound_pair_counts[cid] = cnt

    total_pairs = sum(compound_pair_counts.values())
    target_train = int(total_pairs * ratios["train"])
    target_val = int(total_pairs * ratios["val"])

    # Greedy assignment: fill train first, then val, then test
    compound_to_fold: dict[int, str] = {}
    current_train = 0
    current_val = 0

    # Shuffle scaffolds with same size for randomness
    rng = np.random.RandomState(seed)

    # Group scaffolds by size, shuffle within each size group
    from itertools import groupby
    size_groups = []
    for size, group in groupby(sorted_scaffolds, key=lambda x: len(x[1])):
        group_list = list(group)
        rng.shuffle(group_list)
        size_groups.extend(group_list)

    for scaffold_smi, compound_ids in size_groups:
        group_pairs = sum(compound_pair_counts.get(c, 0) for c in compound_ids)

        if current_train + group_pairs <= target_train:
            fold = "train"
            current_train += group_pairs
        elif current_val + group_pairs <= target_val:
            fold = "val"
            current_val += group_pairs
        else:
            fold = "test"

        for cid in compound_ids:
            compound_to_fold[cid] = fold

    # Write via temp table
    conn.execute(
        "CREATE TEMP TABLE _scaffold_folds (compound_id INTEGER PRIMARY KEY, fold TEXT)"
    )
    conn.executemany(
        "INSERT INTO _scaffold_folds (compound_id, fold) VALUES (?, ?)",
        compound_to_fold.items(),
    )
    conn.execute(
        """INSERT INTO split_assignments (pair_id, split_id, fold)
        SELECT ctp.pair_id, ?, sf.fold
        FROM compound_target_pairs ctp
        JOIN _scaffold_folds sf ON ctp.compound_id = sf.compound_id""",
        (split_id,),
    )
    conn.execute("DROP TABLE _scaffold_folds")
    conn.commit()

    counts: dict[str, int] = {}
    for fold, cnt in conn.execute(
        "SELECT fold, COUNT(*) FROM split_assignments WHERE split_id = ? GROUP BY fold",
        (split_id,),
    ).fetchall():
        counts[fold] = cnt

    logger.info("Scaffold split done: %s", counts)
    return {"split_id": split_id, "counts": counts}


def generate_degree_balanced_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    n_bins: int = 10,
) -> dict:
    """Generate degree-distribution-balanced (DDB) split.

    Bins pairs by (log compound_degree, log target_degree) and performs
    stratified sampling so each fold preserves the degree distribution.
    Essential for Experiment 4 (degree bias evaluation).
    """
    if ratios is None:
        ratios = {"train": 0.7, "val": 0.1, "test": 0.2}

    split_id = _register_split(
        conn, "degree_balanced_v1", "degree_balanced", seed, ratios
    )

    # Fetch pair_id, compound_degree, target_degree
    rows = conn.execute(
        """SELECT pair_id, COALESCE(compound_degree, 1), COALESCE(target_degree, 1)
        FROM compound_target_pairs ORDER BY pair_id"""
    ).fetchall()

    pair_ids = np.array([r[0] for r in rows], dtype=np.int64)
    c_deg = np.array([r[1] for r in rows], dtype=np.float64)
    t_deg = np.array([r[2] for r in rows], dtype=np.float64)

    # Log-scale binning
    c_log = np.log1p(c_deg)
    t_log = np.log1p(t_deg)

    c_bins = np.minimum(
        (c_log / (c_log.max() + 1e-9) * n_bins).astype(int), n_bins - 1
    )
    t_bins = np.minimum(
        (t_log / (t_log.max() + 1e-9) * n_bins).astype(int), n_bins - 1
    )

    # Combined bin label
    bin_labels = c_bins * n_bins + t_bins

    # Stratified split within each bin
    rng = np.random.RandomState(seed)
    fold_labels = np.empty(len(pair_ids), dtype="U5")

    for bin_id in np.unique(bin_labels):
        mask = bin_labels == bin_id
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        fold_labels[idx[:n_train]] = "train"
        fold_labels[idx[n_train:n_train + n_val]] = "val"
        fold_labels[idx[n_train + n_val:]] = "test"

    # Batch insert
    for start in range(0, len(pair_ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(pair_ids))
        batch = [
            (int(pair_ids[i]), split_id, fold_labels[i])
            for i in range(start, end)
        ]
        conn.executemany(
            "INSERT INTO split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
            batch,
        )
    conn.commit()

    counts: dict[str, int] = {}
    for fold, cnt in conn.execute(
        "SELECT fold, COUNT(*) FROM split_assignments WHERE split_id = ? GROUP BY fold",
        (split_id,),
    ).fetchall():
        counts[fold] = cnt

    logger.info("Degree-balanced split done: %s", counts)
    return {"split_id": split_id, "counts": counts}
