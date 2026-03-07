"""ML dataset export pipeline for NegBioDB."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np

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
