"""ML dataset export pipeline for NegBioDB GE (Gene Essentiality) domain.

Provides:
  - DB-level split generation (random, cold_gene, cold_cell_line, cold_both)
  - Positive source: essential genes from CRISPR data
  - GE-M1: binary essential vs non-essential
  - GE-M2: 3-way (common essential / selective essential / non-essential)
  - Control negative generation (uniform random, degree-matched)
  - Conflict resolution (essential in CRISPR but non-essential in RNAi → exclude both)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BATCH_SIZE = 500_000

SPLIT_STRATEGIES = ["random", "cold_gene", "cold_cell_line", "cold_both", "degree_balanced"]

_DEFAULT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}


# ------------------------------------------------------------------
# DB-level split helpers
# ------------------------------------------------------------------

def _register_ge_split(
    conn: sqlite3.Connection,
    name: str,
    strategy: str,
    seed: int | None,
    ratios: dict[str, float],
) -> int:
    """Insert or retrieve a GE split definition and return split_id."""
    row = conn.execute(
        "SELECT split_id FROM ge_split_definitions WHERE split_name = ?",
        (name,),
    ).fetchone()

    if row is not None:
        split_id = int(row[0])
        conn.execute(
            "DELETE FROM ge_split_assignments WHERE split_id = ?",
            (split_id,),
        )
        return split_id

    conn.execute(
        """INSERT INTO ge_split_definitions
        (split_name, split_strategy, random_seed,
         train_ratio, val_ratio, test_ratio)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (name, strategy, seed,
         ratios["train"], ratios["val"], ratios["test"]),
    )
    row = conn.execute(
        "SELECT split_id FROM ge_split_definitions WHERE split_name = ?",
        (name,),
    ).fetchone()
    return int(row[0])


def _fold_counts(conn: sqlite3.Connection, split_id: int) -> dict[str, int]:
    """Return {fold: count} for a split."""
    rows = conn.execute(
        "SELECT fold, COUNT(*) FROM ge_split_assignments WHERE split_id = ? GROUP BY fold",
        (split_id,),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


# ------------------------------------------------------------------
# DB-level split generators
# ------------------------------------------------------------------

def generate_random_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict:
    """Generate random 70/10/20 split across all GE pairs."""
    if ratios is None:
        ratios = _DEFAULT_RATIOS

    split_id = _register_ge_split(conn, "random_v1", "random", seed, ratios)

    pair_ids = np.array(
        [r[0] for r in conn.execute(
            "SELECT pair_id FROM gene_cell_pairs ORDER BY pair_id"
        ).fetchall()],
        dtype=np.int64,
    )
    n = len(pair_ids)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    fold_labels = np.empty(n, dtype="U5")
    fold_labels[indices[:n_train]] = "train"
    fold_labels[indices[n_train:n_train + n_val]] = "val"
    fold_labels[indices[n_train + n_val:]] = "test"

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = [
            (int(pair_ids[i]), split_id, fold_labels[i])
            for i in range(start, end)
        ]
        conn.executemany(
            "INSERT INTO ge_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
            batch,
        )
    conn.commit()

    counts = _fold_counts(conn, split_id)
    logger.info("Random split done: %s", counts)
    return {"split_id": split_id, "counts": counts}


def generate_cold_gene_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    gene_ratios: dict[str, float] | None = None,
) -> dict:
    """Generate cold-gene split: test genes unseen in train."""
    if gene_ratios is None:
        gene_ratios = {"train": 0.80, "val": 0.05, "test": 0.15}

    split_id = _register_ge_split(
        conn, "cold_gene_v1", "cold_gene", seed, _DEFAULT_RATIOS
    )

    genes = [r[0] for r in conn.execute(
        "SELECT DISTINCT gene_id FROM gene_cell_pairs ORDER BY gene_id"
    ).fetchall()]
    n_genes = len(genes)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_genes)

    n_train = int(n_genes * gene_ratios["train"])
    n_val = int(n_genes * gene_ratios["val"])

    gene_fold = {}
    for i in perm[:n_train]:
        gene_fold[genes[i]] = "train"
    for i in perm[n_train:n_train + n_val]:
        gene_fold[genes[i]] = "val"
    for i in perm[n_train + n_val:]:
        gene_fold[genes[i]] = "test"

    # Assign pairs: fold = gene's fold
    pairs = conn.execute(
        "SELECT pair_id, gene_id FROM gene_cell_pairs"
    ).fetchall()

    batch = [(int(pid), split_id, gene_fold[gid]) for pid, gid in pairs]
    for start in range(0, len(batch), BATCH_SIZE):
        conn.executemany(
            "INSERT INTO ge_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
            batch[start:start + BATCH_SIZE],
        )
    conn.commit()

    counts = _fold_counts(conn, split_id)
    logger.info("Cold-gene split done: %s", counts)
    return {"split_id": split_id, "counts": counts}


def generate_cold_cell_line_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    cl_ratios: dict[str, float] | None = None,
) -> dict:
    """Generate cold-cell-line split: test cell lines unseen in train."""
    if cl_ratios is None:
        cl_ratios = {"train": 0.80, "val": 0.05, "test": 0.15}

    split_id = _register_ge_split(
        conn, "cold_cell_line_v1", "cold_cell_line", seed, _DEFAULT_RATIOS
    )

    cls = [r[0] for r in conn.execute(
        "SELECT DISTINCT cell_line_id FROM gene_cell_pairs ORDER BY cell_line_id"
    ).fetchall()]
    n_cls = len(cls)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_cls)

    n_train = int(n_cls * cl_ratios["train"])
    n_val = int(n_cls * cl_ratios["val"])

    cl_fold = {}
    for i in perm[:n_train]:
        cl_fold[cls[i]] = "train"
    for i in perm[n_train:n_train + n_val]:
        cl_fold[cls[i]] = "val"
    for i in perm[n_train + n_val:]:
        cl_fold[cls[i]] = "test"

    pairs = conn.execute(
        "SELECT pair_id, cell_line_id FROM gene_cell_pairs"
    ).fetchall()

    batch = [(int(pid), split_id, cl_fold[clid]) for pid, clid in pairs]
    for start in range(0, len(batch), BATCH_SIZE):
        conn.executemany(
            "INSERT INTO ge_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
            batch[start:start + BATCH_SIZE],
        )
    conn.commit()

    counts = _fold_counts(conn, split_id)
    logger.info("Cold-cell-line split done: %s", counts)
    return {"split_id": split_id, "counts": counts}


def generate_cold_both_split(
    conn: sqlite3.Connection,
    seed: int = 42,
) -> dict:
    """Generate cold-both split: neither gene nor cell line in train appears in test.

    Uses Metis-style partitioning:
      - Assign genes to folds, assign cell lines to folds
      - Pair fold = max(gene_fold, cl_fold) where test > val > train
    """
    split_id = _register_ge_split(
        conn, "cold_both_v1", "cold_both", seed, _DEFAULT_RATIOS
    )
    _fold_rank = {"train": 0, "val": 1, "test": 2}
    _rank_fold = {0: "train", 1: "val", 2: "test"}

    rng = np.random.RandomState(seed)

    # Partition genes
    genes = [r[0] for r in conn.execute(
        "SELECT DISTINCT gene_id FROM gene_cell_pairs ORDER BY gene_id"
    ).fetchall()]
    gene_perm = rng.permutation(len(genes))
    n_g_train = int(len(genes) * 0.80)
    n_g_val = int(len(genes) * 0.05)

    gene_rank = {}
    for i in gene_perm[:n_g_train]:
        gene_rank[genes[i]] = 0
    for i in gene_perm[n_g_train:n_g_train + n_g_val]:
        gene_rank[genes[i]] = 1
    for i in gene_perm[n_g_train + n_g_val:]:
        gene_rank[genes[i]] = 2

    # Partition cell lines
    cls = [r[0] for r in conn.execute(
        "SELECT DISTINCT cell_line_id FROM gene_cell_pairs ORDER BY cell_line_id"
    ).fetchall()]
    cl_perm = rng.permutation(len(cls))
    n_c_train = int(len(cls) * 0.80)
    n_c_val = int(len(cls) * 0.05)

    cl_rank = {}
    for i in cl_perm[:n_c_train]:
        cl_rank[cls[i]] = 0
    for i in cl_perm[n_c_train:n_c_train + n_c_val]:
        cl_rank[cls[i]] = 1
    for i in cl_perm[n_c_train + n_c_val:]:
        cl_rank[cls[i]] = 2

    # Assign pairs
    pairs = conn.execute(
        "SELECT pair_id, gene_id, cell_line_id FROM gene_cell_pairs"
    ).fetchall()

    batch = [
        (int(pid), split_id, _rank_fold[max(gene_rank[gid], cl_rank[clid])])
        for pid, gid, clid in pairs
    ]
    for start in range(0, len(batch), BATCH_SIZE):
        conn.executemany(
            "INSERT INTO ge_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
            batch[start:start + BATCH_SIZE],
        )
    conn.commit()

    counts = _fold_counts(conn, split_id)
    logger.info("Cold-both split done: %s", counts)
    return {"split_id": split_id, "counts": counts}


def generate_degree_balanced_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict:
    """Generate degree-balanced split: stratify by gene_degree bins."""
    if ratios is None:
        ratios = _DEFAULT_RATIOS

    split_id = _register_ge_split(
        conn, "degree_balanced_v1", "degree_balanced", seed, ratios
    )

    pairs = conn.execute(
        "SELECT pair_id, gene_degree FROM gene_cell_pairs ORDER BY pair_id"
    ).fetchall()

    pair_ids = np.array([r[0] for r in pairs], dtype=np.int64)
    degrees = np.array([r[1] or 0 for r in pairs], dtype=np.int64)

    # Bin degrees into quantiles
    n_bins = min(10, len(np.unique(degrees)))
    try:
        bins = pd.qcut(degrees, n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = np.zeros(len(degrees), dtype=int)

    rng = np.random.RandomState(seed)
    fold_labels = np.empty(len(pair_ids), dtype="U5")

    for b in np.unique(bins):
        mask = bins == b
        idx = np.where(mask)[0]
        perm = rng.permutation(len(idx))

        n_train = int(len(idx) * ratios["train"])
        n_val = int(len(idx) * ratios["val"])

        for j in perm[:n_train]:
            fold_labels[idx[j]] = "train"
        for j in perm[n_train:n_train + n_val]:
            fold_labels[idx[j]] = "val"
        for j in perm[n_train + n_val:]:
            fold_labels[idx[j]] = "test"

    batch = [
        (int(pair_ids[i]), split_id, fold_labels[i])
        for i in range(len(pair_ids))
    ]
    for start in range(0, len(batch), BATCH_SIZE):
        conn.executemany(
            "INSERT INTO ge_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
            batch[start:start + BATCH_SIZE],
        )
    conn.commit()

    counts = _fold_counts(conn, split_id)
    logger.info("Degree-balanced split done: %s", counts)
    return {"split_id": split_id, "counts": counts}


# ------------------------------------------------------------------
# Negative export
# ------------------------------------------------------------------

def export_ge_negatives(
    conn: sqlite3.Connection,
    output_path: Path,
    min_confidence: str = "bronze",
) -> int:
    """Export gene_cell_pairs to Parquet with split assignments.

    Args:
        conn: SQLite connection to GE database.
        output_path: Path for output Parquet file.
        min_confidence: Minimum confidence tier to include.

    Returns:
        Number of pairs exported.
    """
    tier_filter = {
        "gold": "('gold')",
        "silver": "('gold', 'silver')",
        "bronze": "('gold', 'silver', 'bronze')",
    }
    tier_sql = tier_filter.get(min_confidence, "('gold', 'silver', 'bronze')")

    query = f"""
    SELECT
        p.pair_id, p.gene_id, p.cell_line_id,
        g.entrez_id, g.gene_symbol, g.is_common_essential, g.is_reference_nonessential,
        c.model_id, c.ccle_name, c.lineage, c.primary_disease,
        p.num_screens, p.num_sources, p.best_confidence, p.best_evidence_type,
        p.min_gene_effect, p.max_gene_effect, p.mean_gene_effect,
        p.gene_degree, p.cell_line_degree
    FROM gene_cell_pairs p
    JOIN genes g ON p.gene_id = g.gene_id
    JOIN cell_lines c ON p.cell_line_id = c.cell_line_id
    WHERE p.best_confidence IN {tier_sql}
    ORDER BY p.pair_id
    """

    df = pd.read_sql_query(query, conn)

    # Add split columns
    splits = conn.execute(
        "SELECT split_id, split_name FROM ge_split_definitions"
    ).fetchall()

    for split_id, split_name in splits:
        assignments = pd.read_sql_query(
            "SELECT pair_id, fold FROM ge_split_assignments WHERE split_id = ?",
            conn,
            params=(split_id,),
        )
        col_name = f"split_{split_name}"
        df = df.merge(
            assignments.rename(columns={"fold": col_name}),
            on="pair_id",
            how="left",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Exported %d GE pairs to %s", len(df), output_path)
    return len(df)


# ------------------------------------------------------------------
# Positive source: essential genes
# ------------------------------------------------------------------

def load_essential_positives(
    conn: sqlite3.Connection,
    gene_effect_file: Path,
    dependency_file: Path,
    dep_prob_threshold: float = 0.5,
    gene_effect_threshold: float = -1.0,
) -> pd.DataFrame:
    """Load essential gene-cell_line pairs as positives for ML classification.

    A gene is "essential" in a cell line if dep_prob >= dep_prob_threshold.

    Returns DataFrame with columns:
        gene_id, cell_line_id, entrez_id, gene_symbol, model_id,
        gene_effect_score, dependency_probability, essentiality_type
    """
    from negbiodb_depmap.etl_depmap import parse_gene_column

    dep_df = pd.read_csv(dependency_file, index_col=0)
    ge_df = pd.read_csv(gene_effect_file, index_col=0)

    # Build lookups
    gene_lookup = {
        row[1]: (row[0], row[2])
        for row in conn.execute(
            "SELECT gene_id, entrez_id, gene_symbol FROM genes WHERE entrez_id IS NOT NULL"
        ).fetchall()
    }
    cl_lookup = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT model_id, cell_line_id FROM cell_lines"
        ).fetchall()
    }

    # Common essential gene set
    common_essential = {
        row[0]
        for row in conn.execute(
            "SELECT entrez_id FROM genes WHERE is_common_essential = 1"
        ).fetchall()
    }

    records = []
    for col_name in dep_df.columns:
        parsed = parse_gene_column(col_name)
        if parsed is None:
            continue
        symbol, entrez_id = parsed
        if entrez_id not in gene_lookup:
            continue
        gene_id, db_symbol = gene_lookup[entrez_id]

        for model_id in dep_df.index:
            model_id_str = str(model_id).strip()
            cl_id = cl_lookup.get(model_id_str)
            if cl_id is None:
                continue

            dp = dep_df.at[model_id, col_name]
            if pd.isna(dp) or dp < dep_prob_threshold:
                continue

            ge = ge_df.at[model_id, col_name] if col_name in ge_df.columns else None
            if ge is not None and pd.isna(ge):
                ge = None

            # Classify essentiality type
            if entrez_id in common_essential and dp >= dep_prob_threshold:
                ess_type = "common_essential"
            elif ge is not None and ge < gene_effect_threshold:
                ess_type = "selective_essential"
            else:
                ess_type = "selective_essential"

            records.append({
                "gene_id": gene_id,
                "cell_line_id": cl_id,
                "entrez_id": entrez_id,
                "gene_symbol": db_symbol,
                "model_id": model_id_str,
                "gene_effect_score": float(ge) if ge is not None else None,
                "dependency_probability": float(dp),
                "essentiality_type": ess_type,
            })

    df = pd.DataFrame(records)
    logger.info(
        "Loaded %d essential positives (%d common, %d selective)",
        len(df),
        (df["essentiality_type"] == "common_essential").sum() if len(df) > 0 else 0,
        (df["essentiality_type"] == "selective_essential").sum() if len(df) > 0 else 0,
    )
    return df


# ------------------------------------------------------------------
# GE-M1: Binary classification (essential vs non-essential)
# ------------------------------------------------------------------

def build_ge_m1(
    conn: sqlite3.Connection,
    positives_df: pd.DataFrame,
    negatives_df: pd.DataFrame,
    balanced: bool = True,
    ratio: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build GE-M1 binary dataset with conflict resolution.

    Args:
        conn: SQLite connection.
        positives_df: Essential gene-cell_line pairs (from load_essential_positives).
        negatives_df: Non-essential pairs (from export_ge_negatives or gene_cell_pairs).
        balanced: If True, sample negatives to match positives.
        ratio: Negative:positive ratio (1.0 for balanced, 10.0 for realistic).
        seed: Random seed.

    Returns:
        DataFrame with label column (1=essential, 0=non-essential).
    """
    # Conflict resolution: remove pairs that appear in both
    pos_keys = set(zip(positives_df["gene_id"], positives_df["cell_line_id"]))
    neg_keys = set(zip(negatives_df["gene_id"], negatives_df["cell_line_id"]))
    conflicts = pos_keys & neg_keys

    if conflicts:
        logger.warning("Removing %d conflicting pairs from both sides", len(conflicts))
        pos_mask = ~positives_df.apply(
            lambda r: (r["gene_id"], r["cell_line_id"]) in conflicts, axis=1
        )
        neg_mask = ~negatives_df.apply(
            lambda r: (r["gene_id"], r["cell_line_id"]) in conflicts, axis=1
        )
        positives_df = positives_df[pos_mask].copy()
        negatives_df = negatives_df[neg_mask].copy()

    n_pos = len(positives_df)
    n_neg_target = int(n_pos * ratio) if balanced else len(negatives_df)
    n_neg_target = min(n_neg_target, len(negatives_df))

    rng = np.random.RandomState(seed)
    if n_neg_target < len(negatives_df):
        neg_idx = rng.choice(len(negatives_df), n_neg_target, replace=False)
        negatives_df = negatives_df.iloc[neg_idx].copy()

    positives_df = positives_df.copy()
    positives_df["label"] = 1
    negatives_df = negatives_df.copy()
    negatives_df["label"] = 0

    combined = pd.concat([positives_df, negatives_df], ignore_index=True)
    logger.info(
        "GE-M1: %d positive + %d negative = %d total (%d conflicts removed)",
        n_pos, n_neg_target, len(combined), len(conflicts),
    )
    return combined


# ------------------------------------------------------------------
# GE-M2: 3-way classification
# ------------------------------------------------------------------

def build_ge_m2(
    conn: sqlite3.Connection,
    positives_df: pd.DataFrame,
    negatives_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Build GE-M2 three-way dataset.

    Classes:
      0: common_essential (dep_prob >= 0.5 AND gene in common essential set)
      1: selective_essential (dep_prob >= 0.5 AND gene NOT in common essential set)
      2: non_essential (our curated negatives)
    """
    # Conflict resolution
    pos_keys = set(zip(positives_df["gene_id"], positives_df["cell_line_id"]))
    neg_keys = set(zip(negatives_df["gene_id"], negatives_df["cell_line_id"]))
    conflicts = pos_keys & neg_keys

    if conflicts:
        logger.warning("GE-M2: removing %d conflicts", len(conflicts))
        pos_mask = ~positives_df.apply(
            lambda r: (r["gene_id"], r["cell_line_id"]) in conflicts, axis=1
        )
        neg_mask = ~negatives_df.apply(
            lambda r: (r["gene_id"], r["cell_line_id"]) in conflicts, axis=1
        )
        positives_df = positives_df[pos_mask].copy()
        negatives_df = negatives_df[neg_mask].copy()

    # Assign M2 labels
    positives_df = positives_df.copy()
    positives_df["label"] = positives_df["essentiality_type"].map(
        {"common_essential": 0, "selective_essential": 1}
    )

    negatives_df = negatives_df.copy()
    negatives_df["label"] = 2

    combined = pd.concat([positives_df, negatives_df], ignore_index=True)
    logger.info(
        "GE-M2: common=%d, selective=%d, non-essential=%d",
        (combined["label"] == 0).sum(),
        (combined["label"] == 1).sum(),
        (combined["label"] == 2).sum(),
    )
    return combined


# ------------------------------------------------------------------
# Control negatives
# ------------------------------------------------------------------

def generate_uniform_random_negatives(
    conn: sqlite3.Connection,
    n_samples: int,
    seed: int = 42,
    exclude_essential: bool = True,
) -> pd.DataFrame:
    """Generate uniform random gene-cell_line pairs as control negatives.

    Samples pairs NOT in any essential set (dep_prob < 0.5 or unknown).
    These are random pairs, not from the curated negative DB.
    """
    genes = conn.execute(
        "SELECT gene_id, entrez_id, gene_symbol FROM genes"
    ).fetchall()
    cell_lines = conn.execute(
        "SELECT cell_line_id, model_id FROM cell_lines"
    ).fetchall()

    # Exclude known essential pairs (from gene_cell_pairs)
    existing = set()
    for row in conn.execute(
        "SELECT gene_id, cell_line_id FROM gene_cell_pairs"
    ).fetchall():
        existing.add((row[0], row[1]))

    rng = np.random.RandomState(seed)
    records = []
    attempts = 0
    max_attempts = n_samples * 20

    while len(records) < n_samples and attempts < max_attempts:
        g_idx = rng.randint(0, len(genes))
        c_idx = rng.randint(0, len(cell_lines))
        gene_id = genes[g_idx][0]
        cl_id = cell_lines[c_idx][0]

        if (gene_id, cl_id) not in existing:
            records.append({
                "gene_id": gene_id,
                "cell_line_id": cl_id,
                "entrez_id": genes[g_idx][1],
                "gene_symbol": genes[g_idx][2],
                "model_id": cell_lines[c_idx][1],
                "neg_source": "uniform_random",
            })
            existing.add((gene_id, cl_id))

        attempts += 1

    df = pd.DataFrame(records)
    logger.info("Generated %d uniform random control negatives", len(df))
    return df


def generate_degree_matched_negatives(
    conn: sqlite3.Connection,
    target_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate degree-matched control negatives.

    Matches the gene_degree distribution of DB negatives.
    """
    if "gene_degree" not in target_df.columns or len(target_df) == 0:
        logger.warning("No gene_degree column for degree matching, falling back to uniform")
        return generate_uniform_random_negatives(conn, len(target_df), seed)

    genes = conn.execute(
        "SELECT gene_id, entrez_id, gene_symbol FROM genes"
    ).fetchall()
    cell_lines = conn.execute(
        "SELECT cell_line_id, model_id FROM cell_lines"
    ).fetchall()

    existing = set()
    for row in conn.execute(
        "SELECT gene_id, cell_line_id FROM gene_cell_pairs"
    ).fetchall():
        existing.add((row[0], row[1]))

    # Gene degrees from DB
    gene_degrees = {}
    for row in conn.execute(
        "SELECT gene_id, COUNT(DISTINCT cell_line_id) as deg FROM gene_cell_pairs GROUP BY gene_id"
    ).fetchall():
        gene_degrees[row[0]] = row[1]

    # Bin target degrees
    target_degrees = target_df["gene_degree"].fillna(0).astype(int).values
    n_bins = min(10, len(np.unique(target_degrees)))
    try:
        target_bins = pd.qcut(target_degrees, n_bins, labels=False, duplicates="drop")
    except ValueError:
        target_bins = np.zeros(len(target_degrees), dtype=int)

    bin_counts = pd.Series(target_bins).value_counts().to_dict()

    # Sample per bin
    rng = np.random.RandomState(seed)
    records = []

    bin_edges = np.percentile(
        target_degrees, np.linspace(0, 100, n_bins + 1)
    )

    # Group genes by degree bin
    gene_by_bin: dict[int, list] = {b: [] for b in range(n_bins)}
    for g in genes:
        deg = gene_degrees.get(g[0], 0)
        b = np.searchsorted(bin_edges[1:], deg, side="right")
        b = min(b, n_bins - 1)
        gene_by_bin[b].append(g)

    for b, count in bin_counts.items():
        candidates = gene_by_bin.get(b, [])
        if not candidates:
            continue
        for _ in range(count):
            g = candidates[rng.randint(0, len(candidates))]
            c = cell_lines[rng.randint(0, len(cell_lines))]
            if (g[0], c[0]) not in existing:
                records.append({
                    "gene_id": g[0],
                    "cell_line_id": c[0],
                    "entrez_id": g[1],
                    "gene_symbol": g[2],
                    "model_id": c[1],
                    "neg_source": "degree_matched",
                })
                existing.add((g[0], c[0]))

    df = pd.DataFrame(records)
    logger.info("Generated %d degree-matched control negatives", len(df))
    return df
