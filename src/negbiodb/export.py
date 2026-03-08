"""ML dataset export pipeline for NegBioDB."""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

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
    """Insert or retrieve a split definition and return its split_id.

    If a split with this name already exists, deletes its old assignments
    so the split can be regenerated cleanly.
    """
    row = conn.execute(
        "SELECT split_id FROM split_definitions WHERE split_name = ?",
        (name,),
    ).fetchone()

    if row is not None:
        # Clear old assignments for re-entrancy
        split_id = int(row[0])
        conn.execute(
            "DELETE FROM split_assignments WHERE split_id = ?",
            (split_id,),
        )
        return split_id

    conn.execute(
        """INSERT INTO split_definitions
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
    conn.execute("DROP TABLE IF EXISTS _group_folds")
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
    if total > 0 and counts.get("test", 0) / total < 0.05:
        logger.warning(
            "Temporal test set is very small (%.1f%%). "
            "Consider adjusting cutoff years.",
            counts.get("test", 0) / total * 100,
        )

    return {"split_id": split_id, "counts": counts}


def _compute_scaffolds(
    conn: sqlite3.Connection,
) -> dict[str, list[int]]:
    """Compute Murcko scaffolds for all compounds, return scaffold→[compound_ids].

    Uses Murcko frameworks (ring systems + linkers, heteroatoms preserved).
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
    conn.execute("DROP TABLE IF EXISTS _scaffold_folds")
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


# ------------------------------------------------------------------
# Dataset export
# ------------------------------------------------------------------

EXPORT_CHUNKSIZE = 500_000

# Split strategies in the order they appear as inline columns
SPLIT_STRATEGIES = [
    "random", "cold_compound", "cold_target",
    "temporal", "scaffold", "degree_balanced",
]


def _build_export_query(split_strategies: list[str] | None = None) -> str:
    """Build the pivot SQL for exporting pairs with inline split columns."""
    if split_strategies is None:
        split_strategies = SPLIT_STRATEGIES

    split_cols = []
    join_clauses = []
    for i, strategy in enumerate(split_strategies):
        alias_sa = f"sa{i}"
        alias_sd = f"sd{i}"
        col_name = f"split_{strategy}"
        split_cols.append(f"{alias_sa}.fold AS {col_name}")
        join_clauses.append(
            f"LEFT JOIN split_definitions {alias_sd} "
            f"ON {alias_sd}.split_strategy = '{strategy}' "
            f"LEFT JOIN split_assignments {alias_sa} "
            f"ON ctp.pair_id = {alias_sa}.pair_id "
            f"AND {alias_sa}.split_id = {alias_sd}.split_id"
        )

    join_sql = "\n".join(join_clauses)

    base_cols = """ctp.pair_id,
       c.canonical_smiles AS smiles,
       c.inchikey,
       t.uniprot_accession AS uniprot_id,
       t.amino_acid_sequence AS target_sequence,
       t.gene_symbol,
       0 AS Y,
       ctp.best_confidence AS confidence_tier,
       ctp.best_result_type,
       ctp.num_assays,
       ctp.num_sources,
       ctp.earliest_year,
       ctp.compound_degree,
       ctp.target_degree"""

    if split_cols:
        select_clause = base_cols + ",\n       " + ",\n       ".join(split_cols)
    else:
        select_clause = base_cols

    query = f"""SELECT
       {select_clause}
FROM compound_target_pairs ctp
JOIN compounds c ON ctp.compound_id = c.compound_id
JOIN targets t ON ctp.target_id = t.target_id
{join_sql}
ORDER BY ctp.pair_id"""

    return query


def export_negative_dataset(
    db_path: str | Path,
    output_dir: str | Path,
    split_strategies: list[str] | None = None,
    chunksize: int = EXPORT_CHUNKSIZE,
) -> dict:
    """Export negative DTI pairs as Parquet and lightweight CSV.

    Produces:
      - negbiodb_dti_pairs.parquet (full dataset with sequences)
      - negbiodb_splits.csv (pair_id + smiles + uniprot + split columns only)

    Returns dict with file paths and row count.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "negbiodb_dti_pairs.parquet"
    splits_csv_path = output_dir / "negbiodb_splits.csv"

    if split_strategies is None:
        split_strategies = SPLIT_STRATEGIES

    query = _build_export_query(split_strategies)
    split_cols = [f"split_{s}" for s in split_strategies]

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")

    total_rows = 0
    pq_writer = None

    try:
        for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
            total_rows += len(chunk)

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(
                    str(parquet_path), table.schema, compression="zstd"
                )
            pq_writer.write_table(table)

            logger.info("Exported %d rows so far...", total_rows)

        if pq_writer is not None:
            pq_writer.close()

        # Lightweight splits CSV (no target_sequence)
        if total_rows > 0:
            splits_columns = ["pair_id", "smiles", "inchikey", "uniprot_id"] + split_cols
            # Read back from parquet for splits CSV (avoids re-querying)
            pf = pq.ParquetFile(str(parquet_path))
            first = True
            for batch in pf.iter_batches(
                batch_size=chunksize, columns=splits_columns
            ):
                df = batch.to_pandas()
                df.to_csv(
                    str(splits_csv_path),
                    mode="w" if first else "a",
                    header=first,
                    index=False,
                )
                first = False

    finally:
        conn.close()

    logger.info(
        "Export complete: %d rows → %s (%.1f MB), %s",
        total_rows,
        parquet_path.name,
        parquet_path.stat().st_size / 1e6 if parquet_path.exists() else 0,
        splits_csv_path.name,
    )

    return {
        "total_rows": total_rows,
        "parquet_path": str(parquet_path),
        "splits_csv_path": str(splits_csv_path),
    }


# ------------------------------------------------------------------
# ChEMBL positive extraction + M1 merge
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# DataFrame-level split helpers (for M1 / random negative datasets)
# ------------------------------------------------------------------

_DEFAULT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}


def add_random_split(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add split_random column with deterministic 70/10/20 assignment."""
    if ratios is None:
        ratios = _DEFAULT_RATIOS
    rng = np.random.RandomState(seed)
    n = len(df)
    if n == 0:
        df = df.copy()
        df["split_random"] = pd.Series(dtype=str)
        return df
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.empty(n, dtype=object)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    folds[indices[:n_train]] = "train"
    folds[indices[n_train:n_train + n_val]] = "val"
    folds[indices[n_train + n_val:]] = "test"
    df = df.copy()
    df["split_random"] = folds
    return df


def add_cold_compound_split(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add split_cold_compound column — group by InChIKey connectivity.

    All rows sharing the same InChIKey[:14] get the same fold,
    ensuring no compound leaks between train and test.
    """
    if ratios is None:
        ratios = _DEFAULT_RATIOS
    if len(df) == 0:
        df = df.copy()
        df["split_cold_compound"] = pd.Series(dtype=str)
        return df
    compounds = np.array(df["inchikey"].str[:14].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(compounds)
    n = len(compounds)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    comp_fold = {}
    for i, c in enumerate(compounds):
        if i < n_train:
            comp_fold[c] = "train"
        elif i < n_train + n_val:
            comp_fold[c] = "val"
        else:
            comp_fold[c] = "test"
    df = df.copy()
    df["split_cold_compound"] = df["inchikey"].str[:14].map(comp_fold)
    return df


def add_cold_target_split(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add split_cold_target column — group by UniProt accession.

    All rows sharing the same uniprot_id get the same fold,
    ensuring no target leaks between train and test.
    """
    if ratios is None:
        ratios = _DEFAULT_RATIOS
    if len(df) == 0:
        df = df.copy()
        df["split_cold_target"] = pd.Series(dtype=str)
        return df
    targets = np.array(df["uniprot_id"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(targets)
    n = len(targets)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    tgt_fold = {}
    for i, t in enumerate(targets):
        if i < n_train:
            tgt_fold[t] = "train"
        elif i < n_train + n_val:
            tgt_fold[t] = "val"
        else:
            tgt_fold[t] = "test"
    df = df.copy()
    df["split_cold_target"] = df["uniprot_id"].map(tgt_fold)
    return df


def apply_m1_splits(
    df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply all three M1 split strategies to a DataFrame.

    Adds columns: split_random, split_cold_compound, split_cold_target.
    Uses the same seed for reproducibility across conditions.
    """
    df = add_random_split(df, seed=seed)
    df = add_cold_compound_split(df, seed=seed)
    df = add_cold_target_split(df, seed=seed)
    return df


_CHEMBL_POSITIVE_SQL = """
SELECT
    cs.canonical_smiles,
    cs.standard_inchi_key AS inchikey,
    cp.accession AS uniprot_id,
    cp.sequence AS target_sequence,
    a.pchembl_value,
    a.standard_type AS activity_type,
    a.standard_value AS activity_value_nm,
    docs.year AS publication_year
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN target_components tc ON td.tid = tc.tid
JOIN component_sequences cp ON tc.component_id = cp.component_id
JOIN molecule_dictionary md ON a.molregno = md.molregno
JOIN compound_structures cs ON md.molregno = cs.molregno
LEFT JOIN docs ON a.doc_id = docs.doc_id
WHERE a.pchembl_value >= ?
  AND a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
  AND td.target_type = 'SINGLE PROTEIN'
  AND td.organism = 'Homo sapiens'
  AND a.data_validity_comment IS NULL
  AND cs.canonical_smiles IS NOT NULL
  AND cp.accession IS NOT NULL
"""


def extract_chembl_positives(
    chembl_db_path: str | Path,
    negbiodb_path: str | Path,
    pchembl_min: float = 6.0,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Extract active compounds from ChEMBL for M1 binary task.

    Filters:
    - pChEMBL >= pchembl_min (default 6.0 = IC50 <= 1 uM)
    - Single protein, human, valid data
    - Target must exist in NegBioDB target pool
    - Deduplicates by (inchikey_connectivity, uniprot_id), keeping max pChEMBL

    Returns DataFrame with columns matching negative export format.
    """
    from negbiodb.standardize import standardize_smiles

    # Load NegBioDB target pool
    neg_conn = sqlite3.connect(str(negbiodb_path))
    neg_targets = {
        r[0] for r in neg_conn.execute(
            "SELECT uniprot_accession FROM targets"
        ).fetchall()
    }
    neg_conn.close()
    logger.info("NegBioDB target pool: %d targets", len(neg_targets))

    # Query ChEMBL
    chembl_conn = sqlite3.connect(str(chembl_db_path))
    rows = []

    for chunk in pd.read_sql_query(
        _CHEMBL_POSITIVE_SQL, chembl_conn,
        params=(pchembl_min,), chunksize=chunksize,
    ):
        # Filter to shared target pool
        chunk = chunk[chunk["uniprot_id"].isin(neg_targets)]
        if chunk.empty:
            continue

        # Standardize SMILES → canonical + InChIKey
        std_results = []
        for _, row in chunk.iterrows():
            result = standardize_smiles(row["canonical_smiles"])
            if result is None:
                continue
            std_results.append({
                "smiles": result["canonical_smiles"],
                "inchikey": result["inchikey"],
                "inchikey_connectivity": result["inchikey"][:14],
                "uniprot_id": row["uniprot_id"],
                "target_sequence": row["target_sequence"],
                "pchembl_value": row["pchembl_value"],
                "activity_type": row["activity_type"],
                "activity_value_nm": row["activity_value_nm"],
                "publication_year": row["publication_year"],
            })
        if std_results:
            rows.extend(std_results)

        logger.info("Processed %d positive candidates so far...", len(rows))

    chembl_conn.close()

    if not rows:
        logger.warning("No positive records extracted from ChEMBL")
        return pd.DataFrame(columns=[
            "smiles", "inchikey", "uniprot_id", "target_sequence",
            "pchembl_value", "activity_type", "activity_value_nm",
            "publication_year",
        ])

    df = pd.DataFrame(rows)

    # Deduplicate: keep highest pChEMBL per (inchikey_connectivity, uniprot_id)
    df = df.sort_values("pchembl_value", ascending=False)
    df = df.drop_duplicates(subset=["inchikey_connectivity", "uniprot_id"], keep="first")
    df = df.drop(columns=["inchikey_connectivity"])

    logger.info("ChEMBL positives after dedup: %d unique pairs", len(df))
    return df


def merge_positive_negative(
    positives: pd.DataFrame,
    negbiodb_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> dict:
    """Merge positives (Y=1) and negatives (Y=0) for M1 binary DTI task.

    Validates zero overlap between positives and negatives by InChIKey
    connectivity × UniProt, then creates:
    - Balanced (1:1) dataset
    - Realistic (1:10 pos:neg) dataset

    Returns dict with file paths and statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load negatives from NegBioDB
    neg_conn = sqlite3.connect(str(negbiodb_path))
    negatives = pd.read_sql_query(
        """SELECT c.canonical_smiles AS smiles, c.inchikey,
                  t.uniprot_accession AS uniprot_id,
                  t.amino_acid_sequence AS target_sequence,
                  ctp.best_confidence AS confidence_tier,
                  ctp.num_assays, ctp.num_sources, ctp.earliest_year
        FROM compound_target_pairs ctp
        JOIN compounds c ON ctp.compound_id = c.compound_id
        JOIN targets t ON ctp.target_id = t.target_id""",
        neg_conn,
    )
    neg_conn.close()

    # Overlap check: InChIKey connectivity × UniProt
    pos_keys = set(
        zip(
            positives["inchikey"].str[:14],
            positives["uniprot_id"],
        )
    )
    neg_keys = set(
        zip(
            negatives["inchikey"].str[:14],
            negatives["uniprot_id"],
        )
    )
    overlap = pos_keys & neg_keys
    if overlap:
        logger.warning(
            "Found %d overlapping (inchikey_conn, uniprot) pairs! "
            "Removing from positives.",
            len(overlap),
        )
        keep_mask = ~pd.Series(
            [k in overlap for k in zip(positives["inchikey"].str[:14], positives["uniprot_id"])],
            index=positives.index,
        )
        positives = positives[keep_mask]
        logger.info("Positives after overlap removal: %d", len(positives))

    # Prepare label columns
    positives = positives.copy()
    positives["Y"] = 1
    negatives = negatives.copy()
    negatives["Y"] = 0

    # Common columns for merge
    common_cols = ["smiles", "inchikey", "uniprot_id", "target_sequence", "Y"]
    pos_export = positives[common_cols]
    neg_export = negatives[common_cols]

    rng = np.random.RandomState(seed)
    results = {}

    # Balanced (1:1)
    n_pos = len(pos_export)
    n_neg = len(neg_export)
    n_balanced = min(n_pos, n_neg)

    pos_balanced = pos_export.sample(n=n_balanced, random_state=rng)
    neg_balanced = neg_export.sample(n=n_balanced, random_state=rng)
    balanced = pd.concat([pos_balanced, neg_balanced], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=rng).reset_index(drop=True)

    # Apply M1 splits (random, cold-compound, cold-target)
    balanced = apply_m1_splits(balanced, seed=seed)

    balanced_path = output_dir / "negbiodb_m1_balanced.parquet"
    balanced.to_parquet(str(balanced_path), index=False, compression="zstd")
    results["balanced"] = {
        "path": str(balanced_path),
        "n_pos": n_balanced,
        "n_neg": n_balanced,
        "total": len(balanced),
    }

    # Realistic (1:10 pos:neg)
    n_realistic_neg = min(n_pos * 10, n_neg)
    n_realistic_pos = min(n_pos, n_realistic_neg // 10) if n_realistic_neg >= 10 else n_pos

    rng2 = np.random.RandomState(seed)
    pos_realistic = pos_export.sample(n=n_realistic_pos, random_state=rng2)
    neg_realistic = neg_export.sample(n=n_realistic_neg, random_state=rng2)
    realistic = pd.concat([pos_realistic, neg_realistic], ignore_index=True)
    realistic = realistic.sample(frac=1, random_state=rng2).reset_index(drop=True)

    # Apply M1 splits (random, cold-compound, cold-target)
    realistic = apply_m1_splits(realistic, seed=seed)

    realistic_path = output_dir / "negbiodb_m1_realistic.parquet"
    realistic.to_parquet(str(realistic_path), index=False, compression="zstd")
    results["realistic"] = {
        "path": str(realistic_path),
        "n_pos": n_realistic_pos,
        "n_neg": n_realistic_neg,
        "total": len(realistic),
    }

    logger.info(
        "M1 merge done: balanced=%d (1:1), realistic=%d (1:%d)",
        len(balanced),
        len(realistic),
        n_realistic_neg // n_realistic_pos if n_realistic_pos > 0 else 0,
    )

    return results


# ------------------------------------------------------------------
# Random negative generation (Exp 1)
# ------------------------------------------------------------------

def _load_tested_pairs(
    negbiodb_path: str | Path,
    positives: pd.DataFrame | None = None,
) -> set[tuple[str, str]]:
    """Load all tested (compound, target) pairs as (inchikey_conn, uniprot_id).

    Includes NegBioDB negative pairs and optionally positive pairs.
    """
    conn = sqlite3.connect(str(negbiodb_path))
    tested = set()
    for row in conn.execute(
        """SELECT c.inchikey_connectivity, t.uniprot_accession
        FROM compound_target_pairs ctp
        JOIN compounds c ON ctp.compound_id = c.compound_id
        JOIN targets t ON ctp.target_id = t.target_id"""
    ):
        tested.add((row[0], row[1]))
    conn.close()

    if positives is not None:
        for ik, uid in zip(positives["inchikey"].str[:14], positives["uniprot_id"]):
            tested.add((ik, uid))

    return tested


def _load_compound_target_pools(
    negbiodb_path: str | Path,
    positives: pd.DataFrame | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load compound and target pools with SMILES/sequence for output.

    Returns (compounds_list, targets_list) where each element is a dict
    with the fields needed for M1 output.
    """
    conn = sqlite3.connect(str(negbiodb_path))

    # Compounds: inchikey_connectivity → {smiles, inchikey}
    compound_map: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT inchikey_connectivity, canonical_smiles, inchikey FROM compounds"
    ):
        compound_map[row[0]] = {"smiles": row[1], "inchikey": row[2]}

    # Add compounds from positives that might not be in NegBioDB
    if positives is not None:
        for _, r in positives[["smiles", "inchikey"]].drop_duplicates(
            subset=["inchikey"]
        ).iterrows():
            ik_conn = r["inchikey"][:14]
            if ik_conn not in compound_map:
                compound_map[ik_conn] = {
                    "smiles": r["smiles"],
                    "inchikey": r["inchikey"],
                }

    # Targets: uniprot_accession → {target_sequence}
    target_map: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT uniprot_accession, amino_acid_sequence FROM targets"
    ):
        target_map[row[0]] = {"target_sequence": row[1]}

    conn.close()

    compounds = [
        {"inchikey_conn": k, **v} for k, v in compound_map.items()
    ]
    targets = [
        {"uniprot_id": k, **v} for k, v in target_map.items()
    ]
    return compounds, targets


def generate_uniform_random_negatives(
    negbiodb_path: str | Path,
    positives: pd.DataFrame,
    n_samples: int,
    output_dir: str | Path,
    seed: int = 42,
) -> dict:
    """Generate uniform random negative pairs for Exp 1 control.

    Samples untested compound-target pairs uniformly from the cross-product
    of all compounds × all targets, excluding any tested pairs (both
    NegBioDB negatives and ChEMBL positives).

    Merges with the same positive set and applies M1 splits.

    Returns dict with file path and statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tested pairs for exclusion...")
    tested = _load_tested_pairs(negbiodb_path, positives)
    logger.info("Tested pairs: %d", len(tested))

    logger.info("Loading compound/target pools...")
    compounds, targets = _load_compound_target_pools(negbiodb_path, positives)
    logger.info("Compound pool: %d, Target pool: %d", len(compounds), len(targets))

    # Rejection sampling
    rng = np.random.RandomState(seed)
    neg_rows = []
    attempts = 0
    max_attempts = n_samples * 100  # safety limit

    while len(neg_rows) < n_samples and attempts < max_attempts:
        batch = min((n_samples - len(neg_rows)) * 2, 1_000_000)
        c_idx = rng.randint(0, len(compounds), batch)
        t_idx = rng.randint(0, len(targets), batch)

        for ci, ti in zip(c_idx, t_idx):
            comp = compounds[ci]
            tgt = targets[ti]
            key = (comp["inchikey_conn"], tgt["uniprot_id"])
            if key not in tested:
                neg_rows.append({
                    "smiles": comp["smiles"],
                    "inchikey": comp["inchikey"],
                    "uniprot_id": tgt["uniprot_id"],
                    "target_sequence": tgt["target_sequence"],
                    "Y": 0,
                })
                tested.add(key)  # prevent duplicate negatives
                if len(neg_rows) >= n_samples:
                    break
            attempts += 1

    if len(neg_rows) == 0:
        logger.warning("Uniform random: 0 negatives generated (pool exhausted)")
    else:
        logger.info(
            "Uniform random: generated %d negatives (rejection rate: %.2f%%)",
            len(neg_rows),
            (1 - len(neg_rows) / max(attempts, 1)) * 100,
        )

    neg_df = pd.DataFrame(neg_rows)

    # Merge with positives (same as M1 balanced)
    pos_export = positives[["smiles", "inchikey", "uniprot_id", "target_sequence"]].copy()
    pos_export["Y"] = 1

    n_balanced = min(len(pos_export), len(neg_df))
    rng2 = np.random.RandomState(seed)
    pos_sample = pos_export.sample(n=n_balanced, random_state=rng2)
    neg_sample = neg_df.sample(n=n_balanced, random_state=rng2)

    merged = pd.concat([pos_sample, neg_sample], ignore_index=True)
    merged = merged.sample(frac=1, random_state=rng2).reset_index(drop=True)
    merged = apply_m1_splits(merged, seed=seed)

    out_path = output_dir / "negbiodb_m1_uniform_random.parquet"
    merged.to_parquet(str(out_path), index=False, compression="zstd")

    result = {
        "path": str(out_path),
        "n_pos": n_balanced,
        "n_neg": n_balanced,
        "total": len(merged),
    }
    logger.info("Uniform random M1: %d total → %s", len(merged), out_path.name)
    return result


def generate_degree_matched_negatives(
    negbiodb_path: str | Path,
    positives: pd.DataFrame,
    n_samples: int,
    output_dir: str | Path,
    seed: int = 42,
) -> dict:
    """Generate degree-matched random negatives for Exp 1 control.

    Samples untested pairs whose compounds and targets have degree
    distributions matching NegBioDB's, isolating the effect of
    experimental confirmation vs. degree bias.

    Uses log-scale binning of compound_degree × target_degree.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tested pairs and degree info...")
    tested = _load_tested_pairs(negbiodb_path, positives)

    conn = sqlite3.connect(str(negbiodb_path))

    # Load compound/target degree from DB
    degree_df = pd.read_sql_query(
        """SELECT c.inchikey_connectivity, c.canonical_smiles, c.inchikey,
                  t.uniprot_accession AS uniprot_id,
                  t.amino_acid_sequence AS target_sequence,
                  ctp.compound_degree, ctp.target_degree
        FROM compound_target_pairs ctp
        JOIN compounds c ON ctp.compound_id = c.compound_id
        JOIN targets t ON ctp.target_id = t.target_id""",
        conn,
    )
    conn.close()

    # Build per-compound and per-target degree lookup
    comp_deg = degree_df.groupby("inchikey_connectivity")["compound_degree"].first()
    tgt_deg = degree_df.groupby("uniprot_id")["target_degree"].first()

    # Log-scale binning of NegBioDB degree distribution
    degree_df["cdeg_bin"] = np.floor(
        np.log2(degree_df["compound_degree"].clip(lower=1))
    ).astype(int)
    degree_df["tdeg_bin"] = np.floor(
        np.log2(degree_df["target_degree"].clip(lower=1))
    ).astype(int)

    bin_counts = degree_df.groupby(["cdeg_bin", "tdeg_bin"]).size()
    total_pairs = bin_counts.sum()

    # Compute target samples per bin
    bin_targets = {}
    remaining = n_samples
    for (cb, tb), count in bin_counts.items():
        n = int(round(count / total_pairs * n_samples))
        bin_targets[(cb, tb)] = n
        remaining -= n
    # Distribute remainder to largest bins
    if remaining > 0:
        for key in bin_counts.sort_values(ascending=False).index:
            if remaining <= 0:
                break
            bin_targets[key] = bin_targets.get(key, 0) + 1
            remaining -= 1

    # Build per-bin compound and target pools
    compounds_by_bin: dict[int, list[dict]] = defaultdict(list)
    for ik_conn, row in degree_df.drop_duplicates("inchikey_connectivity").iterrows():
        cb = int(np.floor(np.log2(max(comp_deg.get(row["inchikey_connectivity"], 1), 1))))
        compounds_by_bin[cb].append({
            "inchikey_conn": row["inchikey_connectivity"],
            "smiles": row["canonical_smiles"],
            "inchikey": row["inchikey"],
        })

    targets_by_bin: dict[int, list[dict]] = defaultdict(list)
    for uid, row in degree_df.drop_duplicates("uniprot_id").set_index("uniprot_id").iterrows():
        tb = int(np.floor(np.log2(max(tgt_deg.get(uid, 1), 1))))
        targets_by_bin[tb].append({
            "uniprot_id": uid,
            "target_sequence": row["target_sequence"],
        })

    # Rejection sampling per bin
    rng = np.random.RandomState(seed)
    neg_rows = []

    for (cb, tb), target_n in bin_targets.items():
        if target_n <= 0:
            continue
        c_pool = compounds_by_bin.get(cb, [])
        t_pool = targets_by_bin.get(tb, [])
        if not c_pool or not t_pool:
            logger.warning("Empty pool for bin (%d, %d), skipping %d samples", cb, tb, target_n)
            continue

        sampled = 0
        attempts = 0
        max_attempts = target_n * 200

        while sampled < target_n and attempts < max_attempts:
            ci = rng.randint(0, len(c_pool))
            ti = rng.randint(0, len(t_pool))
            comp = c_pool[ci]
            tgt = t_pool[ti]
            key = (comp["inchikey_conn"], tgt["uniprot_id"])
            if key not in tested:
                neg_rows.append({
                    "smiles": comp["smiles"],
                    "inchikey": comp["inchikey"],
                    "uniprot_id": tgt["uniprot_id"],
                    "target_sequence": tgt["target_sequence"],
                    "Y": 0,
                })
                tested.add(key)  # prevent duplicate negatives
                sampled += 1
            attempts += 1

    if len(neg_rows) == 0:
        logger.warning("Degree-matched: 0 negatives generated (pool exhausted)")
    else:
        logger.info("Degree-matched: generated %d negatives", len(neg_rows))

    neg_df = pd.DataFrame(neg_rows)

    # Merge with positives (same as M1 balanced)
    pos_export = positives[["smiles", "inchikey", "uniprot_id", "target_sequence"]].copy()
    pos_export["Y"] = 1

    n_balanced = min(len(pos_export), len(neg_df))
    rng2 = np.random.RandomState(seed)
    pos_sample = pos_export.sample(n=n_balanced, random_state=rng2)
    neg_sample = neg_df.iloc[:n_balanced]

    merged = pd.concat([pos_sample, neg_sample], ignore_index=True)
    merged = merged.sample(frac=1, random_state=rng2).reset_index(drop=True)
    merged = apply_m1_splits(merged, seed=seed)

    out_path = output_dir / "negbiodb_m1_degree_matched.parquet"
    merged.to_parquet(str(out_path), index=False, compression="zstd")

    result = {
        "path": str(out_path),
        "n_pos": n_balanced,
        "n_neg": n_balanced,
        "total": len(merged),
    }
    logger.info("Degree-matched M1: %d total → %s", len(merged), out_path.name)
    return result


# ------------------------------------------------------------------
# Data leakage check
# ------------------------------------------------------------------

def check_cold_split_integrity(
    conn: sqlite3.Connection,
) -> dict:
    """Verify cold splits have zero entity leakage between train and test.

    Returns dict with per-strategy leak counts (should all be 0).
    """
    results = {}

    for strategy, entity_col in [
        ("cold_compound", "compound_id"),
        ("cold_target", "target_id"),
    ]:
        sid_row = conn.execute(
            "SELECT split_id FROM split_definitions WHERE split_strategy = ?",
            (strategy,),
        ).fetchone()
        if sid_row is None:
            results[strategy] = {"status": "not_found"}
            continue

        sid = sid_row[0]
        leaks = conn.execute(
            f"""SELECT COUNT(DISTINCT ctp1.{entity_col})
            FROM split_assignments sa1
            JOIN compound_target_pairs ctp1 ON sa1.pair_id = ctp1.pair_id
            WHERE sa1.split_id = ? AND sa1.fold = 'train'
            AND ctp1.{entity_col} IN (
                SELECT ctp2.{entity_col}
                FROM split_assignments sa2
                JOIN compound_target_pairs ctp2 ON sa2.pair_id = ctp2.pair_id
                WHERE sa2.split_id = ? AND sa2.fold = 'test'
            )""",
            (sid, sid),
        ).fetchone()[0]

        results[strategy] = {"split_id": sid, "leaks": leaks}

    return results


def check_cross_db_overlap(
    conn: sqlite3.Connection,
) -> dict:
    """Check overlap between sources at compound×target pair level.

    Returns overlap statistics between each pair of sources.
    """
    sources = [r[0] for r in conn.execute(
        "SELECT DISTINCT source_db FROM negative_results ORDER BY source_db"
    ).fetchall()]

    overlaps = {}
    for i, s1 in enumerate(sources):
        for s2 in sources[i + 1:]:
            count = conn.execute(
                """SELECT COUNT(DISTINCT nr1.compound_id || ':' || nr1.target_id)
                FROM negative_results nr1
                WHERE nr1.source_db = ?
                AND EXISTS (
                    SELECT 1 FROM negative_results nr2
                    WHERE nr2.compound_id = nr1.compound_id
                    AND nr2.target_id = nr1.target_id
                    AND nr2.source_db = ?
                )""",
                (s1, s2),
            ).fetchone()[0]
            overlaps[f"{s1}_vs_{s2}"] = count

    return overlaps


def generate_leakage_report(
    db_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """Generate comprehensive data leakage and integrity report.

    Checks:
    1. Cold split integrity (zero entity leakage)
    2. Split fold counts and ratios
    3. Database summary statistics

    Returns report dict, optionally writes to JSON file.
    """
    import json

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")

    report: dict = {}

    # 1. DB summary
    report["db_summary"] = {
        "compounds": conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0],
        "targets": conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0],
        "negative_results": conn.execute("SELECT COUNT(*) FROM negative_results").fetchone()[0],
        "pairs": conn.execute("SELECT COUNT(*) FROM compound_target_pairs").fetchone()[0],
    }

    # 2. Source breakdown
    source_counts = {}
    for source, cnt in conn.execute(
        "SELECT source_db, COUNT(*) FROM negative_results GROUP BY source_db"
    ).fetchall():
        source_counts[source] = cnt
    report["source_counts"] = source_counts

    # 3. Split summary
    split_summary = {}
    for sid, name, strategy in conn.execute(
        "SELECT split_id, split_name, split_strategy FROM split_definitions"
    ).fetchall():
        fold_counts = {}
        for fold, cnt in conn.execute(
            "SELECT fold, COUNT(*) FROM split_assignments WHERE split_id = ? GROUP BY fold",
            (sid,),
        ).fetchall():
            fold_counts[fold] = cnt
        total = sum(fold_counts.values())
        split_summary[name] = {
            "strategy": strategy,
            "fold_counts": fold_counts,
            "total": total,
            "ratios": {
                f: round(c / total, 4) if total > 0 else 0
                for f, c in fold_counts.items()
            },
        }
    report["splits"] = split_summary

    # 4. Cold split integrity
    report["cold_split_integrity"] = check_cold_split_integrity(conn)

    # 5. Cross-source overlap
    report["cross_source_overlap"] = check_cross_db_overlap(conn)

    # 6. Pairs by num_sources
    multi_source = {}
    for ns, cnt in conn.execute(
        "SELECT num_sources, COUNT(*) FROM compound_target_pairs GROUP BY num_sources ORDER BY num_sources"
    ).fetchall():
        multi_source[str(ns)] = cnt
    report["pairs_by_num_sources"] = multi_source

    conn.close()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Leakage report written to %s", output_path)

    return report
