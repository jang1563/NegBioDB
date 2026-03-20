"""ML dataset export pipeline for NegBioDB PPI domain.

Provides:
  - DB-level split generation (random, cold_protein, cold_both, degree_balanced)
  - DataFrame-level split helpers for merged pos+neg datasets
  - Negative dataset export to Parquet
  - HuRI positive loading + M1 merge with conflict resolution
  - Control negative generation (uniform random, degree-matched)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

BATCH_SIZE = 500_000

# Split strategies used for DB-level inline export columns
SPLIT_STRATEGIES = ["random", "cold_protein", "cold_both", "degree_balanced"]

_DEFAULT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}


# ------------------------------------------------------------------
# DB-level split helpers
# ------------------------------------------------------------------

def _register_ppi_split(
    conn: sqlite3.Connection,
    name: str,
    strategy: str,
    seed: int | None,
    ratios: dict[str, float],
) -> int:
    """Insert or retrieve a PPI split definition and return split_id.

    Targets the ppi_split_definitions table (NOT the DTI split_definitions).
    """
    row = conn.execute(
        "SELECT split_id FROM ppi_split_definitions WHERE split_name = ?",
        (name,),
    ).fetchone()

    if row is not None:
        split_id = int(row[0])
        conn.execute(
            "DELETE FROM ppi_split_assignments WHERE split_id = ?",
            (split_id,),
        )
        return split_id

    conn.execute(
        """INSERT INTO ppi_split_definitions
        (split_name, split_strategy, random_seed,
         train_ratio, val_ratio, test_ratio)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (name, strategy, seed,
         ratios["train"], ratios["val"], ratios["test"]),
    )
    row = conn.execute(
        "SELECT split_id FROM ppi_split_definitions WHERE split_name = ?",
        (name,),
    ).fetchone()
    return int(row[0])


# ------------------------------------------------------------------
# DB-level split generators
# ------------------------------------------------------------------

def generate_random_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict:
    """Generate random 70/10/20 split across all PPI pairs."""
    if ratios is None:
        ratios = _DEFAULT_RATIOS

    split_id = _register_ppi_split(conn, "random_v1", "random", seed, ratios)

    pair_ids = np.array(
        [r[0] for r in conn.execute(
            "SELECT pair_id FROM protein_protein_pairs ORDER BY pair_id"
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

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = [
            (int(pair_ids[i]), split_id, fold_labels[i])
            for i in range(start, end)
        ]
        conn.executemany(
            "INSERT INTO ppi_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
            batch,
        )
    conn.commit()

    counts = _fold_counts(conn, split_id)
    logger.info("Random split done: %s", counts)
    return {"split_id": split_id, "counts": counts}


def generate_cold_protein_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    protein_ratios: dict[str, float] | None = None,
) -> dict:
    """Generate cold-protein split: test proteins unseen in train.

    Proteins are assigned to folds at protein level, then each pair
    gets fold = max(fold_P1, fold_P2) where test > val > train.
    This causes fold inflation: 70/10/20 protein-level != pair-level.
    """
    if protein_ratios is None:
        protein_ratios = {"train": 0.80, "val": 0.05, "test": 0.15}

    ratios_for_record = _DEFAULT_RATIOS  # record target pair-level ratios
    split_id = _register_ppi_split(
        conn, "cold_protein_v1", "cold_protein", seed, ratios_for_record
    )

    # Collect all unique protein IDs from both sides
    proteins = [r[0] for r in conn.execute(
        """SELECT protein_id FROM (
            SELECT protein1_id AS protein_id FROM protein_protein_pairs
            UNION
            SELECT protein2_id FROM protein_protein_pairs
        ) ORDER BY protein_id"""
    ).fetchall()]

    rng = np.random.RandomState(seed)
    rng.shuffle(proteins)

    n = len(proteins)
    n_train = int(n * protein_ratios["train"])
    n_val = int(n * protein_ratios["val"])

    protein_fold = {}
    for i, pid in enumerate(proteins):
        if i < n_train:
            protein_fold[pid] = "train"
        elif i < n_train + n_val:
            protein_fold[pid] = "val"
        else:
            protein_fold[pid] = "test"

    # Write protein folds to temp table
    conn.execute("DROP TABLE IF EXISTS _protein_folds")
    conn.execute(
        "CREATE TEMP TABLE _protein_folds (protein_id INTEGER PRIMARY KEY, fold TEXT)"
    )
    conn.executemany(
        "INSERT INTO _protein_folds (protein_id, fold) VALUES (?, ?)",
        protein_fold.items(),
    )

    # Assign pair folds: fold = max(fold_P1, fold_P2)
    conn.execute(
        """INSERT INTO ppi_split_assignments (pair_id, split_id, fold)
        SELECT ppp.pair_id, ?,
            CASE WHEN gf1.fold = 'test' OR gf2.fold = 'test' THEN 'test'
                 WHEN gf1.fold = 'val'  OR gf2.fold = 'val'  THEN 'val'
                 ELSE 'train' END
        FROM protein_protein_pairs ppp
        JOIN _protein_folds gf1 ON ppp.protein1_id = gf1.protein_id
        JOIN _protein_folds gf2 ON ppp.protein2_id = gf2.protein_id""",
        (split_id,),
    )
    conn.execute("DROP TABLE _protein_folds")
    conn.commit()

    counts = _fold_counts(conn, split_id)

    # Integrity check: no TEST-GROUP protein appears in a TRAIN-FOLD pair.
    # A train-group protein CAN appear in test-fold pairs (expected: max rule).
    # But a test-group protein should NEVER appear in train-fold pairs.
    conn.execute("DROP TABLE IF EXISTS _test_group_proteins")
    conn.execute(
        "CREATE TEMP TABLE _test_group_proteins (protein_id INTEGER PRIMARY KEY)"
    )
    # Re-derive test group from protein_ratios
    test_group_pids = [pid for pid, fold in protein_fold.items() if fold == "test"]
    conn.executemany(
        "INSERT INTO _test_group_proteins (protein_id) VALUES (?)",
        [(pid,) for pid in test_group_pids],
    )

    leaked = conn.execute(
        """SELECT COUNT(DISTINCT protein_id) FROM (
            SELECT protein1_id AS protein_id FROM protein_protein_pairs ppp
            JOIN ppi_split_assignments sa ON ppp.pair_id = sa.pair_id AND sa.split_id = ?
            WHERE sa.fold = 'train'
            UNION ALL
            SELECT protein2_id FROM protein_protein_pairs ppp
            JOIN ppi_split_assignments sa ON ppp.pair_id = sa.pair_id AND sa.split_id = ?
            WHERE sa.fold = 'train'
        ) WHERE protein_id IN (SELECT protein_id FROM _test_group_proteins)""",
        (split_id, split_id),
    ).fetchone()[0]
    conn.execute("DROP TABLE _test_group_proteins")

    if leaked > 0:
        logger.warning("Cold protein leakage: %d test-group proteins in train-fold pairs", leaked)
    else:
        logger.info("Cold protein integrity verified: 0 test-group proteins in train pairs")

    logger.info("Cold-protein split done: %s (protein ratios: %.0f/%.0f/%.0f)",
                counts, protein_ratios["train"] * 100,
                protein_ratios["val"] * 100, protein_ratios["test"] * 100)
    return {"split_id": split_id, "counts": counts, "leaked_proteins": leaked}


def generate_cold_both_partition(
    conn: sqlite3.Connection,
    seed: int = 42,
    nparts: int = 10,
) -> dict:
    """Generate cold-both split via Metis k-way graph partitioning.

    Uses pymetis to partition the protein graph into nparts equal partitions.
    Partitions 0-(nparts*0.7-1) → train, next → val, rest → test.
    Cross-partition pairs are EXCLUDED to preserve cold-both property.

    Registered with strategy='bfs_cluster' (reuses existing CHECK constraint).
    """
    import pymetis

    ratios = _DEFAULT_RATIOS
    split_id = _register_ppi_split(
        conn, "cold_both_v1", "bfs_cluster", seed, ratios
    )

    # Build adjacency list
    rows = conn.execute(
        "SELECT protein1_id, protein2_id FROM protein_protein_pairs"
    ).fetchall()

    # Map protein IDs to contiguous 0..N-1 indices
    protein_ids = sorted({r[0] for r in rows} | {r[1] for r in rows})
    pid_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
    n_nodes = len(protein_ids)

    adjacency = [[] for _ in range(n_nodes)]
    for p1, p2 in rows:
        i, j = pid_to_idx[p1], pid_to_idx[p2]
        adjacency[i].append(j)
        adjacency[j].append(i)

    logger.info("Metis partition: %d nodes, %d edges, nparts=%d",
                n_nodes, len(rows), nparts)

    # Partition with deterministic seed
    opts = pymetis.Options(seed=seed)
    n_cuts, membership = pymetis.part_graph(nparts, adjacency=adjacency, options=opts)

    # Map partitions to folds: 70% train, 10% val, 20% test
    n_train_parts = int(nparts * 0.7)  # 7
    n_val_parts = int(nparts * 0.1)    # 1
    # rest = test (2)

    # Shuffle partition labels so assignment isn't size-biased
    rng = np.random.RandomState(seed)
    part_labels = list(range(nparts))
    rng.shuffle(part_labels)

    part_to_fold = {}
    for i, pl in enumerate(part_labels):
        if i < n_train_parts:
            part_to_fold[pl] = "train"
        elif i < n_train_parts + n_val_parts:
            part_to_fold[pl] = "val"
        else:
            part_to_fold[pl] = "test"

    # Protein-level fold assignment
    protein_fold = {}
    for pid, idx in pid_to_idx.items():
        protein_fold[pid] = part_to_fold[membership[idx]]

    # Write protein folds to temp table
    conn.execute("DROP TABLE IF EXISTS _metis_folds")
    conn.execute(
        "CREATE TEMP TABLE _metis_folds (protein_id INTEGER PRIMARY KEY, fold TEXT)"
    )
    conn.executemany(
        "INSERT INTO _metis_folds (protein_id, fold) VALUES (?, ?)",
        protein_fold.items(),
    )

    # Only include same-partition pairs (both proteins in same fold)
    conn.execute(
        """INSERT INTO ppi_split_assignments (pair_id, split_id, fold)
        SELECT ppp.pair_id, ?, gf1.fold
        FROM protein_protein_pairs ppp
        JOIN _metis_folds gf1 ON ppp.protein1_id = gf1.protein_id
        JOIN _metis_folds gf2 ON ppp.protein2_id = gf2.protein_id
        WHERE gf1.fold = gf2.fold""",
        (split_id,),
    )
    conn.execute("DROP TABLE _metis_folds")
    conn.commit()

    counts = _fold_counts(conn, split_id)
    total_assigned = sum(counts.values())
    total_pairs = conn.execute(
        "SELECT COUNT(*) FROM protein_protein_pairs"
    ).fetchone()[0]
    excluded = total_pairs - total_assigned

    logger.info("Cold-both (Metis) split: %s, excluded=%d (%.1f%%), edge_cuts=%d",
                counts, excluded, 100 * excluded / total_pairs, n_cuts)
    return {
        "split_id": split_id, "counts": counts,
        "excluded": excluded, "edge_cuts": n_cuts,
    }


def generate_degree_balanced_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    n_bins: int = 10,
) -> dict:
    """Generate degree-distribution-balanced (DDB) split.

    Uses 1D log(degree_sum) quantile binning, then stratified 70/10/20
    within each bin.
    """
    if ratios is None:
        ratios = _DEFAULT_RATIOS

    split_id = _register_ppi_split(
        conn, "degree_balanced_v1", "degree_balanced", seed, ratios
    )

    rows = conn.execute(
        """SELECT pair_id,
                  COALESCE(protein1_degree, 1),
                  COALESCE(protein2_degree, 1)
        FROM protein_protein_pairs ORDER BY pair_id"""
    ).fetchall()

    pair_ids = np.array([r[0] for r in rows], dtype=np.int64)
    deg1 = np.array([r[1] for r in rows], dtype=np.float64)
    deg2 = np.array([r[2] for r in rows], dtype=np.float64)

    # 1D degree-sum binning (not 2D)
    degree_sum = deg1 + deg2
    log_deg_sum = np.log1p(degree_sum)

    # Quantile-based bins for equal-frequency
    bin_edges = np.quantile(log_deg_sum, np.linspace(0, 1, n_bins + 1))
    bin_labels = np.digitize(log_deg_sum, bin_edges[1:-1])  # 0..n_bins-1

    rng = np.random.RandomState(seed)
    fold_labels = np.empty(len(pair_ids), dtype="U5")

    for bin_id in np.unique(bin_labels):
        idx = np.where(bin_labels == bin_id)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        fold_labels[idx[:n_train]] = "train"
        fold_labels[idx[n_train:n_train + n_val]] = "val"
        fold_labels[idx[n_train + n_val:]] = "test"

    for start in range(0, len(pair_ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(pair_ids))
        batch = [
            (int(pair_ids[i]), split_id, fold_labels[i])
            for i in range(start, end)
        ]
        conn.executemany(
            "INSERT INTO ppi_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
            batch,
        )
    conn.commit()

    counts = _fold_counts(conn, split_id)
    logger.info("Degree-balanced split done: %s", counts)
    return {"split_id": split_id, "counts": counts}


def _fold_counts(conn: sqlite3.Connection, split_id: int) -> dict[str, int]:
    """Return {fold: count} for a split."""
    counts: dict[str, int] = {}
    for fold, cnt in conn.execute(
        "SELECT fold, COUNT(*) FROM ppi_split_assignments WHERE split_id = ? GROUP BY fold",
        (split_id,),
    ).fetchall():
        counts[fold] = cnt
    return counts


# ------------------------------------------------------------------
# Negative dataset export
# ------------------------------------------------------------------

def _resolve_split_id(conn: sqlite3.Connection, strategy: str) -> int | None:
    """Resolve the preferred split_id for a strategy."""
    rows = conn.execute(
        """SELECT split_id, split_name, version
        FROM ppi_split_definitions
        WHERE split_strategy = ?""",
        (strategy,),
    ).fetchall()
    if not rows:
        return None

    def sort_key(row):
        split_id, split_name, version = row
        version_num = -1
        if version:
            try:
                version_num = int(str(version).split(".")[0])
            except ValueError:
                version_num = -1
        if "_v" in split_name:
            suffix = split_name.rsplit("_v", 1)[1]
            if suffix.isdigit():
                version_num = max(version_num, int(suffix))
        return (version_num, split_name, split_id)

    return sorted(rows, key=sort_key)[-1][0]


def _build_export_query(split_ids: dict[str, int | None]) -> str:
    """Build SQL for exporting PPI pairs with inline split columns.

    Double-JOINs the proteins table for both protein1 and protein2.
    """
    split_cols = []
    join_clauses = []
    for i, strategy in enumerate(split_ids):
        alias_sa = f"sa{i}"
        col_name = f"split_{strategy}"
        split_cols.append(f"{alias_sa}.fold AS {col_name}")
        split_id = split_ids[strategy]
        split_id_sql = "NULL" if split_id is None else str(int(split_id))
        join_clauses.append(
            f"LEFT JOIN ppi_split_assignments {alias_sa} "
            f"ON ppp.pair_id = {alias_sa}.pair_id "
            f"AND {alias_sa}.split_id = {split_id_sql}"
        )

    base_cols = """ppp.pair_id,
       p1.uniprot_accession AS uniprot_id_1,
       p1.amino_acid_sequence AS sequence_1,
       p1.gene_symbol AS gene_symbol_1,
       p1.subcellular_location AS subcellular_location_1,
       p2.uniprot_accession AS uniprot_id_2,
       p2.amino_acid_sequence AS sequence_2,
       p2.gene_symbol AS gene_symbol_2,
       p2.subcellular_location AS subcellular_location_2,
       0 AS Y,
       ppp.best_confidence AS confidence_tier,
       ppp.num_sources,
       ppp.protein1_degree,
       ppp.protein2_degree"""

    if split_cols:
        select_clause = base_cols + ",\n       " + ",\n       ".join(split_cols)
    else:
        select_clause = base_cols

    join_sql = "\n".join(join_clauses)

    query = f"""SELECT
       {select_clause}
FROM protein_protein_pairs ppp
JOIN proteins p1 ON ppp.protein1_id = p1.protein_id
JOIN proteins p2 ON ppp.protein2_id = p2.protein_id
{join_sql}
ORDER BY ppp.pair_id"""

    return query


def export_negative_dataset(
    db_path: str | Path,
    output_dir: str | Path,
    split_strategies: list[str] | None = None,
    chunksize: int = BATCH_SIZE,
    exclude_source: str | None = None,
) -> dict:
    """Export negative PPI pairs as Parquet.

    Produces: negbiodb_ppi_pairs.parquet

    Args:
        db_path: Path to negbiodb_ppi.db.
        output_dir: Output directory.
        split_strategies: List of split strategies to include.
        chunksize: Rows per batch for Parquet writing.
        exclude_source: If set (e.g., 'huri'), exclude negatives from that source.

    Returns dict with file path and row count.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "negbiodb_ppi_pairs.parquet"

    if split_strategies is None:
        split_strategies = SPLIT_STRATEGIES

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")

    split_ids = {
        strategy: _resolve_split_id(conn, strategy if strategy != "cold_both" else "bfs_cluster")
        for strategy in split_strategies
    }
    query = _build_export_query(split_ids)

    # Optional source filtering
    if exclude_source:
        # Get pair_ids to exclude
        excluded_pairs = {r[0] for r in conn.execute(
            """SELECT DISTINCT pair_id FROM protein_protein_pairs ppp
            WHERE NOT EXISTS (
                SELECT 1 FROM ppi_negative_results nr
                WHERE nr.protein1_id = ppp.protein1_id
                  AND nr.protein2_id = ppp.protein2_id
                  AND nr.source_db != ?
            )""",
            (exclude_source,),
        ).fetchall()}
        logger.info("Excluding %d pairs from source '%s'",
                     len(excluded_pairs), exclude_source)

    total_rows = 0
    pq_writer = None

    try:
        for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
            if exclude_source:
                chunk = chunk[~chunk["pair_id"].isin(excluded_pairs)]

            total_rows += len(chunk)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(
                    str(parquet_path), table.schema, compression="zstd"
                )
            pq_writer.write_table(table)
            logger.info("Exported %d rows so far...", total_rows)
    finally:
        if pq_writer is not None:
            pq_writer.close()
        conn.close()

    logger.info(
        "Export complete: %d rows → %s (%.1f MB)",
        total_rows, parquet_path.name,
        parquet_path.stat().st_size / 1e6 if parquet_path.exists() else 0,
    )

    return {"total_rows": total_rows, "parquet_path": str(parquet_path)}


# ------------------------------------------------------------------
# HuRI positive loading + conflict resolution
# ------------------------------------------------------------------

def load_huri_positives_df(
    ppi_path: str | Path,
    ensg_mapping_path: str | Path,
    db_path: str | Path,
) -> pd.DataFrame:
    """Load HuRI positives and join with protein sequences from DB.

    Returns DataFrame with columns matching the negative export schema
    (uniprot_id_1, sequence_1, ..., uniprot_id_2, sequence_2, ..., Y=1).
    """
    from negbiodb_ppi.etl_huri import load_huri_positives
    from negbiodb_ppi.protein_mapper import load_ensg_mapping

    ensg_map = load_ensg_mapping(ensg_mapping_path)
    positive_pairs = load_huri_positives(ppi_path, ensg_mapping=ensg_map)
    logger.info("Loaded %d HuRI positive pairs", len(positive_pairs))

    # Load protein metadata from DB
    conn = sqlite3.connect(str(db_path))
    proteins = {}
    for row in conn.execute(
        """SELECT uniprot_accession, amino_acid_sequence,
                  gene_symbol, subcellular_location
        FROM proteins"""
    ).fetchall():
        proteins[row[0]] = {
            "sequence": row[1],
            "gene_symbol": row[2],
            "subcellular_location": row[3],
        }
    conn.close()

    records = []
    skipped_no_seq = 0
    for acc1, acc2 in positive_pairs:
        p1 = proteins.get(acc1)
        p2 = proteins.get(acc2)
        if not p1 or not p2:
            skipped_no_seq += 1
            continue
        if not p1["sequence"] or not p2["sequence"]:
            skipped_no_seq += 1
            continue
        records.append({
            "pair_id": None,
            "uniprot_id_1": acc1,
            "sequence_1": p1["sequence"],
            "gene_symbol_1": p1["gene_symbol"],
            "subcellular_location_1": p1["subcellular_location"],
            "uniprot_id_2": acc2,
            "sequence_2": p2["sequence"],
            "gene_symbol_2": p2["gene_symbol"],
            "subcellular_location_2": p2["subcellular_location"],
            "Y": 1,
            "confidence_tier": None,
            "num_sources": None,
            "protein1_degree": None,
            "protein2_degree": None,
        })

    if skipped_no_seq:
        logger.warning("Skipped %d positive pairs (missing protein/sequence in DB)",
                        skipped_no_seq)

    df = pd.DataFrame(records)
    logger.info("HuRI positives with sequences: %d", len(df))
    return df


def resolve_conflicts(
    neg_df: pd.DataFrame,
    pos_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Remove pairs that appear as both positive and negative.

    Returns (clean_neg, clean_pos, n_conflicts).
    """
    neg_pairs = set(
        zip(neg_df["uniprot_id_1"], neg_df["uniprot_id_2"])
    )
    pos_pairs = set(
        zip(pos_df["uniprot_id_1"], pos_df["uniprot_id_2"])
    )

    conflicts = neg_pairs & pos_pairs
    n_conflicts = len(conflicts)

    if n_conflicts > 0:
        logger.warning("Found %d pos/neg conflicts — removing from both sides",
                        n_conflicts)
        conflict_set = conflicts
        neg_mask = ~neg_df.apply(
            lambda r: (r["uniprot_id_1"], r["uniprot_id_2"]) in conflict_set,
            axis=1,
        )
        pos_mask = ~pos_df.apply(
            lambda r: (r["uniprot_id_1"], r["uniprot_id_2"]) in conflict_set,
            axis=1,
        )
        neg_df = neg_df[neg_mask].reset_index(drop=True)
        pos_df = pos_df[pos_mask].reset_index(drop=True)

    return neg_df, pos_df, n_conflicts


# ------------------------------------------------------------------
# Control negative generation
# ------------------------------------------------------------------

def _load_tested_pairs(
    db_path: str | Path,
    positive_pairs: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    """Load all tested pairs (NegBioDB negatives + HuRI positives) for exclusion."""
    conn = sqlite3.connect(str(db_path))
    neg_pairs = set()
    for r in conn.execute(
        """SELECT p1.uniprot_accession, p2.uniprot_accession
        FROM protein_protein_pairs ppp
        JOIN proteins p1 ON ppp.protein1_id = p1.protein_id
        JOIN proteins p2 ON ppp.protein2_id = p2.protein_id"""
    ).fetchall():
        neg_pairs.add((r[0], r[1]) if r[0] < r[1] else (r[1], r[0]))
    conn.close()

    return neg_pairs | positive_pairs


def _get_sequenced_proteins(db_path: str | Path) -> list[str]:
    """Get all protein accessions that have sequences."""
    conn = sqlite3.connect(str(db_path))
    accs = [r[0] for r in conn.execute(
        "SELECT uniprot_accession FROM proteins WHERE amino_acid_sequence IS NOT NULL"
    ).fetchall()]
    conn.close()
    return sorted(accs)


def generate_uniform_random_negatives(
    db_path: str | Path,
    positive_pairs: set[tuple[str, str]],
    n_samples: int,
    seed: int = 42,
) -> set[tuple[str, str]]:
    """Generate uniform random negative pairs from sequenced proteins.

    Excludes all NegBioDB negatives, HuRI positives, and pairs with NULL sequences.
    """
    tested = _load_tested_pairs(db_path, positive_pairs)
    proteins = _get_sequenced_proteins(db_path)

    rng = np.random.RandomState(seed)
    n_proteins = len(proteins)
    generated = set()

    attempts = 0
    max_attempts = n_samples * 20

    while len(generated) < n_samples and attempts < max_attempts:
        batch_size = min((n_samples - len(generated)) * 3, 1_000_000)
        idx1 = rng.randint(0, n_proteins, size=batch_size)
        idx2 = rng.randint(0, n_proteins, size=batch_size)

        for i in range(batch_size):
            if idx1[i] == idx2[i]:
                continue
            a, b = proteins[idx1[i]], proteins[idx2[i]]
            pair = (a, b) if a < b else (b, a)
            if pair not in tested and pair not in generated:
                generated.add(pair)
                if len(generated) >= n_samples:
                    break
        attempts += batch_size

    logger.info("Generated %d uniform random negatives (%d attempts)",
                len(generated), attempts)
    return generated


def generate_degree_matched_negatives(
    db_path: str | Path,
    positive_pairs: set[tuple[str, str]],
    n_samples: int,
    seed: int = 42,
) -> set[tuple[str, str]]:
    """Generate degree-matched negative pairs.

    Samples proportional to protein degree (number of known interactions).
    """
    tested = _load_tested_pairs(db_path, positive_pairs)

    conn = sqlite3.connect(str(db_path))
    # Compute degree per protein (from both sides of pairs table)
    degree = {}
    for r in conn.execute(
        """SELECT protein_id, deg FROM (
            SELECT protein_id, COUNT(DISTINCT partner_id) AS deg FROM (
                SELECT protein1_id AS protein_id, protein2_id AS partner_id
                FROM protein_protein_pairs
                UNION ALL
                SELECT protein2_id, protein1_id
                FROM protein_protein_pairs
            ) GROUP BY protein_id
        )"""
    ).fetchall():
        degree[r[0]] = r[1]

    # Map protein_id → accession, filter to sequenced
    pid_to_acc = {}
    for r in conn.execute(
        "SELECT protein_id, uniprot_accession FROM proteins WHERE amino_acid_sequence IS NOT NULL"
    ).fetchall():
        pid_to_acc[r[0]] = r[1]
    conn.close()

    # Build weighted sampling array
    pids = [pid for pid in pid_to_acc if pid in degree]
    accs = [pid_to_acc[pid] for pid in pids]
    weights = np.array([degree[pid] for pid in pids], dtype=np.float64)
    weights /= weights.sum()

    rng = np.random.RandomState(seed)
    generated = set()
    attempts = 0
    max_attempts = n_samples * 20

    while len(generated) < n_samples and attempts < max_attempts:
        batch_size = min((n_samples - len(generated)) * 3, 1_000_000)
        idx1 = rng.choice(len(accs), size=batch_size, p=weights)
        idx2 = rng.choice(len(accs), size=batch_size, p=weights)

        for i in range(batch_size):
            if idx1[i] == idx2[i]:
                continue
            a, b = accs[idx1[i]], accs[idx2[i]]
            pair = (a, b) if a < b else (b, a)
            if pair not in tested and pair not in generated:
                generated.add(pair)
                if len(generated) >= n_samples:
                    break
        attempts += batch_size

    logger.info("Generated %d degree-matched negatives (%d attempts)",
                len(generated), attempts)
    return generated


def control_pairs_to_df(
    pairs: set[tuple[str, str]],
    db_path: str | Path,
) -> pd.DataFrame:
    """Convert control negative pairs to DataFrame with sequences from DB."""
    conn = sqlite3.connect(str(db_path))
    proteins = {}
    for row in conn.execute(
        """SELECT uniprot_accession, amino_acid_sequence,
                  gene_symbol, subcellular_location
        FROM proteins WHERE amino_acid_sequence IS NOT NULL"""
    ).fetchall():
        proteins[row[0]] = {
            "sequence": row[1],
            "gene_symbol": row[2],
            "subcellular_location": row[3],
        }
    conn.close()

    records = []
    for acc1, acc2 in pairs:
        p1 = proteins.get(acc1)
        p2 = proteins.get(acc2)
        if not p1 or not p2:
            continue
        records.append({
            "pair_id": None,
            "uniprot_id_1": acc1,
            "sequence_1": p1["sequence"],
            "gene_symbol_1": p1["gene_symbol"],
            "subcellular_location_1": p1["subcellular_location"],
            "uniprot_id_2": acc2,
            "sequence_2": p2["sequence"],
            "gene_symbol_2": p2["gene_symbol"],
            "subcellular_location_2": p2["subcellular_location"],
            "Y": 0,
            "confidence_tier": None,
            "num_sources": None,
            "protein1_degree": None,
            "protein2_degree": None,
        })

    return pd.DataFrame(records)


# ------------------------------------------------------------------
# DataFrame-level split helpers (for merged pos+neg)
# ------------------------------------------------------------------

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


def add_cold_protein_split(
    df: pd.DataFrame,
    seed: int = 42,
    protein_ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add split_cold_protein column — group by protein identity.

    Collects unique proteins from UNION of uniprot_id_1 and uniprot_id_2,
    assigns each to a fold, then pair fold = max(fold_P1, fold_P2).
    """
    if protein_ratios is None:
        protein_ratios = {"train": 0.80, "val": 0.05, "test": 0.15}
    if len(df) == 0:
        df = df.copy()
        df["split_cold_protein"] = pd.Series(dtype=str)
        return df

    # Collect all unique proteins from both columns
    all_proteins = np.array(sorted(
        set(df["uniprot_id_1"].unique()) | set(df["uniprot_id_2"].unique())
    ))
    rng = np.random.RandomState(seed)
    rng.shuffle(all_proteins)

    n = len(all_proteins)
    n_train = int(n * protein_ratios["train"])
    n_val = int(n * protein_ratios["val"])

    protein_fold = {}
    for i, p in enumerate(all_proteins):
        if i < n_train:
            protein_fold[p] = "train"
        elif i < n_train + n_val:
            protein_fold[p] = "val"
        else:
            protein_fold[p] = "test"

    fold_priority = {"train": 0, "val": 1, "test": 2}
    fold_names = ["train", "val", "test"]

    fold1 = df["uniprot_id_1"].map(protein_fold).map(fold_priority)
    fold2 = df["uniprot_id_2"].map(protein_fold).map(fold_priority)
    max_fold = np.maximum(fold1.to_numpy(), fold2.to_numpy())

    df = df.copy()
    df["split_cold_protein"] = [fold_names[int(f)] for f in max_fold]
    return df


def add_cold_both_partition_split(
    df: pd.DataFrame,
    seed: int = 42,
    nparts: int = 10,
) -> pd.DataFrame:
    """Add split_cold_both column via Metis partitioning.

    Cross-partition pairs get NaN (excluded).
    """
    import pymetis

    if len(df) == 0:
        df = df.copy()
        df["split_cold_both"] = pd.Series(dtype=str)
        return df

    # Build adjacency from both columns
    all_proteins = sorted(
        set(df["uniprot_id_1"].unique()) | set(df["uniprot_id_2"].unique())
    )
    pid_to_idx = {p: i for i, p in enumerate(all_proteins)}

    adjacency = [[] for _ in range(len(all_proteins))]
    for _, row in df[["uniprot_id_1", "uniprot_id_2"]].drop_duplicates().iterrows():
        i = pid_to_idx[row["uniprot_id_1"]]
        j = pid_to_idx[row["uniprot_id_2"]]
        adjacency[i].append(j)
        adjacency[j].append(i)

    opts = pymetis.Options(seed=seed)
    n_cuts, membership = pymetis.part_graph(nparts, adjacency=adjacency, options=opts)

    n_train_parts = int(nparts * 0.7)
    n_val_parts = int(nparts * 0.1)

    rng = np.random.RandomState(seed)
    part_labels = list(range(nparts))
    rng.shuffle(part_labels)

    part_to_fold = {}
    for i, pl in enumerate(part_labels):
        if i < n_train_parts:
            part_to_fold[pl] = "train"
        elif i < n_train_parts + n_val_parts:
            part_to_fold[pl] = "val"
        else:
            part_to_fold[pl] = "test"

    protein_fold = {p: part_to_fold[membership[idx]] for p, idx in pid_to_idx.items()}

    fold1 = df["uniprot_id_1"].map(protein_fold)
    fold2 = df["uniprot_id_2"].map(protein_fold)

    df = df.copy()
    # Same fold → that fold; different → NaN (excluded)
    df["split_cold_both"] = np.where(fold1 == fold2, fold1, None)
    return df


def add_degree_balanced_split(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Add split_degree_balanced using 1D log(degree_sum) quantile bins.

    For merged datasets, degree is computed from the merged graph
    (value_counts of each protein across both columns).
    """
    if ratios is None:
        ratios = _DEFAULT_RATIOS
    if len(df) == 0:
        df = df.copy()
        df["split_degree_balanced"] = pd.Series(dtype=str)
        return df

    df = df.copy()

    # Compute degree from merged graph
    all_ids = pd.concat([df["uniprot_id_1"], df["uniprot_id_2"]])
    degree_map = all_ids.value_counts()

    deg1 = df["uniprot_id_1"].map(degree_map).to_numpy(dtype=np.float64)
    deg2 = df["uniprot_id_2"].map(degree_map).to_numpy(dtype=np.float64)

    degree_sum = deg1 + deg2
    log_deg = np.log1p(degree_sum)

    bin_edges = np.quantile(log_deg, np.linspace(0, 1, n_bins + 1))
    bin_labels = np.digitize(log_deg, bin_edges[1:-1])

    rng = np.random.RandomState(seed)
    fold_labels = np.empty(len(df), dtype=object)

    for bin_id in np.unique(bin_labels):
        idx = np.where(bin_labels == bin_id)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        fold_labels[idx[:n_train]] = "train"
        fold_labels[idx[n_train:n_train + n_val]] = "val"
        fold_labels[idx[n_train + n_val:]] = "test"

    df["split_degree_balanced"] = fold_labels
    return df


def apply_ppi_m1_splits(
    df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply all PPI M1 split strategies to a DataFrame.

    Adds: split_random, split_cold_protein, split_cold_both, split_degree_balanced.
    """
    df = add_random_split(df, seed=seed)
    df = add_cold_protein_split(df, seed=seed)
    df = add_cold_both_partition_split(df, seed=seed)
    df = add_degree_balanced_split(df, seed=seed)
    return df


# ------------------------------------------------------------------
# M1 dataset builders
# ------------------------------------------------------------------

def _recompute_pair_degrees(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute protein1_degree and protein2_degree from the merged graph.

    Avoids pos/neg asymmetry: DB negatives have degree from the pairs table,
    but positives and control negatives have None.  After merge, degree is
    defined uniformly as the number of rows each protein appears in.
    """
    all_ids = pd.concat([df["uniprot_id_1"], df["uniprot_id_2"]])
    degree_map = all_ids.value_counts()
    df["protein1_degree"] = df["uniprot_id_1"].map(degree_map).astype(float)
    df["protein2_degree"] = df["uniprot_id_2"].map(degree_map).astype(float)
    return df


def build_m1_balanced(
    neg_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Build M1 balanced dataset (1:1 pos/neg ratio).

    Samples negatives to match positive count, then merges and applies splits.
    """
    n_pos = len(pos_df)
    if len(neg_df) <= n_pos:
        sampled_neg = neg_df
    else:
        sampled_neg = neg_df.sample(n=n_pos, random_state=seed).reset_index(drop=True)

    df = pd.concat([pos_df, sampled_neg], ignore_index=True)
    df = _recompute_pair_degrees(df)
    df = apply_ppi_m1_splits(df, seed=seed)
    logger.info("M1 balanced: %d rows (%d pos, %d neg)",
                len(df), n_pos, len(sampled_neg))
    return df


def build_m1_realistic(
    neg_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    ratio: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Build M1 realistic dataset (1:ratio pos/neg ratio).

    Samples negatives at ratio×positive count, then merges and applies splits.
    """
    n_pos = len(pos_df)
    n_neg_target = n_pos * ratio
    if len(neg_df) <= n_neg_target:
        sampled_neg = neg_df
    else:
        sampled_neg = neg_df.sample(n=n_neg_target, random_state=seed).reset_index(drop=True)

    df = pd.concat([pos_df, sampled_neg], ignore_index=True)
    df = _recompute_pair_degrees(df)
    df = apply_ppi_m1_splits(df, seed=seed)
    logger.info("M1 realistic (1:%d): %d rows (%d pos, %d neg)",
                ratio, len(df), n_pos, len(sampled_neg))
    return df
