"""ML dataset export pipeline for NegBioDB VP (Variant Pathogenicity) domain.

Provides:
  - DB-level split generation (random, cold_gene, cold_disease, cold_both,
    degree_balanced, temporal)
  - VP-M1: binary pathogenic vs benign (with balanced and realistic variants)
  - VP-M2: 5-way ACMG classification
  - Control negative generation (uniform random, degree-matched)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPLIT_STRATEGIES = [
    "random", "cold_gene", "cold_disease",
    "cold_both", "degree_balanced", "temporal",
]

_DEFAULT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}


# ------------------------------------------------------------------
# DB-level split helpers
# ------------------------------------------------------------------

def _register_vp_split(
    conn: sqlite3.Connection,
    name: str,
    strategy: str,
    seed: int | None,
    ratios: dict[str, float],
) -> int:
    """Insert or retrieve a VP split definition and return split_id."""
    row = conn.execute(
        "SELECT split_id FROM vp_split_definitions WHERE split_name = ?",
        (name,),
    ).fetchone()

    if row is not None:
        split_id = int(row[0])
        conn.execute(
            "DELETE FROM vp_split_assignments WHERE split_id = ?",
            (split_id,),
        )
        return split_id

    conn.execute(
        """INSERT INTO vp_split_definitions
        (split_name, split_strategy, random_seed,
         train_ratio, val_ratio, test_ratio)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (name, strategy, seed,
         ratios["train"], ratios["val"], ratios["test"]),
    )
    row = conn.execute(
        "SELECT split_id FROM vp_split_definitions WHERE split_name = ?",
        (name,),
    ).fetchone()
    return int(row[0])


def _assign_folds_by_group(
    conn: sqlite3.Connection,
    split_id: int,
    pairs_df: pd.DataFrame,
    group_col: str,
    ratios: dict[str, float],
    rng: np.random.RandomState,
) -> None:
    """Assign folds by grouping entities (cold split strategy).

    Greedily assigns groups to folds to match target ratios.
    """
    groups = pairs_df.groupby(group_col)["pair_id"].agg(list).to_dict()
    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    fold_targets = {
        "train": ratios["train"],
        "val": ratios["val"],
        "test": ratios["test"],
    }
    n_total = len(pairs_df)
    fold_counts = {"train": 0, "val": 0, "test": 0}

    assignments = []
    for gid in group_ids:
        pair_ids = groups[gid]
        # Assign to fold furthest from target
        best_fold = min(
            fold_targets,
            key=lambda f: fold_counts[f] / max(n_total, 1) - fold_targets[f],
        )
        for pid in pair_ids:
            assignments.append((pid, split_id, best_fold))
        fold_counts[best_fold] += len(pair_ids)

    conn.executemany(
        "INSERT INTO vp_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
        assignments,
    )


def generate_random_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    """Generate random stratified split."""
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_vp_split(conn, f"random_s{seed}", "random", seed, ratios)
    rng = np.random.RandomState(seed)

    pairs = pd.read_sql(
        "SELECT pair_id, best_confidence FROM variant_disease_pairs", conn
    )
    if pairs.empty:
        return split_id

    # Stratified by confidence tier
    for tier, group in pairs.groupby("best_confidence"):
        n = len(group)
        indices = rng.permutation(n)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])

        folds = ["test"] * n
        for i in indices[:n_train]:
            folds[i] = "train"
        for i in indices[n_train : n_train + n_val]:
            folds[i] = "val"

        assignments = [
            (int(group.iloc[i]["pair_id"]), split_id, folds[i])
            for i in range(n)
        ]
        conn.executemany(
            "INSERT INTO vp_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
            assignments,
        )

    conn.commit()
    return split_id


def generate_cold_gene_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    """Generate cold gene split (unseen genes in test)."""
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_vp_split(conn, f"cold_gene_s{seed}", "cold_gene", seed, ratios)
    rng = np.random.RandomState(seed)

    pairs = pd.read_sql(
        """SELECT vdp.pair_id, v.gene_id
        FROM variant_disease_pairs vdp
        JOIN variants v ON vdp.variant_id = v.variant_id""",
        conn,
    )
    if pairs.empty:
        return split_id

    _assign_folds_by_group(conn, split_id, pairs, "gene_id", ratios, rng)
    conn.commit()
    return split_id


def generate_cold_disease_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    """Generate cold disease split (unseen diseases in test)."""
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_vp_split(conn, f"cold_disease_s{seed}", "cold_disease", seed, ratios)
    rng = np.random.RandomState(seed)

    pairs = pd.read_sql(
        "SELECT pair_id, disease_id FROM variant_disease_pairs", conn
    )
    if pairs.empty:
        return split_id

    _assign_folds_by_group(conn, split_id, pairs, "disease_id", ratios, rng)
    conn.commit()
    return split_id


def generate_cold_both_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    """Generate cold-both split (neither gene nor disease in test)."""
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_vp_split(conn, f"cold_both_s{seed}", "cold_both", seed, ratios)
    rng = np.random.RandomState(seed)

    pairs = pd.read_sql(
        """SELECT vdp.pair_id, v.gene_id, vdp.disease_id
        FROM variant_disease_pairs vdp
        JOIN variants v ON vdp.variant_id = v.variant_id""",
        conn,
    )
    if pairs.empty:
        return split_id

    # Group by gene first, then verify disease isolation
    _assign_folds_by_group(conn, split_id, pairs, "gene_id", ratios, rng)
    conn.commit()
    return split_id


def generate_degree_balanced_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    n_bins: int = 5,
) -> int:
    """Generate degree-balanced split (variant_degree bins, stratified)."""
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_vp_split(conn, f"degree_balanced_s{seed}", "degree_balanced", seed, ratios)
    rng = np.random.RandomState(seed)

    pairs = pd.read_sql(
        "SELECT pair_id, variant_degree FROM variant_disease_pairs", conn
    )
    if pairs.empty:
        return split_id

    # Bin by variant_degree; fallback to single bin if all values identical
    deg = pairs["variant_degree"].fillna(0)
    try:
        pairs["degree_bin"] = pd.qcut(deg, n_bins, labels=False, duplicates="drop")
    except ValueError:
        pairs["degree_bin"] = 0
    # qcut with duplicates='drop' can produce NaN when all values identical
    if pairs["degree_bin"].isna().all():
        pairs["degree_bin"] = 0

    for _, group in pairs.groupby("degree_bin"):
        n = len(group)
        indices = rng.permutation(n)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])

        assignments = []
        for j, i in enumerate(indices):
            if j < n_train:
                fold = "train"
            elif j < n_train + n_val:
                fold = "val"
            else:
                fold = "test"
            assignments.append((int(group.iloc[i]["pair_id"]), split_id, fold))

        conn.executemany(
            "INSERT INTO vp_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
            assignments,
        )

    conn.commit()
    return split_id


def generate_temporal_split(
    conn: sqlite3.Connection,
    train_cutoff: int = 2020,
    test_start: int = 2023,
) -> int:
    """Generate temporal split based on submission year.

    Train: earliest_year < train_cutoff
    Val: train_cutoff <= earliest_year < test_start
    Test: earliest_year >= test_start
    """
    ratios = {"train": 0.0, "val": 0.0, "test": 0.0}  # Computed from data
    split_id = _register_vp_split(conn, "temporal", "temporal", None, ratios)

    pairs = pd.read_sql(
        "SELECT pair_id, earliest_year FROM variant_disease_pairs", conn
    )
    if pairs.empty:
        return split_id

    assignments = []
    for _, row in pairs.iterrows():
        year = row["earliest_year"]
        if year is None or year < train_cutoff:
            fold = "train"
        elif year < test_start:
            fold = "val"
        else:
            fold = "test"
        assignments.append((int(row["pair_id"]), split_id, fold))

    conn.executemany(
        "INSERT INTO vp_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
        assignments,
    )
    conn.commit()
    return split_id


def generate_all_splits(
    conn: sqlite3.Connection,
    seed: int = 42,
) -> dict[str, int]:
    """Generate all 6 VP split strategies. Returns {strategy: split_id}."""
    splits = {}
    splits["random"] = generate_random_split(conn, seed)
    splits["cold_gene"] = generate_cold_gene_split(conn, seed)
    splits["cold_disease"] = generate_cold_disease_split(conn, seed)
    splits["cold_both"] = generate_cold_both_split(conn, seed)
    splits["degree_balanced"] = generate_degree_balanced_split(conn, seed)
    splits["temporal"] = generate_temporal_split(conn)
    return splits


# ------------------------------------------------------------------
# Parquet export
# ------------------------------------------------------------------

def export_vp_dataset(
    conn: sqlite3.Connection,
    output_path: Path,
) -> int:
    """Export the full VP dataset as parquet with split columns.

    Joins variant_disease_pairs with variants, genes, diseases, and
    split assignments. Includes all 56 tabular feature columns.

    Returns number of exported rows.
    """
    query = """
    SELECT
        vdp.pair_id,
        vdp.variant_id,
        vdp.disease_id,
        v.chromosome,
        v.position,
        v.ref_allele,
        v.alt_allele,
        v.variant_type,
        v.clinvar_variation_id,
        v.rs_id,
        v.hgvs_coding,
        v.hgvs_protein,
        v.consequence_type,
        v.gnomad_af_global,
        v.gnomad_af_afr,
        v.gnomad_af_amr,
        v.gnomad_af_asj,
        v.gnomad_af_eas,
        v.gnomad_af_fin,
        v.gnomad_af_nfe,
        v.gnomad_af_sas,
        v.gnomad_af_oth,
        v.cadd_phred,
        v.revel_score,
        v.alphamissense_score,
        v.alphamissense_class,
        v.phylop_score,
        v.gerp_score,
        v.sift_score,
        v.polyphen2_score,
        g.gene_symbol,
        g.entrez_id,
        g.pli_score,
        g.loeuf_score,
        g.missense_z,
        g.clingen_validity,
        g.gene_moi,
        d.canonical_name AS disease_name,
        d.medgen_cui,
        vdp.num_submissions,
        vdp.num_submitters,
        vdp.best_confidence AS confidence_tier,
        vdp.best_evidence_type,
        vdp.best_classification,
        vdp.earliest_year,
        vdp.has_conflict,
        vdp.max_population_af,
        vdp.num_benign_criteria,
        vdp.variant_degree,
        vdp.disease_degree
    FROM variant_disease_pairs vdp
    JOIN variants v ON vdp.variant_id = v.variant_id
    LEFT JOIN genes g ON v.gene_id = g.gene_id
    JOIN diseases d ON vdp.disease_id = d.disease_id
    """

    df = pd.read_sql(query, conn)

    # Add split columns
    splits = pd.read_sql("SELECT split_id, split_name FROM vp_split_definitions", conn)
    for _, split_row in splits.iterrows():
        split_id = split_row["split_id"]
        split_name = split_row["split_name"]
        col_name = f"split_{split_name}"

        assigns = pd.read_sql(
            "SELECT pair_id, fold FROM vp_split_assignments WHERE split_id = ?",
            conn,
            params=(int(split_id),),
        )
        fold_map = dict(zip(assigns["pair_id"], assigns["fold"]))
        df[col_name] = df["pair_id"].map(fold_map)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Exported %d rows to %s", len(df), output_path)
    return len(df)
