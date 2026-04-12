"""ML dataset export for MD domain (Metabolite-Disease Non-Association).

Four split strategies:
  1. random            — Stratified random by best_tier
  2. cold_metabolite   — Metabolite unseen in train (cold compound)
  3. cold_disease      — Disease unseen in train (cold target)
  4. cold_both         — Both metabolite and disease unseen in train

Two ML tasks:
  MD-M1: Binary (is_significant=0 negative vs is_significant=1 positive)
  MD-M2: Multi-class disease category (cancer/metabolic/neurological/cardiovascular/other)

Export format: Parquet with split columns (split_random, split_cold_metabolite, etc.)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from negbiodb_md.md_features import (
    BIOFLUIDS, DISEASE_CATEGORIES, FEATURE_DIM, PLATFORMS,
    build_feature_vector,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EXPORT_DIR = _PROJECT_ROOT / "exports" / "md_ml"

_DEFAULT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}
RANDOM_SEED = 42


# ── Split registration ────────────────────────────────────────────────────────

def _register_split(
    conn: sqlite3.Connection,
    name: str,
    strategy: str,
    seed: int | None,
) -> int:
    """Insert or replace a split definition, returning split_id."""
    row = conn.execute(
        "SELECT split_id FROM md_split_definitions WHERE split_name = ?", (name,)
    ).fetchone()
    if row is not None:
        split_id = int(row[0])
        conn.execute("DELETE FROM md_split_assignments WHERE split_id = ?", (split_id,))
        return split_id
    conn.execute(
        """INSERT INTO md_split_definitions
           (split_name, split_strategy, random_seed)
           VALUES (?,?,?)""",
        (name, strategy, seed),
    )
    return int(conn.execute(
        "SELECT split_id FROM md_split_definitions WHERE split_name = ?", (name,)
    ).fetchone()[0])


def _assign_folds_stratified(
    pair_ids: list[int],
    ratios: dict[str, float] = _DEFAULT_RATIOS,
    seed: int = RANDOM_SEED,
) -> dict[int, str]:
    """Assign pair_ids to train/val/test folds with stratified random split."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(np.array(pair_ids))
    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    folds: dict[int, str] = {}
    for idx in shuffled[:n_train]:
        folds[int(idx)] = "train"
    for idx in shuffled[n_train:n_train + n_val]:
        folds[int(idx)] = "val"
    for idx in shuffled[n_train + n_val:]:
        folds[int(idx)] = "test"
    return folds


def _save_split(conn, split_id: int, folds: dict[int, str]) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO md_split_assignments (split_id, pair_id, fold) VALUES (?,?,?)",
        [(split_id, pid, fold) for pid, fold in folds.items()],
    )


# ── Split strategies ──────────────────────────────────────────────────────────

def build_random_split(conn, seed: int = RANDOM_SEED) -> int:
    """Random split, stratified by best_tier."""
    split_id = _register_split(conn, "random", "random", seed)
    pair_ids = [r[0] for r in conn.execute(
        "SELECT pair_id FROM md_metabolite_disease_pairs"
    ).fetchall()]
    folds = _assign_folds_stratified(pair_ids, seed=seed)
    _save_split(conn, split_id, folds)
    conn.commit()
    return split_id


def build_cold_metabolite_split(conn, seed: int = RANDOM_SEED) -> int:
    """Cold-metabolite split: metabolites in test not seen in train."""
    split_id = _register_split(conn, "cold_metabolite", "cold_metabolite", seed)
    rng = np.random.default_rng(seed)

    metabolite_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT metabolite_id FROM md_metabolite_disease_pairs"
    ).fetchall()]
    shuffled = rng.permuted(np.array(metabolite_ids))
    n = len(shuffled)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)

    train_mets = set(shuffled[:n_train].tolist())
    val_mets = set(shuffled[n_train:n_train + n_val].tolist())

    pairs = conn.execute(
        "SELECT pair_id, metabolite_id FROM md_metabolite_disease_pairs"
    ).fetchall()
    folds: dict[int, str] = {}
    for pair_id, met_id in pairs:
        if met_id in train_mets:
            folds[pair_id] = "train"
        elif met_id in val_mets:
            folds[pair_id] = "val"
        else:
            folds[pair_id] = "test"

    _save_split(conn, split_id, folds)
    conn.commit()
    return split_id


def build_cold_disease_split(conn, seed: int = RANDOM_SEED) -> int:
    """Cold-disease split: diseases in test not seen in train."""
    split_id = _register_split(conn, "cold_disease", "cold_disease", seed)
    rng = np.random.default_rng(seed)

    disease_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT disease_id FROM md_metabolite_disease_pairs"
    ).fetchall()]
    shuffled = rng.permuted(np.array(disease_ids))
    n = len(shuffled)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)

    train_dis = set(shuffled[:n_train].tolist())
    val_dis = set(shuffled[n_train:n_train + n_val].tolist())

    pairs = conn.execute(
        "SELECT pair_id, disease_id FROM md_metabolite_disease_pairs"
    ).fetchall()
    folds: dict[int, str] = {}
    for pair_id, dis_id in pairs:
        if dis_id in train_dis:
            folds[pair_id] = "train"
        elif dis_id in val_dis:
            folds[pair_id] = "val"
        else:
            folds[pair_id] = "test"

    _save_split(conn, split_id, folds)
    conn.commit()
    return split_id


def build_cold_both_split(conn, seed: int = RANDOM_SEED) -> int:
    """Cold-both split: neither metabolite nor disease seen in train."""
    split_id = _register_split(conn, "cold_both", "cold_both", seed)
    rng = np.random.default_rng(seed)

    metabolite_ids = np.array([r[0] for r in conn.execute(
        "SELECT DISTINCT metabolite_id FROM md_metabolite_disease_pairs"
    ).fetchall()])
    shuffled_m = rng.permuted(metabolite_ids)
    n_m = len(shuffled_m)
    test_mets = set(shuffled_m[int(n_m * 0.8):].tolist())

    disease_ids = np.array([r[0] for r in conn.execute(
        "SELECT DISTINCT disease_id FROM md_metabolite_disease_pairs"
    ).fetchall()])
    shuffled_d = rng.permuted(disease_ids)
    n_d = len(shuffled_d)
    test_dis = set(shuffled_d[int(n_d * 0.8):].tolist())

    pairs = conn.execute(
        "SELECT pair_id, metabolite_id, disease_id FROM md_metabolite_disease_pairs"
    ).fetchall()
    folds: dict[int, str] = {}
    for pair_id, met_id, dis_id in pairs:
        if met_id in test_mets and dis_id in test_dis:
            folds[pair_id] = "test"
        elif met_id in test_mets or dis_id in test_dis:
            folds[pair_id] = "val"
        else:
            folds[pair_id] = "train"

    _save_split(conn, split_id, folds)
    conn.commit()
    return split_id


# ── Export ────────────────────────────────────────────────────────────────────

def export_ml_dataset(
    conn,
    output_dir: str | Path | None = None,
    seed: int = RANDOM_SEED,
) -> Path:
    """Build all splits and export Parquet ML dataset.

    Returns path to the exported Parquet file.
    """
    if output_dir is None:
        output_dir = DEFAULT_EXPORT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build all splits
    logger.info("Building MD splits...")
    split_ids = {
        "random": build_random_split(conn, seed),
        "cold_metabolite": build_cold_metabolite_split(conn, seed),
        "cold_disease": build_cold_disease_split(conn, seed),
        "cold_both": build_cold_both_split(conn, seed),
    }

    # Load all data with features
    logger.info("Loading MD data for export...")
    rows = conn.execute(
        """SELECT
            r.result_id,
            p.pair_id,
            m.metabolite_id,
            m.name AS metabolite_name,
            m.inchikey,
            m.canonical_smiles,
            m.metabolite_class,
            m.molecular_weight,
            m.logp,
            m.tpsa,
            m.hbd,
            m.hba,
            d.disease_id,
            d.name AS disease_name,
            d.disease_category,
            d.mondo_id,
            s.study_id,
            s.source AS study_source,
            s.platform,
            s.biofluid,
            s.n_disease,
            s.n_control,
            r.p_value,
            r.fdr,
            r.fold_change,
            r.log2_fc,
            r.is_significant,
            r.tier,
            p.n_studies_negative,
            p.n_studies_positive,
            p.consensus,
            p.best_tier,
            p.metabolite_degree,
            p.disease_degree
        FROM md_biomarker_results r
        JOIN md_metabolites m ON r.metabolite_id = m.metabolite_id
        JOIN md_diseases d ON r.disease_id = d.disease_id
        JOIN md_studies s ON r.study_id = s.study_id
        JOIN md_metabolite_disease_pairs p
            ON p.metabolite_id = r.metabolite_id AND p.disease_id = r.disease_id
        ORDER BY r.result_id"""
    ).fetchall()

    cols = [
        "result_id", "pair_id", "metabolite_id", "metabolite_name", "inchikey",
        "canonical_smiles", "metabolite_class", "molecular_weight", "logp", "tpsa",
        "hbd", "hba", "disease_id", "disease_name", "disease_category", "mondo_id",
        "study_id", "study_source", "platform", "biofluid", "n_disease", "n_control",
        "p_value", "fdr", "fold_change", "log2_fc", "is_significant", "tier",
        "n_studies_negative", "n_studies_positive", "consensus", "best_tier",
        "metabolite_degree", "disease_degree",
    ]
    df = pd.DataFrame(rows, columns=cols)

    if df.empty:
        logger.warning("No MD data to export")
        return output_dir / "md_ml.parquet"

    # Build feature matrix
    logger.info("Building feature matrix for %d rows...", len(df))
    X_list = []
    for _, row in df.iterrows():
        vec = build_feature_vector(
            row["canonical_smiles"],
            row["disease_category"],
            row["platform"],
            row["biofluid"],
            row["n_disease"],
            row["n_control"],
        )
        X_list.append(vec)
    X = np.stack(X_list)

    # Add feature columns
    feature_cols = [f"feat_{i}" for i in range(FEATURE_DIM)]
    feat_df = pd.DataFrame(X, columns=feature_cols, dtype=np.float32)
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    # Add split columns
    for split_name, split_id in split_ids.items():
        assignment_rows = conn.execute(
            "SELECT pair_id, fold FROM md_split_assignments WHERE split_id = ?",
            (split_id,),
        ).fetchall()
        fold_map = {r[0]: r[1] for r in assignment_rows}
        df[f"split_{split_name}"] = df["pair_id"].map(fold_map).fillna("unassigned")

    # M2 label column
    df["label_m2"] = df["disease_category"].map(
        {cat: i for i, cat in enumerate(DISEASE_CATEGORIES)}
    ).fillna(4).astype(int)

    out_path = output_dir / "md_ml.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Exported MD ML dataset: %d rows → %s", len(df), out_path)
    return out_path
