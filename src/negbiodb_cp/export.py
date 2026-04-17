"""ML export pipeline for the NegBioDB Cell Painting domain."""

from __future__ import annotations

import json
import logging
import sqlite3
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CP_SPLIT_STRATEGIES = ["random", "cold_compound", "scaffold", "batch_holdout"]
CP_M2_LABEL_TO_INT = {
    "inactive": 0,
    "weak_phenotype": 1,
    "strong_phenotype": 2,
    "toxic_or_artifact": 3,
}

_DEFAULT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}


def _register_cp_split(
    conn: sqlite3.Connection,
    name: str,
    strategy: str,
    seed: int | None,
    ratios: dict[str, float],
) -> int:
    row = conn.execute(
        "SELECT split_id FROM cp_split_definitions WHERE split_name = ?",
        (name,),
    ).fetchone()
    if row is not None:
        split_id = int(row[0])
        conn.execute("DELETE FROM cp_split_assignments WHERE split_id = ?", (split_id,))
        return split_id

    conn.execute(
        """INSERT INTO cp_split_definitions
        (split_name, split_strategy, random_seed, train_ratio, val_ratio, test_ratio)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (name, strategy, seed, ratios["train"], ratios["val"], ratios["test"]),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _load_result_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            r.cp_result_id,
            r.compound_id,
            r.cell_line_id,
            r.assay_context_id,
            r.batch_id,
            r.dose,
            r.dose_unit,
            r.timepoint_h,
            r.num_observations,
            r.num_valid_observations,
            r.dmso_distance_mean,
            r.replicate_reproducibility,
            r.viability_ratio,
            r.outcome_label,
            r.confidence_tier,
            r.has_orthogonal_evidence,
            c.compound_name,
            c.canonical_smiles,
            c.inchikey,
            c.inchikey_connectivity,
            cl.cell_line_name,
            cl.tissue,
            cl.disease,
            b.batch_name
        FROM cp_perturbation_results r
        JOIN compounds c ON r.compound_id = c.compound_id
        JOIN cp_cell_lines cl ON r.cell_line_id = cl.cell_line_id
        JOIN cp_batches b ON r.batch_id = b.batch_id
        ORDER BY r.cp_result_id
        """,
        conn,
    )


def _assign_folds_by_group(
    conn: sqlite3.Connection,
    split_id: int,
    frame: pd.DataFrame,
    group_col: str,
    ratios: dict[str, float],
    rng: np.random.RandomState,
) -> None:
    groups = frame.groupby(group_col)["cp_result_id"].agg(list).to_dict()
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    group_ids = [item[0] for item in group_items]

    fold_targets = {
        "train": ratios["train"],
        "val": ratios["val"],
        "test": ratios["test"],
    }
    n_total = len(frame)
    fold_counts = {"train": 0, "val": 0, "test": 0}
    assignments = []

    if len(group_ids) >= 3:
        seeded_folds = ["train", "val", "test"]
    elif len(group_ids) == 2:
        seeded_folds = ["train", "test"]
    else:
        seeded_folds = ["train"]

    seeded_groups = group_ids[:len(seeded_folds)]
    remaining_groups = group_ids[len(seeded_folds):]

    for gid, fold in zip(seeded_groups, seeded_folds):
        result_ids = groups[gid]
        for cp_result_id in result_ids:
            assignments.append((int(cp_result_id), split_id, fold))
        fold_counts[fold] += len(result_ids)

    for gid in remaining_groups:
        result_ids = groups[gid]
        best_fold = min(
            fold_targets,
            key=lambda fold: fold_counts[fold] / max(n_total, 1) - fold_targets[fold],
        )
        for cp_result_id in result_ids:
            assignments.append((int(cp_result_id), split_id, best_fold))
        fold_counts[best_fold] += len(result_ids)

    conn.executemany(
        """INSERT INTO cp_split_assignments (cp_result_id, split_id, fold)
        VALUES (?, ?, ?)""",
        assignments,
    )


def generate_random_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_cp_split(conn, f"random_s{seed}", "random", seed, ratios)
    frame = _load_result_frame(conn)
    if frame.empty:
        conn.commit()
        return split_id

    rng = np.random.RandomState(seed)
    assignments = []
    for label, group in frame.groupby("outcome_label"):
        ids = group["cp_result_id"].to_numpy(dtype=int).copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        for i, cp_result_id in enumerate(ids):
            fold = "test"
            if i < n_train:
                fold = "train"
            elif i < n_train + n_val:
                fold = "val"
            assignments.append((int(cp_result_id), split_id, fold))
        logger.info("CP random split label=%s count=%d", label, n)

    conn.executemany(
        "INSERT INTO cp_split_assignments (cp_result_id, split_id, fold) VALUES (?, ?, ?)",
        assignments,
    )
    conn.commit()
    return split_id


def generate_cold_compound_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_cp_split(
        conn, f"cold_compound_s{seed}", "cold_compound", seed, ratios
    )
    frame = _load_result_frame(conn)
    if frame.empty:
        conn.commit()
        return split_id

    rng = np.random.RandomState(seed)
    _assign_folds_by_group(conn, split_id, frame, "compound_id", ratios, rng)
    conn.commit()
    return split_id


def _compute_scaffold_map(frame: pd.DataFrame) -> dict[int, str]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
    except ModuleNotFoundError:
        return {
            int(row["compound_id"]): "NONE"
            for row in frame[["compound_id"]].drop_duplicates().to_dict(orient="records")
        }

    scaffold_map: dict[int, str] = {}
    compounds = frame[["compound_id", "canonical_smiles"]].drop_duplicates()
    for row in compounds.to_dict(orient="records"):
        smiles = row.get("canonical_smiles")
        if not isinstance(smiles, str) or not smiles:
            scaffold_map[int(row["compound_id"])] = "NONE"
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scaffold_map[int(row["compound_id"])] = "NONE"
            continue
        try:
            scaf = GetScaffoldForMol(mol)
            scaffold_map[int(row["compound_id"])] = Chem.MolToSmiles(scaf) if scaf else "NONE"
        except Exception:
            scaffold_map[int(row["compound_id"])] = "NONE"
    return scaffold_map


def generate_scaffold_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_cp_split(conn, f"scaffold_s{seed}", "scaffold", seed, ratios)
    frame = _load_result_frame(conn)
    if frame.empty:
        conn.commit()
        return split_id

    frame = frame.copy()
    frame["scaffold"] = frame["compound_id"].map(_compute_scaffold_map(frame))

    scaffold_groups = frame.groupby("scaffold")["cp_result_id"].agg(list).to_dict()
    sorted_scaffolds = sorted(scaffold_groups.items(), key=lambda item: (-len(item[1]), item[0]))
    rng = np.random.RandomState(seed)

    size_groups = []
    for _size, group in groupby(sorted_scaffolds, key=lambda item: len(item[1])):
        grouped = list(group)
        rng.shuffle(grouped)
        size_groups.extend(grouped)

    targets = {
        "train": int(len(frame) * ratios["train"]),
        "val": int(len(frame) * ratios["val"]),
        "test": len(frame),
    }
    counts = {"train": 0, "val": 0, "test": 0}
    assignments = []

    for scaffold, ids in size_groups:
        if counts["train"] < targets["train"]:
            fold = "train"
        elif counts["val"] < targets["val"]:
            fold = "val"
        else:
            fold = "test"
        for cp_result_id in ids:
            assignments.append((int(cp_result_id), split_id, fold))
        counts[fold] += len(ids)
        logger.debug("CP scaffold %s -> %s (%d rows)", scaffold, fold, len(ids))

    conn.executemany(
        "INSERT INTO cp_split_assignments (cp_result_id, split_id, fold) VALUES (?, ?, ?)",
        assignments,
    )
    conn.commit()
    return split_id


def generate_batch_holdout_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_cp_split(
        conn, "batch_holdout", "batch_holdout", seed, ratios
    )
    frame = _load_result_frame(conn)
    if frame.empty:
        conn.commit()
        return split_id

    rng = np.random.RandomState(seed)
    _assign_folds_by_group(conn, split_id, frame, "batch_id", ratios, rng)
    conn.commit()
    return split_id


def generate_all_splits(conn: sqlite3.Connection, seed: int = 42) -> dict[str, int]:
    """Generate all required CP v1 split strategies."""
    return {
        "random": generate_random_split(conn, seed=seed),
        "cold_compound": generate_cold_compound_split(conn, seed=seed),
        "scaffold": generate_scaffold_split(conn, seed=seed),
        "batch_holdout": generate_batch_holdout_split(conn, seed=seed),
    }


def build_cp_m1_labels(df: pd.DataFrame) -> pd.Series:
    """Return binary CP-M1 labels: inactive=0, everything else=1."""
    return (df["outcome_label"] != "inactive").astype(int)


def build_cp_m2_labels(df: pd.DataFrame) -> pd.Series:
    """Return CP-M2 integer labels."""
    return df["outcome_label"].map(CP_M2_LABEL_TO_INT)


def _resolve_split_columns(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    rows = conn.execute(
        "SELECT split_name, split_id FROM cp_split_definitions ORDER BY split_id"
    ).fetchall()
    return [(str(name), int(split_id)) for name, split_id in rows]


def _attach_split_columns(frame: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    result = frame.copy()
    for split_name, split_id in _resolve_split_columns(conn):
        split_frame = pd.read_sql_query(
            "SELECT cp_result_id, fold FROM cp_split_assignments WHERE split_id = ?",
            conn,
            params=(split_id,),
        )
        result = result.merge(
            split_frame.rename(columns={"fold": f"split_{split_name}"}),
            on="cp_result_id",
            how="left",
        )
    return result


def export_cp_m1_dataset(conn: sqlite3.Connection, output_dir: str | Path) -> tuple[Path, int]:
    """Export the CP-M1 binary dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "negbiodb_cp_pairs.parquet"

    frame = _attach_split_columns(_load_result_frame(conn), conn)
    frame["Y"] = build_cp_m1_labels(frame)
    frame.to_parquet(path, index=False)
    return path, len(frame)


def export_cp_m2_dataset(conn: sqlite3.Connection, output_dir: str | Path) -> tuple[Path, int]:
    """Export the CP-M2 4-way dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "negbiodb_cp_m2.parquet"

    frame = _attach_split_columns(_load_result_frame(conn), conn)
    frame["Y"] = build_cp_m2_labels(frame)
    frame.to_parquet(path, index=False)
    return path, len(frame)


def _expand_feature_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"SELECT cp_result_id, feature_source, storage_uri, feature_json, n_features FROM {table_name}",
        conn,
    )
    if df.empty:
        return pd.DataFrame(columns=["cp_result_id"])

    feature_rows = []
    for row in df.to_dict(orient="records"):
        payload = json.loads(row["feature_json"]) if row.get("feature_json") else {}
        payload["cp_result_id"] = int(row["cp_result_id"])
        payload["feature_source"] = row.get("feature_source")
        payload["storage_uri"] = row.get("storage_uri")
        feature_rows.append(payload)
    return pd.DataFrame(feature_rows)


def export_cp_feature_tables(conn: sqlite3.Connection, output_dir: str | Path) -> dict[str, Path]:
    """Export CP profile and image-feature matrices keyed by cp_result_id."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_path = output_dir / "negbiodb_cp_profile_features.parquet"
    image_path = output_dir / "negbiodb_cp_image_features.parquet"

    _expand_feature_table(conn, "cp_profile_features").to_parquet(profile_path, index=False)
    _expand_feature_table(conn, "cp_image_features").to_parquet(image_path, index=False)
    return {"profile": profile_path, "image": image_path}
