"""ML dataset export for DC domain (Drug Combination Synergy).

Six split strategies:
  1. random            — Stratified random by best_confidence tier
  2. cold_compound      — Neither Drug A nor Drug B seen in train
  3. cold_cell_line     — Cell lines unseen in train
  4. cold_both          — Unseen compounds AND unseen cell lines
  5. scaffold           — Bemis-Murcko scaffold of Drug A
  6. leave_one_tissue_out — Entire tissue type held out for test

Two ML tasks:
  DC-M1: Binary classification (antagonistic+additive vs synergistic)
  DC-M2: 3-class classification (antagonistic vs additive vs synergistic)
"""

import logging
import sqlite3
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


# ------------------------------------------------------------------
# Split registration helpers
# ------------------------------------------------------------------

def _register_dc_split(
    conn: sqlite3.Connection,
    name: str,
    strategy: str,
    seed: int | None,
    ratios: dict[str, float],
) -> int:
    """Insert or retrieve a DC split definition and return split_id."""
    row = conn.execute(
        "SELECT split_id FROM dc_split_definitions WHERE split_name = ?",
        (name,),
    ).fetchone()

    if row is not None:
        split_id = int(row[0])
        conn.execute(
            "DELETE FROM dc_split_assignments WHERE split_id = ?",
            (split_id,),
        )
        return split_id

    conn.execute(
        """INSERT INTO dc_split_definitions
        (split_name, split_strategy, random_seed,
         train_ratio, val_ratio, test_ratio)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (name, strategy, seed,
         ratios["train"], ratios["val"], ratios["test"]),
    )
    row = conn.execute(
        "SELECT split_id FROM dc_split_definitions WHERE split_name = ?",
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
        best_fold = min(
            fold_targets,
            key=lambda f: fold_counts[f] / max(n_total, 1) - fold_targets[f],
        )
        for pid in pair_ids:
            assignments.append((pid, split_id, best_fold))
        fold_counts[best_fold] += len(pair_ids)

    conn.executemany(
        "INSERT INTO dc_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
        assignments,
    )


# ------------------------------------------------------------------
# Split generators (6 strategies)
# ------------------------------------------------------------------

def generate_random_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    """Generate random split stratified by best_confidence tier."""
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_dc_split(conn, f"random_s{seed}", "random", seed, ratios)
    rng = np.random.RandomState(seed)

    pairs = pd.read_sql(
        "SELECT pair_id, best_confidence FROM drug_drug_pairs", conn
    )
    if pairs.empty:
        conn.commit()
        return split_id

    for _tier, group in pairs.groupby("best_confidence"):
        n = len(group)
        indices = rng.permutation(n)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])

        folds = ["test"] * n
        for i in indices[:n_train]:
            folds[i] = "train"
        for i in indices[n_train: n_train + n_val]:
            folds[i] = "val"

        assignments = [
            (int(group.iloc[i]["pair_id"]), split_id, folds[i])
            for i in range(n)
        ]
        conn.executemany(
            "INSERT INTO dc_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
            assignments,
        )

    conn.commit()
    return split_id


def generate_cold_compound_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    """Generate cold compound split — neither Drug A nor Drug B seen in train.

    Partitions compounds into train/val/test, then assigns each pair to the
    fold of its 'hardest' compound (max rank).
    """
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_dc_split(
        conn, f"cold_compound_s{seed}", "cold_compound", seed, ratios
    )
    rng = np.random.RandomState(seed)

    pairs = pd.read_sql(
        "SELECT pair_id, compound_a_id, compound_b_id FROM drug_drug_pairs", conn
    )
    if pairs.empty:
        conn.commit()
        return split_id

    # Get all unique compounds
    compound_ids = sorted(set(pairs["compound_a_id"]) | set(pairs["compound_b_id"]))
    rng.shuffle(compound_ids)

    n = len(compound_ids)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    # Assign compounds to ranks: 0=train, 1=val, 2=test
    compound_rank = {}
    for i, cid in enumerate(compound_ids):
        if i < n_train:
            compound_rank[cid] = 0
        elif i < n_train + n_val:
            compound_rank[cid] = 1
        else:
            compound_rank[cid] = 2

    rank_to_fold = {0: "train", 1: "val", 2: "test"}

    # Pair fold = max(rank_a, rank_b) — ensures cold: both compounds must be in train
    assignments = []
    for _, row in pairs.iterrows():
        rank_a = compound_rank[row["compound_a_id"]]
        rank_b = compound_rank[row["compound_b_id"]]
        fold = rank_to_fold[max(rank_a, rank_b)]
        assignments.append((int(row["pair_id"]), split_id, fold))

    conn.executemany(
        "INSERT INTO dc_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
        assignments,
    )
    conn.commit()
    return split_id


def generate_cold_cell_line_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    """Generate cold cell line split — cell lines unseen in train.

    For pair-level splits, we assign each pair to the fold corresponding
    to the most common cell line it was tested in. This approximates
    cell-line-based isolation at the pair level.
    """
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_dc_split(
        conn, f"cold_cell_line_s{seed}", "cold_cell_line", seed, ratios
    )
    rng = np.random.RandomState(seed)

    # Get pair → primary cell line mapping (most measurements)
    triples = pd.read_sql(
        """SELECT pair_id, cell_line_id, num_measurements
        FROM drug_drug_cell_line_triples""", conn
    )
    if triples.empty:
        conn.commit()
        return split_id

    # For each pair, pick the cell line with most measurements
    idx = triples.groupby("pair_id")["num_measurements"].idxmax()
    pair_primary_cl = triples.loc[idx, ["pair_id", "cell_line_id"]].copy()

    # Partition cell lines
    cl_ids = sorted(pair_primary_cl["cell_line_id"].unique())
    rng.shuffle(cl_ids)
    n = len(cl_ids)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    cl_fold = {}
    for i, cid in enumerate(cl_ids):
        if i < n_train:
            cl_fold[cid] = "train"
        elif i < n_train + n_val:
            cl_fold[cid] = "val"
        else:
            cl_fold[cid] = "test"

    # Assign pairs based on primary cell line fold
    assignments = [
        (int(row["pair_id"]), split_id, cl_fold[row["cell_line_id"]])
        for _, row in pair_primary_cl.iterrows()
    ]
    conn.executemany(
        "INSERT INTO dc_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
        assignments,
    )
    conn.commit()
    return split_id


def generate_cold_both_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    """Generate cold-both split — unseen compounds AND unseen cell lines.

    Uses max-rank merging (GE pattern):
    - Partition compounds → rank 0/1/2
    - Partition cell lines → rank 0/1/2
    - pair_fold = rank_to_fold[max(compound_rank_a, compound_rank_b, cell_line_rank)]
    """
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_dc_split(
        conn, f"cold_both_s{seed}", "cold_both", seed, ratios
    )
    rng = np.random.RandomState(seed)

    pairs = pd.read_sql(
        "SELECT pair_id, compound_a_id, compound_b_id FROM drug_drug_pairs", conn
    )
    triples = pd.read_sql(
        """SELECT pair_id, cell_line_id, num_measurements
        FROM drug_drug_cell_line_triples""", conn
    )
    if pairs.empty:
        conn.commit()
        return split_id

    # Partition compounds
    compound_ids = sorted(set(pairs["compound_a_id"]) | set(pairs["compound_b_id"]))
    rng.shuffle(compound_ids)
    n_c = len(compound_ids)
    n_c_train = int(n_c * ratios["train"])
    n_c_val = int(n_c * ratios["val"])

    compound_rank = {}
    for i, cid in enumerate(compound_ids):
        if i < n_c_train:
            compound_rank[cid] = 0
        elif i < n_c_train + n_c_val:
            compound_rank[cid] = 1
        else:
            compound_rank[cid] = 2

    # Partition cell lines
    if not triples.empty:
        idx = triples.groupby("pair_id")["num_measurements"].idxmax()
        pair_primary_cl = dict(zip(
            triples.loc[idx, "pair_id"],
            triples.loc[idx, "cell_line_id"],
        ))

        cl_ids = sorted(set(pair_primary_cl.values()))
        rng.shuffle(cl_ids)
        n_cl = len(cl_ids)
        n_cl_train = int(n_cl * ratios["train"])
        n_cl_val = int(n_cl * ratios["val"])

        cl_rank = {}
        for i, cid in enumerate(cl_ids):
            if i < n_cl_train:
                cl_rank[cid] = 0
            elif i < n_cl_train + n_cl_val:
                cl_rank[cid] = 1
            else:
                cl_rank[cid] = 2
    else:
        pair_primary_cl = {}
        cl_rank = {}

    rank_to_fold = {0: "train", 1: "val", 2: "test"}

    assignments = []
    for _, row in pairs.iterrows():
        rank_a = compound_rank[row["compound_a_id"]]
        rank_b = compound_rank[row["compound_b_id"]]
        cl_id = pair_primary_cl.get(row["pair_id"])
        rank_cl = cl_rank.get(cl_id, 0) if cl_id is not None else 0
        fold = rank_to_fold[max(rank_a, rank_b, rank_cl)]
        assignments.append((int(row["pair_id"]), split_id, fold))

    conn.executemany(
        "INSERT INTO dc_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
        assignments,
    )
    conn.commit()
    return split_id


def _compute_scaffolds(conn: sqlite3.Connection) -> dict[str, list[int]]:
    """Compute Murcko scaffolds for all compounds, return scaffold→[compound_ids].

    Uses Murcko frameworks (ring systems + linkers, heteroatoms preserved).
    Compounds that fail RDKit parsing get scaffold='NONE'.
    """
    if not _RDKIT_AVAILABLE:
        # Without RDKit, put all compounds in one scaffold group
        rows = conn.execute("SELECT compound_id FROM compounds").fetchall()
        return {"NONE": [r[0] for r in rows]}

    scaffold_to_compounds: dict[str, list[int]] = defaultdict(list)

    rows = conn.execute(
        "SELECT compound_id, canonical_smiles FROM compounds ORDER BY compound_id"
    ).fetchall()

    for compound_id, smiles in rows:
        if not smiles:
            scaffold_to_compounds["NONE"].append(compound_id)
            continue
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
        len(rows), len(scaffold_to_compounds),
    )
    return dict(scaffold_to_compounds)


def generate_scaffold_split(
    conn: sqlite3.Connection,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> int:
    """Generate scaffold split using Bemis-Murcko scaffolds.

    Groups compounds by scaffold, then assigns groups to folds using
    a greedy size-based approach (largest scaffolds first → train).
    All pairs for a compound inherit its scaffold's fold.
    """
    ratios = ratios or _DEFAULT_RATIOS
    split_id = _register_dc_split(
        conn, f"scaffold_s{seed}", "scaffold", seed, ratios
    )

    scaffold_to_compounds = _compute_scaffolds(conn)

    # Sort by size descending (tie-break by scaffold SMILES for determinism)
    sorted_scaffolds = sorted(
        scaffold_to_compounds.items(),
        key=lambda x: (-len(x[1]), x[0]),
    )

    # Count total pairs per compound
    compound_pair_counts: dict[int, int] = {}
    for cid, cnt in conn.execute(
        """SELECT compound_id, COUNT(*) FROM (
            SELECT compound_a_id AS compound_id FROM drug_drug_pairs
            UNION ALL
            SELECT compound_b_id AS compound_id FROM drug_drug_pairs
        ) GROUP BY compound_id"""
    ):
        compound_pair_counts[cid] = cnt

    total_pairs = conn.execute("SELECT COUNT(*) FROM drug_drug_pairs").fetchone()[0]
    target_train = int(total_pairs * ratios["train"])
    target_val = int(total_pairs * (ratios["train"] + ratios["val"]))

    # Shuffle within same-size groups for randomness
    rng = np.random.RandomState(seed)
    size_groups = []
    for _size, group in groupby(sorted_scaffolds, key=lambda x: len(x[1])):
        group_list = list(group)
        rng.shuffle(group_list)
        size_groups.extend(group_list)

    compound_to_fold: dict[int, str] = {}
    current_train = 0
    current_val = 0

    for _scaffold_smi, compound_ids in size_groups:
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

    # Write assignments via temp table
    conn.execute("DROP TABLE IF EXISTS _scaffold_folds")
    conn.execute(
        "CREATE TEMP TABLE _scaffold_folds (compound_id INTEGER PRIMARY KEY, fold TEXT)"
    )
    conn.executemany(
        "INSERT INTO _scaffold_folds (compound_id, fold) VALUES (?, ?)",
        compound_to_fold.items(),
    )
    # Pair fold = fold of compound_a (Drug A scaffold)
    conn.execute(
        """INSERT INTO dc_split_assignments (pair_id, split_id, fold)
        SELECT ddp.pair_id, ?, sf.fold
        FROM drug_drug_pairs ddp
        JOIN _scaffold_folds sf ON ddp.compound_a_id = sf.compound_id""",
        (split_id,),
    )
    conn.execute("DROP TABLE _scaffold_folds")
    conn.commit()
    return split_id


def generate_leave_one_tissue_out_split(
    conn: sqlite3.Connection,
    held_out_tissue: str | None = None,
    seed: int = 42,
) -> int:
    """Generate leave-one-tissue-out split.

    Test: all pairs whose primary cell line is from held_out_tissue.
    Val: 10% of remaining pairs (random).
    Train: rest.

    If held_out_tissue is None, picks the largest tissue by pair count.
    """
    ratios = {"train": 0.0, "val": 0.0, "test": 0.0}  # computed from data

    # Get pair → primary cell line → tissue mapping
    triples = pd.read_sql(
        """SELECT t.pair_id, t.cell_line_id, t.num_measurements, cl.tissue
        FROM drug_drug_cell_line_triples t
        JOIN cell_lines cl ON t.cell_line_id = cl.cell_line_id""", conn
    )
    if triples.empty:
        split_id = _register_dc_split(
            conn, "leave_one_tissue_out", "leave_one_tissue_out", seed, ratios
        )
        conn.commit()
        return split_id

    # Pick primary cell line per pair
    idx = triples.groupby("pair_id")["num_measurements"].idxmax()
    pair_tissue = triples.loc[idx, ["pair_id", "tissue"]].copy()

    # Auto-select held-out tissue if not specified
    if held_out_tissue is None:
        tissue_counts = pair_tissue["tissue"].value_counts()
        tissue_counts = tissue_counts[tissue_counts.index.notna()]
        if tissue_counts.empty:
            held_out_tissue = "Unknown"
        else:
            held_out_tissue = tissue_counts.index[0]

    split_name = f"loto_{held_out_tissue.lower().replace(' ', '_')}_s{seed}"
    split_id = _register_dc_split(
        conn, split_name, "leave_one_tissue_out", seed, ratios
    )

    rng = np.random.RandomState(seed)

    # Separate test (held-out tissue) and non-test
    test_mask = pair_tissue["tissue"] == held_out_tissue
    test_pairs = pair_tissue.loc[test_mask, "pair_id"].tolist()
    non_test = pair_tissue.loc[~test_mask, "pair_id"].tolist()

    # Split non-test into train/val (90/10)
    rng.shuffle(non_test)
    n_val = max(1, int(len(non_test) * 0.1))
    val_pairs = non_test[:n_val]
    train_pairs = non_test[n_val:]

    assignments = []
    for pid in train_pairs:
        assignments.append((int(pid), split_id, "train"))
    for pid in val_pairs:
        assignments.append((int(pid), split_id, "val"))
    for pid in test_pairs:
        assignments.append((int(pid), split_id, "test"))

    conn.executemany(
        "INSERT INTO dc_split_assignments (pair_id, split_id, fold) VALUES (?, ?, ?)",
        assignments,
    )
    conn.commit()
    return split_id


def generate_all_splits(
    conn: sqlite3.Connection,
    seed: int = 42,
) -> dict[str, int]:
    """Generate all 6 DC split strategies. Returns {strategy: split_id}."""
    splits = {}
    splits["random"] = generate_random_split(conn, seed)
    splits["cold_compound"] = generate_cold_compound_split(conn, seed)
    splits["cold_cell_line"] = generate_cold_cell_line_split(conn, seed)
    splits["cold_both"] = generate_cold_both_split(conn, seed)
    splits["scaffold"] = generate_scaffold_split(conn, seed)
    splits["leave_one_tissue_out"] = generate_leave_one_tissue_out_split(
        conn, seed=seed
    )
    return splits


# ------------------------------------------------------------------
# Parquet export
# ------------------------------------------------------------------

SPLIT_STRATEGIES = [
    "random", "cold_compound", "cold_cell_line",
    "cold_both", "scaffold", "leave_one_tissue_out",
]


def export_dc_dataset(
    conn: sqlite3.Connection,
    output_path: Path,
) -> int:
    """Export the full DC pair-level dataset as parquet with split columns.

    Joins drug_drug_pairs with compound info and split assignments.
    Returns number of exported rows.
    """
    query = """
    SELECT
        ddp.pair_id,
        ddp.compound_a_id,
        ddp.compound_b_id,
        ca.drug_name AS drug_a_name,
        cb.drug_name AS drug_b_name,
        ca.canonical_smiles AS smiles_a,
        cb.canonical_smiles AS smiles_b,
        ca.pubchem_cid AS pubchem_cid_a,
        cb.pubchem_cid AS pubchem_cid_b,
        ddp.num_cell_lines,
        ddp.num_sources,
        ddp.num_measurements,
        ddp.median_zip,
        ddp.median_bliss,
        ddp.antagonism_fraction,
        ddp.synergy_fraction,
        ddp.consensus_class,
        ddp.best_confidence AS confidence_tier,
        ddp.num_shared_targets,
        ddp.target_jaccard,
        ddp.compound_a_degree,
        ddp.compound_b_degree
    FROM drug_drug_pairs ddp
    JOIN compounds ca ON ddp.compound_a_id = ca.compound_id
    JOIN compounds cb ON ddp.compound_b_id = cb.compound_id
    """

    df = pd.read_sql(query, conn)

    # Add split columns
    splits = pd.read_sql(
        "SELECT split_id, split_name FROM dc_split_definitions", conn
    )
    for _, split_row in splits.iterrows():
        split_id = split_row["split_id"]
        split_name = split_row["split_name"]
        col_name = f"split_{split_name}"

        assigns = pd.read_sql(
            "SELECT pair_id, fold FROM dc_split_assignments WHERE split_id = ?",
            conn,
            params=(int(split_id),),
        )
        fold_map = dict(zip(assigns["pair_id"], assigns["fold"]))
        df[col_name] = df["pair_id"].map(fold_map)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Exported %d rows to %s", len(df), output_path)
    return len(df)


# ------------------------------------------------------------------
# ML task builders (M1 binary, M2 3-class)
# ------------------------------------------------------------------

def build_dc_m1_labels(df: pd.DataFrame) -> pd.Series:
    """Build DC-M1 binary labels: 0=negative (antagonistic+additive), 1=positive (synergistic).

    Uses consensus_class from drug_drug_pairs.
    """
    label_map = {
        "antagonistic": 0,
        "additive": 0,
        "synergistic": 1,
        "context_dependent": 0,  # Majority are not synergistic → negative
    }
    return df["consensus_class"].map(label_map)


def build_dc_m2_labels(df: pd.DataFrame) -> pd.Series:
    """Build DC-M2 3-class labels: 0=antagonistic, 1=additive, 2=synergistic.

    Drops context_dependent pairs (returns NaN for those).
    """
    label_map = {
        "antagonistic": 0,
        "additive": 1,
        "synergistic": 2,
    }
    return df["consensus_class"].map(label_map)
