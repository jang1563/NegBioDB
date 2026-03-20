"""Feature encoding for CT domain ML benchmarks.

Encodes drug, condition, and trial features from exported parquet DataFrames
into NumPy arrays suitable for XGBoost, MLP, and GNN+Tab models.

Feature dimensions:
  Drug (with FP):  1030 = FP(1024) + MolProps(6)
  Drug (tabular):  13 = MolType(10) + target_count(1) + drug_degree(1) + phase(1)
  Condition:       1 = condition_degree
  Trial (M2):      22 = trial_phase(9) + blinding(6) + sponsor(5) + randomized(1) + enrollment(1)
  Total M1:        1044 = 1030 + 13 + 1
  Total M2:        1066 = 1044 + 22
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Morgan fingerprint
FP_RADIUS = 2
FP_NBITS = 1024

# Molecular property names (RDKit descriptors)
MOL_PROPS = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "NumRotatableBonds",
]
N_MOL_PROPS = len(MOL_PROPS)

# Molecular type vocabulary — from DB CHECK constraint (migration 002, line 24-27)
MOLECULAR_TYPES: list[str] = [
    "small_molecule",
    "monoclonal_antibody",
    "antibody_drug_conjugate",
    "peptide",
    "oligonucleotide",
    "cell_therapy",
    "gene_therapy",
    "other_biologic",
    "unknown",
]  # 9 types → 10-dim one-hot (9 + unknown bucket)

# Phase ordinal encoding — from DB CHECK (migration 001, line 89-91)
# Ordinal (not one-hot): phase is naturally ordered.
PHASE_ORDER: dict[str, float] = {
    "early_phase_1": 0.5,
    "phase_1": 1.0,
    "phase_1_2": 1.5,
    "phase_2": 2.0,
    "phase_2_3": 2.5,
    "phase_3": 3.0,
    "phase_4": 4.0,
    "not_applicable": 0.0,
}

# Trial phase vocabulary for M2 one-hot — from DB CHECK (migration 001, line 89-91)
TRIAL_PHASES: list[str] = [
    "early_phase_1",
    "phase_1",
    "phase_1_2",
    "phase_2",
    "phase_2_3",
    "phase_3",
    "phase_4",
    "not_applicable",
]  # 8 phases → 9-dim one-hot (8 + unknown bucket)

# Blinding — case-insensitive SUBSTRING matching on raw AACT masking field
# Values like "Double (Participant, Investigator)" → .lower() then match
BLINDING_KEYWORDS: list[str] = ["none", "single", "double", "triple", "quadruple"]
# 5 keywords → 6-dim one-hot (5 + unrecognized bucket)

# Sponsor type vocabulary — from DB CHECK (migration 001, line 101-102)
SPONSOR_TYPES: list[str] = ["industry", "academic", "government", "other"]
# 4 types → 5-dim one-hot (4 + unknown bucket)
# Note: "academic" is in DB CHECK but etl_aact maps all non-industry to
# "government" or "other"; academic dimension may always be 0. Kept for
# schema completeness.

# Feature dimensions
DRUG_FP_DIM = FP_NBITS + N_MOL_PROPS  # 1024 + 6 = 1030
DRUG_TAB_DIM = (len(MOLECULAR_TYPES) + 1) + 1 + 1 + 1
# mol_type(10) + target_count(1) + drug_degree(1) + phase(1) = 13

CONDITION_DIM = 1  # condition_degree only

M2_TRIAL_DIM = (
    (len(TRIAL_PHASES) + 1)       # trial_phase one-hot: 9
    + (len(BLINDING_KEYWORDS) + 1)  # blinding: 6
    + (len(SPONSOR_TYPES) + 1)      # sponsor: 5
    + 1                             # randomized: 1
    + 1                             # enrollment: 1
)  # = 22

TOTAL_M1_DIM = DRUG_FP_DIM + DRUG_TAB_DIM + CONDITION_DIM  # 1030 + 13 + 1 = 1044
TOTAL_M2_DIM = TOTAL_M1_DIM + M2_TRIAL_DIM                 # 1044 + 22 = 1066

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_hot(value: str | None, choices: Sequence[str]) -> list[int]:
    """One-hot encode with unknown bucket (last element).

    Returns a vector of length len(choices)+1.
    """
    vec = [0] * (len(choices) + 1)
    if value is not None:
        try:
            vec[choices.index(value)] = 1
        except ValueError:
            vec[-1] = 1  # unknown bucket
    else:
        vec[-1] = 1  # unknown bucket
    return vec


def _encode_blinding(blinding_str: str | None) -> list[int]:
    """Case-insensitive substring match on raw AACT masking field.

    Returns 6-dim vector (5 keywords + unrecognized bucket).
    """
    vec = [0] * (len(BLINDING_KEYWORDS) + 1)
    if blinding_str is None or pd.isna(blinding_str):
        vec[-1] = 1
        return vec
    lower = str(blinding_str).lower()
    matched = False
    for i, kw in enumerate(BLINDING_KEYWORDS):
        if kw in lower:
            vec[i] = 1
            matched = True
            break
    if not matched:
        vec[-1] = 1
    return vec


def _safe_log1p(value: float | None, default: float = 0.0) -> float:
    """Safe log1p: handles NaN/None/pd.NA → default, then np.log1p."""
    if value is None or pd.isna(value):
        return np.log1p(default)
    return float(np.log1p(float(value)))


# ---------------------------------------------------------------------------
# Feature Encoding Functions
# ---------------------------------------------------------------------------


def _compute_fp_and_props(smiles_series: pd.Series) -> np.ndarray:
    """Compute Morgan FP (1024-bit) + 6 molecular properties from SMILES.

    Returns (N, 1030) array. Rows with invalid/missing SMILES get NaN.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
    except ImportError as e:
        raise RuntimeError("rdkit required for fingerprint computation") from e

    n = len(smiles_series)
    result = np.full((n, DRUG_FP_DIM), np.nan, dtype=np.float32)

    descriptor_funcs = [getattr(Descriptors, name) for name in MOL_PROPS]

    for i, smi in enumerate(smiles_series):
        if pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        # Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
        arr = np.zeros(FP_NBITS, dtype=np.float32)
        fp_array = fp.ToList()
        for j, bit in enumerate(fp_array):
            arr[j] = float(bit)
        result[i, :FP_NBITS] = arr
        # Molecular properties
        for k, func in enumerate(descriptor_funcs):
            result[i, FP_NBITS + k] = func(mol)

    return result


def encode_drug_features(df: pd.DataFrame, *, include_fp: bool = True) -> np.ndarray:
    """Encode drug features from DataFrame.

    Parameters
    ----------
    df : DataFrame with columns: smiles, molecular_type, target_count (optional),
         intervention_degree (optional), highest_phase_reached (optional)
    include_fp : if True, include Morgan FP + mol props (1030-dim)

    Returns
    -------
    np.ndarray of shape (N, 1043) with FP or (N, 13) without FP.
    NaN preserved for XGBoost; callers should zero-pad for MLP.
    """
    n = len(df)
    parts: list[np.ndarray] = []

    # --- Fingerprint + molecular properties (1030-dim) ---
    if include_fp:
        smiles_col = df["smiles"] if "smiles" in df.columns else pd.Series([None] * n)
        fp_props = _compute_fp_and_props(smiles_col)
        parts.append(fp_props)

    # --- Molecular type one-hot (10-dim) ---
    mol_type_arr = np.zeros((n, len(MOLECULAR_TYPES) + 1), dtype=np.float32)
    mol_type_col = df.get("molecular_type")
    for i in range(n):
        val = mol_type_col.iloc[i] if mol_type_col is not None else None
        if pd.isna(val):
            val = None
        mol_type_arr[i] = _one_hot(val, MOLECULAR_TYPES)
    parts.append(mol_type_arr)

    # --- target_count (1-dim, log1p) ---
    tc_col = df.get("target_count")
    tc_arr = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        val = tc_col.iloc[i] if tc_col is not None else 0
        tc_arr[i, 0] = _safe_log1p(val, default=0.0)
    parts.append(tc_arr)

    # --- intervention_degree (1-dim, log1p) ---
    deg_col = df.get("intervention_degree")
    deg_arr = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        val = deg_col.iloc[i] if deg_col is not None else 1
        deg_arr[i, 0] = _safe_log1p(val, default=1.0)
    parts.append(deg_arr)

    # --- highest_phase_reached ordinal (1-dim) ---
    phase_col = df.get("highest_phase_reached")
    phase_arr = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        val = phase_col.iloc[i] if phase_col is not None else None
        if pd.isna(val):
            phase_arr[i, 0] = 0.0
        else:
            phase_arr[i, 0] = PHASE_ORDER.get(str(val), 0.0)
    parts.append(phase_arr)

    return np.hstack(parts)


def encode_condition_features(df: pd.DataFrame) -> np.ndarray:
    """Encode condition features: condition_degree (log1p).

    Returns (N, 1) array.
    """
    n = len(df)
    result = np.zeros((n, CONDITION_DIM), dtype=np.float32)
    deg_col = df.get("condition_degree")
    for i in range(n):
        val = deg_col.iloc[i] if deg_col is not None else 1
        result[i, 0] = _safe_log1p(val, default=1.0)
    return result


def encode_trial_features(df: pd.DataFrame) -> np.ndarray:
    """Encode trial-level features for CT-M2.

    Columns used: trial_phase, blinding, sponsor_type, randomized, enrollment_actual.

    Returns (N, 22) array.
    """
    n = len(df)
    parts: list[np.ndarray] = []

    # --- trial_phase one-hot (9-dim) ---
    tp_arr = np.zeros((n, len(TRIAL_PHASES) + 1), dtype=np.float32)
    tp_col = df.get("trial_phase")
    for i in range(n):
        val = tp_col.iloc[i] if tp_col is not None else None
        if pd.isna(val):
            val = None
        tp_arr[i] = _one_hot(val, TRIAL_PHASES)
    parts.append(tp_arr)

    # --- blinding (6-dim, case-insensitive substring) ---
    bl_arr = np.zeros((n, len(BLINDING_KEYWORDS) + 1), dtype=np.float32)
    bl_col = df.get("blinding")
    for i in range(n):
        val = bl_col.iloc[i] if bl_col is not None else None
        if pd.isna(val):
            val = None
        bl_arr[i] = _encode_blinding(val)
    parts.append(bl_arr)

    # --- sponsor_type one-hot (5-dim) ---
    sp_arr = np.zeros((n, len(SPONSOR_TYPES) + 1), dtype=np.float32)
    sp_col = df.get("sponsor_type")
    for i in range(n):
        val = sp_col.iloc[i] if sp_col is not None else None
        if pd.isna(val):
            val = None
        sp_arr[i] = _one_hot(val, SPONSOR_TYPES)
    parts.append(sp_arr)

    # --- randomized (1-dim, binary) ---
    rand_arr = np.zeros((n, 1), dtype=np.float32)
    rand_col = df.get("randomized")
    for i in range(n):
        val = rand_col.iloc[i] if rand_col is not None else 0
        if pd.isna(val):
            val = 0
        rand_arr[i, 0] = float(int(val))
    parts.append(rand_arr)

    # --- enrollment_actual (1-dim, log1p) ---
    enr_arr = np.zeros((n, 1), dtype=np.float32)
    enr_col = df.get("enrollment_actual")
    for i in range(n):
        val = enr_col.iloc[i] if enr_col is not None else 0
        enr_arr[i, 0] = _safe_log1p(val, default=0.0)
    parts.append(enr_arr)

    return np.hstack(parts)


# ---------------------------------------------------------------------------
# Composite Feature Builders
# ---------------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    task: str = "m1",
    *,
    include_fp: bool = True,
) -> np.ndarray:
    """Build full feature matrix for CT-M1 or CT-M2.

    Parameters
    ----------
    df : exported parquet DataFrame
    task : "m1" or "m2"
    include_fp : include Morgan FP + mol props

    Returns
    -------
    np.ndarray: (N, 1044) for M1 with FP, (N, 14) without FP,
                (N, 1066) for M2 with FP, (N, 36) without FP.
    """
    parts = [
        encode_drug_features(df, include_fp=include_fp),
        encode_condition_features(df),
    ]
    if task == "m2":
        parts.append(encode_trial_features(df))
    return np.hstack(parts)


def build_xgboost_features(df: pd.DataFrame, task: str = "m1") -> np.ndarray:
    """Build features for XGBoost. NaN preserved (handled natively)."""
    return build_feature_matrix(df, task, include_fp=True)


def build_mlp_features(df: pd.DataFrame, task: str = "m1") -> np.ndarray:
    """Build features for MLP. NaN → 0.0."""
    X = build_feature_matrix(df, task, include_fp=True)
    return np.nan_to_num(X, nan=0.0)


def build_gnn_tab_features(df: pd.DataFrame, task: str = "m1") -> np.ndarray:
    """Build tabular features for GNN+Tab (no FP). NaN → 0.0."""
    X = build_feature_matrix(df, task, include_fp=False)
    return np.nan_to_num(X, nan=0.0)
