"""Feature engineering for MD domain metabolite-disease biomarker prediction.

Feature vector design (2,068-dim, consistent with DTI/DC raw fingerprint approach):
  - Metabolite ECFP4 (2048-bit Morgan FP, radius=2): primary structural representation
  - Metabolite physicochemical (5): MW, logP, TPSA, HBD, HBA
  - Disease category one-hot (5): cancer/metabolic/neurological/cardiovascular/other
  - Platform one-hot (4): nmr/lc_ms/gc_ms/other
  - Biofluid one-hot (4): blood/urine/csf/tissue (tissue+other → 'tissue')
  - Study size features (2): log10(n_disease), log10(n_control)
  Total: 2048 + 5 + 5 + 4 + 4 + 2 = 2068

Labels:
  - M1: binary is_significant (0=negative, 1=positive)
  - M2: disease_category (5-class)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False

MISSING_SENTINEL = -1.0

# Ordered categories for one-hot encoding (must match schema CHECK constraints)
DISEASE_CATEGORIES = ["cancer", "metabolic", "neurological", "cardiovascular", "other"]
PLATFORMS = ["nmr", "lc_ms", "gc_ms", "other"]
BIOFLUIDS = ["blood", "urine", "csf", "tissue"]  # 'other' → 'tissue' bucket

FEATURE_DIM = 2068


# ── Molecular fingerprint ────────────────────────────────────────────────────

def compute_ecfp4(smiles: str | None, n_bits: int = 2048) -> np.ndarray:
    """Compute ECFP4 (Morgan FP radius=2) bit vector.

    Returns array of n_bits floats (0/1); all zeros for invalid/missing SMILES.
    """
    if not _RDKIT_AVAILABLE or not smiles:
        return np.zeros(n_bits, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.frombuffer(fp.ToBitString().encode(), dtype="u1") - ord("0")


# ── Physicochemical descriptors ──────────────────────────────────────────────

def compute_physico(smiles: str | None) -> np.ndarray:
    """Compute 5-dim physicochemical descriptors from SMILES.

    Returns [MW, logP, TPSA, HBD, HBA] — NaN for missing/invalid.
    """
    if not _RDKIT_AVAILABLE or not smiles:
        return np.full(5, np.nan, dtype=np.float64)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(5, np.nan, dtype=np.float64)

    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
    ], dtype=np.float64)


# ── One-hot encoders ─────────────────────────────────────────────────────────

def one_hot(value: str | None, categories: list[str]) -> np.ndarray:
    """Return one-hot vector for value in categories (all-zero if None/unknown)."""
    vec = np.zeros(len(categories), dtype=np.float32)
    if value and value in categories:
        vec[categories.index(value)] = 1.0
    return vec


# ── Study size features ──────────────────────────────────────────────────────

def study_size_features(n_disease: int | None, n_control: int | None) -> np.ndarray:
    """Return [log10(n_disease), log10(n_control)] — 0.0 if None."""
    import math
    nd = math.log10(max(n_disease, 1)) if n_disease else 0.0
    nc = math.log10(max(n_control, 1)) if n_control else 0.0
    return np.array([nd, nc], dtype=np.float32)


# ── Full feature vector ──────────────────────────────────────────────────────

def build_feature_vector(
    smiles: str | None,
    disease_category: str | None,
    platform: str | None,
    biofluid: str | None,
    n_disease: int | None = None,
    n_control: int | None = None,
) -> np.ndarray:
    """Build the full 2068-dim feature vector for one metabolite-disease-study triple.

    Args:
        smiles:           Metabolite canonical SMILES (may be None)
        disease_category: One of DISEASE_CATEGORIES (may be None → zeros)
        platform:         One of PLATFORMS (may be None → zeros)
        biofluid:         One of BIOFLUIDS (tissue for 'other'; may be None → zeros)
        n_disease:        Number of disease-group subjects
        n_control:        Number of control-group subjects

    Returns:
        np.ndarray of shape (2068,) dtype float32
    """
    # Normalize biofluid: 'other' → 'tissue' (catch-all bucket)
    bf = biofluid if biofluid in BIOFLUIDS else "tissue"

    parts = [
        compute_ecfp4(smiles).astype(np.float32),            # 2048
        compute_physico(smiles).astype(np.float32),           # 5
        one_hot(disease_category, DISEASE_CATEGORIES),        # 5
        one_hot(platform, PLATFORMS),                         # 4
        one_hot(bf, BIOFLUIDS),                               # 4
        study_size_features(n_disease, n_control),            # 2
    ]
    vec = np.concatenate(parts)

    # Replace NaN with MISSING_SENTINEL for tree models
    np.nan_to_num(vec, nan=MISSING_SENTINEL, copy=False)
    return vec


# ── Batch feature matrix builder ─────────────────────────────────────────────

def build_feature_matrix(conn, pair_ids: list[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y_m1, y_m2) feature matrices from the MD database.

    Args:
        conn:     sqlite3 connection to negbiodb_md.db
        pair_ids: optional list of pair_ids to include; if None, uses all pairs

    Returns:
        X:    np.ndarray of shape (N, 2068) float32
        y_m1: np.ndarray of shape (N,) int32 — 0=negative, 1=positive (M1 label)
        y_m2: np.ndarray of shape (N,) int32 — disease_category index (M2 label)
    """
    where_clause = ""
    params: tuple = ()
    if pair_ids:
        placeholders = ",".join("?" * len(pair_ids))
        where_clause = f"WHERE p.pair_id IN ({placeholders})"
        params = tuple(pair_ids)

    rows = conn.execute(
        f"""SELECT
                r.result_id,
                m.canonical_smiles,
                d.disease_category,
                s.platform,
                s.biofluid,
                s.n_disease,
                s.n_control,
                r.is_significant
            FROM md_biomarker_results r
            JOIN md_metabolites m ON r.metabolite_id = m.metabolite_id
            JOIN md_diseases d ON r.disease_id = d.disease_id
            JOIN md_studies s ON r.study_id = s.study_id
            JOIN md_metabolite_disease_pairs p
                ON p.metabolite_id = r.metabolite_id AND p.disease_id = r.disease_id
            {where_clause}""",
        params,
    ).fetchall()

    n = len(rows)
    if n == 0:
        return (
            np.empty((0, FEATURE_DIM), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    X = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    y_m1 = np.zeros(n, dtype=np.int32)
    y_m2 = np.zeros(n, dtype=np.int32)

    for i, row in enumerate(rows):
        _, smiles, dis_cat, platform, biofluid, n_dis, n_ctrl, is_sig = row
        X[i] = build_feature_vector(smiles, dis_cat, platform, biofluid, n_dis, n_ctrl)
        y_m1[i] = int(is_sig)
        y_m2[i] = DISEASE_CATEGORIES.index(dis_cat) if dis_cat in DISEASE_CATEGORIES else 4

    return X, y_m1, y_m2
