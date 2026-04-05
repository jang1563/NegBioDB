"""Feature engineering for DC domain drug combination synergy prediction.

Feature vector design:
  DC-Tabular (variable ~37-dim): descriptors + target overlap + cell context + pair stats
  DC-Full (variable ~293-dim): DC-Tabular + Morgan FP PCA (128+128)
  DC-DeepSynergy (4146-dim): raw Morgan FP (2048+2048) + cell line expr (50)

Features available from DB schema (no external data needed):
  - Drug A/B descriptors (8+8): MW, LogP, HBA, HBD, TPSA, rotatable bonds, num_rings, Fsp3
  - Target overlap (5): shared_targets, jaccard_targets, drug_a_n_targets, drug_b_n_targets, same_pathway_proxy
  - Cell line context (5+): tissue one-hot (top tissues)
  - Pair-level stats (8): compound_a_degree, compound_b_degree, num_measurements,
    num_sources, num_cell_lines, antagonism_fraction, synergy_fraction, median_zip
  - Drug similarity (2): tanimoto_similarity, target_jaccard (=target_jaccard from pairs)
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False

# Sentinel for missing values (MLP/DNN models; XGBoost handles NaN natively)
MISSING_SENTINEL = -1.0


# ── Molecular descriptors ────────────────────────────────────────────

DESCRIPTOR_NAMES = [
    "molecular_weight",
    "logp",
    "hba",
    "hbd",
    "tpsa",
    "rotatable_bonds",
    "num_rings",
    "fsp3",
]


def compute_mol_descriptors(smiles: str | None) -> np.ndarray:
    """Compute 8-dim molecular descriptors from SMILES.

    Returns array of 8 floats (NaN for missing/invalid SMILES).
    """
    if not _RDKIT_AVAILABLE or not smiles:
        return np.full(8, np.nan)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(8, np.nan)

    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        Descriptors.FractionCSP3(mol),
    ], dtype=np.float64)


def compute_morgan_fp(smiles: str | None, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Compute Morgan fingerprint bit vector.

    Returns array of n_bits floats (0.0/1.0), all zeros for invalid SMILES.
    """
    if not _RDKIT_AVAILABLE or not smiles:
        return np.zeros(n_bits, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


def compute_tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Tanimoto similarity between two binary fingerprint vectors."""
    intersection = np.sum(fp_a * fp_b)
    union = np.sum(fp_a) + np.sum(fp_b) - intersection
    if union == 0:
        return 0.0
    return float(intersection / union)


# ── Feature assembly ─────────────────────────────────────────────────


def build_tabular_features(
    conn,
    pair_ids: list[int] | None = None,
    use_sentinel: bool = False,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Build tabular feature matrix for drug_drug_pairs.

    Args:
        conn: Database connection.
        pair_ids: Optional list of pair_ids to include. If None, all pairs.
        use_sentinel: Replace NaN with MISSING_SENTINEL (-1.0).

    Returns:
        (X, feature_names, pair_ids_out) where X is (n_pairs, n_features).
    """
    # Load pair data
    if pair_ids is not None:
        placeholders = ",".join("?" * len(pair_ids))
        query = f"""
            SELECT p.pair_id, p.compound_a_id, p.compound_b_id,
                   p.num_cell_lines, p.num_sources, p.num_measurements,
                   p.median_zip, p.median_bliss,
                   p.antagonism_fraction, p.synergy_fraction,
                   p.num_shared_targets, p.target_jaccard,
                   p.compound_a_degree, p.compound_b_degree,
                   ca.canonical_smiles AS smiles_a,
                   cb.canonical_smiles AS smiles_b
            FROM drug_drug_pairs p
            JOIN compounds ca ON p.compound_a_id = ca.compound_id
            JOIN compounds cb ON p.compound_b_id = cb.compound_id
            WHERE p.pair_id IN ({placeholders})
            ORDER BY p.pair_id
        """
        rows = conn.execute(query, pair_ids).fetchall()
    else:
        query = """
            SELECT p.pair_id, p.compound_a_id, p.compound_b_id,
                   p.num_cell_lines, p.num_sources, p.num_measurements,
                   p.median_zip, p.median_bliss,
                   p.antagonism_fraction, p.synergy_fraction,
                   p.num_shared_targets, p.target_jaccard,
                   p.compound_a_degree, p.compound_b_degree,
                   ca.canonical_smiles AS smiles_a,
                   cb.canonical_smiles AS smiles_b
            FROM drug_drug_pairs p
            JOIN compounds ca ON p.compound_a_id = ca.compound_id
            JOIN compounds cb ON p.compound_b_id = cb.compound_id
            ORDER BY p.pair_id
        """
        rows = conn.execute(query).fetchall()

    if not rows:
        return np.empty((0, 0)), [], []

    # Build drug target count cache
    target_counts = {}
    for row in conn.execute(
        "SELECT compound_id, COUNT(*) FROM drug_targets GROUP BY compound_id"
    ):
        target_counts[row[0]] = row[1]

    # Build tissue lookup for cell line context (top tissues)
    tissue_counts = {}
    for row in conn.execute(
        "SELECT tissue, COUNT(*) FROM cell_lines WHERE tissue IS NOT NULL GROUP BY tissue ORDER BY COUNT(*) DESC"
    ):
        tissue_counts[row[0]] = row[1]
    top_tissues = list(tissue_counts.keys())[:5]

    # Build feature matrix
    feature_names = []

    # Drug A descriptors (8)
    feature_names.extend([f"drug_a_{n}" for n in DESCRIPTOR_NAMES])
    # Drug B descriptors (8)
    feature_names.extend([f"drug_b_{n}" for n in DESCRIPTOR_NAMES])
    # Target overlap (5)
    feature_names.extend([
        "shared_targets", "target_jaccard",
        "drug_a_n_targets", "drug_b_n_targets",
        "tanimoto_similarity",
    ])
    # Pair-level (8)
    feature_names.extend([
        "compound_a_degree", "compound_b_degree",
        "num_cell_lines", "num_sources", "num_measurements",
        "antagonism_fraction", "synergy_fraction", "median_zip",
    ])
    # Cell line tissue context (top 5 tissue fractions)
    feature_names.extend([f"tissue_frac_{t}" for t in top_tissues])

    n_features = len(feature_names)
    X = np.full((len(rows), n_features), np.nan, dtype=np.float64)
    pair_ids_out = []

    for i, row in enumerate(rows):
        pair_id = row[0]
        cid_a, cid_b = row[1], row[2]
        smiles_a, smiles_b = row[14], row[15]

        pair_ids_out.append(pair_id)
        col = 0

        # Drug A descriptors (8)
        desc_a = compute_mol_descriptors(smiles_a)
        X[i, col:col + 8] = desc_a
        col += 8

        # Drug B descriptors (8)
        desc_b = compute_mol_descriptors(smiles_b)
        X[i, col:col + 8] = desc_b
        col += 8

        # Target overlap (5)
        X[i, col] = row[10] or 0  # num_shared_targets
        X[i, col + 1] = row[11] or 0.0  # target_jaccard
        X[i, col + 2] = target_counts.get(cid_a, 0)
        X[i, col + 3] = target_counts.get(cid_b, 0)

        # Tanimoto similarity (compute from Morgan FP)
        if _RDKIT_AVAILABLE and smiles_a and smiles_b:
            fp_a = compute_morgan_fp(smiles_a, n_bits=2048)
            fp_b = compute_morgan_fp(smiles_b, n_bits=2048)
            X[i, col + 4] = compute_tanimoto(fp_a, fp_b)
        else:
            X[i, col + 4] = np.nan
        col += 5

        # Pair-level (8)
        X[i, col] = row[12] or 0  # compound_a_degree
        X[i, col + 1] = row[13] or 0  # compound_b_degree
        X[i, col + 2] = row[3] or 0  # num_cell_lines
        X[i, col + 3] = row[4] or 0  # num_sources
        X[i, col + 4] = row[5] or 0  # num_measurements
        X[i, col + 5] = row[8] or 0.0  # antagonism_fraction
        X[i, col + 6] = row[9] or 0.0  # synergy_fraction
        X[i, col + 7] = row[6] if row[6] is not None else np.nan  # median_zip
        col += 8

        # Tissue context: fraction of measurements per top tissue
        # Query triple-level tissue distribution for this pair
        tissue_fracs = _get_tissue_fractions(conn, pair_id, top_tissues)
        for j, t in enumerate(top_tissues):
            X[i, col + j] = tissue_fracs.get(t, 0.0)
        col += len(top_tissues)

    if use_sentinel:
        X = np.where(np.isnan(X), MISSING_SENTINEL, X)

    logger.info(
        "Built tabular features: %d pairs × %d features",
        X.shape[0], X.shape[1],
    )
    return X, feature_names, pair_ids_out


def _get_tissue_fractions(
    conn, pair_id: int, top_tissues: list[str]
) -> dict[str, float]:
    """Get fraction of cell lines per tissue for a drug pair."""
    rows = conn.execute(
        """SELECT cl.tissue, COUNT(*)
        FROM drug_drug_cell_line_triples t
        JOIN cell_lines cl ON t.cell_line_id = cl.cell_line_id
        WHERE t.pair_id = ?
        GROUP BY cl.tissue""",
        (pair_id,),
    ).fetchall()

    total = sum(r[1] for r in rows)
    if total == 0:
        return {}
    return {r[0]: r[1] / total for r in rows if r[0] in top_tissues}


def build_deepsynergy_features(
    conn,
    pair_ids: list[int] | None = None,
    fp_bits: int = 2048,
) -> tuple[np.ndarray, list[int]]:
    """Build DeepSynergy-style feature vectors: Drug A FP + Drug B FP.

    Args:
        conn: Database connection.
        pair_ids: Optional pair_id filter.
        fp_bits: Morgan fingerprint bits (default 2048).

    Returns:
        (X, pair_ids_out) where X is (n_pairs, fp_bits*2).
    """
    if pair_ids is not None:
        placeholders = ",".join("?" * len(pair_ids))
        query = f"""
            SELECT p.pair_id, ca.canonical_smiles, cb.canonical_smiles
            FROM drug_drug_pairs p
            JOIN compounds ca ON p.compound_a_id = ca.compound_id
            JOIN compounds cb ON p.compound_b_id = cb.compound_id
            WHERE p.pair_id IN ({placeholders})
            ORDER BY p.pair_id
        """
        rows = conn.execute(query, pair_ids).fetchall()
    else:
        query = """
            SELECT p.pair_id, ca.canonical_smiles, cb.canonical_smiles
            FROM drug_drug_pairs p
            JOIN compounds ca ON p.compound_a_id = ca.compound_id
            JOIN compounds cb ON p.compound_b_id = cb.compound_id
            ORDER BY p.pair_id
        """
        rows = conn.execute(query).fetchall()

    n = len(rows)
    X = np.zeros((n, fp_bits * 2), dtype=np.float32)
    pair_ids_out = []

    for i, (pair_id, smiles_a, smiles_b) in enumerate(rows):
        pair_ids_out.append(pair_id)
        X[i, :fp_bits] = compute_morgan_fp(smiles_a, fp_bits)
        X[i, fp_bits:] = compute_morgan_fp(smiles_b, fp_bits)

    logger.info("Built DeepSynergy features: %d pairs × %d", n, fp_bits * 2)
    return X, pair_ids_out


# ── Feature name list (for interpretability) ─────────────────────────

TABULAR_FEATURE_CATEGORIES = {
    "drug_a_descriptors": 8,
    "drug_b_descriptors": 8,
    "target_overlap": 5,
    "pair_stats": 8,
    "tissue_context": 5,  # variable (top_tissues)
}
