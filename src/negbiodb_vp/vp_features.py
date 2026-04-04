"""VP domain feature engineering — 56-dimensional tabular feature vector.

Feature categories (56 total):
  7  Variant scores: CADD, REVEL, AlphaMissense, PhyloP, GERP, SIFT, PolyPhen2
  6  Variant type one-hot: snv, insertion, deletion, indel, duplication, other
  8  Consequence one-hot: missense, nonsense, synonymous, frameshift, splice,
     inframe_indel, intronic, other
  9  Population AF: log10(AF + 1e-8) for gnomAD global + 8 ancestry-specific
  3  Gene constraint: pLI, LOEUF, missense_z
  5  Inheritance one-hot: AD, AR, XL, other, unknown
  5  Amino acid change: Blosum62, Grantham distance, Δhydrophobicity, Δcharge, Δsize
  5  Pair-level: variant_degree, disease_degree, num_submissions, num_submitters,
     max_population_af
  8  Domain context: is_in_known_domain, gene_disease_count, ClinGen validity (ordinal),
     exon_flag, has_functional_evidence, BA1_met, BS1_met, num_benign_criteria
"""

import math
import re

import numpy as np

# Feature vector dimension
TABULAR_DIM = 56

# Sentinel for missing values (MLP/ESM2 models — XGBoost handles NaN natively)
MISSING_SENTINEL = -1.0

# ── Variant type categories ───────────────────────────────────────────

VARIANT_TYPES = ["single nucleotide variant", "Insertion", "Deletion", "Indel", "Duplication", "other"]

# ── Consequence categories ────────────────────────────────────────────

CONSEQUENCE_TYPES = [
    "missense", "nonsense", "synonymous", "frameshift",
    "splice", "inframe_indel", "intronic", "other",
]

# ── Inheritance categories ────────────────────────────────────────────

INHERITANCE_MODES = ["AD", "AR", "XL", "Other", "Unknown"]

# ── ClinGen validity ordinal ─────────────────────────────────────────

CLINGEN_ORDINAL = {
    "Definitive": 5,
    "Strong": 4,
    "Moderate": 3,
    "Limited": 2,
    "Disputed": 1,
    "Refuted": 0,
    "No Known Disease Relationship": 0,
    None: 0,
}

# ── Amino acid properties ────────────────────────────────────────────

# Standard amino acid single-letter codes
_AA_CODES = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale
_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Amino acid charge at pH 7
_CHARGE = {
    "R": 1, "K": 1, "H": 0.1, "D": -1, "E": -1,
}

# Amino acid molecular weight (approximate Da)
_MW = {
    "G": 57, "A": 71, "V": 99, "L": 113, "I": 113,
    "P": 97, "F": 147, "W": 186, "M": 131, "S": 87,
    "T": 101, "C": 103, "Y": 163, "H": 137, "D": 115,
    "E": 129, "N": 114, "Q": 128, "K": 128, "R": 156,
}

# Blosum62 matrix (selected substitution scores)
_BLOSUM62 = {
    ("A", "A"): 4, ("A", "R"): -1, ("A", "N"): -2, ("A", "D"): -2, ("A", "C"): 0,
    ("A", "Q"): -1, ("A", "E"): -1, ("A", "G"): 0, ("A", "H"): -2, ("A", "I"): -1,
    ("A", "L"): -1, ("A", "K"): -1, ("A", "M"): -1, ("A", "F"): -2, ("A", "P"): -1,
    ("A", "S"): 1, ("A", "T"): 0, ("A", "W"): -3, ("A", "Y"): -2, ("A", "V"): 0,
    ("R", "R"): 5, ("R", "N"): 0, ("R", "D"): -2, ("R", "C"): -3, ("R", "Q"): 1,
    ("R", "E"): 0, ("R", "G"): -2, ("R", "H"): 0, ("R", "I"): -3, ("R", "L"): -2,
    ("R", "K"): 2, ("R", "M"): -1, ("R", "F"): -3, ("R", "P"): -2, ("R", "S"): -1,
    ("R", "T"): -1, ("R", "W"): -3, ("R", "Y"): -2, ("R", "V"): -3,
    ("N", "N"): 6, ("N", "D"): 1, ("N", "C"): -3, ("N", "Q"): 0, ("N", "E"): 0,
    ("N", "G"): 0, ("N", "H"): 1, ("N", "I"): -3, ("N", "L"): -3, ("N", "K"): 0,
    ("N", "M"): -2, ("N", "F"): -3, ("N", "P"): -2, ("N", "S"): 1, ("N", "T"): 0,
    ("N", "W"): -4, ("N", "Y"): -2, ("N", "V"): -3,
    ("D", "D"): 6, ("D", "C"): -3, ("D", "Q"): 0, ("D", "E"): 2, ("D", "G"): -1,
    ("D", "H"): -1, ("D", "I"): -3, ("D", "L"): -4, ("D", "K"): -1, ("D", "M"): -3,
    ("D", "F"): -3, ("D", "P"): -1, ("D", "S"): 0, ("D", "T"): -1, ("D", "W"): -4,
    ("D", "Y"): -3, ("D", "V"): -3,
    ("C", "C"): 9, ("C", "Q"): -3, ("C", "E"): -4, ("C", "G"): -3, ("C", "H"): -3,
    ("C", "I"): -1, ("C", "L"): -1, ("C", "K"): -3, ("C", "M"): -1, ("C", "F"): -2,
    ("C", "P"): -3, ("C", "S"): -1, ("C", "T"): -1, ("C", "W"): -2, ("C", "Y"): -2,
    ("C", "V"): -1,
}

# Grantham distance matrix (subset — most common substitutions)
_GRANTHAM = {
    ("A", "G"): 60, ("A", "V"): 64, ("A", "D"): 126, ("A", "E"): 107,
    ("A", "P"): 27, ("A", "S"): 99, ("A", "T"): 58, ("G", "V"): 109,
    ("G", "D"): 94, ("G", "E"): 98, ("G", "R"): 125, ("G", "S"): 56,
    ("D", "E"): 45, ("D", "N"): 23, ("D", "H"): 81, ("I", "V"): 29,
    ("I", "L"): 5, ("I", "M"): 10, ("L", "V"): 32, ("L", "M"): 15,
    ("K", "R"): 26, ("K", "Q"): 53, ("K", "N"): 94, ("K", "E"): 56,
    ("F", "Y"): 22, ("F", "W"): 40, ("S", "T"): 58, ("S", "N"): 46,
    ("R", "Q"): 43, ("R", "H"): 29, ("R", "K"): 26,
}


def _get_blosum62(aa1: str, aa2: str) -> float:
    """Get Blosum62 score for amino acid substitution."""
    if aa1 == aa2:
        return _BLOSUM62.get((aa1, aa2), 4)  # Default self-match
    key = (aa1, aa2) if (aa1, aa2) in _BLOSUM62 else (aa2, aa1)
    return float(_BLOSUM62.get(key, -1))  # Default mismatch


def _get_grantham(aa1: str, aa2: str) -> float:
    """Get Grantham distance for amino acid substitution."""
    if aa1 == aa2:
        return 0.0
    key = (aa1, aa2) if (aa1, aa2) in _GRANTHAM else (aa2, aa1)
    return float(_GRANTHAM.get(key, 100))  # Default moderate distance


def _parse_aa_change(hgvs_protein: str | None) -> tuple[str | None, str | None]:
    """Extract reference and alternate amino acids from HGVS protein notation.

    Examples: 'p.Ala1708Asp' → ('A', 'D'), 'p.A1708D' → ('A', 'D')
    """
    if not hgvs_protein:
        return None, None

    # Three-letter code mapping
    aa3to1 = {
        "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
        "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
        "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
        "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
        "Ter": "*",
    }

    # Try 3-letter: p.Ala1708Asp
    match = re.match(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", hgvs_protein)
    if match:
        ref = aa3to1.get(match.group(1))
        alt = aa3to1.get(match.group(3))
        return ref, alt

    # Try 1-letter: p.A1708D
    match = re.match(r"p\.([A-Z])(\d+)([A-Z*])", hgvs_protein)
    if match:
        ref = match.group(1) if match.group(1) in _AA_CODES else None
        alt = match.group(3) if match.group(3) in _AA_CODES else None
        return ref, alt

    return None, None


def compute_aa_features(hgvs_protein: str | None) -> list[float]:
    """Compute 5 amino acid change features.

    Returns: [blosum62, grantham, delta_hydrophobicity, delta_charge, delta_size]
    All 0.0 for non-missense (when AA change cannot be parsed).
    """
    ref_aa, alt_aa = _parse_aa_change(hgvs_protein)

    if ref_aa is None or alt_aa is None or ref_aa not in _AA_CODES or alt_aa not in _AA_CODES:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    blosum = _get_blosum62(ref_aa, alt_aa)
    grantham = _get_grantham(ref_aa, alt_aa)
    delta_hydro = _HYDROPHOBICITY.get(alt_aa, 0) - _HYDROPHOBICITY.get(ref_aa, 0)
    delta_charge = _CHARGE.get(alt_aa, 0) - _CHARGE.get(ref_aa, 0)
    delta_size = _MW.get(alt_aa, 100) - _MW.get(ref_aa, 100)

    return [blosum, grantham, delta_hydro, delta_charge, float(delta_size)]


def compute_features(
    row: dict,
    use_sentinel: bool = False,
) -> np.ndarray:
    """Compute 56-dimensional tabular feature vector from a variant-disease pair row.

    Args:
        row: Dict with variant + gene + pair data (from SQL query or parquet)
        use_sentinel: If True, replace NaN with MISSING_SENTINEL (-1.0) for MLP/ESM2.
                      If False, keep NaN for XGBoost.

    Returns:
        numpy array of shape (56,)
    """
    fill = MISSING_SENTINEL if use_sentinel else float("nan")
    features = []

    # ── 7 Variant scores ──────────────────────────────────────────
    for col in ["cadd_phred", "revel_score", "alphamissense_score",
                "phylop_score", "gerp_score", "sift_score", "polyphen2_score"]:
        val = row.get(col)
        features.append(float(val) if val is not None else fill)

    # ── 6 Variant type one-hot ────────────────────────────────────
    vtype = row.get("variant_type", "other")
    for t in VARIANT_TYPES:
        features.append(1.0 if vtype == t else 0.0)

    # ── 8 Consequence one-hot ─────────────────────────────────────
    consequence = row.get("consequence_type", "other")
    for c in CONSEQUENCE_TYPES:
        features.append(1.0 if consequence == c else 0.0)

    # ── 9 Population AF (log-transformed) ─────────────────────────
    for col in ["gnomad_af_global", "gnomad_af_afr", "gnomad_af_amr",
                "gnomad_af_asj", "gnomad_af_eas", "gnomad_af_fin",
                "gnomad_af_nfe", "gnomad_af_sas", "gnomad_af_oth"]:
        val = row.get(col)
        if val is not None:
            features.append(math.log10(float(val) + 1e-8))
        else:
            features.append(fill)

    # ── 3 Gene constraint ─────────────────────────────────────────
    for col in ["pli_score", "loeuf_score", "missense_z"]:
        val = row.get(col)
        features.append(float(val) if val is not None else fill)

    # ── 5 Inheritance one-hot ─────────────────────────────────────
    moi = row.get("gene_moi", "Unknown")
    if moi is None:
        moi = "Unknown"
    for mode in INHERITANCE_MODES:
        features.append(1.0 if moi == mode else 0.0)

    # ── 5 Amino acid change ───────────────────────────────────────
    hgvs_p = row.get("hgvs_protein")
    aa_feats = compute_aa_features(hgvs_p)
    features.extend(aa_feats)

    # ── 5 Pair-level ──────────────────────────────────────────────
    for col in ["variant_degree", "disease_degree", "num_submissions",
                "num_submitters", "max_population_af"]:
        val = row.get(col)
        features.append(float(val) if val is not None else 0.0)

    # ── 8 Domain context ──────────────────────────────────────────
    features.append(1.0 if row.get("is_in_known_domain") else 0.0)
    features.append(float(row.get("gene_disease_count", 0)))
    clingen_val = row.get("clingen_validity")
    features.append(float(CLINGEN_ORDINAL.get(clingen_val, 0)))
    features.append(1.0 if row.get("exon_flag") else 0.0)
    features.append(1.0 if row.get("has_functional_evidence") else 0.0)

    # BA1 met: global AF > 0.01 (standalone benign)
    af_global = row.get("gnomad_af_global")
    features.append(1.0 if af_global is not None and float(af_global) > 0.01 else 0.0)

    # BS1 met: AF > 0.001 (expected for disorder) — simplified
    features.append(1.0 if af_global is not None and float(af_global) > 0.001 else 0.0)

    features.append(float(row.get("num_benign_criteria", 0)))

    result = np.array(features, dtype=np.float32)
    assert result.shape == (TABULAR_DIM,), f"Expected {TABULAR_DIM} dims, got {result.shape[0]}"
    return result


# Feature names for interpretability
FEATURE_NAMES = (
    # 7 scores
    ["cadd_phred", "revel_score", "alphamissense_score",
     "phylop_score", "gerp_score", "sift_score", "polyphen2_score"]
    # 6 variant type
    + [f"vtype_{t.replace(' ', '_')}" for t in VARIANT_TYPES]
    # 8 consequence
    + [f"csq_{c}" for c in CONSEQUENCE_TYPES]
    # 9 population AF
    + [f"log_af_{p}" for p in ["global", "afr", "amr", "asj", "eas", "fin", "nfe", "sas", "oth"]]
    # 3 gene constraint
    + ["pli_score", "loeuf_score", "missense_z"]
    # 5 inheritance
    + [f"moi_{m}" for m in INHERITANCE_MODES]
    # 5 AA change
    + ["blosum62", "grantham", "delta_hydrophobicity", "delta_charge", "delta_size"]
    # 5 pair-level
    + ["variant_degree", "disease_degree", "num_submissions", "num_submitters", "max_population_af"]
    # 8 domain context
    + ["is_in_known_domain", "gene_disease_count", "clingen_validity_ord",
       "exon_flag", "has_functional_evidence", "BA1_met", "BS1_met", "num_benign_criteria"]
)

assert len(FEATURE_NAMES) == TABULAR_DIM, f"Feature names mismatch: {len(FEATURE_NAMES)} != {TABULAR_DIM}"
