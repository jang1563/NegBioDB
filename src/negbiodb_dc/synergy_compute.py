"""Synergy score computation via R SynergyFinder (rpy2 bridge).

Computes ZIP, Bliss, Loewe, and HSA synergy scores from dose-response
inhibition matrices using the R SynergyFinder package (gold standard).

Requires:
    - R >= 4.1 installed
    - R package: SynergyFinder (Bioconductor)
    - Python package: rpy2

Falls back to a pure-Python Bliss independence approximation when R/rpy2
is not available (for testing and environments without R).
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Try to import rpy2; set flag for availability
_RPY2_AVAILABLE = False
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    _RPY2_AVAILABLE = True
except ImportError:
    pass


def is_r_available() -> bool:
    """Check if R and SynergyFinder are available via rpy2."""
    if not _RPY2_AVAILABLE:
        return False
    try:
        importr("synergyfinder")
        return True
    except Exception:
        return False


@dataclass
class SynergyScores:
    """Container for synergy scores computed from a dose-response matrix."""

    zip_score: float | None = None
    bliss_score: float | None = None
    loewe_score: float | None = None
    hsa_score: float | None = None


def compute_synergy_r(
    drug_row: str,
    drug_col: str,
    conc_row: list[float],
    conc_col: list[float],
    response_matrix: list[list[float]],
) -> SynergyScores:
    """Compute synergy scores using R SynergyFinder.

    Args:
        drug_row: Name of the drug in rows.
        drug_col: Name of the drug in columns.
        conc_row: Concentrations for row drug (ascending).
        conc_col: Concentrations for column drug (ascending).
        response_matrix: 2D inhibition values (rows=conc_row, cols=conc_col).
            Values should be % inhibition (0-100).

    Returns:
        SynergyScores with ZIP, Bliss, Loewe, HSA values.

    Raises:
        RuntimeError: If R/SynergyFinder is not available.
    """
    if not _RPY2_AVAILABLE:
        raise RuntimeError(
            "rpy2 not available. Install with: pip install rpy2"
        )

    synergyfinder = importr("synergyfinder")
    base = importr("base")

    # Build input data frame for SynergyFinder
    # Format: long-form with columns: BlockID, DrugRow, DrugCol, ConcRow, ConcCol, Response
    rows = []
    for i, cr in enumerate(conc_row):
        for j, cc in enumerate(conc_col):
            rows.append({
                "BlockID": 1,
                "DrugRow": drug_row,
                "DrugCol": drug_col,
                "ConcRow": cr,
                "ConcCol": cc,
                "Response": response_matrix[i][j],
                "ConcRowUnit": "uM",
                "ConcColUnit": "uM",
            })

    # Convert to R data frame via pandas
    import pandas as pd

    df = pd.DataFrame(rows)
    pandas2ri.activate()
    r_df = pandas2ri.py2rpy(df)

    try:
        # ReshapeData → CalculateSynergy
        reshaped = synergyfinder.ReshapeData(
            r_df,
            data_type="inhibition",
            impute=True,
            noise=True,
        )

        scores = SynergyScores()

        for method in ("ZIP", "Bliss", "Loewe", "HSA"):
            try:
                result = synergyfinder.CalculateSynergy(
                    reshaped, method=method
                )
                # Extract average synergy score
                score_matrix = base.slot(result, "scores")
                avg_score = float(np.mean(
                    pandas2ri.rpy2py(score_matrix[0])
                ))
                setattr(scores, f"{method.lower()}_score", avg_score)
            except Exception as e:
                logger.warning("Failed to compute %s: %s", method, e)

        return scores
    finally:
        pandas2ri.deactivate()


def compute_synergy_bliss_python(
    conc_row: list[float],
    conc_col: list[float],
    response_matrix: list[list[float]],
) -> SynergyScores:
    """Compute Bliss independence synergy using pure Python (fallback).

    This is a simplified approximation used when R is not available.
    The Bliss independence model predicts:
        E_ab = E_a + E_b - E_a * E_b
    Synergy score = E_observed - E_predicted

    Positive = synergistic, negative = antagonistic.

    Args:
        conc_row: Concentrations for row drug.
        conc_col: Concentrations for column drug.
        response_matrix: 2D inhibition values (rows=conc_row, cols=conc_col).

    Returns:
        SynergyScores with bliss_score only.
    """
    mat = np.array(response_matrix) / 100.0  # Convert to fraction

    # Mono-therapy responses: first row (drug_col only) and first column (drug_row only)
    mono_row = mat[:, 0]  # Drug A alone at each concentration
    mono_col = mat[0, :]  # Drug B alone at each concentration

    # Bliss independence prediction for combination entries
    bliss_scores = []
    for i in range(1, len(conc_row)):
        for j in range(1, len(conc_col)):
            e_a = mono_row[i]
            e_b = mono_col[j]
            e_predicted = e_a + e_b - e_a * e_b
            e_observed = mat[i, j]
            bliss_scores.append((e_observed - e_predicted) * 100.0)

    if not bliss_scores:
        return SynergyScores(bliss_score=None)

    return SynergyScores(bliss_score=float(np.mean(bliss_scores)))


def compute_synergy(
    drug_row: str,
    drug_col: str,
    conc_row: list[float],
    conc_col: list[float],
    response_matrix: list[list[float]],
    use_r: bool | None = None,
) -> SynergyScores:
    """Compute synergy scores, using R SynergyFinder if available.

    Args:
        drug_row: Name of the row drug.
        drug_col: Name of the column drug.
        conc_row: Row drug concentrations.
        conc_col: Column drug concentrations.
        response_matrix: 2D inhibition matrix.
        use_r: Force R (True), force Python (False), or auto-detect (None).

    Returns:
        SynergyScores with available scores.
    """
    if use_r is None:
        use_r = is_r_available()

    if use_r:
        return compute_synergy_r(
            drug_row, drug_col, conc_row, conc_col, response_matrix
        )
    else:
        return compute_synergy_bliss_python(conc_row, conc_col, response_matrix)
