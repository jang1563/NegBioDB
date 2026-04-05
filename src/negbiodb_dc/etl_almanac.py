"""ETL module for NCI-ALMANAC data (secondary DC domain source).

Parses ComboDrugGrowth_Nov2017.csv with ComboScore values.
NCI-ALMANAC provides pre-computed ComboScore (positive = synergistic).

ComboScore → ZIP-equivalent mapping (approximate):
    ComboScore > 50   → strongly_synergistic
    20 < ComboScore ≤ 50 → synergistic
    -20 ≤ ComboScore ≤ 20 → additive
    -50 ≤ ComboScore < -20 → antagonistic
    ComboScore < -50  → strongly_antagonistic

License: Public domain (US government work).
"""

import logging
import math
from pathlib import Path

import pandas as pd

from negbiodb_dc.dc_db import get_connection, normalize_pair

logger = logging.getLogger(__name__)


def classify_combo_score(combo_score: float | None) -> str | None:
    """Classify NCI-ALMANAC ComboScore into synergy class."""
    if combo_score is None or math.isnan(combo_score):
        return None
    if combo_score > 50:
        return "strongly_synergistic"
    elif combo_score > 20:
        return "synergistic"
    elif combo_score >= -20:
        return "additive"
    elif combo_score >= -50:
        return "antagonistic"
    else:
        return "strongly_antagonistic"


def parse_almanac_csv(csv_path: Path) -> pd.DataFrame:
    """Parse NCI-ALMANAC ComboDrugGrowth CSV.

    Reads the CSV, groups by (drug pair, cell line), and computes
    the mean ComboScore for each combination.

    Returns:
        DataFrame with columns: DRUG_A, DRUG_B, CELLNAME, COMBO_SCORE, N_CONC
    """
    logger.info("Reading NCI-ALMANAC CSV: %s", csv_path)

    # NCI-ALMANAC CSV columns include:
    # NSC1, NSC2, CELLNAME, PANEL, COMBODRUGSEQ, SAMPLE_ID,
    # CONC1, CONC2, TESTVALUE, CONTROLVALUE, TZVALUE, PERCENTGROWTH,
    # PERCENTGROWTHNOTZ, EXPECTEDGROWTH, SCORE
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]

    # Identify key columns
    drug_a_col = next(
        (c for c in ("NSC1", "DRUG1", "DRUG_A") if c in df.columns), None
    )
    drug_b_col = next(
        (c for c in ("NSC2", "DRUG2", "DRUG_B") if c in df.columns), None
    )
    cell_col = next(
        (c for c in ("CELLNAME", "CELL_LINE", "CELL") if c in df.columns), None
    )
    score_col = next(
        (c for c in ("SCORE", "COMBOSCORE", "COMBO_SCORE") if c in df.columns), None
    )

    if not all([drug_a_col, drug_b_col, cell_col]):
        raise ValueError(
            f"Cannot find required columns. Found: {list(df.columns[:10])}"
        )

    if not score_col:
        logger.warning("No SCORE column found; ComboScore will be NULL for all rows")

    # Aggregate by drug pair + cell line
    agg_dict = {"N_CONC": (drug_a_col, "count")}
    if score_col:
        agg_dict["COMBO_SCORE"] = (score_col, "mean")
    agg = df.groupby([drug_a_col, drug_b_col, cell_col]).agg(**agg_dict).reset_index()
    if "COMBO_SCORE" not in agg.columns:
        agg["COMBO_SCORE"] = None

    agg = agg.rename(columns={
        drug_a_col: "DRUG_A",
        drug_b_col: "DRUG_B",
        cell_col: "CELLNAME",
    })

    logger.info("Parsed %d unique combinations from NCI-ALMANAC", len(agg))
    return agg


def load_almanac_synergy(
    conn,
    agg_df: pd.DataFrame,
    compound_cache: dict[str, int],
    cell_line_cache: dict[str, int],
    batch_size: int = 5000,
) -> dict[str, int]:
    """Load NCI-ALMANAC aggregated data into dc_synergy_results.

    Args:
        conn: Database connection.
        agg_df: Aggregated DataFrame from parse_almanac_csv.
        compound_cache: drug_name/NSC → compound_id mapping.
        cell_line_cache: cell_line_name → cell_line_id mapping.
        batch_size: Commit every N inserts.

    Returns:
        Stats dict.
    """
    stats = {
        "results_inserted": 0,
        "skipped_unknown_drug": 0,
        "skipped_unknown_cell_line": 0,
        "skipped_self_combination": 0,
    }

    for idx, row in agg_df.iterrows():
        drug_a = str(row["DRUG_A"]).strip()
        drug_b = str(row["DRUG_B"]).strip()
        cl_name = str(row["CELLNAME"]).strip()

        # Resolve compound IDs (auto-create if not in cache)
        for drug_name in (drug_a, drug_b):
            if drug_name not in compound_cache:
                conn.execute(
                    "INSERT OR IGNORE INTO compounds (drug_name) VALUES (?)",
                    (drug_name,),
                )
                r = conn.execute(
                    "SELECT compound_id FROM compounds WHERE drug_name = ?",
                    (drug_name,),
                ).fetchone()
                if r:
                    compound_cache[drug_name] = r[0]

        if drug_a not in compound_cache or drug_b not in compound_cache:
            stats["skipped_unknown_drug"] += 1
            continue

        if cl_name not in cell_line_cache:
            conn.execute(
                "INSERT OR IGNORE INTO cell_lines (cell_line_name) VALUES (?)",
                (cl_name,),
            )
            r = conn.execute(
                "SELECT cell_line_id FROM cell_lines WHERE cell_line_name = ?",
                (cl_name,),
            ).fetchone()
            if r:
                cell_line_cache[cl_name] = r[0]

        if cl_name not in cell_line_cache:
            stats["skipped_unknown_cell_line"] += 1
            continue

        cid_a = compound_cache[drug_a]
        cid_b = compound_cache[drug_b]

        if cid_a == cid_b:
            stats["skipped_self_combination"] += 1
            continue

        cid_a, cid_b = normalize_pair(cid_a, cid_b)
        cl_id = cell_line_cache[cl_name]

        combo_score = float(row["COMBO_SCORE"]) if pd.notna(row.get("COMBO_SCORE")) else None
        n_conc = int(row.get("N_CONC", 1))
        synergy_class = classify_combo_score(combo_score)
        has_matrix = n_conc >= 9  # NCI-ALMANAC uses 3x3 = 9 dose combos
        tier = "bronze" if has_matrix else "copper"
        evidence = "dose_response_matrix" if has_matrix else "single_concentration"

        conn.execute(
            """INSERT INTO dc_synergy_results
            (compound_a_id, compound_b_id, cell_line_id,
             combo_score, synergy_class, confidence_tier, evidence_type,
             source_db, num_concentrations, has_dose_matrix)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'nci_almanac', ?, ?)""",
            (cid_a, cid_b, cl_id,
             combo_score, synergy_class, tier, evidence,
             n_conc, int(has_matrix)),
        )
        stats["results_inserted"] += 1

        if stats["results_inserted"] % batch_size == 0:
            conn.commit()

    conn.commit()
    logger.info(
        "NCI-ALMANAC ETL: %d results inserted, %d unknown drugs, %d unknown cell lines",
        stats["results_inserted"],
        stats["skipped_unknown_drug"],
        stats["skipped_unknown_cell_line"],
    )
    return stats


def run_almanac_etl(
    db_path: Path,
    data_dir: Path,
    batch_size: int = 5000,
) -> dict[str, int]:
    """Run full NCI-ALMANAC ETL pipeline."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    csv_path = data_dir / "ComboDrugGrowth_Nov2017.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"NCI-ALMANAC CSV not found: {csv_path}")

    agg_df = parse_almanac_csv(csv_path)

    conn = get_connection(db_path)
    try:
        # Load existing caches
        compound_cache = {
            row[1]: row[0]
            for row in conn.execute("SELECT compound_id, drug_name FROM compounds")
        }
        cell_line_cache = {
            row[1]: row[0]
            for row in conn.execute("SELECT cell_line_id, cell_line_name FROM cell_lines")
        }

        stats = load_almanac_synergy(
            conn, agg_df, compound_cache, cell_line_cache, batch_size
        )
        return stats
    finally:
        conn.close()
