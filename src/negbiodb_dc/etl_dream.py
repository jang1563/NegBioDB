"""ETL module for AstraZeneca-Sanger DREAM Challenge data (tertiary DC source).

Parses:
  - ch1_train_combination_and_monoTherapy.csv — Combination and monotherapy data
  - OI_combinations_synergy_scores_final.txt — Pre-computed Loewe/Bliss scores

The DREAM Challenge provides pre-computed synergy scores for ~11,576 experiments,
910 drug combinations, 85 cell lines.

License: CC BY 4.0 (requires Synapse account for download).
"""

import logging
from pathlib import Path

import pandas as pd

from negbiodb_dc.dc_db import classify_synergy, get_connection, normalize_pair

logger = logging.getLogger(__name__)


def parse_dream_synergy_scores(scores_path: Path) -> pd.DataFrame:
    """Parse DREAM Challenge synergy scores file.

    Expected format: Tab-separated with columns including:
        COMBINATION_ID, CELL_LINE, DRUG_A, DRUG_B,
        SYNERGY_SCORE (Loewe-based), SYNERGY_SCORE_BLISS

    Returns:
        DataFrame with columns: DRUG_A, DRUG_B, CELL_LINE,
        LOEWE_SCORE, BLISS_SCORE
    """
    logger.info("Reading DREAM synergy scores: %s", scores_path)

    # Try tab-separated first, then comma
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(scores_path, sep=sep, low_memory=False)
            if len(df.columns) > 1:
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Cannot parse DREAM scores file: {scores_path}")

    # Normalize column names
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

    # Identify key columns with flexible naming
    drug_a_col = next(
        (c for c in ("DRUG_A", "COMPOUND_A", "DRUG_ROW") if c in df.columns), None
    )
    drug_b_col = next(
        (c for c in ("DRUG_B", "COMPOUND_B", "DRUG_COL") if c in df.columns), None
    )
    cl_col = next(
        (c for c in ("CELL_LINE", "CELL_LINE_NAME", "CELLNAME") if c in df.columns),
        None,
    )
    loewe_col = next(
        (c for c in ("SYNERGY_SCORE", "LOEWE_SCORE", "LOEWE", "SYNERGY_LOEWE")
         if c in df.columns),
        None,
    )
    bliss_col = next(
        (c for c in ("SYNERGY_SCORE_BLISS", "BLISS_SCORE", "BLISS", "SYNERGY_BLISS")
         if c in df.columns),
        None,
    )

    if not all([drug_a_col, drug_b_col, cl_col]):
        raise ValueError(
            f"Cannot find required columns. Found: {list(df.columns[:10])}"
        )

    rename = {drug_a_col: "DRUG_A", drug_b_col: "DRUG_B", cl_col: "CELL_LINE"}
    if loewe_col:
        rename[loewe_col] = "LOEWE_SCORE"
    if bliss_col:
        rename[bliss_col] = "BLISS_SCORE"
    df = df.rename(columns=rename)

    logger.info("Parsed %d DREAM synergy records", len(df))
    return df


def parse_dream_combination_csv(csv_path: Path) -> pd.DataFrame:
    """Parse DREAM ch1_train_combination_and_monoTherapy.csv.

    This file contains combination and monotherapy dose-response data.
    We extract unique drug pair × cell line × synergy score entries.

    Returns:
        DataFrame with columns: DRUG_A, DRUG_B, CELL_LINE,
        LOEWE_SCORE, BLISS_SCORE (if available)
    """
    logger.info("Reading DREAM combination CSV: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

    # Filter to combination rows only (not monotherapy)
    combo_col = next(
        (c for c in ("IC_TYPE", "TREATMENT_TYPE", "COMBINATION") if c in df.columns),
        None,
    )
    if combo_col:
        df = df[df[combo_col].str.lower().str.contains("combo", na=False)]

    # Identify columns
    drug_a_col = next(
        (c for c in ("COMPOUND_A", "DRUG_A", "DRUG_ROW") if c in df.columns), None
    )
    drug_b_col = next(
        (c for c in ("COMPOUND_B", "DRUG_B", "DRUG_COL") if c in df.columns), None
    )
    cl_col = next(
        (c for c in ("CELL_LINE_NAME", "CELL_LINE", "CELLNAME") if c in df.columns),
        None,
    )

    if not all([drug_a_col, drug_b_col, cl_col]):
        raise ValueError(
            f"Cannot find required columns. Found: {list(df.columns[:10])}"
        )

    # Group by drug pair + cell line
    group_cols = [drug_a_col, drug_b_col, cl_col]
    agg = df.groupby(group_cols).size().reset_index(name="N_MEASUREMENTS")
    agg = agg.rename(columns={
        drug_a_col: "DRUG_A", drug_b_col: "DRUG_B", cl_col: "CELL_LINE",
    })

    # Extract synergy scores if present in this file
    for score_src, score_dst in [
        ("SYNERGY_SCORE", "LOEWE_SCORE"), ("SYNERGY_LOEWE", "LOEWE_SCORE"),
        ("SYNERGY_SCORE_BLISS", "BLISS_SCORE"), ("SYNERGY_BLISS", "BLISS_SCORE"),
    ]:
        if score_src in df.columns:
            score_agg = df.groupby(group_cols)[score_src].mean().reset_index()
            score_agg = score_agg.rename(columns={
                drug_a_col: "DRUG_A", drug_b_col: "DRUG_B",
                cl_col: "CELL_LINE", score_src: score_dst,
            })
            agg = agg.merge(score_agg, on=["DRUG_A", "DRUG_B", "CELL_LINE"], how="left")

    logger.info("Parsed %d DREAM combinations", len(agg))
    return agg


def load_dream_synergy(
    conn,
    df: pd.DataFrame,
    compound_cache: dict[str, int],
    cell_line_cache: dict[str, int],
    batch_size: int = 5000,
) -> dict[str, int]:
    """Load DREAM synergy data into dc_synergy_results.

    Args:
        conn: Database connection.
        df: DataFrame from parse_dream_synergy_scores or parse_dream_combination_csv.
        compound_cache: drug_name → compound_id mapping.
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

    for _, row in df.iterrows():
        drug_a = str(row["DRUG_A"]).strip()
        drug_b = str(row["DRUG_B"]).strip()
        cl_name = str(row["CELL_LINE"]).strip()

        # Auto-create unknown compounds
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

        # Auto-create unknown cell lines
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

        loewe = float(row["LOEWE_SCORE"]) if pd.notna(row.get("LOEWE_SCORE")) else None
        bliss = float(row["BLISS_SCORE"]) if pd.notna(row.get("BLISS_SCORE")) else None

        # Use Loewe for classification (DREAM primary metric), fallback to Bliss
        primary_score = loewe if loewe is not None else bliss
        synergy_class = classify_synergy(primary_score)

        # DREAM provides full dose-response matrices → bronze tier
        tier = "bronze"
        evidence = "dose_response_matrix"

        conn.execute(
            """INSERT INTO dc_synergy_results
            (compound_a_id, compound_b_id, cell_line_id,
             loewe_score, bliss_score,
             synergy_class, confidence_tier, evidence_type,
             source_db, has_dose_matrix)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'az_dream', 1)""",
            (cid_a, cid_b, cl_id,
             loewe, bliss,
             synergy_class, tier, evidence),
        )
        stats["results_inserted"] += 1

        if stats["results_inserted"] % batch_size == 0:
            conn.commit()

    conn.commit()
    logger.info(
        "DREAM ETL: %d results inserted, %d unknown drugs, %d unknown cell lines",
        stats["results_inserted"],
        stats["skipped_unknown_drug"],
        stats["skipped_unknown_cell_line"],
    )
    return stats


def run_dream_etl(
    db_path: Path,
    data_dir: Path,
    batch_size: int = 5000,
) -> dict[str, int]:
    """Run full AZ-DREAM ETL pipeline.

    Looks for synergy scores file first; falls back to combination CSV.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    conn = get_connection(db_path)
    try:
        # Load existing caches
        compound_cache = {
            row[1]: row[0]
            for row in conn.execute("SELECT compound_id, drug_name FROM compounds")
        }
        cell_line_cache = {
            row[1]: row[0]
            for row in conn.execute(
                "SELECT cell_line_id, cell_line_name FROM cell_lines"
            )
        }

        # Try pre-computed scores first (preferred)
        scores_path = data_dir / "OI_combinations_synergy_scores_final.txt"
        if scores_path.exists():
            df = parse_dream_synergy_scores(scores_path)
        else:
            # Fall back to combination CSV
            csv_path = data_dir / "ch1_train_combination_and_monoTherapy.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"No DREAM data found in {data_dir}. Expected either "
                    f"OI_combinations_synergy_scores_final.txt or "
                    f"ch1_train_combination_and_monoTherapy.csv"
                )
            df = parse_dream_combination_csv(csv_path)

        stats = load_dream_synergy(
            conn, df, compound_cache, cell_line_cache, batch_size
        )
        return stats
    finally:
        conn.close()
