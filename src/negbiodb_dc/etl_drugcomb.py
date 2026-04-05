"""ETL module for DrugComb data (primary DC domain source).

Parses:
  - drugcomb_data_v1.4.csv (2 GB) — Raw dose-response inhibition data
  - DrugComb_drug_identifiers.xlsx — Drug name → PubChem CID, InChIKey, SMILES
  - DrugComb_cell_line_identifiers.xlsx — Cell line → COSMIC ID, tissue

Loading strategy:
  1. Parse drug/cell line identifiers first (populate entities)
  2. Parse main CSV in chunks (100K rows) to avoid memory issues
  3. Group by block_id (experiment), compute synergy scores per block
  4. Insert results with pair normalization (compound_a_id < compound_b_id)
"""

import logging
from pathlib import Path

import pandas as pd

from negbiodb_dc.dc_db import classify_synergy, get_connection, normalize_pair, refresh_all_drug_pairs

logger = logging.getLogger(__name__)


def parse_drug_identifiers(xlsx_path: Path) -> list[dict]:
    """Parse DrugComb drug identifiers XLSX.

    Returns list of dicts with keys:
        drug_name, pubchem_cid, inchikey, canonical_smiles
    """
    df = pd.read_excel(xlsx_path)
    # Normalize column names (DrugComb uses various conventions)
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("dname", "drug_name") or ("drug" in cl and "name" in cl):
            col_map[col] = "drug_name"
        elif cl == "cid" or cl == "pubchem_cid" or cl == "pubchemcid":
            # Only exact 'cid' column — not cid_m / cid_s variants
            col_map[col] = "pubchem_cid"
        elif "pubchem" in cl and "cid" in cl:
            col_map[col] = "pubchem_cid"
        elif "inchikey" in cl:
            col_map[col] = "inchikey"
        elif "smiles" in cl:
            col_map[col] = "canonical_smiles"
    df = df.rename(columns=col_map)

    records = []
    for _, row in df.iterrows():
        raw_name = row.get("drug_name")
        if raw_name is None or (hasattr(raw_name, '__class__') and pd.isna(raw_name)):
            continue
        name = str(raw_name).strip()
        if not name or name.lower() == "nan":
            continue
        rec = {"drug_name": name}
        for field in ("pubchem_cid", "inchikey", "canonical_smiles"):
            val = row.get(field)
            if pd.notna(val):
                if field == "pubchem_cid":
                    try:
                        rec[field] = int(float(val))
                    except (ValueError, TypeError):
                        rec[field] = None
                else:
                    rec[field] = str(val).strip()
            else:
                rec[field] = None
        records.append(rec)

    logger.info("Parsed %d drug identifiers from %s", len(records), xlsx_path.name)
    return records


def parse_cell_line_identifiers(xlsx_path: Path) -> list[dict]:
    """Parse DrugComb cell line identifiers XLSX.

    Returns list of dicts with keys:
        cell_line_name, cosmic_id, tissue, cancer_type
    """
    df = pd.read_excel(xlsx_path)
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("name", "cell_line_name", "cell_line") or ("cell" in cl and "line" in cl and "name" in cl):
            col_map[col] = "cell_line_name"
        elif "cosmic" in cl:
            col_map[col] = "cosmic_id"
        elif "tissue" in cl:
            col_map[col] = "tissue"
        elif "cancer" in cl and "type" in cl:
            col_map[col] = "cancer_type"
    df = df.rename(columns=col_map)

    records = []
    for _, row in df.iterrows():
        raw_name = row.get("cell_line_name")
        if raw_name is None or (hasattr(raw_name, '__class__') and pd.isna(raw_name)):
            continue
        name = str(raw_name).strip()
        if not name or name.lower() == "nan":
            continue
        rec = {"cell_line_name": name}
        for field in ("cosmic_id", "tissue", "cancer_type"):
            val = row.get(field)
            if pd.notna(val):
                if field == "cosmic_id":
                    try:
                        rec[field] = int(float(val))
                    except (ValueError, TypeError):
                        rec[field] = None
                else:
                    rec[field] = str(val).strip()
            else:
                rec[field] = None
        records.append(rec)

    logger.info("Parsed %d cell line identifiers from %s", len(records), xlsx_path.name)
    return records


def load_compounds(conn, drug_records: list[dict]) -> dict[str, int]:
    """Insert compounds and return name → compound_id cache."""
    cache = {}
    for rec in drug_records:
        conn.execute(
            """INSERT OR IGNORE INTO compounds
            (drug_name, pubchem_cid, inchikey, canonical_smiles)
            VALUES (?, ?, ?, ?)""",
            (rec["drug_name"], rec.get("pubchem_cid"),
             rec.get("inchikey"), rec.get("canonical_smiles")),
        )
    conn.commit()

    for row in conn.execute("SELECT compound_id, drug_name FROM compounds"):
        cache[row[1]] = row[0]

    logger.info("Loaded %d compounds", len(cache))
    return cache


def load_cell_lines(conn, cl_records: list[dict]) -> dict[str, int]:
    """Insert cell lines and return name → cell_line_id cache."""
    cache = {}
    for rec in cl_records:
        conn.execute(
            """INSERT OR IGNORE INTO cell_lines
            (cell_line_name, cosmic_id, tissue, cancer_type)
            VALUES (?, ?, ?, ?)""",
            (rec["cell_line_name"], rec.get("cosmic_id"),
             rec.get("tissue"), rec.get("cancer_type")),
        )
    conn.commit()

    for row in conn.execute("SELECT cell_line_id, cell_line_name FROM cell_lines"):
        cache[row[1]] = row[0]

    logger.info("Loaded %d cell lines", len(cache))
    return cache


def _assign_tier(num_concentrations: int, has_dose_matrix: bool) -> tuple[str, str]:
    """Assign confidence tier and evidence type for a DrugComb result.

    DrugComb provides dose-response matrices, so:
    - Full matrix (>= 3 concentrations) → bronze / dose_response_matrix
    - Single point → copper / single_concentration
    """
    if has_dose_matrix and num_concentrations >= 3:
        return "bronze", "dose_response_matrix"
    return "copper", "single_concentration"


def load_drugcomb_synergy(
    conn,
    csv_path: Path,
    compound_cache: dict[str, int],
    cell_line_cache: dict[str, int],
    synergy_scores: dict | None = None,
    batch_size: int = 5000,
) -> dict[str, int]:
    """Load DrugComb synergy data from pre-aggregated summary or raw CSV.

    If synergy_scores dict is provided (block_id → SynergyScores), uses those.
    Otherwise reads the CSV and groups by (drug_row, drug_col, cell_line) to
    count concentrations and determine if a dose-response matrix exists.

    Args:
        conn: Database connection.
        csv_path: Path to drugcomb_data_v1.4.csv.
        compound_cache: drug_name → compound_id mapping.
        cell_line_cache: cell_line_name → cell_line_id mapping.
        synergy_scores: Optional pre-computed synergy scores keyed by block_id.
        batch_size: Commit every N inserts.

    Returns:
        Stats dict with insertion counts.
    """
    stats = {
        "results_inserted": 0,
        "skipped_unknown_drug": 0,
        "skipped_unknown_cell_line": 0,
        "chunks_processed": 0,
    }

    # Read CSV in chunks to handle 2 GB file
    reader = pd.read_csv(csv_path, chunksize=100_000, low_memory=False)

    for chunk_num, chunk in enumerate(reader):
        # Normalize column names
        chunk.columns = [c.lower().strip() for c in chunk.columns]

        # Group by experiment block
        group_cols = []
        for candidate in ["block_id", "blockid"]:
            if candidate in chunk.columns:
                group_cols = [candidate]
                break
        if not group_cols:
            # Fallback: group by drug pair + cell line
            for candidate in [
                ("drug_row", "drug_col", "cell_line_name"),
                ("drug_row", "drug_col", "cell_line"),
            ]:
                if all(c in chunk.columns for c in candidate):
                    group_cols = list(candidate)
                    break

        if not group_cols:
            logger.warning("Chunk %d: Cannot find grouping columns, skipping", chunk_num)
            continue

        # Detect drug/cell line column names
        drug_row_col = "drug_row" if "drug_row" in chunk.columns else None
        drug_col_col = "drug_col" if "drug_col" in chunk.columns else None
        cl_col = next(
            (c for c in ("cell_line_name", "cell_line") if c in chunk.columns),
            None,
        )

        if not all([drug_row_col, drug_col_col, cl_col]):
            logger.warning("Chunk %d: Missing required columns", chunk_num)
            continue

        for group_key, group_df in chunk.groupby(group_cols):
            row0 = group_df.iloc[0]
            drug_a_name = str(row0[drug_row_col]).strip()
            drug_b_name = str(row0[drug_col_col]).strip()
            cl_name = str(row0[cl_col]).strip()

            # Look up entity IDs
            # First ensure compounds exist (auto-create if not in identifiers)
            if drug_a_name not in compound_cache:
                conn.execute(
                    "INSERT OR IGNORE INTO compounds (drug_name) VALUES (?)",
                    (drug_a_name,),
                )
                row = conn.execute(
                    "SELECT compound_id FROM compounds WHERE drug_name = ?",
                    (drug_a_name,),
                ).fetchone()
                if row:
                    compound_cache[drug_a_name] = row[0]
                else:
                    stats["skipped_unknown_drug"] += 1
                    continue

            if drug_b_name not in compound_cache:
                conn.execute(
                    "INSERT OR IGNORE INTO compounds (drug_name) VALUES (?)",
                    (drug_b_name,),
                )
                row = conn.execute(
                    "SELECT compound_id FROM compounds WHERE drug_name = ?",
                    (drug_b_name,),
                ).fetchone()
                if row:
                    compound_cache[drug_b_name] = row[0]
                else:
                    stats["skipped_unknown_drug"] += 1
                    continue

            if cl_name not in cell_line_cache:
                conn.execute(
                    "INSERT OR IGNORE INTO cell_lines (cell_line_name) VALUES (?)",
                    (cl_name,),
                )
                row = conn.execute(
                    "SELECT cell_line_id FROM cell_lines WHERE cell_line_name = ?",
                    (cl_name,),
                ).fetchone()
                if row:
                    cell_line_cache[cl_name] = row[0]
                else:
                    stats["skipped_unknown_cell_line"] += 1
                    continue

            cid_a = compound_cache[drug_a_name]
            cid_b = compound_cache[drug_b_name]

            # Skip self-combinations
            if cid_a == cid_b:
                continue

            # Normalize pair ordering
            cid_a, cid_b = normalize_pair(cid_a, cid_b)
            cl_id = cell_line_cache[cl_name]

            # Determine concentration counts and matrix status
            conc_r_col = next(
                (c for c in ("conc_r", "conc_row", "concrow") if c in group_df.columns),
                None,
            )
            conc_c_col = next(
                (c for c in ("conc_c", "conc_col", "conccol") if c in group_df.columns),
                None,
            )

            if conc_r_col and conc_c_col:
                n_conc_r = group_df[conc_r_col].nunique()
                n_conc_c = group_df[conc_c_col].nunique()
                num_conc = max(n_conc_r, n_conc_c)
                has_matrix = n_conc_r >= 2 and n_conc_c >= 2
            else:
                num_conc = len(group_df)
                has_matrix = False

            tier, evidence = _assign_tier(num_conc, has_matrix)

            # Get synergy scores if pre-computed
            zip_score = bliss_score = loewe_score = hsa_score = None
            block_id_val = group_key if isinstance(group_key, (int, float, str)) else group_key[0]
            if synergy_scores and block_id_val in synergy_scores:
                sc = synergy_scores[block_id_val]
                zip_score = sc.zip_score
                bliss_score = sc.bliss_score
                loewe_score = sc.loewe_score
                hsa_score = sc.hsa_score

            # Check for pre-computed scores in data (DrugComb sometimes includes them)
            score_vars = {
                "zip_score": zip_score, "bliss_score": bliss_score,
                "loewe_score": loewe_score, "hsa_score": hsa_score,
            }
            for csv_col, target in [
                ("synergy_zip", "zip_score"), ("synergy_bliss", "bliss_score"),
                ("synergy_loewe", "loewe_score"), ("synergy_hsa", "hsa_score"),
                ("zip_synergy", "zip_score"), ("bliss_synergy", "bliss_score"),
            ]:
                if csv_col in group_df.columns and score_vars[target] is None:
                    val = group_df[csv_col].dropna()
                    if len(val) > 0:
                        score_vars[target] = float(val.mean())
            zip_score = score_vars["zip_score"]
            bliss_score = score_vars["bliss_score"]
            loewe_score = score_vars["loewe_score"]
            hsa_score = score_vars["hsa_score"]

            synergy_class = classify_synergy(zip_score)

            conn.execute(
                """INSERT INTO dc_synergy_results
                (compound_a_id, compound_b_id, cell_line_id,
                 zip_score, bliss_score, loewe_score, hsa_score,
                 synergy_class, confidence_tier, evidence_type,
                 source_db, num_concentrations, has_dose_matrix)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'drugcomb', ?, ?)""",
                (cid_a, cid_b, cl_id,
                 zip_score, bliss_score, loewe_score, hsa_score,
                 synergy_class, tier, evidence,
                 num_conc, int(has_matrix)),
            )
            stats["results_inserted"] += 1

            if stats["results_inserted"] % batch_size == 0:
                conn.commit()

        stats["chunks_processed"] += 1
        if (chunk_num + 1) % 10 == 0:
            logger.info(
                "Processed %d chunks, %d results inserted",
                chunk_num + 1, stats["results_inserted"],
            )

    conn.commit()
    logger.info(
        "DrugComb ETL complete: %d results, %d unknown drugs, %d unknown cell lines",
        stats["results_inserted"],
        stats["skipped_unknown_drug"],
        stats["skipped_unknown_cell_line"],
    )
    return stats


def run_drugcomb_etl(
    db_path: Path,
    data_dir: Path,
    batch_size: int = 5000,
    synergy_scores: dict | None = None,
) -> dict[str, int]:
    """Run full DrugComb ETL pipeline.

    Args:
        db_path: Path to DC database.
        data_dir: Directory containing DrugComb files.
        batch_size: Commit every N inserts.
        synergy_scores: Optional pre-computed synergy scores (block_id → SynergyScores)
            from compute_synergy_scores.py. If None, scores are derived from raw CSV.

    Returns:
        Combined stats dict.
    """
    conn = get_connection(db_path)
    try:
        # Step 1: Load drug identifiers
        drug_xlsx = data_dir / "DrugComb_drug_identifiers.xlsx"
        if drug_xlsx.exists():
            drug_records = parse_drug_identifiers(drug_xlsx)
        else:
            logger.warning("Drug identifiers not found: %s", drug_xlsx)
            drug_records = []

        compound_cache = load_compounds(conn, drug_records)

        # Step 2: Load cell line identifiers
        cl_xlsx = data_dir / "DrugComb_cell_line_identifiers.xlsx"
        if cl_xlsx.exists():
            cl_records = parse_cell_line_identifiers(cl_xlsx)
        else:
            logger.warning("Cell line identifiers not found: %s", cl_xlsx)
            cl_records = []

        cell_line_cache = load_cell_lines(conn, cl_records)

        # Step 3: Load synergy data
        csv_path = data_dir / "drugcomb_data_v1.4.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"DrugComb CSV not found: {csv_path}")

        stats = load_drugcomb_synergy(
            conn, csv_path, compound_cache, cell_line_cache,
            synergy_scores=synergy_scores,
            batch_size=batch_size,
        )
        stats["compounds_loaded"] = len(compound_cache)
        stats["cell_lines_loaded"] = len(cell_line_cache)

        # Refresh pair aggregation after loading
        n_pairs = refresh_all_drug_pairs(conn)
        stats["pairs_refreshed"] = n_pairs
        logger.info("DrugComb ETL complete: %d results, %d pairs", stats["results_inserted"], n_pairs)

        return stats
    finally:
        conn.close()
