"""ETL pipeline for loading ChEMBL inactive DTI data into NegBioDB.

Core (default, conservative):
  Type 1: pChEMBL < 4.5 (quantitative, after borderline exclusion)
  Type 2: Right-censored (standard_relation '>'/'>=', value >= 10000 nM)

Optional (disabled by default, recall expansion):
  Type 3: activity_comment in inactive terms (e.g., "Not Active")

Confidence policy:
  - Core types -> silver
  - activity_comment-only route -> bronze

ChEMBL already provides UniProt accessions — no API mapping needed.
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from negbiodb.db import connect, create_database, refresh_all_pairs, _PROJECT_ROOT
from negbiodb.download import load_config
from negbiodb.standardize import standardize_smiles

logger = logging.getLogger(__name__)

# SQL query for extracting conservative inactive records from ChEMBL
_CHEMBL_INACTIVE_CORE_SQL = """
SELECT
    a.activity_id,
    md.molregno,
    md.chembl_id AS chembl_compound_id,
    cs.canonical_smiles,
    cs.standard_inchi_key,
    a.pchembl_value,
    a.standard_type,
    a.standard_value,
    a.standard_relation,
    a.standard_units,
    cp.accession AS uniprot_accession,
    td.chembl_id AS chembl_target_id,
    td.pref_name AS target_name,
    td.organism,
    cp.sequence AS protein_sequence,
    LENGTH(cp.sequence) AS sequence_length,
    ass.chembl_id AS assay_chembl_id,
    docs.year AS publication_year,
    a.activity_comment,
    CASE
      WHEN a.pchembl_value IS NOT NULL AND a.pchembl_value < :borderline_lower
        THEN 'quantitative'
      ELSE 'right_censored'
    END AS inactivity_source
FROM activities a
JOIN molecule_dictionary md ON a.molregno = md.molregno
JOIN compound_structures cs ON a.molregno = cs.molregno
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN target_components tc ON td.tid = tc.tid
JOIN component_sequences cp ON tc.component_id = cp.component_id
LEFT JOIN docs ON a.doc_id = docs.doc_id
WHERE (
    (a.pchembl_value IS NOT NULL AND a.pchembl_value < :borderline_lower)
    OR
    (a.pchembl_value IS NULL
     AND a.standard_relation IN ('>', '>=')
     AND a.standard_value >= :inactivity_threshold
     AND a.standard_units = 'nM')
)
AND a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
AND a.data_validity_comment IS NULL
AND td.target_type = 'SINGLE PROTEIN'
AND td.organism = 'Homo sapiens'
AND cp.accession IS NOT NULL
AND cs.canonical_smiles IS NOT NULL
"""


def _build_activity_comment_sql(comment_terms: list[str]) -> tuple[str, dict]:
    """Build activity_comment extraction SQL with parameterized comment terms."""
    placeholders = ", ".join([f":comment_{i}" for i in range(len(comment_terms))])
    params = {f"comment_{i}": term for i, term in enumerate(comment_terms)}

    sql = f"""
SELECT
    a.activity_id,
    md.molregno,
    md.chembl_id AS chembl_compound_id,
    cs.canonical_smiles,
    cs.standard_inchi_key,
    a.pchembl_value,
    a.standard_type,
    a.standard_value,
    a.standard_relation,
    a.standard_units,
    cp.accession AS uniprot_accession,
    td.chembl_id AS chembl_target_id,
    td.pref_name AS target_name,
    td.organism,
    cp.sequence AS protein_sequence,
    LENGTH(cp.sequence) AS sequence_length,
    ass.chembl_id AS assay_chembl_id,
    docs.year AS publication_year,
    a.activity_comment,
    'activity_comment' AS inactivity_source
FROM activities a
JOIN molecule_dictionary md ON a.molregno = md.molregno
JOIN compound_structures cs ON a.molregno = cs.molregno
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN target_components tc ON td.tid = tc.tid
JOIN component_sequences cp ON tc.component_id = cp.component_id
LEFT JOIN docs ON a.doc_id = docs.doc_id
WHERE a.activity_comment IN ({placeholders})
AND a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
AND a.data_validity_comment IS NULL
AND td.target_type = 'SINGLE PROTEIN'
AND td.organism = 'Homo sapiens'
AND cp.accession IS NOT NULL
AND cs.canonical_smiles IS NOT NULL
"""
    return sql, params


# ============================================================
# EXTRACT
# ============================================================


def find_chembl_db(data_dir: Path | None = None) -> Path:
    """Find ChEMBL SQLite database file in data/chembl/.

    Looks for the symlink created by scripts/download_chembl.py.
    """
    if data_dir is None:
        cfg = load_config()
        data_dir = _PROJECT_ROOT / cfg["downloads"]["chembl"]["dest_dir"]

    candidates = sorted(data_dir.glob("chembl_*.db"))
    if not candidates:
        raise FileNotFoundError(
            f"No ChEMBL database found in {data_dir}. "
            "Run 'make download-chembl' first."
        )
    # Use the latest version (last in sorted order)
    return candidates[-1]


def extract_chembl_inactives(
    chembl_db_path: Path,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Query ChEMBL SQLite for inactive DTI records.

    Returns DataFrame with all columns needed for compound/target/result loading.
    """
    if cfg is None:
        cfg = load_config()

    borderline_lower = cfg["borderline_exclusion"]["lower"]
    inactivity_threshold = cfg["inactivity_threshold_nm"]
    chembl_cfg = cfg.get("chembl_etl", {})
    include_activity_comment = bool(chembl_cfg.get("include_activity_comment", False))
    comment_terms = chembl_cfg.get("inactive_activity_comments", ["Not Active", "Inactive"])

    logger.info(
        "Querying ChEMBL core inactives (pChEMBL < %.1f or right-censored >= %d nM)...",
        borderline_lower, inactivity_threshold,
    )

    conn = sqlite3.connect(str(chembl_db_path))
    try:
        core_df = pd.read_sql_query(
            _CHEMBL_INACTIVE_CORE_SQL,
            conn,
            params={
                "borderline_lower": borderline_lower,
                "inactivity_threshold": inactivity_threshold,
            },
        )

        if include_activity_comment and comment_terms:
            logger.info(
                "Including activity_comment inactive route (terms=%s)",
                ", ".join(comment_terms),
            )
            comment_sql, comment_params = _build_activity_comment_sql(comment_terms)
            comment_df = pd.read_sql_query(comment_sql, conn, params=comment_params)

            # Keep core route first; if the same activity appears in both routes,
            # preserve the conservative classification.
            df = pd.concat([core_df, comment_df], ignore_index=True)
            df = df.drop_duplicates(subset=["activity_id"], keep="first")
        elif include_activity_comment:
            logger.warning(
                "include_activity_comment=True but inactive_activity_comments is empty; skipping comment route."
            )
            df = core_df
        else:
            df = core_df
    finally:
        conn.close()

    n_type1 = ((df["inactivity_source"] == "quantitative")).sum()
    n_type2 = ((df["inactivity_source"] == "right_censored")).sum()
    n_type3 = ((df["inactivity_source"] == "activity_comment")).sum()
    logger.info(
        "Extracted %d records: %d quantitative, %d right-censored, %d activity_comment",
        len(df), int(n_type1), int(n_type2), int(n_type3),
    )

    return df


# ============================================================
# TRANSFORM: Compounds
# ============================================================


def standardize_chembl_compounds(
    df: pd.DataFrame,
) -> tuple[list[dict], dict[int, str]]:
    """Standardize unique ChEMBL compounds with RDKit.

    Returns:
        compounds: list of dicts ready for DB insertion
        molregno_to_inchikey: mapping from ChEMBL molregno to computed InChIKey
    """
    unique = df.drop_duplicates("molregno")[["molregno", "chembl_compound_id", "canonical_smiles"]]
    logger.info("Standardizing %d unique compounds...", len(unique))

    compounds = []
    molregno_to_inchikey: dict[int, str] = {}
    failed = 0

    for _, row in tqdm(unique.iterrows(), total=len(unique), desc="RDKit"):
        result = standardize_smiles(row["canonical_smiles"])
        if result is None:
            failed += 1
            continue
        result["chembl_id"] = row["chembl_compound_id"]
        compounds.append(result)
        molregno_to_inchikey[row["molregno"]] = result["inchikey"]

    logger.info(
        "Standardized %d / %d compounds (%d failed)",
        len(compounds), len(unique), failed,
    )
    return compounds, molregno_to_inchikey


# ============================================================
# TRANSFORM: Targets
# ============================================================


def prepare_chembl_targets(df: pd.DataFrame) -> list[dict]:
    """Deduplicate and prepare target dicts from ChEMBL query results.

    UniProt accessions come directly from ChEMBL — no API mapping needed.
    """
    target_cols = [
        "uniprot_accession", "chembl_target_id", "target_name",
        "protein_sequence", "sequence_length",
    ]
    unique = df.drop_duplicates("uniprot_accession")[target_cols]
    logger.info("Preparing %d unique targets...", len(unique))

    targets = []
    for _, row in unique.iterrows():
        targets.append({
            "uniprot_accession": row["uniprot_accession"],
            "chembl_target_id": row["chembl_target_id"],
            "amino_acid_sequence": row["protein_sequence"],
            "sequence_length": int(row["sequence_length"]) if pd.notna(row["sequence_length"]) else None,
        })

    return targets


# ============================================================
# LOAD
# ============================================================


def insert_chembl_compounds(
    conn: sqlite3.Connection,
    compounds: list[dict],
) -> dict[str, int]:
    """Insert standardized compounds into the database.

    Returns dict mapping InChIKey -> database compound_id.
    Uses INSERT OR IGNORE for cross-DB dedup with DAVIS via InChIKey UNIQUE.
    """
    for comp in compounds:
        conn.execute(
            """INSERT OR IGNORE INTO compounds
            (canonical_smiles, inchikey, inchikey_connectivity, inchi,
             chembl_id, molecular_weight, logp, hbd, hba, tpsa,
             rotatable_bonds, num_heavy_atoms, qed, pains_alert,
             lipinski_violations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                comp["canonical_smiles"],
                comp["inchikey"],
                comp["inchikey_connectivity"],
                comp["inchi"],
                comp["chembl_id"],
                comp["molecular_weight"],
                comp["logp"],
                comp["hbd"],
                comp["hba"],
                comp["tpsa"],
                comp["rotatable_bonds"],
                comp["num_heavy_atoms"],
                comp["qed"],
                comp["pains_alert"],
                comp["lipinski_violations"],
            ),
        )

    # Build inchikey -> compound_id mapping
    inchikeys = [c["inchikey"] for c in compounds]
    inchikey_to_cid: dict[str, int] = {}
    # Query in batches to avoid SQL variable limit
    batch_size = 500
    for i in range(0, len(inchikeys), batch_size):
        batch = inchikeys[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT inchikey, compound_id FROM compounds WHERE inchikey IN ({placeholders})",
            batch,
        ).fetchall()
        for ik, cid in rows:
            inchikey_to_cid[ik] = cid

    return inchikey_to_cid


def insert_chembl_targets(
    conn: sqlite3.Connection,
    targets: list[dict],
) -> dict[str, int]:
    """Insert standardized targets into the database.

    Returns dict mapping UniProt accession -> database target_id.
    Uses INSERT OR IGNORE for cross-DB dedup with DAVIS via uniprot_accession UNIQUE.
    """
    for tgt in targets:
        conn.execute(
            """INSERT OR IGNORE INTO targets
            (uniprot_accession, chembl_target_id,
             amino_acid_sequence, sequence_length)
            VALUES (?, ?, ?, ?)""",
            (
                tgt["uniprot_accession"],
                tgt["chembl_target_id"],
                tgt["amino_acid_sequence"],
                tgt["sequence_length"],
            ),
        )

    # Build accession -> target_id mapping
    accessions = [t["uniprot_accession"] for t in targets]
    acc_to_tid: dict[str, int] = {}
    batch_size = 500
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT uniprot_accession, target_id FROM targets WHERE uniprot_accession IN ({placeholders})",
            batch,
        ).fetchall()
        for acc, tid in rows:
            acc_to_tid[acc] = tid

    return acc_to_tid


def insert_chembl_negative_results(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    molregno_to_inchikey: dict[int, str],
    inchikey_to_cid: dict[str, int],
    uniprot_to_tid: dict[str, int],
) -> tuple[int, int]:
    """Insert ChEMBL inactive records into negative_results table.

    Returns (inserted_count, skipped_count).
    """
    params = []
    skipped = 0

    for _, row in df.iterrows():
        # Resolve compound_id via molregno -> inchikey -> compound_id
        inchikey = molregno_to_inchikey.get(row["molregno"])
        if inchikey is None:
            skipped += 1
            continue
        compound_id = inchikey_to_cid.get(inchikey)
        if compound_id is None:
            skipped += 1
            continue

        # Resolve target_id via uniprot_accession
        target_id = uniprot_to_tid.get(row["uniprot_accession"])
        if target_id is None:
            skipped += 1
            continue

        activity_id = int(row["activity_id"])
        pchembl = float(row["pchembl_value"]) if pd.notna(row["pchembl_value"]) else None
        activity_value = float(row["standard_value"]) if pd.notna(row["standard_value"]) else None
        activity_relation = row["standard_relation"] if pd.notna(row["standard_relation"]) else "="
        pub_year = int(row["publication_year"]) if pd.notna(row["publication_year"]) else None

        inactivity_source = row.get("inactivity_source", "quantitative")
        confidence_tier = "bronze" if inactivity_source == "activity_comment" else "silver"

        params.append((
            compound_id,
            target_id,
            confidence_tier,
            row["standard_type"],
            activity_value,
            row["standard_units"] if pd.notna(row["standard_units"]) else "nM",
            activity_relation,
            pchembl,
            f"CHEMBL:{activity_id}",
            pub_year,
        ))

    # Batch insert
    conn.executemany(
        """INSERT OR IGNORE INTO negative_results
        (compound_id, target_id, assay_id,
         result_type, confidence_tier,
         activity_type, activity_value, activity_unit, activity_relation,
         pchembl_value,
         inactivity_threshold, inactivity_threshold_unit,
         source_db, source_record_id, extraction_method,
         curator_validated, publication_year, species_tested)
        VALUES (?, ?, NULL,
                'hard_negative', ?,
                ?, ?, ?, ?,
                ?,
                10000.0, 'nM',
                'chembl', ?, 'database_direct',
                0, ?, 'Homo sapiens')""",
        params,
    )

    return len(params), skipped


# ============================================================
# ORCHESTRATOR
# ============================================================


def run_chembl_etl(
    db_path: Path,
    chembl_db_path: Path | None = None,
) -> dict:
    """Run the full ChEMBL ETL pipeline.

    Returns dict with ETL statistics.
    """
    cfg = load_config()

    if chembl_db_path is None:
        chembl_db_path = find_chembl_db()

    logger.info("Using ChEMBL database: %s", chembl_db_path)

    # === EXTRACT ===
    df = extract_chembl_inactives(chembl_db_path, cfg)

    # === TRANSFORM: Compounds ===
    compounds, molregno_to_inchikey = standardize_chembl_compounds(df)

    # === TRANSFORM: Targets ===
    targets = prepare_chembl_targets(df)

    # === LOAD ===
    create_database(db_path)

    with connect(db_path) as conn:
        logger.info("Inserting %d compounds...", len(compounds))
        inchikey_to_cid = insert_chembl_compounds(conn, compounds)

        logger.info("Inserting %d targets...", len(targets))
        uniprot_to_tid = insert_chembl_targets(conn, targets)

        logger.info("Inserting negative results...")
        inserted, skipped = insert_chembl_negative_results(
            conn, df, molregno_to_inchikey, inchikey_to_cid, uniprot_to_tid,
        )

        logger.info("Refreshing compound-target pairs (all sources)...")
        n_pairs = refresh_all_pairs(conn)

        conn.commit()

    stats = {
        "records_extracted": len(df),
        "compounds_standardized": len(compounds),
        "compounds_failed_rdkit": len(df["molregno"].unique()) - len(compounds),
        "targets_prepared": len(targets),
        "results_inserted": inserted,
        "results_skipped": skipped,
        "pairs_total": n_pairs,
    }

    logger.info("ChEMBL ETL complete: %s", stats)
    return stats
