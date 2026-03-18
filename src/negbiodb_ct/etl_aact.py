"""ETL pipeline for loading AACT clinical trial data into NegBioDB-CT.

Loads 13 pipe-delimited AACT tables into the CT SQLite database.
Handles studies, interventions, conditions, outcomes, and sponsors.
Does NOT perform failure classification (see etl_classify.py).
Does NOT perform drug name resolution (see drug_resolver.py).

AACT pipe-delimited format: '|' separator, one file per table,
column names in header row. Files extracted from monthly ZIP snapshot.
"""

import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from negbiodb_ct.ct_db import (
    connect,
    create_ct_database,
    get_connection,
    refresh_all_ct_pairs,
    DEFAULT_CT_DB_PATH,
    DEFAULT_CT_MIGRATIONS_DIR,
)
from negbiodb.download import load_config

logger = logging.getLogger(__name__)

SEPARATOR = "|"
BATCH_SIZE = 500

# AACT phase strings -> schema trial_phase CHECK values
# Supports both old Title Case and new UPPER_CASE AACT formats.
_PHASE_MAP = {
    "Phase 1": "phase_1",
    "Phase 1/Phase 2": "phase_1_2",
    "Phase 2": "phase_2",
    "Phase 2/Phase 3": "phase_2_3",
    "Phase 3": "phase_3",
    "Phase 4": "phase_4",
    "Early Phase 1": "early_phase_1",
    "Not Applicable": "not_applicable",
    # New AACT format (2026+)
    "PHASE1": "phase_1",
    "PHASE1/PHASE2": "phase_1_2",
    "PHASE2": "phase_2",
    "PHASE2/PHASE3": "phase_2_3",
    "PHASE3": "phase_3",
    "PHASE4": "phase_4",
    "EARLY_PHASE1": "early_phase_1",
    "NA": "not_applicable",
}

# AACT agency_class -> schema sponsor_type CHECK values
_SPONSOR_TYPE_MAP = {
    "Industry": "industry",
    "NIH": "government",
    "U.S. Fed": "government",
    "Other": "other",
    # New AACT format (2026+)
    "INDUSTRY": "industry",
    "NIH": "government",
    "FED": "government",
    "OTHER_GOV": "government",
    "NETWORK": "other",
    "INDIV": "other",
    "AMBIG": "other",
    "UNKNOWN": "other",
    "OTHER": "other",
}

# AACT intervention_type -> schema intervention_type CHECK values
_INTERVENTION_TYPE_MAP = {
    "Drug": "drug",
    "Biological": "biologic",
    "Device": "device",
    "Procedure": "procedure",
    "Behavioral": "behavioral",
    "Dietary Supplement": "dietary",
    "Genetic": "genetic",
    "Radiation": "radiation",
    "Combination Product": "combination",
    "Diagnostic Test": "other",
    "Other": "other",
    # New AACT format
    "DRUG": "drug",
    "BIOLOGICAL": "biologic",
    "DEVICE": "device",
    "PROCEDURE": "procedure",
    "BEHAVIORAL": "behavioral",
    "DIETARY_SUPPLEMENT": "dietary",
    "GENETIC": "genetic",
    "RADIATION": "radiation",
    "COMBINATION_PRODUCT": "combination",
    "DIAGNOSTIC_TEST": "other",
    "OTHER": "other",
}

# Drug/biologic types that we focus on for the failure pipeline
DRUG_TYPES = {"Drug", "Biological", "Combination Product",
              "DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"}


# ============================================================
# EXTRACT
# ============================================================


def load_aact_table(
    data_dir: Path,
    table_name: str,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a single AACT pipe-delimited file.

    Args:
        data_dir: Directory containing extracted .txt files.
        table_name: AACT table name (e.g., 'studies').
        usecols: Optional list of columns to read.

    Returns:
        DataFrame with the table data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = data_dir / f"{table_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"AACT table not found: {path}")

    logger.info("Loading AACT table: %s", table_name)
    df = pd.read_csv(
        path,
        sep=SEPARATOR,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=usecols,
    )
    logger.info("  %s: %d rows, %d columns", table_name, len(df), len(df.columns))
    return df


def extract_drug_trials(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and filter AACT data to interventional drug/biologic trials.

    Filters studies to:
      - study_type = 'Interventional'
      - At least one intervention with type in ('Drug', 'Biological',
        'Combination Product')

    Returns dict of DataFrames keyed by table name, all filtered to
    matching nct_ids.
    """
    # Load core tables
    studies = load_aact_table(data_dir, "studies", usecols=[
        "nct_id", "overall_status", "phase", "study_type",
        "enrollment", "start_date", "primary_completion_date",
        "completion_date", "results_first_submitted_date",
        "why_stopped", "has_dmc",
    ])
    interventions = load_aact_table(data_dir, "interventions", usecols=[
        "nct_id", "intervention_type", "name", "description",
    ])
    conditions = load_aact_table(data_dir, "conditions", usecols=[
        "nct_id", "name",
    ])

    # Filter to interventional studies (handle both old/new AACT format)
    studies = studies[
        studies["study_type"].str.upper() == "INTERVENTIONAL"
    ].copy()
    logger.info("Interventional studies: %d", len(studies))

    # Filter to studies with at least one drug/biologic intervention
    drug_ncts = set(
        interventions[interventions["intervention_type"].isin(DRUG_TYPES)]["nct_id"]
    )
    studies = studies[studies["nct_id"].isin(drug_ncts)].copy()
    logger.info("Drug/biologic studies: %d", len(studies))

    nct_ids = set(studies["nct_id"])

    # Filter all other tables to matching nct_ids
    interventions = interventions[interventions["nct_id"].isin(nct_ids)].copy()
    conditions = conditions[conditions["nct_id"].isin(nct_ids)].copy()

    # Load supplementary tables (filtered)
    result = {
        "studies": studies,
        "interventions": interventions,
        "conditions": conditions,
    }

    # Optional tables — load what's available
    optional_tables = {
        "designs": ["nct_id", "allocation", "masking", "intervention_model"],
        "sponsors": ["nct_id", "agency_class", "lead_or_collaborator", "name"],
        "calculated_values": ["nct_id", "number_of_facilities", "has_us_facility"],
        "browse_interventions": ["nct_id", "mesh_term"],
        "browse_conditions": ["nct_id", "mesh_term"],
        "documents": ["nct_id", "document_id", "document_type", "url"],
    }

    for table_name, cols in optional_tables.items():
        try:
            df = load_aact_table(data_dir, table_name, usecols=cols)
            result[table_name] = df[df["nct_id"].isin(nct_ids)].copy()
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipping optional table %s: %s", table_name, e)
            result[table_name] = pd.DataFrame(columns=cols)

    return result


# ============================================================
# TRANSFORM
# ============================================================


def normalize_phase(phase_str: str | None) -> str | None:
    """Map AACT phase string to schema trial_phase CHECK value."""
    if phase_str is None or (isinstance(phase_str, float)):
        return None
    phase_str = str(phase_str).strip()
    if not phase_str:
        return None
    return _PHASE_MAP.get(phase_str)


# Map AACT overall_status to Title Case for consistency.
# New AACT format uses UPPER_SNAKE (e.g., TERMINATED, ACTIVE_NOT_RECRUITING).
_STATUS_MAP = {
    "COMPLETED": "Completed",
    "TERMINATED": "Terminated",
    "WITHDRAWN": "Withdrawn",
    "SUSPENDED": "Suspended",
    "RECRUITING": "Recruiting",
    "ACTIVE_NOT_RECRUITING": "Active, not recruiting",
    "NOT_YET_RECRUITING": "Not yet recruiting",
    "ENROLLING_BY_INVITATION": "Enrolling by invitation",
    "UNKNOWN": "Unknown status",
    "WITHHELD": "Withheld",
    "NO_LONGER_AVAILABLE": "No longer available",
    "AVAILABLE": "Available",
    "APPROVED_FOR_MARKETING": "Approved for marketing",
    "TEMPORARILY_NOT_AVAILABLE": "Temporarily not available",
}


def _normalize_status(raw: str | None) -> str:
    """Normalize AACT overall_status to Title Case."""
    if raw is None or (isinstance(raw, float)):
        return "Unknown"
    s = str(raw).strip()
    return _STATUS_MAP.get(s, s)


def normalize_sponsor_type(agency_class: str | None) -> str:
    """Map AACT agency_class to schema sponsor_type."""
    if agency_class is None or (isinstance(agency_class, float)):
        return "other"
    return _SPONSOR_TYPE_MAP.get(str(agency_class).strip(), "other")


def normalize_intervention_type(aact_type: str | None) -> str:
    """Map AACT intervention_type to schema intervention_type CHECK value."""
    if aact_type is None or (isinstance(aact_type, float)):
        return "other"
    return _INTERVENTION_TYPE_MAP.get(str(aact_type).strip(), "other")


def parse_aact_date(date_str: str | None) -> str | None:
    """Parse AACT date strings to ISO 8601 YYYY-MM-DD.

    Handles: 'January 2020', 'March 15, 2021', '2022-05-01', None/NaN.
    """
    if date_str is None or (isinstance(date_str, float)):
        return None
    date_str = str(date_str).strip()
    if not date_str:
        return None

    # Already ISO format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    # "March 15, 2021"
    try:
        from datetime import datetime
        dt = datetime.strptime(date_str, "%B %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    # "January 2020" -> first of month
    try:
        from datetime import datetime
        dt = datetime.strptime(date_str, "%B %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    # "January 1, 2020" variant
    try:
        from datetime import datetime
        dt = datetime.strptime(date_str, "%B %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    logger.debug("Unparseable date: %s", date_str)
    return None


def prepare_interventions(
    interventions_df: pd.DataFrame,
    browse_df: pd.DataFrame,
) -> list[dict]:
    """Deduplicate and prepare intervention dicts from AACT data.

    Deduplication key: (lowercased_name, intervention_type).
    Enriches with MeSH term from browse_interventions where available.
    """
    # Build MeSH lookup: nct_id -> set of mesh_terms
    mesh_by_nct: dict[str, set[str]] = {}
    if len(browse_df) > 0 and "mesh_term" in browse_df.columns:
        for _, row in browse_df.iterrows():
            nct = row.get("nct_id")
            mesh = row.get("mesh_term")
            if pd.notna(nct) and pd.notna(mesh):
                mesh_by_nct.setdefault(str(nct), set()).add(str(mesh))

    seen: dict[tuple[str, str], dict] = {}
    for _, row in tqdm(interventions_df.iterrows(), total=len(interventions_df),
                       desc="Prepare interventions"):
        name = row.get("name")
        itype = row.get("intervention_type")
        nct_id = row.get("nct_id")

        if pd.isna(name) or str(name).strip() == "":
            continue

        name_str = str(name).strip()
        itype_str = normalize_intervention_type(itype)
        key = (name_str.lower(), itype_str)

        if key not in seen:
            mesh_id = None
            if nct_id and str(nct_id) in mesh_by_nct:
                # Use first available mesh term for this trial
                mesh_terms = mesh_by_nct[str(nct_id)]
                if mesh_terms:
                    mesh_id = sorted(mesh_terms)[0]

            seen[key] = {
                "intervention_type": itype_str,
                "intervention_name": name_str,
                "mesh_id": mesh_id,
            }

    logger.info("Unique interventions: %d (from %d raw rows)",
                len(seen), len(interventions_df))
    return list(seen.values())


def prepare_conditions(
    conditions_df: pd.DataFrame,
    browse_df: pd.DataFrame,
) -> list[dict]:
    """Deduplicate and prepare condition dicts from AACT data.

    Deduplication key: lowercased condition name.
    Enriches with MeSH term from browse_conditions where available.
    """
    mesh_by_nct: dict[str, set[str]] = {}
    if len(browse_df) > 0 and "mesh_term" in browse_df.columns:
        for _, row in browse_df.iterrows():
            nct = row.get("nct_id")
            mesh = row.get("mesh_term")
            if pd.notna(nct) and pd.notna(mesh):
                mesh_by_nct.setdefault(str(nct), set()).add(str(mesh))

    seen: dict[str, dict] = {}
    for _, row in conditions_df.iterrows():
        name = row.get("name")
        nct_id = row.get("nct_id")

        if pd.isna(name) or str(name).strip() == "":
            continue

        name_str = str(name).strip()
        key = name_str.lower()

        if key not in seen:
            mesh_id = None
            if nct_id and str(nct_id) in mesh_by_nct:
                mesh_terms = mesh_by_nct[str(nct_id)]
                if mesh_terms:
                    mesh_id = sorted(mesh_terms)[0]

            seen[key] = {
                "condition_name": name_str,
                "mesh_id": mesh_id,
            }

    logger.info("Unique conditions: %d (from %d raw rows)",
                len(seen), len(conditions_df))
    return list(seen.values())


def prepare_trials(
    studies_df: pd.DataFrame,
    designs_df: pd.DataFrame,
    sponsors_df: pd.DataFrame,
) -> list[dict]:
    """Prepare clinical_trials records by joining studies + designs + sponsors."""
    # Build design lookup
    design_by_nct: dict[str, dict] = {}
    if len(designs_df) > 0:
        for _, row in designs_df.iterrows():
            nct = row.get("nct_id")
            if pd.notna(nct):
                design_by_nct[str(nct)] = {
                    "study_design": str(row.get("intervention_model", "")) or None,
                    "blinding": str(row.get("masking", "")) or None,
                    "randomized": 1 if str(row.get("allocation", "")).lower() == "randomized" else 0,
                }

    # Build sponsor lookup (lead sponsors only)
    sponsor_by_nct: dict[str, dict] = {}
    if len(sponsors_df) > 0:
        leads = sponsors_df[sponsors_df["lead_or_collaborator"] == "lead"]
        for _, row in leads.iterrows():
            nct = row.get("nct_id")
            if pd.notna(nct):
                sponsor_by_nct[str(nct)] = {
                    "sponsor_type": normalize_sponsor_type(row.get("agency_class")),
                    "sponsor_name": str(row.get("name", "")) or None,
                }

    trials = []
    for _, row in tqdm(studies_df.iterrows(), total=len(studies_df),
                       desc="Prepare trials"):
        nct_id = str(row["nct_id"])
        design = design_by_nct.get(nct_id, {})
        sponsor = sponsor_by_nct.get(nct_id, {})

        has_results = 1 if pd.notna(row.get("results_first_submitted_date")) else 0

        trial = {
            "source_db": "clinicaltrials_gov",
            "source_trial_id": nct_id,
            "overall_status": _normalize_status(row.get("overall_status", "Unknown")),
            "trial_phase": normalize_phase(row.get("phase")),
            "study_design": design.get("study_design"),
            "blinding": design.get("blinding"),
            "randomized": design.get("randomized", 0),
            "enrollment_actual": int(row["enrollment"]) if pd.notna(row.get("enrollment")) else None,
            "sponsor_type": sponsor.get("sponsor_type", "other"),
            "sponsor_name": sponsor.get("sponsor_name"),
            "start_date": parse_aact_date(row.get("start_date")),
            "primary_completion_date": parse_aact_date(row.get("primary_completion_date")),
            "completion_date": parse_aact_date(row.get("completion_date")),
            "why_stopped": str(row["why_stopped"]) if pd.notna(row.get("why_stopped")) else None,
            "has_results": has_results,
        }
        trials.append(trial)

    logger.info("Prepared %d trials", len(trials))
    return trials


# ============================================================
# LOAD
# ============================================================


def insert_interventions(
    conn: sqlite3.Connection,
    interventions: list[dict],
) -> dict[str, int]:
    """INSERT interventions into the database.

    Deduplicates in Python before insertion (no UNIQUE constraint on name).
    Returns dict mapping lowered_name -> intervention_id.
    """
    name_to_id: dict[str, int] = {}

    for i in tqdm(range(0, len(interventions), BATCH_SIZE),
                  desc="Insert interventions"):
        batch = interventions[i:i + BATCH_SIZE]
        for item in batch:
            key = item["intervention_name"].lower()
            if key in name_to_id:
                continue
            conn.execute(
                "INSERT INTO interventions (intervention_type, intervention_name, mesh_id) "
                "VALUES (?, ?, ?)",
                (item["intervention_type"], item["intervention_name"], item.get("mesh_id")),
            )
            name_to_id[key] = conn.execute(
                "SELECT last_insert_rowid()"
            ).fetchone()[0]

    conn.commit()
    logger.info("Inserted %d interventions", len(name_to_id))
    return name_to_id


def insert_conditions(
    conn: sqlite3.Connection,
    conditions: list[dict],
) -> dict[str, int]:
    """INSERT conditions into the database.

    Returns dict mapping lowered_name -> condition_id.
    """
    name_to_id: dict[str, int] = {}

    for i in tqdm(range(0, len(conditions), BATCH_SIZE),
                  desc="Insert conditions"):
        batch = conditions[i:i + BATCH_SIZE]
        for item in batch:
            key = item["condition_name"].lower()
            if key in name_to_id:
                continue
            conn.execute(
                "INSERT INTO conditions (condition_name, mesh_id) "
                "VALUES (?, ?)",
                (item["condition_name"], item.get("mesh_id")),
            )
            name_to_id[key] = conn.execute(
                "SELECT last_insert_rowid()"
            ).fetchone()[0]

    conn.commit()
    logger.info("Inserted %d conditions", len(name_to_id))
    return name_to_id


def insert_trials(
    conn: sqlite3.Connection,
    trials: list[dict],
) -> dict[str, int]:
    """Insert clinical_trials records.

    Uses INSERT OR IGNORE with UNIQUE(source_db, source_trial_id).
    Returns dict mapping nct_id -> trial_id.
    """
    nct_to_id: dict[str, int] = {}

    cols = [
        "source_db", "source_trial_id", "overall_status", "trial_phase",
        "study_design", "blinding", "randomized", "enrollment_actual",
        "sponsor_type", "sponsor_name", "start_date",
        "primary_completion_date", "completion_date", "why_stopped",
        "has_results",
    ]
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO clinical_trials ({col_names}) VALUES ({placeholders})"

    for i in tqdm(range(0, len(trials), BATCH_SIZE), desc="Insert trials"):
        batch = trials[i:i + BATCH_SIZE]
        rows = [tuple(t[c] for c in cols) for t in batch]
        conn.executemany(sql, rows)

    conn.commit()

    # Build nct_id -> trial_id mapping
    cursor = conn.execute(
        "SELECT source_trial_id, trial_id FROM clinical_trials "
        "WHERE source_db = 'clinicaltrials_gov'"
    )
    for row in cursor:
        nct_to_id[row[0]] = row[1]

    logger.info("Inserted/found %d trials", len(nct_to_id))
    return nct_to_id


def insert_trial_junctions(
    conn: sqlite3.Connection,
    raw_interventions_df: pd.DataFrame,
    raw_conditions_df: pd.DataFrame,
    nct_to_trial_id: dict[str, int],
    name_to_intervention_id: dict[str, int],
    name_to_condition_id: dict[str, int],
) -> tuple[int, int]:
    """Populate trial_interventions and trial_conditions junction tables.

    Returns (n_trial_interventions, n_trial_conditions).
    """
    # Trial-intervention links
    ti_rows = []
    seen_ti = set()
    for _, row in raw_interventions_df.iterrows():
        nct = str(row.get("nct_id", ""))
        name = row.get("name")
        if pd.isna(name) or nct not in nct_to_trial_id:
            continue
        trial_id = nct_to_trial_id[nct]
        interv_id = name_to_intervention_id.get(str(name).strip().lower())
        if interv_id is None:
            continue
        key = (trial_id, interv_id)
        if key not in seen_ti:
            seen_ti.add(key)
            ti_rows.append((trial_id, interv_id))

    for i in range(0, len(ti_rows), BATCH_SIZE):
        conn.executemany(
            "INSERT OR IGNORE INTO trial_interventions (trial_id, intervention_id) "
            "VALUES (?, ?)",
            ti_rows[i:i + BATCH_SIZE],
        )
    conn.commit()
    logger.info("Inserted %d trial-intervention links", len(ti_rows))

    # Trial-condition links
    tc_rows = []
    seen_tc = set()
    for _, row in raw_conditions_df.iterrows():
        nct = str(row.get("nct_id", ""))
        name = row.get("name")
        if pd.isna(name) or nct not in nct_to_trial_id:
            continue
        trial_id = nct_to_trial_id[nct]
        cond_id = name_to_condition_id.get(str(name).strip().lower())
        if cond_id is None:
            continue
        key = (trial_id, cond_id)
        if key not in seen_tc:
            seen_tc.add(key)
            tc_rows.append((trial_id, cond_id))

    for i in range(0, len(tc_rows), BATCH_SIZE):
        conn.executemany(
            "INSERT OR IGNORE INTO trial_conditions (trial_id, condition_id) "
            "VALUES (?, ?)",
            tc_rows[i:i + BATCH_SIZE],
        )
    conn.commit()
    logger.info("Inserted %d trial-condition links", len(tc_rows))

    return len(ti_rows), len(tc_rows)


def insert_trial_publications(
    conn: sqlite3.Connection,
    documents_df: pd.DataFrame,
    nct_to_trial_id: dict[str, int],
) -> int:
    """Insert trial-publication links from AACT documents table.

    Filters to rows with PubMed-like URLs or document_type containing 'pubmed'.
    Returns count inserted.
    """
    count = 0
    seen = set()

    for _, row in documents_df.iterrows():
        nct = str(row.get("nct_id", ""))
        url = str(row.get("url", ""))
        doc_type = str(row.get("document_type", "")).lower()

        if nct not in nct_to_trial_id:
            continue

        # Try to extract PubMed ID from URL
        pubmed_id = None
        if "pubmed" in url or "ncbi.nlm.nih.gov" in url:
            match = re.search(r"(\d{6,10})", url)
            if match:
                pubmed_id = int(match.group(1))
        elif "pubmed" in doc_type:
            doc_id = row.get("document_id")
            if pd.notna(doc_id):
                try:
                    pubmed_id = int(doc_id)
                except (ValueError, TypeError):
                    pass

        if pubmed_id is None:
            continue

        trial_id = nct_to_trial_id[nct]
        key = (trial_id, pubmed_id)
        if key in seen:
            continue
        seen.add(key)

        conn.execute(
            "INSERT OR IGNORE INTO trial_publications (trial_id, pubmed_id) "
            "VALUES (?, ?)",
            (trial_id, pubmed_id),
        )
        count += 1

    conn.commit()
    logger.info("Inserted %d trial-publication links", count)
    return count


# ============================================================
# ORCHESTRATOR
# ============================================================


def run_aact_etl(
    db_path: Path = DEFAULT_CT_DB_PATH,
    data_dir: Path | None = None,
) -> dict:
    """Run the full AACT ETL pipeline.

    Steps:
      1. Ensure DB exists (create if needed)
      2. Load and filter AACT pipe-delimited files
      3. Prepare (deduplicate, normalize) interventions/conditions/trials
      4. Insert into CT database
      5. Populate junction tables
      6. Insert publications

    Returns dict with ETL statistics.
    """
    if data_dir is None:
        cfg = load_config()
        data_dir = Path(cfg["ct_domain"]["downloads"]["aact"]["dest_dir"])

    # Ensure database exists
    create_ct_database(db_path, DEFAULT_CT_MIGRATIONS_DIR)

    # Step 1: Extract
    logger.info("=== AACT ETL: Extract ===")
    tables = extract_drug_trials(data_dir)
    studies_read = len(tables["studies"])

    # Step 2: Transform
    logger.info("=== AACT ETL: Transform ===")
    interventions = prepare_interventions(
        tables["interventions"],
        tables.get("browse_interventions", pd.DataFrame()),
    )
    conditions = prepare_conditions(
        tables["conditions"],
        tables.get("browse_conditions", pd.DataFrame()),
    )
    trials = prepare_trials(
        tables["studies"],
        tables.get("designs", pd.DataFrame()),
        tables.get("sponsors", pd.DataFrame()),
    )

    # Step 3: Load
    logger.info("=== AACT ETL: Load ===")
    conn = get_connection(db_path)
    try:
        name_to_interv_id = insert_interventions(conn, interventions)
        name_to_cond_id = insert_conditions(conn, conditions)
        nct_to_trial_id = insert_trials(conn, trials)

        # Step 4: Junction tables
        n_ti, n_tc = insert_trial_junctions(
            conn,
            tables["interventions"],
            tables["conditions"],
            nct_to_trial_id,
            name_to_interv_id,
            name_to_cond_id,
        )

        # Step 5: Publications
        n_pubs = insert_trial_publications(
            conn,
            tables.get("documents", pd.DataFrame()),
            nct_to_trial_id,
        )
    finally:
        conn.close()

    stats = {
        "studies_read": studies_read,
        "interventions_inserted": len(name_to_interv_id),
        "conditions_inserted": len(name_to_cond_id),
        "trials_inserted": len(nct_to_trial_id),
        "trial_interventions_linked": n_ti,
        "trial_conditions_linked": n_tc,
        "publications_linked": n_pubs,
    }

    logger.info("=== AACT ETL Complete ===")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    return stats
