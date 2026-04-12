"""ETL module for MetaboLights (EBI) metabolomics studies.

MetaboLights data is CC0 — fully redistributable.

API: https://www.ebi.ac.uk/metabolights/ws/
FTP: ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/

Strategy:
  1. List all public studies via GET /studies/public
  2. For each study, fetch study details (organism, title, description)
  3. Filter to human disease-comparison studies (disease vs healthy)
  4. For each qualifying study, fetch MAF (Metabolite Assignment File) rows
     — only studies with p_value/FDR columns in MAF are kept (full stats)
  5. Standardize metabolite names → md_metabolites via etl_standardize
  6. Map disease names → md_diseases
  7. Insert md_biomarker_results rows (both significant and non-significant)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.ebi.ac.uk/metabolights/ws"
_REQUEST_TIMEOUT = 30  # seconds
_RETRY_DELAY = 5       # seconds between retries
_MAX_RETRIES = 3

# MAF column names that indicate presence of statistical results
_PVAL_COLS = {"p_value", "p-value", "pvalue", "p_val", "adjusted_p_value",
              "adj_p_value", "fdr", "q_value", "qvalue"}
_FC_COLS = {"fold_change", "fc", "log2_fold_change", "log2fc",
            "log2_fc", "log_fold_change", "logfc"}

# Human organism identifiers in MetaboLights metadata
_HUMAN_TAXA = {"homo sapiens", "human", "9606"}


def _get(url: str, params: dict | None = None) -> dict | list | None:
    """HTTP GET with retry."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                logger.warning("GET %s failed (%s), retrying in %ds", url, exc, _RETRY_DELAY)
                time.sleep(_RETRY_DELAY)
            else:
                logger.error("GET %s failed after %d retries: %s", url, _MAX_RETRIES, exc)
                return None


# ── Study discovery ─────────────────────────────────────────────────────────

def list_public_studies() -> list[str]:
    """Return list of all public MetaboLights study IDs (e.g. ['MTBLS1', ...])."""
    data = _get(f"{_BASE_URL}/studies/public")
    if data is None:
        return []
    # Response is {"content": ["MTBLS1", "MTBLS2", ...], ...}
    if isinstance(data, dict):
        studies = data.get("content", data.get("studies", []))
    elif isinstance(data, list):
        studies = data
    else:
        studies = []
    return [str(s) for s in studies if s]


def fetch_study_details(study_id: str) -> dict | None:
    """Fetch high-level details for one MetaboLights study.

    Returns dict with keys:
        study_id, title, description, organism, factors, assay_count
    or None on failure.
    """
    data = _get(f"{_BASE_URL}/studies/{study_id}")
    if data is None:
        return None

    # Navigate MetaboLights JSON structure
    study = data.get("content", data) if isinstance(data, dict) else data
    if not isinstance(study, dict):
        return None

    title = study.get("title") or ""
    desc = (study.get("description") or "").strip()

    # Organism
    organisms = []
    for org_block in study.get("organism", []):
        if isinstance(org_block, dict):
            name = org_block.get("Organism", "") or org_block.get("organism", "")
            if name:
                organisms.append(name.lower().strip())

    # Factors (disease terms, etc.)
    factors: list[str] = []
    for factor in study.get("factors", []):
        if isinstance(factor, dict):
            fname = factor.get("name", "") or factor.get("factorName", "")
            if fname:
                factors.append(fname.strip())

    # Publication / PMID
    pmid = None
    for pub in study.get("publications", []):
        if isinstance(pub, dict):
            raw_pmid = pub.get("pubmedId") or pub.get("pmid")
            if raw_pmid:
                try:
                    pmid = int(raw_pmid)
                    break
                except (ValueError, TypeError):
                    pass

    return {
        "study_id": study_id,
        "title": title,
        "description": desc,
        "organisms": organisms,
        "factors": factors,
        "pmid": pmid,
    }


def is_human_disease_study(details: dict) -> bool:
    """Return True if study is a human disease comparison study."""
    if not details:
        return False
    organisms = details.get("organisms", [])
    if not any(o in _HUMAN_TAXA for o in organisms):
        return False
    # Heuristic: study has disease-related factors or title keywords
    title_lower = (details.get("title") or "").lower()
    desc_lower = (details.get("description") or "").lower()
    disease_keywords = {
        "disease", "disorder", "syndrome", "cancer", "tumor", "tumour",
        "diabetes", "obesity", "hypertension", "cardiovascular", "neurological",
        "alzheimer", "parkinson", "depression", "schizophrenia", "lupus",
        "arthritis", "asthma", "fibrosis", "hepatitis", "cirrhosis",
        "patients", "case-control", "case control", "healthy controls",
    }
    combined = title_lower + " " + desc_lower
    return any(kw in combined for kw in disease_keywords)


# ── MAF (Metabolite Assignment File) parsing ────────────────────────────────

def fetch_maf(study_id: str, assay_name: str | None = None) -> list[dict] | None:
    """Fetch MAF rows for a MetaboLights study.

    Returns list of dicts with raw MAF columns, or None on failure.
    Only returns rows from MAF files that contain p-value/FDR columns.
    """
    # List assays for study
    data = _get(f"{_BASE_URL}/studies/{study_id}/assays")
    if data is None:
        return None

    assay_list = []
    if isinstance(data, dict):
        assay_list = data.get("content", data.get("assays", []))
    elif isinstance(data, list):
        assay_list = data

    all_rows: list[dict] = []
    for assay in assay_list:
        if not isinstance(assay, dict):
            continue
        maf_name = assay.get("metaboliteAssignment", {}).get("metaboliteAssignmentFileName", "")
        if not maf_name:
            continue
        maf_url = f"{_BASE_URL}/studies/{study_id}/download?file={maf_name}"
        rows = _fetch_maf_file(maf_url, study_id, maf_name)
        if rows:
            all_rows.extend(rows)

    return all_rows if all_rows else None


def _fetch_maf_file(url: str, study_id: str, filename: str) -> list[dict]:
    """Download and parse a single MAF TSV/CSV file.

    Filters to MAF files with p-value/FDR columns (required for negatives).
    Returns list of row dicts, empty list if no stats columns found.
    """
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("MAF fetch failed for %s/%s: %s", study_id, filename, exc)
        return []

    lines = resp.text.splitlines()
    if not lines:
        return []

    # Detect delimiter
    header_line = lines[0]
    delimiter = "\t" if "\t" in header_line else ","
    headers = [h.strip().lower() for h in header_line.split(delimiter)]

    # Check for p-value columns (required)
    has_pval = any(h in _PVAL_COLS for h in headers)
    if not has_pval:
        logger.debug("MAF %s/%s: no p-value columns, skipping", study_id, filename)
        return []

    # Detect FC columns
    fc_col = next((h for h in headers if h in _FC_COLS), None)
    pval_col = next(
        (h for h in headers if h in {"p_value", "p-value", "pvalue", "p_val"}),
        next((h for h in headers if "p_val" in h or "p-val" in h), None),
    )
    fdr_col = next(
        (h for h in headers if h in {"fdr", "q_value", "qvalue", "adjusted_p_value", "adj_p_value"}),
        None,
    )
    name_col = next(
        (h for h in headers if h in {"metabolite_identification", "metabolite_name",
                                      "database_identifier", "chemical_formula",
                                      "metabolite", "compound_name"}),
        headers[0] if headers else None,
    )

    rows: list[dict] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split(delimiter)
        row: dict[str, Any] = {}

        def _field(col: str | None) -> str | None:
            if col is None:
                return None
            try:
                idx = headers.index(col)
                val = fields[idx].strip() if idx < len(fields) else ""
                return val if val not in ("", "NA", "N/A", "NaN", "nan", "null") else None
            except (ValueError, IndexError):
                return None

        row["metabolite_name"] = _field(name_col)
        raw_pval = _field(pval_col)
        raw_fdr = _field(fdr_col)
        raw_fc = _field(fc_col)

        def _float(s: str | None) -> float | None:
            if s is None:
                return None
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        row["p_value"] = _float(raw_pval)
        row["fdr"] = _float(raw_fdr)
        row["fold_change"] = _float(raw_fc)
        row["log2_fc"] = None  # computed below if fc available

        if row["fold_change"] is not None and row["fold_change"] > 0:
            import math
            row["log2_fc"] = math.log2(row["fold_change"])
        elif fc_col and fc_col.startswith("log2"):
            row["log2_fc"] = row["fold_change"]
            row["fold_change"] = None

        row["source_file"] = filename
        rows.append(row)

    logger.debug("MAF %s/%s: %d rows with stats", study_id, filename, len(rows))
    return rows


# ── Platform + biofluid detection ───────────────────────────────────────────

_PLATFORM_MAP = {
    "nmr": "nmr",
    "nuclear magnetic resonance": "nmr",
    "lc-ms": "lc_ms",
    "lc/ms": "lc_ms",
    "liquid chromatography": "lc_ms",
    "uplc": "lc_ms",
    "uhplc": "lc_ms",
    "gc-ms": "gc_ms",
    "gc/ms": "gc_ms",
    "gas chromatography": "gc_ms",
}

_BIOFLUID_MAP = {
    "serum": "blood",
    "plasma": "blood",
    "blood": "blood",
    "urine": "urine",
    "urinary": "urine",
    "cerebrospinal fluid": "csf",
    "csf": "csf",
    "tissue": "tissue",
    "biopsy": "tissue",
}


def detect_platform(title: str, description: str) -> str:
    """Infer analytical platform from study text (NMR/LC-MS/GC-MS/other)."""
    text = (title + " " + description).lower()
    for key, val in _PLATFORM_MAP.items():
        if key in text:
            return val
    return "other"


def detect_biofluid(title: str, description: str) -> str:
    """Infer biofluid from study text."""
    text = (title + " " + description).lower()
    for key, val in _BIOFLUID_MAP.items():
        if key in text:
            return val
    return "other"


# ── Disease name extraction ──────────────────────────────────────────────────

def extract_disease_terms(title: str, description: str, factors: list[str]) -> list[str]:
    """Extract candidate disease names from study text and factor names.

    Returns a list of candidate disease strings for MONDO mapping.
    """
    candidates: set[str] = set()

    # Factor names are the most reliable disease source
    for factor in factors:
        if factor and len(factor) > 2:
            candidates.add(factor.strip())

    # Simple keyword extraction from title
    title_words = title.strip()
    if title_words:
        candidates.add(title_words)

    return list(candidates)


# ── High-level ingestion ─────────────────────────────────────────────────────

def ingest_metabolights(
    conn,
    standardizer,
    limit: int | None = None,
    skip_existing: bool = True,
) -> int:
    """Ingest MetaboLights studies into the MD database.

    Args:
        conn:         sqlite3 connection to negbiodb_md.db
        standardizer: negbiodb_md.etl_standardize.Standardizer instance
        limit:        max number of qualifying studies to process
        skip_existing: skip study_ids already in md_studies

    Returns:
        Number of md_biomarker_results rows inserted.
    """
    from negbiodb_md.md_db import assign_tier

    existing_ids: set[str] = set()
    if skip_existing:
        rows = conn.execute(
            "SELECT external_id FROM md_studies WHERE source = 'metabolights'"
        ).fetchall()
        existing_ids = {r[0] for r in rows}

    study_ids = list_public_studies()
    logger.info("MetaboLights: %d public studies found", len(study_ids))

    processed = 0
    inserted = 0

    for sid in study_ids:
        if skip_existing and sid in existing_ids:
            continue

        details = fetch_study_details(sid)
        if details is None or not is_human_disease_study(details):
            continue

        maf_rows = fetch_maf(sid)
        if not maf_rows:
            continue

        title = details.get("title") or sid
        desc = details.get("description") or ""
        platform = detect_platform(title, desc)
        biofluid = detect_biofluid(title, desc)
        disease_terms = extract_disease_terms(title, desc, details.get("factors", []))

        # Upsert study record
        conn.execute(
            """INSERT OR IGNORE INTO md_studies
               (source, external_id, title, description, biofluid, platform,
                comparison, pmid)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("metabolights", sid, title, desc, biofluid, platform,
             "disease_vs_healthy", details.get("pmid")),
        )
        study_db_id = conn.execute(
            "SELECT study_id FROM md_studies WHERE source='metabolights' AND external_id=?",
            (sid,),
        ).fetchone()[0]

        # Map disease term → disease_id (best match)
        disease_id = standardizer.get_or_create_disease(conn, disease_terms)
        if disease_id is None:
            logger.debug("No disease mapped for study %s, skipping", sid)
            continue

        # Get sample sizes from study details (approximate; NMDR provides explicit counts)
        n_disease = n_control = None

        # Insert biomarker results
        study_inserted = 0
        for row in maf_rows:
            name = row.get("metabolite_name")
            if not name:
                continue
            metabolite_id = standardizer.get_or_create_metabolite(conn, name)
            if metabolite_id is None:
                continue

            p_value = row.get("p_value")
            fdr = row.get("fdr")
            is_significant = 0
            if p_value is not None and p_value <= 0.05:
                is_significant = 1
            elif fdr is not None and fdr <= 0.05:
                is_significant = 1

            tier = None
            if not is_significant:
                tier = assign_tier(p_value, fdr, n_disease)

            conn.execute(
                """INSERT OR IGNORE INTO md_biomarker_results
                   (metabolite_id, disease_id, study_id,
                    fold_change, log2_fc, p_value, fdr,
                    is_significant, tier)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (metabolite_id, disease_id, study_db_id,
                 row.get("fold_change"), row.get("log2_fc"),
                 p_value, fdr, is_significant, tier),
            )
            study_inserted += 1

        conn.commit()
        inserted += study_inserted
        processed += 1
        logger.info(
            "MetaboLights %s: %d results (processed %d)", sid, study_inserted, processed
        )

        if limit and processed >= limit:
            break

    logger.info("MetaboLights ingest done: %d studies, %d results", processed, inserted)
    return inserted
