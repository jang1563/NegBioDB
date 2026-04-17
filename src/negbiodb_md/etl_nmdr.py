"""ETL module for Metabolomics Workbench (NMDR) studies.

Metabolomics Workbench / NMDR (National Metabolomics Data Repository) data is
publicly available under open data terms — redistributable.

API: https://www.metabolomicsworkbench.org/rest/
Documentation: https://www.metabolomicsworkbench.org/tools/MWRestAPIv1.0.pdf

Strategy:
  1. List all public studies via /study/studyid/all/summary
  2. Filter to human disease-comparison studies
  3. For each qualifying study, fetch analysis results with p-values
  4. Standardize metabolite names → md_metabolites
  5. Map disease terms → md_diseases
  6. Insert md_biomarker_results rows
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.metabolomicsworkbench.org/rest"
_REQUEST_TIMEOUT = 30
_RETRY_DELAY = 5
_MAX_RETRIES = 3

_HUMAN_TAXA = {"homo sapiens", "human", "9606", "human subjects"}

_PLATFORM_MAP = {
    "nmr": "nmr",
    "lc-ms": "lc_ms",
    "lc/ms": "lc_ms",
    "lcms": "lc_ms",
    "gc-ms": "gc_ms",
    "gc/ms": "gc_ms",
    "gcms": "gc_ms",
    "uplc": "lc_ms",
}

_BIOFLUID_MAP = {
    "serum": "blood",
    "plasma": "blood",
    "blood": "blood",
    "urine": "urine",
    "csf": "csf",
    "cerebrospinal fluid": "csf",
    "tissue": "tissue",
    "biopsy": "tissue",
    "feces": "other",
    "stool": "other",
    "saliva": "other",
}


def _get(endpoint: str) -> dict | list | None:
    """HTTP GET the NMDR REST API with retry."""
    url = f"{_BASE_URL}/{endpoint}"
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                logger.warning("NMDR GET %s failed (%s), retry %d", endpoint, exc, attempt + 1)
                time.sleep(_RETRY_DELAY)
            else:
                logger.error("NMDR GET %s failed after %d retries", endpoint, _MAX_RETRIES)
                return None


# ── Study discovery ─────────────────────────────────────────────────────────

def list_public_studies() -> list[dict]:
    """Return list of all NMDR public study summaries.

    Returns list of dicts with keys: study_id, study_title, subject_species,
    institute, analysis_type, study_summary.
    """
    data = _get("study/study_title/all/summary")
    if data is None:
        return []
    # NMDR returns {"1": {"study_id": "ST...", ...}, "2": {...}, ...} (numeric keys)
    # or {"study": {...}} or a list
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Numeric-key format (current API): {"1": {...}, "2": {...}}
        first_key = next(iter(data), None)
        if first_key is not None and first_key.isdigit():
            return list(data.values())
        # Legacy format: {"study": {"ST000001": {...}, ...}}
        studies_raw = data.get("study", data)
        if isinstance(studies_raw, dict):
            return [{"study_id": k, **v} for k, v in studies_raw.items()]
        elif isinstance(studies_raw, list):
            return studies_raw
    return []


def fetch_study_details(study_id: str) -> dict | None:
    """Fetch full details for one NMDR study.

    Returns dict with keys:
        study_id, title, description, organism, subject_type, analysis_type,
        sample_source (biofluid), pmid, factors
    """
    data = _get(f"study/study_id/{study_id}/summary")
    if data is None:
        return None

    study = data.get("study", data) if isinstance(data, dict) else data
    if isinstance(study, dict) and study_id in study:
        study = study[study_id]
    if not isinstance(study, dict):
        return None

    return {
        "study_id": study_id,
        "title": study.get("study_title") or study.get("title") or "",
        "description": study.get("study_summary") or study.get("description") or "",
        "organism": (study.get("subject_species") or study.get("species") or study.get("organism") or "").lower(),
        "analysis_type": (study.get("analysis_type") or "").lower(),
        "sample_source": (study.get("sample_source") or study.get("subject_type") or "").lower(),
        "pmid": _parse_pmid(study.get("pubmed_id") or study.get("pmid")),
        "factors": _parse_factors(study),
    }


def _parse_pmid(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except (ValueError, TypeError):
        return None


def _parse_factors(study: dict) -> list[str]:
    """Extract factor/disease terms from NMDR study dict."""
    factors: list[str] = []
    for key in ("subject_type", "study_type", "study_design",
                "disease", "condition", "treatment"):
        val = study.get(key)
        if val and isinstance(val, str) and len(val.strip()) > 2:
            factors.append(val.strip())
    return factors


def is_human_disease_study(details: dict) -> bool:
    """Return True if NMDR study is a human disease comparison."""
    if not details:
        return False
    organism = details.get("organism") or ""
    if not any(h in organism for h in _HUMAN_TAXA):
        return False
    text = (
        (details.get("title") or "") + " " +
        (details.get("description") or "") + " " +
        " ".join(details.get("factors", []))
    ).lower()
    disease_keywords = {
        "disease", "disorder", "syndrome", "cancer", "tumor", "diabetes",
        "obesity", "hypertension", "cardiovascular", "neurological",
        "patients", "case-control", "healthy controls", "clinical",
    }
    return any(kw in text for kw in disease_keywords)


def detect_platform(details: dict) -> str:
    """Detect analytical platform from NMDR analysis_type field."""
    analysis = (details.get("analysis_type") or "").lower()
    for key, val in _PLATFORM_MAP.items():
        if key in analysis:
            return val
    return "other"


def detect_biofluid(details: dict) -> str:
    """Detect biofluid from NMDR sample_source field."""
    source = (details.get("sample_source") or "").lower()
    for key, val in _BIOFLUID_MAP.items():
        if key in source:
            return val
    return "other"


# ── Result fetching ─────────────────────────────────────────────────────────

def fetch_results(study_id: str) -> list[dict]:
    """Fetch metabolite result rows for an NMDR study.

    Returns list of dicts with metabolite_name, p_value, fdr, fold_change.
    Only returns results if statistical columns are present.
    """
    data = _get(f"study/study_id/{study_id}/metabolites")
    if data is None:
        return []

    metabolites_raw = data.get("metabolites", data) if isinstance(data, dict) else data
    if not isinstance(metabolites_raw, (list, dict)):
        return []

    if isinstance(metabolites_raw, dict):
        rows_raw = list(metabolites_raw.values())
    else:
        rows_raw = metabolites_raw

    rows: list[dict] = []
    for raw in rows_raw:
        if not isinstance(raw, dict):
            continue
        name = (raw.get("metabolite_name") or raw.get("name") or
                raw.get("metabolite") or "").strip()
        if not name:
            continue

        p_val = _to_float(raw.get("p_value") or raw.get("pvalue") or
                          raw.get("p-value"))
        fdr = _to_float(raw.get("fdr") or raw.get("q_value") or
                        raw.get("adjusted_p_value"))
        fc_raw = raw.get("fold_change") or raw.get("fc") or raw.get("log2fc")
        fc = _to_float(fc_raw)

        # NMDR's /metabolites endpoint rarely includes per-metabolite
        # statistics — keep the row so downstream tier assignment can mark
        # it as copper (no-stats). Studies that happen to include p-values
        # flow through to higher tiers naturally.

        import math
        log2_fc = None
        if fc is not None:
            # If column name starts with log2, value is already log2
            if any(k in str(raw).lower() for k in ("log2fc", "log2_fc", "log2fold")):
                log2_fc = fc
                fc = None
            elif fc > 0:
                log2_fc = math.log2(fc)

        rows.append({
            "metabolite_name": name,
            "p_value": p_val,
            "fdr": fdr,
            "fold_change": fc,
            "log2_fc": log2_fc,
        })

    if rows:
        logger.debug("NMDR %s: %d metabolite rows with stats", study_id, len(rows))
    return rows


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── High-level ingestion ─────────────────────────────────────────────────────

def ingest_nmdr(
    conn,
    standardizer,
    limit: int | None = None,
    skip_existing: bool = True,
) -> int:
    """Ingest NMDR studies into the MD database.

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
            "SELECT external_id FROM md_studies WHERE source = 'nmdr'"
        ).fetchall()
        existing_ids = {r[0] for r in rows}

    all_studies = list_public_studies()
    logger.info("NMDR: %d public studies found", len(all_studies))

    processed = 0
    inserted = 0

    for study_summary in all_studies:
        sid = study_summary.get("study_id") or study_summary.get("ST")
        if not sid:
            continue
        if skip_existing and sid in existing_ids:
            continue

        details = fetch_study_details(sid)
        # Merge summary fields as fallback (species/analysis_type already present)
        if details is not None:
            if not details.get("organism") and study_summary.get("species"):
                details["organism"] = study_summary["species"].lower()
            if not details.get("title") and study_summary.get("study_title"):
                details["title"] = study_summary["study_title"]
        if details is None or not is_human_disease_study(details):
            continue

        result_rows = fetch_results(sid)
        if not result_rows:
            continue

        title = details.get("title") or sid
        desc = details.get("description") or ""
        platform = detect_platform(details)
        biofluid = detect_biofluid(details)
        disease_terms = details.get("factors", []) + [title]

        conn.execute(
            """INSERT OR IGNORE INTO md_studies
               (source, external_id, title, description, biofluid, platform,
                comparison, pmid)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("nmdr", sid, title, desc, biofluid, platform,
             "disease_vs_healthy", details.get("pmid")),
        )
        study_db_id = conn.execute(
            "SELECT study_id FROM md_studies WHERE source='nmdr' AND external_id=?",
            (sid,),
        ).fetchone()[0]

        disease_id = standardizer.get_or_create_disease(conn, disease_terms)
        if disease_id is None:
            logger.debug("No disease mapped for NMDR study %s, skipping", sid)
            continue

        study_inserted = 0
        for row in result_rows:
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
                tier = assign_tier(p_value, fdr, None)

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
        logger.info("NMDR %s: %d results (total studies: %d)", sid, study_inserted, processed)

        if limit and processed >= limit:
            break

    logger.info("NMDR ingest done: %d studies, %d results", processed, inserted)
    return inserted
