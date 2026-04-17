#!/usr/bin/env python3
"""Ingest hand-curated metabolite-disease studies into the MD DB.

Input CSVs (populated manually by JK):

  exports/md_hand_curated_studies.csv
    columns: study_id, pmid, disease, n_disease, n_control,
             supplementary_url, biofluid, platform, notes

  exports/md_hand_curated_results.csv
    columns: study_id, metabolite_name, p_value, fdr, fold_change,
             direction (up/down/ns), notes

For each study:
  1. Insert row into md_studies (source='hand_curated')
  2. Resolve disease terms → md_diseases
  3. For each result row: resolve metabolite → md_metabolites, compute tier
     via md_db.assign_tier, insert into md_biomarker_results

Designed to be idempotent: INSERT OR IGNORE on md_studies with (source, external_id).
"""
import argparse
import csv
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def to_float(val: str) -> float | None:
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def to_int(val: str) -> int | None:
    if val is None or val.strip() == "":
        return None
    try:
        return int(float(val))  # handle "50.0" → 50
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description="Ingest hand-curated MD studies")
    ap.add_argument("--studies", default="exports/md_hand_curated_studies.csv")
    ap.add_argument("--results", default="exports/md_hand_curated_results.csv")
    ap.add_argument("--db", default="data/negbiodb_md.db")
    ap.add_argument("--hmdb-cache", default=None,
                    help="Optional HMDB cache DB for name resolution")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    from negbiodb_md.md_db import assign_tier, get_md_connection
    from negbiodb_md.etl_standardize import Standardizer

    studies_path = Path(args.studies)
    results_path = Path(args.results)
    if not studies_path.exists():
        raise SystemExit(f"Studies CSV not found: {studies_path}")
    if not results_path.exists():
        raise SystemExit(f"Results CSV not found: {results_path}")

    # Load CSVs
    with open(studies_path) as f:
        studies = list(csv.DictReader(f))
    with open(results_path) as f:
        results = list(csv.DictReader(f))
    logger.info("Loaded %d studies, %d result rows", len(studies), len(results))

    # Group results by study_id
    results_by_study: dict[str, list[dict]] = {}
    for r in results:
        sid = r["study_id"].strip()
        results_by_study.setdefault(sid, []).append(r)

    # Open DB + standardizer
    conn = get_md_connection(args.db)
    hmdb_cache_conn = None
    if args.hmdb_cache and Path(args.hmdb_cache).exists():
        hmdb_cache_conn = sqlite3.connect(args.hmdb_cache, timeout=60)
    std = Standardizer(hmdb_cache_conn=hmdb_cache_conn)

    inserted_results = 0
    processed_studies = 0

    for study in studies:
        sid = study["study_id"].strip()
        pmid = to_int(study.get("pmid", ""))
        disease = study.get("disease", "").strip()
        n_disease = to_int(study.get("n_disease", ""))
        n_control = to_int(study.get("n_control", ""))
        biofluid = study.get("biofluid", "").strip().lower() or "other"
        platform = study.get("platform", "").strip().lower() or "other"

        # Insert study row
        conn.execute(
            """INSERT OR IGNORE INTO md_studies
               (source, external_id, title, description, biofluid, platform,
                comparison, pmid)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("hand_curated", sid,
             study.get("notes", "")[:200] or sid,
             f"Hand-curated {disease} study (n_disease={n_disease}, n_control={n_control})",
             biofluid, platform, "disease_vs_healthy", pmid),
        )
        study_row = conn.execute(
            "SELECT study_id FROM md_studies WHERE source='hand_curated' AND external_id=?",
            (sid,),
        ).fetchone()
        if study_row is None:
            logger.warning("Could not locate inserted study %s; skipping", sid)
            continue
        study_db_id = study_row[0]

        # Resolve disease
        disease_id = std.get_or_create_disease(conn, [disease])
        if disease_id is None:
            logger.warning("No disease mapping for '%s' (study %s); skipping", disease, sid)
            continue

        study_rows = results_by_study.get(sid, [])
        for row in study_rows:
            name = row["metabolite_name"].strip()
            if not name:
                continue
            p_value = to_float(row.get("p_value", ""))
            fdr = to_float(row.get("fdr", ""))
            fc = to_float(row.get("fold_change", ""))

            metabolite_id = std.get_or_create_metabolite(conn, name)
            if metabolite_id is None:
                logger.debug("Skipping metabolite '%s' (no PubChem/HMDB match)", name)
                continue

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
                    fold_change, p_value, fdr,
                    is_significant, tier)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (metabolite_id, disease_id, study_db_id,
                 fc, p_value, fdr, is_significant, tier),
            )
            inserted_results += 1

        conn.commit()
        processed_studies += 1
        logger.info("Study %s: %d results inserted", sid, len(study_rows))

    logger.info(
        "Done. Processed %d studies, inserted %d biomarker_results rows.",
        processed_studies, inserted_results,
    )
    conn.close()
    if hmdb_cache_conn:
        hmdb_cache_conn.close()


if __name__ == "__main__":
    main()
