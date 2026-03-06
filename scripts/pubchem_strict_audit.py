"""Full-scan audit for strict human-only PubChem ETL policy.

This script performs an exhaustive scan over PubChem bioactivities and reports:
- Legacy vs strict human-only filtering counts
- Drop reasons under strict filtering
- UniProt resolvability under each policy
- Optional audit of already-loaded PubChem rows in the SQLite DB

Outputs JSON and Markdown files under exports/.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from negbiodb.db import _PROJECT_ROOT
from negbiodb.download import load_config
from negbiodb.etl_pubchem import (
    _normalize_accession,
    _resolve_pubchem_chunk,
    load_aid_to_uniprot_map,
    load_confirmatory_aids,
    load_confirmatory_human_aids,
)

_HUMAN_TAXID = 9606


def _pct(n: int, d: int) -> float | None:
    if d == 0:
        return None
    return round((100.0 * n) / d, 6)


def _file_meta(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "bytes": int(st.st_size),
        "size_mb": round(st.st_size / (1024 * 1024), 2),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def _maybe_add_example(bucket: list[dict], sample_limit: int, row: pd.Series, reason: str, aid_is_human: bool) -> None:
    if len(bucket) >= sample_limit:
        return
    bucket.append(
        {
            "reason": reason,
            "aid": int(row["aid"]),
            "sid": int(row["sid"]),
            "target_taxid": None if pd.isna(row["target_taxid"]) else int(row["target_taxid"]),
            "protein_accession": None
            if pd.isna(row["protein_accession"])
            else str(row["protein_accession"]),
            "aid_in_confirmatory_human_set": bool(aid_is_human),
        }
    )


def scan_full_bioactivities(
    bioactivities_path: Path,
    confirmatory_aids: set[int],
    confirmatory_human_aids: set[int],
    aid_to_uniprot: dict[int, str],
    chunksize: int,
    sample_limit: int,
    progress_rows: int,
) -> dict:
    rows_seen = 0
    chunks_seen = 0
    next_progress = progress_rows

    stats = {
        "rows_seen": 0,
        "rows_inactive_confirmatory_base": 0,
        "legacy_rows_kept": 0,
        "strict_rows_kept": 0,
        "strict_rows_kept_known_human_taxid": 0,
        "strict_rows_kept_missing_taxid_human_aid": 0,
        "strict_rows_dropped_known_nonhuman_taxid": 0,
        "strict_rows_dropped_missing_taxid_nonhuman_aid": 0,
        "legacy_rows_with_uniprot": 0,
        "strict_rows_with_uniprot": 0,
        "strict_rows_without_uniprot": 0,
    }

    unique = {
        "base_aids": set(),
        "legacy_aids": set(),
        "strict_aids": set(),
        "strict_drop_nonhuman_aids": set(),
        "strict_drop_missing_nonhuman_aids": set(),
    }

    samples = {
        "kept_missing_taxid_human_aid": [],
        "dropped_known_nonhuman_taxid": [],
        "dropped_missing_taxid_nonhuman_aid": [],
    }

    reader = pd.read_csv(
        bioactivities_path,
        sep="\t",
        compression="gzip" if bioactivities_path.suffix == ".gz" else None,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        chunks_seen += 1
        rows_seen += len(chunk)
        stats["rows_seen"] = int(rows_seen)

        base = _resolve_pubchem_chunk(chunk, confirmatory_aids)
        if base.empty:
            if rows_seen >= next_progress:
                print(f"[audit] rows_seen={rows_seen} base=0 strict={stats['strict_rows_kept']}", flush=True)
                next_progress += progress_rows
            continue

        stats["rows_inactive_confirmatory_base"] += len(base)
        unique["base_aids"].update(int(v) for v in base["aid"].unique().tolist())

        known_taxid = base["target_taxid"].notna()
        human_taxid = base["target_taxid"] == _HUMAN_TAXID
        missing_taxid = ~known_taxid
        missing_taxid_human_aid = missing_taxid & base["aid"].isin(confirmatory_human_aids)
        strict_mask = human_taxid | missing_taxid_human_aid

        # Legacy behavior was chunk-level: only filter if chunk has any known taxid.
        legacy = base
        if known_taxid.any():
            legacy = base[human_taxid]
        stats["legacy_rows_kept"] += len(legacy)
        unique["legacy_aids"].update(int(v) for v in legacy["aid"].unique().tolist())

        strict = base[strict_mask]
        stats["strict_rows_kept"] += len(strict)
        unique["strict_aids"].update(int(v) for v in strict["aid"].unique().tolist())

        kept_known_human = int(human_taxid.sum())
        kept_missing_human = int(missing_taxid_human_aid.sum())
        dropped_known_nonhuman = int((known_taxid & (~human_taxid)).sum())
        dropped_missing_nonhuman = int((missing_taxid & (~base["aid"].isin(confirmatory_human_aids))).sum())

        stats["strict_rows_kept_known_human_taxid"] += kept_known_human
        stats["strict_rows_kept_missing_taxid_human_aid"] += kept_missing_human
        stats["strict_rows_dropped_known_nonhuman_taxid"] += dropped_known_nonhuman
        stats["strict_rows_dropped_missing_taxid_nonhuman_aid"] += dropped_missing_nonhuman

        if dropped_known_nonhuman > 0:
            drop_rows = base[known_taxid & (~human_taxid)]
            unique["strict_drop_nonhuman_aids"].update(int(v) for v in drop_rows["aid"].unique().tolist())
            for _, row in drop_rows.head(sample_limit).iterrows():
                _maybe_add_example(
                    samples["dropped_known_nonhuman_taxid"],
                    sample_limit,
                    row,
                    "known_nonhuman_taxid",
                    bool(int(row["aid"]) in confirmatory_human_aids),
                )

        if dropped_missing_nonhuman > 0:
            drop_rows = base[missing_taxid & (~base["aid"].isin(confirmatory_human_aids))]
            unique["strict_drop_missing_nonhuman_aids"].update(
                int(v) for v in drop_rows["aid"].unique().tolist()
            )
            for _, row in drop_rows.head(sample_limit).iterrows():
                _maybe_add_example(
                    samples["dropped_missing_taxid_nonhuman_aid"],
                    sample_limit,
                    row,
                    "missing_taxid_nonhuman_aid",
                    False,
                )

        if kept_missing_human > 0:
            keep_rows = base[missing_taxid_human_aid]
            for _, row in keep_rows.head(sample_limit).iterrows():
                _maybe_add_example(
                    samples["kept_missing_taxid_human_aid"],
                    sample_limit,
                    row,
                    "missing_taxid_human_aid",
                    True,
                )

        aid_mapped = base["aid"].map(aid_to_uniprot).map(_normalize_accession)
        resolved_uniprot = base["protein_accession"].where(base["protein_accession"].notna(), aid_mapped)

        legacy_uniprot = resolved_uniprot.loc[legacy.index]
        strict_uniprot = resolved_uniprot.loc[strict.index]
        stats["legacy_rows_with_uniprot"] += int(legacy_uniprot.notna().sum())
        stats["strict_rows_with_uniprot"] += int(strict_uniprot.notna().sum())
        stats["strict_rows_without_uniprot"] += int(strict_uniprot.isna().sum())

        if rows_seen >= next_progress:
            print(
                f"[audit] rows_seen={rows_seen} base={stats['rows_inactive_confirmatory_base']} "
                f"legacy={stats['legacy_rows_kept']} strict={stats['strict_rows_kept']}",
                flush=True,
            )
            next_progress += progress_rows

    stats["chunks_seen"] = chunks_seen
    stats["rows_seen"] = rows_seen
    stats["unique_base_aids"] = len(unique["base_aids"])
    stats["unique_legacy_aids"] = len(unique["legacy_aids"])
    stats["unique_strict_aids"] = len(unique["strict_aids"])
    stats["unique_strict_drop_nonhuman_aids"] = len(unique["strict_drop_nonhuman_aids"])
    stats["unique_strict_drop_missing_nonhuman_aids"] = len(unique["strict_drop_missing_nonhuman_aids"])

    stats["strict_vs_legacy_row_retention_pct"] = _pct(
        stats["strict_rows_kept"],
        stats["legacy_rows_kept"],
    )
    stats["strict_vs_base_row_retention_pct"] = _pct(
        stats["strict_rows_kept"],
        stats["rows_inactive_confirmatory_base"],
    )
    stats["strict_with_uniprot_pct"] = _pct(
        stats["strict_rows_with_uniprot"],
        stats["strict_rows_kept"],
    )

    return {
        "stats": stats,
        "samples": samples,
    }


def audit_existing_db(db_path: Path, confirmatory_human_aids: set[int]) -> dict:
    out = {
        "db_path": str(db_path),
        "pubchem_rows_total": 0,
        "pubchem_assays_total": 0,
        "pubchem_assays_in_human_aid_set": 0,
        "pubchem_assays_not_in_human_aid_set": 0,
        "pubchem_rows_from_nonhuman_assays": 0,
        "species_top_counts": [],
    }
    if not db_path.exists():
        out["db_exists"] = False
        return out
    out["db_exists"] = True

    conn = sqlite3.connect(str(db_path))
    try:
        out["pubchem_rows_total"] = int(
            conn.execute(
                "SELECT COUNT(*) FROM negative_results WHERE source_db='pubchem'"
            ).fetchone()[0]
        )
        out["pubchem_assays_total"] = int(
            conn.execute("SELECT COUNT(*) FROM assays WHERE source_db='pubchem'").fetchone()[0]
        )
        species_rows = conn.execute(
            """
            SELECT species_tested, COUNT(*)
            FROM negative_results
            WHERE source_db='pubchem'
            GROUP BY species_tested
            ORDER BY COUNT(*) DESC
            LIMIT 20
            """
        ).fetchall()
        out["species_top_counts"] = [
            {"species_tested": row[0], "count": int(row[1])} for row in species_rows
        ]

        assay_ids = conn.execute(
            "SELECT assay_id, source_assay_id FROM assays WHERE source_db='pubchem'"
        ).fetchall()
        if not assay_ids:
            return out

        nonhuman_assay_ids: list[int] = []
        human_assay_count = 0
        for assay_id, source_assay_id in assay_ids:
            try:
                aid = int(source_assay_id)
            except (TypeError, ValueError):
                continue
            if aid in confirmatory_human_aids:
                human_assay_count += 1
            else:
                nonhuman_assay_ids.append(int(assay_id))
        out["pubchem_assays_in_human_aid_set"] = int(human_assay_count)
        out["pubchem_assays_not_in_human_aid_set"] = int(len(nonhuman_assay_ids))

        if nonhuman_assay_ids:
            rows = 0
            batch_size = 500
            for i in range(0, len(nonhuman_assay_ids), batch_size):
                batch = nonhuman_assay_ids[i : i + batch_size]
                placeholders = ",".join("?" * len(batch))
                rows += int(
                    conn.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM negative_results
                        WHERE source_db='pubchem'
                          AND assay_id IN ({placeholders})
                        """,
                        batch,
                    ).fetchone()[0]
                )
            out["pubchem_rows_from_nonhuman_assays"] = int(rows)
    finally:
        conn.close()

    return out


def render_markdown(report: dict) -> str:
    fs = report["full_scan"]["stats"]
    db = report["existing_db_audit"]
    lines = [
        "# PubChem Strict Human-Only Audit",
        "",
        f"- generated_at_utc: {report['generated_at_utc']}",
        "",
        "## File Metadata",
        "",
        "| file | size_mb | mtime_utc |",
        "|---|---:|---|",
    ]
    for key in ["bioactivities", "bioassays", "aid_uniprot"]:
        m = report["files"][key]
        lines.append(f"| {key} | {m['size_mb']} | {m['mtime_utc']} |")

    lines += [
        "",
        "## Key Sets",
        "",
        f"- confirmatory_aids: {report['key_sets']['confirmatory_aids']}",
        f"- confirmatory_human_aids: {report['key_sets']['confirmatory_human_aids']}",
        f"- aid_to_uniprot: {report['key_sets']['aid_to_uniprot']}",
        "",
        "## Full Scan",
        "",
        f"- rows_seen: {fs['rows_seen']}",
        f"- rows_inactive_confirmatory_base: {fs['rows_inactive_confirmatory_base']}",
        f"- legacy_rows_kept: {fs['legacy_rows_kept']}",
        f"- strict_rows_kept: {fs['strict_rows_kept']}",
        f"- strict_vs_legacy_row_retention_pct: {fs['strict_vs_legacy_row_retention_pct']}",
        f"- strict_rows_dropped_known_nonhuman_taxid: {fs['strict_rows_dropped_known_nonhuman_taxid']}",
        f"- strict_rows_dropped_missing_taxid_nonhuman_aid: {fs['strict_rows_dropped_missing_taxid_nonhuman_aid']}",
        f"- strict_rows_kept_missing_taxid_human_aid: {fs['strict_rows_kept_missing_taxid_human_aid']}",
        f"- strict_rows_with_uniprot: {fs['strict_rows_with_uniprot']}",
        f"- strict_rows_without_uniprot: {fs['strict_rows_without_uniprot']}",
        "",
        "## Existing DB Audit",
        "",
        f"- db_exists: {db.get('db_exists')}",
        f"- pubchem_rows_total: {db.get('pubchem_rows_total')}",
        f"- pubchem_assays_total: {db.get('pubchem_assays_total')}",
        f"- pubchem_assays_in_human_aid_set: {db.get('pubchem_assays_in_human_aid_set')}",
        f"- pubchem_assays_not_in_human_aid_set: {db.get('pubchem_assays_not_in_human_aid_set')}",
        f"- pubchem_rows_from_nonhuman_assays: {db.get('pubchem_rows_from_nonhuman_assays')}",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Full audit for strict human-only PubChem ETL policy")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Per-category sample rows in the report",
    )
    parser.add_argument(
        "--progress-rows",
        type=int,
        default=5_000_000,
        help="Print progress every N scanned rows",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="pubchem_strict_audit",
    )
    args = parser.parse_args()

    cfg = load_config()
    pubchem_cfg = cfg["downloads"]["pubchem"]
    bioactivities_path = _PROJECT_ROOT / pubchem_cfg["dest"]
    bioassays_path = _PROJECT_ROOT / pubchem_cfg["bioassays_dest"]
    aid_uniprot_path = _PROJECT_ROOT / pubchem_cfg["aid_uniprot_dest"]
    db_path = _PROJECT_ROOT / cfg["paths"]["database"]

    for p in [bioactivities_path, bioassays_path, aid_uniprot_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    confirmatory_aids = load_confirmatory_aids(bioassays_path, chunksize=args.chunksize)
    confirmatory_human_aids = load_confirmatory_human_aids(bioassays_path, chunksize=args.chunksize)
    aid_to_uniprot = load_aid_to_uniprot_map(aid_uniprot_path, chunksize=args.chunksize)

    print(
        f"[audit] loaded sets: confirm={len(confirmatory_aids)} "
        f"confirm_human={len(confirmatory_human_aids)} aid_map={len(aid_to_uniprot)}",
        flush=True,
    )

    full_scan = scan_full_bioactivities(
        bioactivities_path=bioactivities_path,
        confirmatory_aids=confirmatory_aids,
        confirmatory_human_aids=confirmatory_human_aids,
        aid_to_uniprot=aid_to_uniprot,
        chunksize=args.chunksize,
        sample_limit=args.sample_limit,
        progress_rows=args.progress_rows,
    )

    existing_db = audit_existing_db(db_path, confirmatory_human_aids)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "bioactivities": _file_meta(bioactivities_path),
            "bioassays": _file_meta(bioassays_path),
            "aid_uniprot": _file_meta(aid_uniprot_path),
        },
        "key_sets": {
            "confirmatory_aids": len(confirmatory_aids),
            "confirmatory_human_aids": len(confirmatory_human_aids),
            "aid_to_uniprot": len(aid_to_uniprot),
        },
        "full_scan": full_scan,
        "existing_db_audit": existing_db,
    }

    exports_dir = _PROJECT_ROOT / cfg["paths"]["exports_dir"]
    exports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = exports_dir / f"{args.output_prefix}_{ts}.json"
    md_path = exports_dir / f"{args.output_prefix}_{ts}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    fs = report["full_scan"]["stats"]
    print(f"Audit JSON: {json_path}")
    print(f"Audit Markdown: {md_path}")
    print(
        "[audit] strict rows kept: "
        f"{fs['strict_rows_kept']} / legacy {fs['legacy_rows_kept']} "
        f"({fs['strict_vs_legacy_row_retention_pct']}%)"
    )


if __name__ == "__main__":
    main()
