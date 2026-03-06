"""Generate a lightweight PubChem raw-data QC report.

This script is safe to run while ETL is in progress because it only reads:
- Downloaded PubChem files
- SID lookup SQLite

Output:
- JSON report in exports/
- Markdown summary in exports/
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
    _find_col,
    _normalize_accession,
    _normalize_columns,
)

_HUMAN_TAXID = 9606

_ASSAY_TYPE_CANDIDATES = [
    "assay_type",
    "outcome_type",
    "bioassay_type",
    "bioassay_types",
    "aid_type",
    "activity_type",
    "screening_type",
]
_ACCESSION_COLS = {
    "protein_accession",
    "protein_accessions",
    "proteinaccession",
    "uniprot",
    "uniprot_accession",
    "uniprots_id",
    "uniprots_ids",
}
_TAXID_COLS = {
    "target_taxid",
    "target_taxids",
    "taxonomy_id",
    "taxonomy_ids",
}
_TARGET_COLS = _ACCESSION_COLS | {
    "gene_id",
    "gene_ids",
    "target_id",
} | _TAXID_COLS


def _pct(n: int, d: int) -> float | None:
    if d == 0:
        return None
    return round((100.0 * n) / d, 4)


def _file_meta(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "bytes": int(st.st_size),
        "size_mb": round(st.st_size / (1024 * 1024), 2),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def _lookup_sid_rows(
    lookup_conn: sqlite3.Connection,
    sids: list[int],
) -> dict[int, tuple[int | None, str | None]]:
    if not sids:
        return {}
    unique_sids = list(dict.fromkeys(int(sid) for sid in sids))
    out: dict[int, tuple[int | None, str | None]] = {}
    batch_size = 500
    for i in range(0, len(unique_sids), batch_size):
        batch = unique_sids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = lookup_conn.execute(
            f"SELECT sid, cid, smiles FROM sid_cid_map WHERE sid IN ({placeholders})",
            batch,
        ).fetchall()
        for sid, cid, smiles in rows:
            out[int(sid)] = (None if cid is None else int(cid), smiles)
    return out


def scan_bioassays(path: Path, chunksize: int) -> tuple[dict, set[int]]:
    stats = {
        "rows_total": 0,
        "rows_aid_valid": 0,
        "rows_confirmatory": 0,
        "rows_confirmatory_with_target_annotation": 0,
        "rows_confirmatory_human_taxid": 0,
        "unique_confirmatory_aids": 0,
    }
    confirmatory_aids: set[int] = set()
    aid_col = None
    assay_type_candidates: list[str] = []
    target_cols: list[str] = []
    taxid_cols: list[str] = []

    reader = pd.read_csv(
        path,
        sep="\t",
        compression="gzip" if path.suffix == ".gz" else None,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        chunk = _normalize_columns(chunk)
        cols = list(chunk.columns)
        stats["rows_total"] += len(chunk)

        if aid_col is None:
            aid_col = _find_col(cols, ["aid"])
            if aid_col is None:
                raise ValueError(f"Could not find AID column in {path}")
            assay_type_candidates = [c for c in _ASSAY_TYPE_CANDIDATES if c in cols]
            target_cols = [c for c in cols if c in _TARGET_COLS]
            taxid_cols = [c for c in cols if c in _TAXID_COLS]

        out = chunk[pd.to_numeric(chunk[aid_col], errors="coerce").notna()].copy()
        if out.empty:
            continue
        out[aid_col] = out[aid_col].astype(int)
        stats["rows_aid_valid"] += len(out)

        assay_type_col = None
        for c in assay_type_candidates:
            if out[c].astype(str).str.contains("confirm", case=False, na=False).any():
                assay_type_col = c
                break
        if assay_type_col is None and assay_type_candidates:
            assay_type_col = assay_type_candidates[0]

        if assay_type_col is not None:
            conf = out[out[assay_type_col].astype(str).str.contains("confirm", case=False, na=False)]
        else:
            conf = out
        if conf.empty:
            continue

        stats["rows_confirmatory"] += len(conf)
        confirmatory_aids.update(int(v) for v in conf[aid_col].tolist())

        if target_cols:
            target_mask = pd.Series(False, index=conf.index)
            for c in target_cols:
                col = conf[c]
                if c in _ACCESSION_COLS:
                    target_mask = target_mask | col.map(_normalize_accession).notna()
                else:
                    target_mask = target_mask | col.notna()
            stats["rows_confirmatory_with_target_annotation"] += int(target_mask.sum())

        if taxid_cols:
            human_mask = pd.Series(False, index=conf.index)
            for c in taxid_cols:
                human_mask = human_mask | (pd.to_numeric(conf[c], errors="coerce") == _HUMAN_TAXID)
            stats["rows_confirmatory_human_taxid"] += int(human_mask.sum())

    stats["unique_confirmatory_aids"] = len(confirmatory_aids)
    stats["confirmatory_with_target_pct"] = _pct(
        stats["rows_confirmatory_with_target_annotation"],
        stats["rows_confirmatory"],
    )
    stats["confirmatory_human_taxid_pct"] = _pct(
        stats["rows_confirmatory_human_taxid"],
        stats["rows_confirmatory"],
    )
    return stats, confirmatory_aids


def scan_aid_map(path: Path, chunksize: int) -> tuple[dict, dict[int, str]]:
    stats = {
        "rows_total": 0,
        "rows_aid_valid": 0,
        "rows_accession_valid": 0,
        "unique_aids_with_accession": 0,
        "duplicate_aid_rows": 0,
        "conflicting_duplicate_aid_rows": 0,
    }
    aid_to_uniprot: dict[int, str] = {}
    aid_col = None
    uniprot_col = None

    reader = pd.read_csv(
        path,
        sep="\t",
        compression="gzip" if path.suffix == ".gz" else None,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        chunk = _normalize_columns(chunk)
        cols = list(chunk.columns)
        stats["rows_total"] += len(chunk)

        if aid_col is None:
            aid_col = _find_col(cols, ["aid"])
            if aid_col is None:
                raise ValueError(f"Could not find AID column in {path}")
            uniprot_col = _find_col(
                cols,
                [
                    "uniprot",
                    "uniprotkb_ac_id",
                    "uniprotkb_ac",
                    "uniprotkb_id",
                    "uniprot_accession",
                    "protein_accession",
                    "proteinaccession",
                    "accession",
                ],
            )
            if uniprot_col is None:
                raise ValueError(f"Could not find UniProt/accession column in {path}")

        out = chunk[[aid_col, uniprot_col]].copy()
        out = out[pd.to_numeric(out[aid_col], errors="coerce").notna()]
        if out.empty:
            continue
        out[aid_col] = out[aid_col].astype(int)
        stats["rows_aid_valid"] += len(out)

        out[uniprot_col] = out[uniprot_col].map(_normalize_accession)
        out = out[out[uniprot_col].notna()]
        if out.empty:
            continue
        stats["rows_accession_valid"] += len(out)

        for r in out.itertuples(index=False):
            aid = int(getattr(r, aid_col))
            acc = str(getattr(r, uniprot_col))
            if aid in aid_to_uniprot:
                stats["duplicate_aid_rows"] += 1
                if aid_to_uniprot[aid] != acc:
                    stats["conflicting_duplicate_aid_rows"] += 1
                continue
            aid_to_uniprot[aid] = acc

    stats["unique_aids_with_accession"] = len(aid_to_uniprot)
    stats["aid_valid_pct"] = _pct(stats["rows_aid_valid"], stats["rows_total"])
    stats["accession_valid_pct"] = _pct(stats["rows_accession_valid"], stats["rows_total"])
    return stats, aid_to_uniprot


def summarize_sid_lookup(path: Path) -> dict:
    with sqlite3.connect(str(path)) as conn:
        total = conn.execute("SELECT COUNT(*) FROM sid_cid_map").fetchone()[0]
        with_cid = conn.execute("SELECT COUNT(*) FROM sid_cid_map WHERE cid IS NOT NULL").fetchone()[0]
        with_smiles = conn.execute(
            "SELECT COUNT(*) FROM sid_cid_map WHERE smiles IS NOT NULL AND TRIM(smiles) <> ''"
        ).fetchone()[0]
        with_both = conn.execute(
            "SELECT COUNT(*) FROM sid_cid_map WHERE cid IS NOT NULL AND smiles IS NOT NULL AND TRIM(smiles) <> ''"
        ).fetchone()[0]
    return {
        "rows_total": int(total),
        "rows_with_cid": int(with_cid),
        "rows_with_smiles": int(with_smiles),
        "rows_with_cid_and_smiles": int(with_both),
        "with_cid_pct": _pct(int(with_cid), int(total)),
        "with_smiles_pct": _pct(int(with_smiles), int(total)),
        "with_cid_and_smiles_pct": _pct(int(with_both), int(total)),
    }


def sample_bioactivities(
    bioactivities_path: Path,
    sid_lookup_path: Path,
    confirmatory_aids: set[int],
    aid_to_uniprot: dict[int, str],
    chunksize: int,
    sample_rows: int,
) -> dict:
    stats = {
        "rows_sampled": 0,
        "rows_aid_sid_valid": 0,
        "rows_inactive": 0,
        "rows_inactive_confirmatory": 0,
        "rows_inactive_confirmatory_with_uniprot": 0,
        "rows_inactive_confirmatory_with_cid_smiles": 0,
        "rows_inactive_confirmatory_ready_all": 0,
        "rows_inactive_confirmatory_human_taxid": 0,
    }

    aid_col = None
    sid_col = None
    cid_col = None
    outcome_col = None
    accession_col = None
    taxid_col = None

    with sqlite3.connect(str(sid_lookup_path)) as sid_conn:
        reader = pd.read_csv(
            bioactivities_path,
            sep="\t",
            compression="gzip" if bioactivities_path.suffix == ".gz" else None,
            chunksize=chunksize,
            low_memory=False,
        )
        remaining = sample_rows
        for chunk in reader:
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
            remaining -= len(chunk)

            chunk = _normalize_columns(chunk)
            cols = list(chunk.columns)
            if aid_col is None:
                aid_col = _find_col(cols, ["aid"])
                sid_col = _find_col(cols, ["sid"])
                cid_col = _find_col(cols, ["cid"])
                outcome_col = _find_col(cols, ["activity_outcome", "activityoutcome"])
                accession_col = _find_col(
                    cols,
                    ["protein_accession", "proteinaccession", "uniprot_accession", "uniprot"],
                )
                taxid_col = _find_col(cols, ["target_taxid", "target_tax_id", "taxid", "taxonomy_id"])
                if aid_col is None or sid_col is None or outcome_col is None:
                    raise ValueError(
                        "Missing required bioactivities columns: need AID, SID, Activity Outcome"
                    )

            stats["rows_sampled"] += len(chunk)

            out = pd.DataFrame()
            out["aid"] = pd.to_numeric(chunk[aid_col], errors="coerce")
            out["sid"] = pd.to_numeric(chunk[sid_col], errors="coerce")
            out["cid"] = pd.to_numeric(chunk[cid_col], errors="coerce") if cid_col else pd.NA
            out["activity_outcome"] = chunk[outcome_col].astype(str)
            out["uniprot_direct"] = (
                chunk[accession_col].map(_normalize_accession) if accession_col else None
            )
            out["target_taxid"] = (
                pd.to_numeric(chunk[taxid_col], errors="coerce") if taxid_col else pd.NA
            )

            out = out[out["aid"].notna() & out["sid"].notna()].copy()
            if out.empty:
                continue
            out["aid"] = out["aid"].astype(int)
            out["sid"] = out["sid"].astype(int)
            stats["rows_aid_sid_valid"] += len(out)

            inactive = out[out["activity_outcome"].str.contains("inactive", case=False, na=False)].copy()
            if inactive.empty:
                continue
            stats["rows_inactive"] += len(inactive)

            inactive = inactive[inactive["aid"].isin(confirmatory_aids)].copy()
            if inactive.empty:
                continue
            stats["rows_inactive_confirmatory"] += len(inactive)

            aid_uniprot = inactive["aid"].map(aid_to_uniprot).map(_normalize_accession)
            inactive["resolved_uniprot"] = inactive["uniprot_direct"].where(
                inactive["uniprot_direct"].notna(),
                aid_uniprot,
            )
            stats["rows_inactive_confirmatory_with_uniprot"] += int(
                inactive["resolved_uniprot"].notna().sum()
            )

            sid_lookup = _lookup_sid_rows(sid_conn, inactive["sid"].tolist())
            resolved_cids: list[int | None] = []
            resolved_smiles: list[str | None] = []
            for r in inactive.itertuples(index=False):
                sid_info = sid_lookup.get(int(r.sid))
                cid = int(r.cid) if pd.notna(r.cid) else None
                smiles = None
                if sid_info is not None:
                    lookup_cid, lookup_smiles = sid_info
                    if cid is None:
                        cid = lookup_cid
                    smiles = lookup_smiles
                resolved_cids.append(cid)
                resolved_smiles.append(smiles)
            inactive["resolved_cid"] = resolved_cids
            inactive["resolved_smiles"] = resolved_smiles

            cid_smiles_ok = inactive["resolved_cid"].notna() & inactive["resolved_smiles"].notna()
            stats["rows_inactive_confirmatory_with_cid_smiles"] += int(cid_smiles_ok.sum())

            ready_all = cid_smiles_ok & inactive["resolved_uniprot"].notna()
            stats["rows_inactive_confirmatory_ready_all"] += int(ready_all.sum())

            if taxid_col is not None:
                stats["rows_inactive_confirmatory_human_taxid"] += int(
                    (inactive["target_taxid"] == _HUMAN_TAXID).sum()
                )

    stats["inactive_pct"] = _pct(stats["rows_inactive"], stats["rows_sampled"])
    stats["inactive_confirmatory_pct"] = _pct(
        stats["rows_inactive_confirmatory"],
        stats["rows_sampled"],
    )
    stats["inactive_confirmatory_with_uniprot_pct"] = _pct(
        stats["rows_inactive_confirmatory_with_uniprot"],
        stats["rows_inactive_confirmatory"],
    )
    stats["inactive_confirmatory_with_cid_smiles_pct"] = _pct(
        stats["rows_inactive_confirmatory_with_cid_smiles"],
        stats["rows_inactive_confirmatory"],
    )
    stats["inactive_confirmatory_ready_all_pct"] = _pct(
        stats["rows_inactive_confirmatory_ready_all"],
        stats["rows_inactive_confirmatory"],
    )
    stats["inactive_confirmatory_human_taxid_pct"] = _pct(
        stats["rows_inactive_confirmatory_human_taxid"],
        stats["rows_inactive_confirmatory"],
    )
    return stats


def render_markdown(report: dict) -> str:
    files = report["files"]
    bioassays = report["bioassays"]
    aid_map = report["aid_map"]
    sid_lookup = report["sid_lookup"]
    sample = report["bioactivities_sample"]
    lines = [
        "# PubChem QC Report",
        "",
        f"- generated_at_utc: {report['generated_at_utc']}",
        f"- sample_rows_target: {sample['sample_rows_target']}",
        f"- sample_rows_actual: {sample['rows_sampled']}",
        "",
        "## Files",
        "",
        "| file | size_mb | mtime_utc |",
        "|---|---:|---|",
    ]
    for key in [
        "bioactivities",
        "bioassays",
        "aid_uniprot",
        "sid_cid_smiles",
        "sid_lookup_db",
    ]:
        meta = files[key]
        lines.append(f"| {key} | {meta['size_mb']} | {meta['mtime_utc']} |")

    lines += [
        "",
        "## Bioassays",
        "",
        f"- rows_total: {bioassays['rows_total']}",
        f"- rows_confirmatory: {bioassays['rows_confirmatory']}",
        f"- unique_confirmatory_aids: {bioassays['unique_confirmatory_aids']}",
        f"- rows_confirmatory_with_target_annotation: {bioassays['rows_confirmatory_with_target_annotation']}",
        f"- confirmatory_with_target_pct: {bioassays['confirmatory_with_target_pct']}",
        f"- rows_confirmatory_human_taxid: {bioassays['rows_confirmatory_human_taxid']}",
        f"- confirmatory_human_taxid_pct: {bioassays['confirmatory_human_taxid_pct']}",
        "",
        "## AID Map",
        "",
        f"- rows_total: {aid_map['rows_total']}",
        f"- rows_accession_valid: {aid_map['rows_accession_valid']}",
        f"- unique_aids_with_accession: {aid_map['unique_aids_with_accession']}",
        f"- duplicate_aid_rows: {aid_map['duplicate_aid_rows']}",
        f"- conflicting_duplicate_aid_rows: {aid_map['conflicting_duplicate_aid_rows']}",
        "",
        "## SID Lookup",
        "",
        f"- rows_total: {sid_lookup['rows_total']}",
        f"- rows_with_cid: {sid_lookup['rows_with_cid']}",
        f"- rows_with_smiles: {sid_lookup['rows_with_smiles']}",
        f"- rows_with_cid_and_smiles: {sid_lookup['rows_with_cid_and_smiles']}",
        "",
        "## Bioactivities Sample",
        "",
        f"- rows_sampled: {sample['rows_sampled']}",
        f"- rows_inactive: {sample['rows_inactive']}",
        f"- rows_inactive_confirmatory: {sample['rows_inactive_confirmatory']}",
        f"- rows_inactive_confirmatory_with_uniprot: {sample['rows_inactive_confirmatory_with_uniprot']}",
        f"- rows_inactive_confirmatory_with_cid_smiles: {sample['rows_inactive_confirmatory_with_cid_smiles']}",
        f"- rows_inactive_confirmatory_ready_all: {sample['rows_inactive_confirmatory_ready_all']}",
        f"- inactive_confirmatory_ready_all_pct: {sample['inactive_confirmatory_ready_all_pct']}",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PubChem raw-data QC report")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2_000_000,
        help="Rows to sample from bioactivities.tsv.gz (set 0 to disable sample)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size for streamed TSV reads",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="pubchem_qc",
        help="Output filename prefix under exports/",
    )
    args = parser.parse_args()

    cfg = load_config()
    pubchem_cfg = cfg["downloads"]["pubchem"]

    bioactivities_path = _PROJECT_ROOT / pubchem_cfg["dest"]
    bioassays_path = _PROJECT_ROOT / pubchem_cfg["bioassays_dest"]
    aid_uniprot_path = _PROJECT_ROOT / pubchem_cfg["aid_uniprot_dest"]
    sid_cid_smiles_path = _PROJECT_ROOT / pubchem_cfg["sid_cid_smiles_dest"]
    sid_lookup_path = _PROJECT_ROOT / pubchem_cfg["sid_lookup_db"]

    for p in [bioactivities_path, bioassays_path, aid_uniprot_path, sid_cid_smiles_path, sid_lookup_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    bioassays_stats, confirmatory_aids = scan_bioassays(bioassays_path, chunksize=args.chunksize)
    aid_map_stats, aid_to_uniprot = scan_aid_map(aid_uniprot_path, chunksize=args.chunksize)
    sid_lookup_stats = summarize_sid_lookup(sid_lookup_path)

    sample_stats = {
        "sample_rows_target": int(args.sample_rows),
    }
    if args.sample_rows > 0:
        sample_stats.update(
            sample_bioactivities(
                bioactivities_path=bioactivities_path,
                sid_lookup_path=sid_lookup_path,
                confirmatory_aids=confirmatory_aids,
                aid_to_uniprot=aid_to_uniprot,
                chunksize=args.chunksize,
                sample_rows=args.sample_rows,
            )
        )
    else:
        sample_stats["rows_sampled"] = 0

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "bioactivities": _file_meta(bioactivities_path),
            "bioassays": _file_meta(bioassays_path),
            "aid_uniprot": _file_meta(aid_uniprot_path),
            "sid_cid_smiles": _file_meta(sid_cid_smiles_path),
            "sid_lookup_db": _file_meta(sid_lookup_path),
        },
        "bioassays": bioassays_stats,
        "aid_map": aid_map_stats,
        "sid_lookup": sid_lookup_stats,
        "bioactivities_sample": sample_stats,
    }

    exports_dir = _PROJECT_ROOT / cfg["paths"]["exports_dir"]
    exports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = exports_dir / f"{args.output_prefix}_{ts}.json"
    md_path = exports_dir / f"{args.output_prefix}_{ts}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"QC JSON: {json_path}")
    print(f"QC Markdown: {md_path}")
    print(f"Sample rows: {sample_stats.get('rows_sampled', 0)}")
    if sample_stats.get("rows_inactive_confirmatory_ready_all") is not None:
        print(
            "Ready-all rows (sample): "
            f"{sample_stats.get('rows_inactive_confirmatory_ready_all')} "
            f"({sample_stats.get('inactive_confirmatory_ready_all_pct')}%)"
        )


if __name__ == "__main__":
    main()
