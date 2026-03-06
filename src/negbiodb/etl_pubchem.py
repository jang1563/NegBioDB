"""ETL pipeline for loading PubChem confirmatory inactive DTI data.

Design goals:
- Streaming processing for large `bioactivities.tsv.gz`
- Confirmatory-assay filtering using `bioassays.tsv.gz`
- AID->UniProt mapping via `Aid2GeneidAccessionUniProt.gz`
- SID->CID/SMILES enrichment via `Sid2CidSMILES.gz` lookup DB
"""

from __future__ import annotations

import gzip
import logging
import math
import re
import sqlite3
from pathlib import Path

import pandas as pd

from negbiodb.db import connect, create_database, _PROJECT_ROOT
from negbiodb.download import load_config
from negbiodb.etl_chembl import refresh_all_pairs
from negbiodb.standardize import standardize_smiles

logger = logging.getLogger(__name__)
_HUMAN_TAXID = 9606
_UNIPROT_RE_6 = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$")
_UNIPROT_RE_10 = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9][A-Z0-9]{3}[0-9]$")


def _normalize_col_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _normalize_col_name(str(c)) for c in df.columns})


def _extract_accession_token(text: str) -> str:
    """Normalize common accession encodings to a single token.

    Examples:
      - 'sp|P00533|EGFR_HUMAN' -> 'P00533'
      - 'P00533; Q9Y6K9' -> 'P00533'
    """
    token = text.strip()
    if "|" in token:
        parts = [p.strip() for p in token.split("|") if p.strip()]
        if len(parts) >= 2 and parts[0].lower() in {"sp", "tr", "uniprotkb"}:
            token = parts[1]
        else:
            token = parts[0]
    token = re.split(r"[;,/]", token, maxsplit=1)[0].strip()
    token = token.split()[0] if token else token
    return token


def _normalize_accession(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None
    norm = _extract_accession_token(text)
    return norm or None


def _is_uniprot_accession(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().upper()
    if text == "":
        return False
    return bool(_UNIPROT_RE_6.match(text) or _UNIPROT_RE_10.match(text))


def _normalize_uniprot_accession(value: object) -> str | None:
    accession = _normalize_accession(value)
    if accession is None:
        return None
    accession = accession.upper()
    if not _is_uniprot_accession(accession):
        return None
    return accession


def _is_human_taxid_value(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if text == "":
        return False
    # Handle scalar numerics quickly.
    try:
        return int(float(text)) == _HUMAN_TAXID
    except (TypeError, ValueError):
        pass
    # Handle multi-value cells: "9606;10090", "9606,10090", "9606|10090", etc.
    for token in re.split(r"[;,/|\s]+", text):
        token = token.strip()
        if not token:
            continue
        try:
            if int(float(token)) == _HUMAN_TAXID:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _is_nm_unit(value: object) -> bool:
    if pd.isna(value):
        return False
    unit = str(value).strip().lower().replace(" ", "")
    return unit in {"nm", "nanomolar"}


def _source_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime)


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    colset = set(columns)
    for cand in candidates:
        if cand in colset:
            return cand
    return None


def _open_tsv_chunks(
    path: Path,
    chunksize: int,
) -> pd.io.parsers.TextFileReader:
    return pd.read_csv(
        path,
        sep="\t",
        compression="gzip" if path.suffix == ".gz" else None,
        chunksize=chunksize,
        low_memory=False,
    )


def _is_header_line(first_line: str) -> bool:
    first = first_line.split("\t", 1)[0].strip()
    return not first.isdigit()


def load_confirmatory_aids(
    bioassays_path: Path,
    chunksize: int = 500_000,
) -> set[int]:
    """Load confirmatory AIDs with target annotation from bioassays TSV."""
    reader = pd.read_csv(
        bioassays_path,
        sep="\t",
        compression="gzip" if bioassays_path.suffix == ".gz" else None,
        chunksize=chunksize,
        low_memory=False,
    )
    aids: set[int] = set()
    aid_col = None
    assay_type_candidates: list[str] = []
    target_cols: list[str] = []

    for chunk in reader:
        chunk = _normalize_columns(chunk)
        cols = list(chunk.columns)
        if aid_col is None:
            aid_col = _find_col(cols, ["aid"])
            if aid_col is None:
                raise ValueError(f"Could not find AID column in {bioassays_path}")
            assay_type_candidates = [
                c
                for c in [
                    "assay_type",
                    "outcome_type",
                    "bioassay_type",
                    "bioassay_types",
                    "aid_type",
                    "activity_type",
                    "screening_type",
                ]
                if c in cols
            ]
            target_cols = [
                c
                for c in cols
                if c
                in {
                    "protein_accession",
                    "protein_accessions",
                    "proteinaccession",
                    "uniprot",
                    "uniprot_accession",
                    "uniprots_id",
                    "uniprots_ids",
                    "gene_id",
                    "gene_ids",
                    "target_taxid",
                    "target_taxids",
                    "target_id",
                    "taxonomy_id",
                    "taxonomy_ids",
                }
            ]

        out = chunk[pd.to_numeric(chunk[aid_col], errors="coerce").notna()].copy()
        if out.empty:
            continue
        out[aid_col] = out[aid_col].astype(int)

        assay_type_col = None
        for c in assay_type_candidates:
            if out[c].astype(str).str.contains("confirm", case=False, na=False).any():
                assay_type_col = c
                break
        if assay_type_col is None and assay_type_candidates:
            assay_type_col = assay_type_candidates[0]

        if assay_type_col is not None:
            out = out[
                out[assay_type_col]
                .astype(str)
                .str.contains("confirm", case=False, na=False)
            ]
        if out.empty:
            continue

        if target_cols:
            mask = pd.Series(False, index=out.index)
            for c in target_cols:
                col = out[c]
                if c in {
                    "protein_accession",
                    "protein_accessions",
                    "proteinaccession",
                    "uniprot",
                    "uniprot_accession",
                    "uniprots_id",
                    "uniprots_ids",
                }:
                    mask = mask | col.map(_normalize_accession).notna()
                else:
                    mask = mask | col.notna()
            out = out[mask]
            if out.empty:
                continue

        aids.update(int(v) for v in out[aid_col].tolist())

    if not assay_type_candidates:
        logger.warning("No assay-type column found in bioassays.tsv.gz; using all AIDs.")
    if not target_cols:
        logger.warning("No target annotation columns found in bioassays.tsv.gz.")

    logger.info("Loaded %d confirmatory AIDs from %s", len(aids), bioassays_path)
    return aids


def load_confirmatory_human_aids(
    bioassays_path: Path,
    chunksize: int = 500_000,
) -> set[int]:
    """Load confirmatory AIDs with explicit human taxid evidence."""
    reader = pd.read_csv(
        bioassays_path,
        sep="\t",
        compression="gzip" if bioassays_path.suffix == ".gz" else None,
        chunksize=chunksize,
        low_memory=False,
    )
    aids: set[int] = set()
    aid_col = None
    assay_type_candidates: list[str] = []
    taxid_cols: list[str] = []

    for chunk in reader:
        chunk = _normalize_columns(chunk)
        cols = list(chunk.columns)
        if aid_col is None:
            aid_col = _find_col(cols, ["aid"])
            if aid_col is None:
                raise ValueError(f"Could not find AID column in {bioassays_path}")
            assay_type_candidates = [
                c
                for c in [
                    "assay_type",
                    "outcome_type",
                    "bioassay_type",
                    "bioassay_types",
                    "aid_type",
                    "activity_type",
                    "screening_type",
                ]
                if c in cols
            ]
            taxid_cols = [
                c
                for c in cols
                if c in {"target_taxid", "target_taxids", "taxonomy_id", "taxonomy_ids"}
            ]

        out = chunk[pd.to_numeric(chunk[aid_col], errors="coerce").notna()].copy()
        if out.empty:
            continue
        out[aid_col] = out[aid_col].astype(int)

        assay_type_col = None
        for c in assay_type_candidates:
            if out[c].astype(str).str.contains("confirm", case=False, na=False).any():
                assay_type_col = c
                break
        if assay_type_col is None and assay_type_candidates:
            assay_type_col = assay_type_candidates[0]

        if assay_type_col is not None:
            out = out[
                out[assay_type_col]
                .astype(str)
                .str.contains("confirm", case=False, na=False)
            ]
        if out.empty or not taxid_cols:
            continue

        human_mask = pd.Series(False, index=out.index)
        for c in taxid_cols:
            human_mask = human_mask | out[c].map(_is_human_taxid_value)
        human_rows = out[human_mask]
        if human_rows.empty:
            continue

        aids.update(int(v) for v in human_rows[aid_col].tolist())

    if not assay_type_candidates:
        logger.warning("No assay-type column found in bioassays.tsv.gz; using all AIDs.")
    if not taxid_cols:
        logger.warning("No taxid columns found in bioassays.tsv.gz; strict human filtering may be empty.")

    logger.info("Loaded %d confirmatory human AIDs from %s", len(aids), bioassays_path)
    return aids


def load_aid_to_uniprot_map(
    aid_map_path: Path,
    chunksize: int = 500_000,
) -> dict[int, str]:
    """Load AID->UniProt mapping from Aid2GeneidAccessionUniProt.gz.

    If multiple UniProt accessions map to the same AID, keeps the first one
    encountered and logs how many duplicate AIDs were present.
    """
    reader = pd.read_csv(
        aid_map_path,
        sep="\t",
        compression="gzip" if aid_map_path.suffix == ".gz" else None,
        chunksize=chunksize,
        low_memory=False,
    )
    mapping: dict[int, str] = {}
    duplicate_aids = 0
    aid_col = None
    uniprot_col = None

    for chunk in reader:
        chunk = _normalize_columns(chunk)
        cols = list(chunk.columns)
        if aid_col is None:
            aid_col = _find_col(cols, ["aid"])
            if aid_col is None:
                raise ValueError(f"Could not find AID column in {aid_map_path}")
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
                raise ValueError(f"Could not find UniProt/accession column in {aid_map_path}")

        out = chunk[[aid_col, uniprot_col]].copy()
        out = out[pd.to_numeric(out[aid_col], errors="coerce").notna()]
        if out.empty:
            continue
        out[aid_col] = out[aid_col].astype(int)
        out[uniprot_col] = out[uniprot_col].map(_normalize_uniprot_accession)
        out = out[out[uniprot_col].notna()]

        for r in out.itertuples(index=False):
            aid = int(getattr(r, aid_col))
            accession = str(getattr(r, uniprot_col))
            if aid in mapping:
                if mapping[aid] != accession:
                    duplicate_aids += 1
                continue
            mapping[aid] = accession

    if duplicate_aids > 0:
        logger.warning(
            "AID->UniProt mapping has %d duplicate AIDs with multiple accessions; "
            "kept first accession per AID.",
            duplicate_aids,
        )
    logger.info("Loaded %d AID->UniProt mappings", len(mapping))
    return mapping


def build_sid_lookup_db(sid_map_path: Path, lookup_db_path: Path) -> Path:
    """Build local SQLite lookup DB for SID->CID/SMILES if needed."""
    lookup_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(lookup_db_path))
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sid_cid_map (
                sid INTEGER PRIMARY KEY,
                cid INTEGER,
                smiles TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sid_lookup_meta (
                meta_key TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_size INTEGER NOT NULL,
                source_mtime INTEGER NOT NULL,
                complete INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            )"""
        )
        source_size, source_mtime = _source_signature(sid_map_path)
        meta = conn.execute(
            """SELECT source_size, source_mtime, complete
            FROM sid_lookup_meta WHERE meta_key = 'sid_cid_map'"""
        ).fetchone()
        existing = conn.execute("SELECT COUNT(*) FROM sid_cid_map").fetchone()[0]
        has_nonempty_smiles = conn.execute(
            """SELECT 1
            FROM sid_cid_map
            WHERE smiles IS NOT NULL AND TRIM(smiles) <> ''
            LIMIT 1"""
        ).fetchone() is not None
        if (
            meta is not None
            and int(meta[2]) == 1
            and int(meta[0]) == source_size
            and int(meta[1]) == source_mtime
            and existing > 0
            and has_nonempty_smiles
        ):
            logger.info(
                "Reusing existing SID lookup DB (%d rows): %s",
                existing,
                lookup_db_path,
            )
            return lookup_db_path
        if existing > 0:
            if not has_nonempty_smiles:
                logger.warning(
                    "Rebuilding SID lookup DB because existing lookup has no SMILES values: %s",
                    lookup_db_path,
                )
            else:
                logger.warning(
                    "Rebuilding SID lookup DB because source changed or prior build incomplete: %s",
                    lookup_db_path,
                )
        conn.execute("DELETE FROM sid_cid_map")
        conn.execute(
            """INSERT INTO sid_lookup_meta
            (meta_key, source_path, source_size, source_mtime, complete, updated_at)
            VALUES ('sid_cid_map', ?, ?, ?, 0, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            ON CONFLICT(meta_key) DO UPDATE SET
                source_path=excluded.source_path,
                source_size=excluded.source_size,
                source_mtime=excluded.source_mtime,
                complete=0,
                updated_at=strftime('%Y-%m-%dT%H:%M:%SZ', 'now')""",
            (str(sid_map_path), source_size, source_mtime),
        )
        conn.commit()

        with gzip.open(sid_map_path, "rt") as fh:
            first_line = fh.readline()
        has_header = _is_header_line(first_line)

        reader = pd.read_csv(
            sid_map_path,
            sep="\t",
            compression="gzip" if sid_map_path.suffix == ".gz" else None,
            chunksize=500_000,
            low_memory=False,
            header=0 if has_header else None,
            names=None if has_header else ["sid", "cid", "smiles"],
        )

        total = 0
        for chunk in reader:
            chunk = _normalize_columns(chunk)
            cols = list(chunk.columns)
            sid_col = _find_col(cols, ["sid"])
            cid_col = _find_col(cols, ["cid"])
            smiles_col = _find_col(
                cols,
                ["smiles", "canonical_smiles", "isomeric_smiles"],
            )

            if sid_col is None or cid_col is None:
                # Fallback for malformed headered files
                raw = chunk.copy()
                raw.columns = [f"col_{i}" for i in range(len(raw.columns))]
                sid_col = "col_0"
                cid_col = "col_1"
                smiles_col = "col_2" if len(raw.columns) > 2 else None
                chunk = raw

            if smiles_col is None:
                chunk["smiles"] = None
                smiles_col = "smiles"

            out = chunk[[sid_col, cid_col, smiles_col]].copy()
            out.columns = ["sid", "cid", "smiles"]
            out["sid"] = pd.to_numeric(out["sid"], errors="coerce")
            out["cid"] = pd.to_numeric(out["cid"], errors="coerce")
            out = out[out["sid"].notna()]
            out["sid"] = out["sid"].astype(int)
            out["cid"] = out["cid"].astype("Int64")

            rows = [
                (int(r.sid), int(r.cid) if pd.notna(r.cid) else None, None if pd.isna(r.smiles) else str(r.smiles))
                for r in out.itertuples(index=False)
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO sid_cid_map (sid, cid, smiles) VALUES (?, ?, ?)",
                rows,
            )
            total += len(rows)
            conn.commit()

        conn.execute("CREATE INDEX IF NOT EXISTS idx_sid_cid_map_cid ON sid_cid_map(cid)")
        conn.execute(
            """UPDATE sid_lookup_meta
            SET complete = 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            WHERE meta_key = 'sid_cid_map'"""
        )
        conn.commit()
        logger.info("Built SID lookup DB with %d rows: %s", total, lookup_db_path)
        return lookup_db_path
    finally:
        conn.close()


def _lookup_sid_rows(lookup_conn: sqlite3.Connection, sids: list[int]) -> dict[int, tuple[int | None, str | None]]:
    if not sids:
        return {}
    out: dict[int, tuple[int | None, str | None]] = {}
    batch_size = 500
    for i in range(0, len(sids), batch_size):
        batch = sids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = lookup_conn.execute(
            f"SELECT sid, cid, smiles FROM sid_cid_map WHERE sid IN ({placeholders})",
            batch,
        ).fetchall()
        for sid, cid, smiles in rows:
            out[int(sid)] = (None if cid is None else int(cid), smiles)
    return out


def _resolve_pubchem_chunk(chunk: pd.DataFrame, confirmatory_aids: set[int]) -> pd.DataFrame:
    chunk = _normalize_columns(chunk)
    cols = list(chunk.columns)

    aid_col = _find_col(cols, ["aid"])
    sid_col = _find_col(cols, ["sid"])
    cid_col = _find_col(cols, ["cid"])
    outcome_col = _find_col(cols, ["activity_outcome", "activityoutcome"])
    value_col = _find_col(cols, ["activity_value", "activityvalue"])
    name_col = _find_col(cols, ["activity_name", "activityname"])
    unit_col = _find_col(cols, ["activity_unit", "activityunit"])
    taxid_col = _find_col(cols, ["target_taxid", "target_tax_id", "taxid", "taxonomy_id"])
    accession_col = _find_col(
        cols,
        ["protein_accession", "proteinaccession", "uniprot_accession", "uniprot"],
    )

    if aid_col is None or sid_col is None or outcome_col is None:
        raise ValueError("Missing required PubChem columns (AID/SID/Activity Outcome)")

    out = pd.DataFrame()
    out["aid"] = pd.to_numeric(chunk[aid_col], errors="coerce")
    out["sid"] = pd.to_numeric(chunk[sid_col], errors="coerce")
    out["cid"] = pd.to_numeric(chunk[cid_col], errors="coerce") if cid_col else pd.NA
    out["activity_outcome"] = chunk[outcome_col].astype(str)
    out["activity_value"] = pd.to_numeric(chunk[value_col], errors="coerce") if value_col else pd.NA
    out["activity_name"] = chunk[name_col].astype(str) if name_col else "bioactivity"
    out["activity_unit"] = chunk[unit_col].astype(str) if unit_col else "nM"
    out["target_taxid"] = pd.to_numeric(chunk[taxid_col], errors="coerce") if taxid_col else pd.NA
    out["protein_accession"] = (
        chunk[accession_col].map(_normalize_accession) if accession_col else None
    )

    out = out[out["aid"].notna() & out["sid"].notna()]
    out["aid"] = out["aid"].astype(int)
    out["sid"] = out["sid"].astype(int)
    out = out[out["aid"].isin(confirmatory_aids)]
    out = out[
        out["activity_outcome"]
        .str.contains("inactive", case=False, na=False)
    ]
    return out


def run_pubchem_etl(
    db_path: Path,
    bioactivities_path: Path | None = None,
    bioassays_path: Path | None = None,
    aid_uniprot_path: Path | None = None,
    sid_cid_smiles_path: Path | None = None,
    sid_lookup_db_path: Path | None = None,
    chunksize: int | None = None,
) -> dict:
    """Run full PubChem ETL into NegBioDB."""
    cfg = load_config()
    pubchem_cfg = cfg["downloads"]["pubchem"]

    if bioactivities_path is None:
        bioactivities_path = _PROJECT_ROOT / pubchem_cfg["dest"]
    if bioassays_path is None:
        bioassays_path = _PROJECT_ROOT / pubchem_cfg["bioassays_dest"]
    if aid_uniprot_path is None:
        aid_uniprot_path = _PROJECT_ROOT / pubchem_cfg["aid_uniprot_dest"]
    if sid_cid_smiles_path is None:
        sid_cid_smiles_path = _PROJECT_ROOT / pubchem_cfg["sid_cid_smiles_dest"]
    if sid_lookup_db_path is None:
        sid_lookup_db_path = _PROJECT_ROOT / pubchem_cfg["sid_lookup_db"]
    if chunksize is None:
        chunksize = int(cfg.get("pubchem_etl", {}).get("chunksize", cfg.get("pubchem_chunksize", 100000)))
    human_only = bool(cfg.get("pubchem_etl", {}).get("human_only", True))
    inactivity_threshold_nm = float(cfg.get("inactivity_threshold_nm", 10000))

    confirmatory_aids = load_confirmatory_aids(bioassays_path)
    confirmatory_human_aids: set[int] = set()
    if human_only:
        confirmatory_human_aids = load_confirmatory_human_aids(bioassays_path)
    aid_to_uniprot = load_aid_to_uniprot_map(aid_uniprot_path)
    build_sid_lookup_db(sid_cid_smiles_path, sid_lookup_db_path)

    create_database(db_path)

    rows_read = 0
    rows_filtered = 0
    rows_mapped = 0
    rows_skipped = 0
    rows_attempted_insert = 0

    with connect(db_path) as conn, sqlite3.connect(str(sid_lookup_db_path)) as sid_conn:
        before_results = conn.execute(
            "SELECT COUNT(*) FROM negative_results WHERE source_db='pubchem'"
        ).fetchone()[0]

        compound_cache: dict[tuple[int | None, str], int] = {}
        target_cache: dict[str, int] = {}
        assay_cache: dict[int, int] = {}
        insert_params: list[tuple] = []

        reader = _open_tsv_chunks(bioactivities_path, chunksize=chunksize)
        for chunk in reader:
            rows_read += len(chunk)
            filt = _resolve_pubchem_chunk(chunk, confirmatory_aids)
            if human_only and not filt.empty:
                known_taxid = filt["target_taxid"].notna()
                human_taxid = filt["target_taxid"] == _HUMAN_TAXID
                human_aid_with_missing_taxid = (~known_taxid) & filt["aid"].isin(confirmatory_human_aids)
                filt = filt[human_taxid | human_aid_with_missing_taxid]
            rows_filtered += len(filt)
            if filt.empty:
                continue

            sid_lookup = _lookup_sid_rows(sid_conn, filt["sid"].astype(int).tolist())
            direct_uniprot = filt["protein_accession"].map(_normalize_uniprot_accession)
            aid_mapped = filt["aid"].map(aid_to_uniprot).map(_normalize_uniprot_accession)
            filt["uniprot_accession"] = direct_uniprot.where(
                direct_uniprot.notna(),
                aid_mapped,
            )

            # Fill cid/smiles via SID lookup
            cids: list[int | None] = []
            smiles: list[str | None] = []
            for r in filt.itertuples(index=False):
                sid_info = sid_lookup.get(int(r.sid))
                cid = int(r.cid) if pd.notna(r.cid) else None
                smi = None
                if sid_info is not None:
                    lookup_cid, lookup_smiles = sid_info
                    if cid is None:
                        cid = lookup_cid
                    smi = lookup_smiles
                cids.append(cid)
                smiles.append(smi)

            filt["resolved_cid"] = cids
            filt["smiles"] = smiles

            keep = filt[
                filt["resolved_cid"].notna()
                & filt["smiles"].notna()
                & filt["uniprot_accession"].notna()
            ].copy()
            rows_mapped += len(keep)
            rows_skipped += len(filt) - len(keep)
            if keep.empty:
                continue

            for r in keep.itertuples(index=False):
                cid = int(r.resolved_cid)
                smi = str(r.smiles)
                cache_key = (cid, smi)
                compound_id = compound_cache.get(cache_key)
                if compound_id is None:
                    std = standardize_smiles(smi)
                    if std is None:
                        rows_skipped += 1
                        continue
                    conn.execute(
                        """INSERT OR IGNORE INTO compounds
                        (canonical_smiles, inchikey, inchikey_connectivity, inchi,
                         pubchem_cid, molecular_weight, logp, hbd, hba, tpsa,
                         rotatable_bonds, num_heavy_atoms, qed, pains_alert,
                         lipinski_violations)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            std["canonical_smiles"],
                            std["inchikey"],
                            std["inchikey_connectivity"],
                            std["inchi"],
                            cid,
                            std["molecular_weight"],
                            std["logp"],
                            std["hbd"],
                            std["hba"],
                            std["tpsa"],
                            std["rotatable_bonds"],
                            std["num_heavy_atoms"],
                            std["qed"],
                            std["pains_alert"],
                            std["lipinski_violations"],
                        ),
                    )
                    row = conn.execute(
                        "SELECT compound_id, pubchem_cid FROM compounds WHERE inchikey = ?",
                        (std["inchikey"],),
                    ).fetchone()
                    if row is None:
                        rows_skipped += 1
                        continue
                    compound_id = int(row[0])
                    if row[1] is None:
                        conn.execute(
                            "UPDATE compounds SET pubchem_cid = ? WHERE compound_id = ?",
                            (cid, compound_id),
                        )
                    compound_cache[cache_key] = compound_id

                uniprot = str(r.uniprot_accession).strip()
                target_id = target_cache.get(uniprot)
                if target_id is None:
                    conn.execute(
                        "INSERT OR IGNORE INTO targets (uniprot_accession) VALUES (?)",
                        (uniprot,),
                    )
                    row = conn.execute(
                        "SELECT target_id FROM targets WHERE uniprot_accession = ?",
                        (uniprot,),
                    ).fetchone()
                    if row is None:
                        rows_skipped += 1
                        continue
                    target_id = int(row[0])
                    target_cache[uniprot] = target_id

                aid = int(r.aid)
                assay_id = assay_cache.get(aid)
                if assay_id is None:
                    conn.execute(
                        """INSERT OR IGNORE INTO assays
                        (source_db, source_assay_id, assay_type, screen_type)
                        VALUES ('pubchem', ?, 'confirmatory', 'confirmatory_dose_response')""",
                        (str(aid),),
                    )
                    row = conn.execute(
                        "SELECT assay_id FROM assays WHERE source_db='pubchem' AND source_assay_id=?",
                        (str(aid),),
                    ).fetchone()
                    if row is None:
                        rows_skipped += 1
                        continue
                    assay_id = int(row[0])
                    assay_cache[aid] = assay_id

                activity_value = (
                    float(r.activity_value) if pd.notna(r.activity_value) else None
                )
                pchembl_value = None
                if (
                    activity_value is not None
                    and activity_value > 0
                    and _is_nm_unit(r.activity_unit)
                ):
                    pchembl_value = 9.0 - math.log10(activity_value)

                source_record_id = f"PUBCHEM:{aid}:{int(r.sid)}"
                species_tested = None
                if pd.notna(r.target_taxid):
                    try:
                        taxid = int(r.target_taxid)
                        if taxid == _HUMAN_TAXID:
                            species_tested = "Homo sapiens"
                        elif not human_only:
                            species_tested = f"taxid:{taxid}"
                    except (TypeError, ValueError):
                        pass
                elif human_only and aid in confirmatory_human_aids:
                    species_tested = "Homo sapiens"
                insert_params.append(
                    (
                        compound_id,
                        target_id,
                        assay_id,
                        str(r.activity_name) if pd.notna(r.activity_name) else "bioactivity",
                        activity_value,
                        str(r.activity_unit) if pd.notna(r.activity_unit) else None,
                        pchembl_value,
                        inactivity_threshold_nm,
                        source_record_id,
                        species_tested,
                    )
                )

                if len(insert_params) >= 10000:
                    conn.executemany(
                        """INSERT OR IGNORE INTO negative_results
                        (compound_id, target_id, assay_id,
                         result_type, confidence_tier,
                         activity_type, activity_value, activity_unit, activity_relation,
                         pchembl_value,
                         inactivity_threshold, inactivity_threshold_unit,
                         source_db, source_record_id, extraction_method,
                         curator_validated, species_tested)
                        VALUES (?, ?, ?,
                                'hard_negative', 'silver',
                                ?, ?, ?, '=',
                                ?,
                                ?, 'nM',
                                'pubchem', ?, 'database_direct',
                                0, ?)""",
                        insert_params,
                    )
                    rows_attempted_insert += len(insert_params)
                    insert_params = []
                    conn.commit()

        if insert_params:
            conn.executemany(
                """INSERT OR IGNORE INTO negative_results
                (compound_id, target_id, assay_id,
                 result_type, confidence_tier,
                 activity_type, activity_value, activity_unit, activity_relation,
                 pchembl_value,
                 inactivity_threshold, inactivity_threshold_unit,
                 source_db, source_record_id, extraction_method,
                 curator_validated, species_tested)
                VALUES (?, ?, ?,
                        'hard_negative', 'silver',
                        ?, ?, ?, '=',
                        ?,
                        ?, 'nM',
                        'pubchem', ?, 'database_direct',
                        0, ?)""",
                insert_params,
            )
            rows_attempted_insert += len(insert_params)

        n_pairs = refresh_all_pairs(conn)
        conn.commit()

        after_results = conn.execute(
            "SELECT COUNT(*) FROM negative_results WHERE source_db='pubchem'"
        ).fetchone()[0]

    inserted = int(after_results - before_results)
    return {
        "rows_read": int(rows_read),
        "rows_filtered_inactive_confirmatory": int(rows_filtered),
        "rows_mapped_ready": int(rows_mapped),
        "rows_skipped": int(rows_skipped),
        "rows_attempted_insert": int(rows_attempted_insert),
        "results_inserted": inserted,
        "pairs_total": int(n_pairs),
    }
