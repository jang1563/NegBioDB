"""ETL pipeline for loading BindingDB inactive DTI data."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from pathlib import Path

import pandas as pd

from negbiodb.db import connect, create_database, _PROJECT_ROOT
from negbiodb.download import load_config
from negbiodb.db import refresh_all_pairs
from negbiodb.standardize import standardize_smiles

logger = logging.getLogger(__name__)
_HUMAN_SPECIES = "Homo sapiens"


def _normalize_col_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _normalize_col_name(str(c)) for c in df.columns})


def _normalize_uniprot_accession(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    if "|" in text:
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if len(parts) >= 2 and parts[0].lower() in {"sp", "tr", "uniprotkb"}:
            text = parts[1]
        else:
            text = parts[0]
    text = re.split(r"[;,]", text, maxsplit=1)[0].strip()
    text = text.split()[0] if text else text
    return text or None


def _normalize_reactant_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        fval = float(text)
        if fval.is_integer():
            return str(int(fval))
    except (TypeError, ValueError):
        pass
    return text


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    colset = set(columns)
    for cand in candidates:
        if cand in colset:
            return cand
    return None


def _parse_relation_value(raw: object) -> tuple[str, float | None]:
    """Parse relation and numeric value from BindingDB fields."""
    if pd.isna(raw):
        return "=", None
    text = str(raw).strip()
    if text == "":
        return "=", None

    relation = "="
    if text.startswith(">="):
        relation = ">="
    elif text.startswith("<="):
        relation = "<="
    elif text.startswith(">"):
        relation = ">"
    elif text.startswith("<"):
        relation = "<"
    elif text.startswith("="):
        relation = "="

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not m:
        return relation, None
    return relation, float(m.group(0))


def _row_hash_source_id(row: dict) -> str:
    basis = "|".join(
        [
            row.get("smiles", ""),
            row.get("uniprot_accession", ""),
            row.get("species_tested", ""),
            row.get("activity_type", ""),
            row.get("activity_relation", ""),
            str(row.get("activity_value", "")),
            str(row.get("publication_year", "")),
        ]
    )
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
    return f"BDB:HASH:{digest}"


def _extract_inactive_rows_from_chunk(
    chunk: pd.DataFrame,
    inactivity_threshold_nm: float,
    human_only: bool = True,
) -> list[dict]:
    chunk = _normalize_columns(chunk)
    cols = list(chunk.columns)

    smiles_col = _find_col(cols, ["ligand_smiles", "smiles"])
    uniprot_col = _find_col(
        cols,
        [
            "uniprot_swissprot_primary_id_of_target_chain_1",
            "uniprot_swissprot_primary_id_of_target_chain",
            "uniprot_primary_id_of_target_chain",
            "uniprot_id",
            "target_uniprot",
            "uniprot",
        ],
    )
    organism_col = _find_col(
        cols,
        [
            "target_source_organism_according_to_curator_or_datasource",
            "target_source_organism",
            "target_organism",
            "organism",
        ],
    )
    reactant_col = _find_col(
        cols,
        [
            "bindingdb_reactant_set_id",
            "reactant_set_id",
            "reactant_setid",
            "bindingdb_ligand_id",
        ],
    )
    year_col = _find_col(cols, ["date_of_publication", "publication_year", "year"])

    metric_cols = [
        ("Ki", _find_col(cols, ["ki_nm", "ki_n_m", "ki"])),
        ("Kd", _find_col(cols, ["kd_nm", "kd_n_m", "kd"])),
        ("IC50", _find_col(cols, ["ic50_nm", "ic50_n_m", "ic50"])),
        ("EC50", _find_col(cols, ["ec50_nm", "ec50_n_m", "ec50"])),
    ]
    metric_cols = [(name, col) for name, col in metric_cols if col is not None]

    if smiles_col is None or uniprot_col is None or not metric_cols:
        return []
    if human_only and organism_col is None:
        logger.warning("BindingDB human_only=True but organism column not found; skipping chunk.")
        return []

    out: list[dict] = []
    for r in chunk.itertuples(index=False):
        row = r._asdict()
        smiles = row.get(smiles_col)
        uniprot = _normalize_uniprot_accession(row.get(uniprot_col))
        if pd.isna(smiles) or pd.isna(uniprot):
            continue
        smiles = str(smiles).strip()
        if smiles == "" or uniprot == "":
            continue

        if human_only:
            org = row.get(organism_col)
            if pd.isna(org) or str(org).strip() != _HUMAN_SPECIES:
                continue
        species_tested = None
        if organism_col is not None:
            org = row.get(organism_col)
            if not pd.isna(org):
                species_tested = str(org).strip() or None
        if species_tested is None and human_only:
            species_tested = _HUMAN_SPECIES

        chosen_type = None
        chosen_rel = "="
        chosen_val = None
        for metric_name, metric_col in metric_cols:
            rel, val = _parse_relation_value(row.get(metric_col))
            if val is None:
                continue
            is_inactive = False
            if rel in (">", ">=") and val >= inactivity_threshold_nm:
                is_inactive = True
            elif rel == "=" and val > inactivity_threshold_nm:
                is_inactive = True
            if not is_inactive:
                continue
            chosen_type = metric_name
            chosen_rel = rel
            chosen_val = val
            # One negative result per compound-target pair: break after first
            # qualifying metric to avoid duplicate rows.
            break

        if chosen_type is None or chosen_val is None:
            continue

        rec_id = None
        if reactant_col is not None:
            rv = _normalize_reactant_id(row.get(reactant_col))
            if rv is not None:
                rec_id = f"BDB:{rv}:{uniprot}:{chosen_type}"

        pub_year = None
        if year_col is not None:
            yv = row.get(year_col)
            if not pd.isna(yv):
                text = str(yv).strip()
                try:
                    pub_year = int(float(text))
                except (TypeError, ValueError):
                    # Try extracting 4-digit year from date string (e.g., "8/30/1996")
                    ym = re.search(r"\b((?:19|20)\d{2})\b", text)
                    if ym:
                        pub_year = int(ym.group(1))

        payload = {
            "smiles": smiles,
            "uniprot_accession": uniprot,
            "species_tested": species_tested,
            "activity_type": chosen_type,
            "activity_relation": chosen_rel,
            "activity_value": float(chosen_val),
            "publication_year": pub_year,
        }
        if rec_id is None:
            rec_id = _row_hash_source_id(payload)
        payload["source_record_id"] = rec_id
        out.append(payload)

    return out


def run_bindingdb_etl(
    db_path: Path,
    bindingdb_tsv_path: Path | None = None,
    chunksize: int | None = None,
) -> dict:
    cfg = load_config()
    bcfg = cfg["downloads"]["bindingdb"]
    etl_cfg = cfg.get("bindingdb_etl", {})

    if bindingdb_tsv_path is None:
        bindingdb_tsv_path = _PROJECT_ROOT / bcfg["dest_dir"] / "BindingDB_All.tsv"
    if chunksize is None:
        chunksize = int(etl_cfg.get("chunksize", 100000))

    inactivity_threshold_nm = float(etl_cfg.get("inactive_threshold_nm", cfg["inactivity_threshold_nm"]))
    human_only = bool(etl_cfg.get("human_only", True))

    create_database(db_path)

    rows_read = 0
    rows_filtered = 0
    rows_skipped = 0
    rows_attempted_insert = 0

    with connect(db_path) as conn:
        before_results = conn.execute(
            "SELECT COUNT(*) FROM negative_results WHERE source_db='bindingdb'"
        ).fetchone()[0]

        compound_cache: dict[str, int] = {}
        target_cache: dict[str, int] = {}
        insert_params: list[tuple] = []

        reader = pd.read_csv(bindingdb_tsv_path, sep="\t", chunksize=chunksize, low_memory=False)
        for chunk in reader:
            rows_read += len(chunk)
            rows = _extract_inactive_rows_from_chunk(
                chunk,
                inactivity_threshold_nm=inactivity_threshold_nm,
                human_only=human_only,
            )
            rows_filtered += len(rows)

            for row in rows:
                std = standardize_smiles(row["smiles"])
                if std is None:
                    rows_skipped += 1
                    continue

                inchikey = std["inchikey"]
                compound_id = compound_cache.get(inchikey)
                if compound_id is None:
                    conn.execute(
                        """INSERT OR IGNORE INTO compounds
                        (canonical_smiles, inchikey, inchikey_connectivity, inchi,
                         molecular_weight, logp, hbd, hba, tpsa,
                         rotatable_bonds, num_heavy_atoms, qed, pains_alert,
                         lipinski_violations)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            std["canonical_smiles"],
                            std["inchikey"],
                            std["inchikey_connectivity"],
                            std["inchi"],
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
                    c = conn.execute(
                        "SELECT compound_id FROM compounds WHERE inchikey = ?",
                        (inchikey,),
                    ).fetchone()
                    if c is None:
                        rows_skipped += 1
                        continue
                    compound_id = int(c[0])
                    compound_cache[inchikey] = compound_id

                uniprot = row["uniprot_accession"]
                target_id = target_cache.get(uniprot)
                if target_id is None:
                    conn.execute(
                        "INSERT OR IGNORE INTO targets (uniprot_accession) VALUES (?)",
                        (uniprot,),
                    )
                    t = conn.execute(
                        "SELECT target_id FROM targets WHERE uniprot_accession = ?",
                        (uniprot,),
                    ).fetchone()
                    if t is None:
                        rows_skipped += 1
                        continue
                    target_id = int(t[0])
                    target_cache[uniprot] = target_id

                pchembl_value = None
                if row["activity_value"] > 0:
                    pchembl_value = 9.0 - math.log10(float(row["activity_value"]))

                insert_params.append(
                    (
                        compound_id,
                        target_id,
                        row["activity_type"],
                        float(row["activity_value"]),
                        "nM",
                        row["activity_relation"],
                        pchembl_value,
                        inactivity_threshold_nm,
                        row["source_record_id"],
                        row["publication_year"],
                        row["species_tested"],
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
                         curator_validated, publication_year, species_tested)
                        VALUES (?, ?, NULL,
                                'hard_negative', 'silver',
                                ?, ?, ?, ?,
                                ?,
                                ?, 'nM',
                                'bindingdb', ?, 'database_direct',
                                0, ?, ?)""",
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
                 curator_validated, publication_year, species_tested)
                VALUES (?, ?, NULL,
                        'hard_negative', 'silver',
                        ?, ?, ?, ?,
                        ?,
                        ?, 'nM',
                        'bindingdb', ?, 'database_direct',
                        0, ?, ?)""",
                insert_params,
            )
            rows_attempted_insert += len(insert_params)

        n_pairs = refresh_all_pairs(conn)
        conn.commit()

        after_results = conn.execute(
            "SELECT COUNT(*) FROM negative_results WHERE source_db='bindingdb'"
        ).fetchone()[0]

    inserted = int(after_results - before_results)
    return {
        "rows_read": int(rows_read),
        "rows_filtered_inactive": int(rows_filtered),
        "rows_skipped": int(rows_skipped),
        "rows_attempted_insert": int(rows_attempted_insert),
        "results_inserted": inserted,
        "pairs_total": int(n_pairs),
    }
