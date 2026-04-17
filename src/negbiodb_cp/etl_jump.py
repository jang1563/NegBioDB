"""JUMP-first ETL helpers for the NegBioDB Cell Painting domain."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OBS_ALIAS_MAP = {
    "cell_line": "cell_line_name",
    "batch": "batch_name",
    "plate": "plate_name",
    "well": "well_id",
    "site": "site_id",
    "replicate": "replicate_id",
    "smiles": "canonical_smiles",
}

REQUIRED_OBSERVATION_COLUMNS = {
    "batch_name",
    "plate_name",
    "cell_line_name",
    "well_id",
    "dose",
    "timepoint_h",
    "dmso_distance",
    "replicate_reproducibility",
    "viability_ratio",
}

DEFAULT_TIMEPOINT_H = 48.0
DEFAULT_DOSE_UNIT = "uM"
DEFAULT_DMSO_DISTANCE_THRESHOLD = 0.10
STRONG_DISTANCE_MULTIPLIER = 2.0
TOXIC_VIABILITY_THRESHOLD = 0.50
SILVER_REPRO_THRESHOLD = 0.60


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV/TSV/JSONL/Parquet table."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix in {".tsv", ".txt"} else ","
        return pd.read_csv(path, sep=sep)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table format: {path}")


def normalize_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize observation-level input columns and defaults."""
    renamed = df.rename(columns={k: v for k, v in OBS_ALIAS_MAP.items() if k in df.columns}).copy()
    missing = REQUIRED_OBSERVATION_COLUMNS - set(renamed.columns)
    if missing:
        raise ValueError(f"Missing required observation columns: {sorted(missing)}")

    if "compound_name" not in renamed.columns:
        renamed["compound_name"] = None
    if "canonical_smiles" not in renamed.columns:
        renamed["canonical_smiles"] = None
    if "inchikey" not in renamed.columns:
        renamed["inchikey"] = None
    if "pubchem_cid" not in renamed.columns:
        renamed["pubchem_cid"] = None
    if "chembl_id" not in renamed.columns:
        renamed["chembl_id"] = None
    if "site_id" not in renamed.columns:
        renamed["site_id"] = None
    if "replicate_id" not in renamed.columns:
        renamed["replicate_id"] = None
    if "control_type" not in renamed.columns:
        renamed["control_type"] = "perturbation"
    if "qc_pass" not in renamed.columns:
        renamed["qc_pass"] = 1
    if "dose_unit" not in renamed.columns:
        renamed["dose_unit"] = DEFAULT_DOSE_UNIT
    if "source_record_id" not in renamed.columns:
        renamed["source_record_id"] = None
    if "image_uri" not in renamed.columns:
        renamed["image_uri"] = None
    if "assay_name" not in renamed.columns:
        renamed["assay_name"] = "Cell Painting"
    if "cell_painting_version" not in renamed.columns:
        renamed["cell_painting_version"] = "v3"
    if "tissue" not in renamed.columns:
        renamed["tissue"] = None
    if "disease" not in renamed.columns:
        renamed["disease"] = None
    if "stain_channels" not in renamed.columns:
        renamed["stain_channels"] = "DNA,ER,RNA,AGP,Mito"

    renamed["dose_unit"] = renamed["dose_unit"].fillna(DEFAULT_DOSE_UNIT).astype(str)
    renamed["timepoint_h"] = renamed["timepoint_h"].fillna(DEFAULT_TIMEPOINT_H).astype(float)
    renamed["qc_pass"] = renamed["qc_pass"].fillna(0).astype(int)
    renamed["control_type"] = renamed["control_type"].fillna("perturbation").astype(str)
    return renamed


def _ensure_dataset_version(
    conn,
    name: str,
    version: str,
    annotation_mode: str = "annotated",
    source_url: str | None = None,
    row_count: int | None = None,
) -> int:
    row = conn.execute(
        "SELECT dataset_id FROM dataset_versions WHERE name = ? AND version = ?",
        (name, version),
    ).fetchone()
    if row is not None:
        dataset_id = int(row[0])
        conn.execute(
            """
            UPDATE dataset_versions
            SET annotation_mode = ?,
                source_url = COALESCE(?, source_url),
                download_date = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                row_count = COALESCE(?, row_count)
            WHERE dataset_id = ?
            """,
            (annotation_mode, source_url, row_count, dataset_id),
        )
        return dataset_id

    conn.execute(
        """INSERT INTO dataset_versions
        (name, version, annotation_mode, source_url, download_date, row_count)
        VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), ?)""",
        (name, version, annotation_mode, source_url, row_count),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _resolve_or_insert_compound(conn, row: pd.Series) -> int:
    compound_name = row.get("compound_name")
    smiles = row.get("canonical_smiles")
    inchikey = row.get("inchikey")
    chembl_id = row.get("chembl_id")
    pubchem_cid = row.get("pubchem_cid")

    normalized = None
    if isinstance(smiles, str) and smiles.strip():
        try:
            from negbiodb.standardize import standardize_smiles
        except ModuleNotFoundError:
            standardize_smiles = None
        if standardize_smiles is not None:
            normalized = standardize_smiles(smiles)
            if normalized:
                smiles = normalized["canonical_smiles"]
                inchikey = normalized["inchikey"]

    if isinstance(inchikey, str) and inchikey.strip():
        found = conn.execute(
            "SELECT compound_id FROM compounds WHERE inchikey = ?",
            (inchikey,),
        ).fetchone()
        if found is not None:
            return int(found[0])

    if compound_name:
        found = conn.execute(
            "SELECT compound_id FROM compounds WHERE compound_name = ?",
            (compound_name,),
        ).fetchone()
        if found is not None:
            return int(found[0])

    inchikey_connectivity = inchikey[:14] if isinstance(inchikey, str) and len(inchikey) >= 14 else None
    inchi = normalized.get("inchi") if normalized else None
    molecular_weight = normalized.get("molecular_weight") if normalized else None
    logp = normalized.get("logp") if normalized else None
    hbd = normalized.get("hbd") if normalized else None
    hba = normalized.get("hba") if normalized else None
    tpsa = normalized.get("tpsa") if normalized else None
    rotatable_bonds = normalized.get("rotatable_bonds") if normalized else None
    num_heavy_atoms = normalized.get("num_heavy_atoms") if normalized else None
    qed = normalized.get("qed") if normalized else None
    pains_alert = normalized.get("pains_alert") if normalized else 0
    lipinski_violations = normalized.get("lipinski_violations") if normalized else 0

    conn.execute(
        """INSERT INTO compounds
        (compound_name, canonical_smiles, inchikey, inchikey_connectivity, inchi,
         pubchem_cid, chembl_id, molecular_weight, logp, hbd, hba, tpsa,
         rotatable_bonds, num_heavy_atoms, qed, pains_alert, lipinski_violations)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            compound_name,
            smiles,
            inchikey,
            inchikey_connectivity,
            inchi,
            pubchem_cid,
            chembl_id,
            molecular_weight,
            logp,
            hbd,
            hba,
            tpsa,
            rotatable_bonds,
            num_heavy_atoms,
            qed,
            pains_alert,
            lipinski_violations,
        ),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _ensure_cell_line(conn, row: pd.Series) -> int:
    row_out = conn.execute(
        "SELECT cell_line_id FROM cp_cell_lines WHERE cell_line_name = ?",
        (row["cell_line_name"],),
    ).fetchone()
    if row_out is not None:
        return int(row_out[0])

    conn.execute(
        """INSERT INTO cp_cell_lines
        (cell_line_name, tissue, disease, assay_protocol_version)
        VALUES (?, ?, ?, ?)""",
        (
            row["cell_line_name"],
            row.get("tissue"),
            row.get("disease"),
            f"Cell Painting {row.get('cell_painting_version', 'v3')}",
        ),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _ensure_assay_context(conn, row: pd.Series, cell_line_id: int) -> int:
    existing = conn.execute(
        """SELECT assay_context_id FROM cp_assay_contexts
        WHERE cell_line_id = ? AND assay_name = ? AND cell_painting_version = ?
          AND timepoint_h = ?""",
        (
            cell_line_id,
            row.get("assay_name", "Cell Painting"),
            row.get("cell_painting_version", "v3"),
            float(row.get("timepoint_h", DEFAULT_TIMEPOINT_H)),
        ),
    ).fetchone()
    if existing is not None:
        return int(existing[0])

    conn.execute(
        """INSERT INTO cp_assay_contexts
        (cell_line_id, assay_name, cell_painting_version, timepoint_h, stain_channels)
        VALUES (?, ?, ?, ?, ?)""",
        (
            cell_line_id,
            row.get("assay_name", "Cell Painting"),
            row.get("cell_painting_version", "v3"),
            float(row.get("timepoint_h", DEFAULT_TIMEPOINT_H)),
            row.get("stain_channels"),
        ),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _ensure_batch(conn, dataset_id: int, batch_name: str) -> int:
    existing = conn.execute(
        "SELECT batch_id FROM cp_batches WHERE batch_name = ?",
        (batch_name,),
    ).fetchone()
    if existing is not None:
        return int(existing[0])

    conn.execute(
        "INSERT INTO cp_batches (dataset_id, batch_name, source_name) VALUES (?, ?, ?)",
        (dataset_id, batch_name, "cpg0016-jump"),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _ensure_plate(conn, batch_id: int, plate_name: str) -> int:
    existing = conn.execute(
        "SELECT plate_id FROM cp_plates WHERE batch_id = ? AND plate_name = ?",
        (batch_id, plate_name),
    ).fetchone()
    if existing is not None:
        return int(existing[0])

    conn.execute(
        "INSERT INTO cp_plates (batch_id, plate_name) VALUES (?, ?)",
        (batch_id, plate_name),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def assign_outcome_label(
    dmso_distance: float | None,
    replicate_reproducibility: float | None,
    viability_ratio: float | None,
    inactive_threshold: float,
) -> str:
    """Assign an assay-context outcome label from aggregated signals."""
    if viability_ratio is not None and np.isfinite(viability_ratio):
        if viability_ratio < TOXIC_VIABILITY_THRESHOLD:
            return "toxic_or_artifact"

    if dmso_distance is None or not np.isfinite(dmso_distance):
        return "weak_phenotype"

    strong_threshold = inactive_threshold * STRONG_DISTANCE_MULTIPLIER
    if dmso_distance <= inactive_threshold:
        return "inactive"
    if dmso_distance >= strong_threshold and (
        replicate_reproducibility is None
        or not np.isfinite(replicate_reproducibility)
        or replicate_reproducibility >= SILVER_REPRO_THRESHOLD
    ):
        return "strong_phenotype"
    return "weak_phenotype"


def assign_confidence_tier(
    num_valid_observations: int,
    dmso_distance: float | None,
    replicate_reproducibility: float | None,
    viability_ratio: float | None,
) -> str:
    """Assign a within-batch CP confidence tier."""
    if any(x is None or not np.isfinite(x) for x in (dmso_distance, replicate_reproducibility, viability_ratio)):
        return "copper"
    if num_valid_observations >= 2 and replicate_reproducibility >= SILVER_REPRO_THRESHOLD:
        return "silver"
    if num_valid_observations >= 1:
        return "bronze"
    return "copper"


def refresh_cp_perturbation_results(conn) -> int:
    """Rebuild batch-consensus perturbation rows from observation-level inputs."""
    conn.execute("DELETE FROM cp_perturbation_results")

    obs = pd.read_sql_query(
        """
        SELECT
            o.observation_id,
            o.compound_id,
            p.batch_id,
            o.plate_id,
            o.assay_context_id,
            a.cell_line_id,
            o.dose,
            o.dose_unit,
            o.timepoint_h,
            o.control_type,
            o.dmso_distance,
            o.replicate_reproducibility,
            o.viability_ratio,
            o.qc_pass
        FROM cp_observations o
        JOIN cp_plates p ON o.plate_id = p.plate_id
        JOIN cp_assay_contexts a ON o.assay_context_id = a.assay_context_id
        """,
        conn,
    )
    if obs.empty:
        conn.commit()
        return 0

    valid = obs[obs["qc_pass"] == 1].copy()
    if valid.empty:
        conn.commit()
        return 0

    dmso = valid[valid["control_type"] == "dmso"].copy()
    batch_cutoffs = (
        dmso.groupby("batch_id")["dmso_distance"].quantile(0.95).to_dict()
        if not dmso.empty else {}
    )

    perturb = valid[valid["control_type"] != "dmso"].copy()
    if perturb.empty:
        conn.commit()
        return 0

    group_cols = [
        "compound_id", "cell_line_id", "assay_context_id",
        "batch_id", "dose", "dose_unit", "timepoint_h",
    ]
    agg = (
        perturb.groupby(group_cols, dropna=False)
        .agg(
            num_observations=("observation_id", "size"),
            num_valid_observations=("observation_id", "size"),
            dmso_distance_mean=("dmso_distance", "mean"),
            replicate_reproducibility=("replicate_reproducibility", "mean"),
            viability_ratio=("viability_ratio", "mean"),
        )
        .reset_index()
    )

    rows = []
    for rec in agg.to_dict(orient="records"):
        inactive_threshold = batch_cutoffs.get(rec["batch_id"], DEFAULT_DMSO_DISTANCE_THRESHOLD)
        label = assign_outcome_label(
            rec["dmso_distance_mean"],
            rec["replicate_reproducibility"],
            rec["viability_ratio"],
            inactive_threshold,
        )
        tier = assign_confidence_tier(
            int(rec["num_valid_observations"]),
            rec["dmso_distance_mean"],
            rec["replicate_reproducibility"],
            rec["viability_ratio"],
        )
        rows.append((
            rec["compound_id"],
            rec["cell_line_id"],
            rec["assay_context_id"],
            rec["batch_id"],
            rec["dose"],
            rec["dose_unit"] or DEFAULT_DOSE_UNIT,
            rec["timepoint_h"],
            int(rec["num_observations"]),
            int(rec["num_valid_observations"]),
            float(rec["dmso_distance_mean"]) if pd.notna(rec["dmso_distance_mean"]) else None,
            float(rec["replicate_reproducibility"]) if pd.notna(rec["replicate_reproducibility"]) else None,
            float(rec["viability_ratio"]) if pd.notna(rec["viability_ratio"]) else None,
            label,
            tier,
        ))

    conn.executemany(
        """INSERT INTO cp_perturbation_results
        (compound_id, cell_line_id, assay_context_id, batch_id, dose, dose_unit, timepoint_h,
         num_observations, num_valid_observations, dmso_distance_mean,
         replicate_reproducibility, viability_ratio, outcome_label, confidence_tier)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )

    agreement = conn.execute(
        """
        SELECT compound_id, cell_line_id, dose, dose_unit, timepoint_h, outcome_label
        FROM cp_perturbation_results
        GROUP BY compound_id, cell_line_id, dose, dose_unit, timepoint_h, outcome_label
        HAVING COUNT(DISTINCT batch_id) >= 2
        """
    ).fetchall()
    for compound_id, cell_line_id, dose, dose_unit, timepoint_h, outcome_label in agreement:
        conn.execute(
            """UPDATE cp_perturbation_results
            SET confidence_tier = 'gold'
            WHERE compound_id = ? AND cell_line_id = ?
              AND (dose = ? OR (dose IS NULL AND ? IS NULL))
              AND dose_unit = ? AND timepoint_h = ? AND outcome_label = ?""",
            (compound_id, cell_line_id, dose, dose,
             dose_unit, timepoint_h, outcome_label),
        )

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM cp_perturbation_results").fetchone()[0]
    return int(count)


def _load_result_key_map(conn) -> tuple[dict[tuple, int], dict[tuple, int]]:
    df = pd.read_sql_query(
        """
        SELECT
            r.cp_result_id,
            r.dose,
            r.dose_unit,
            r.timepoint_h,
            b.batch_name,
            c.compound_name,
            c.inchikey,
            c.canonical_smiles,
            cl.cell_line_name
        FROM cp_perturbation_results r
        JOIN cp_batches b ON r.batch_id = b.batch_id
        JOIN compounds c ON r.compound_id = c.compound_id
        JOIN cp_cell_lines cl ON r.cell_line_id = cl.cell_line_id
        """
        ,
        conn,
    )
    by_inchikey = {}
    by_name = {}
    for row in df.to_dict(orient="records"):
        base = (
            row["cell_line_name"],
            float(row["dose"]) if pd.notna(row["dose"]) else None,
            row["dose_unit"],
            float(row["timepoint_h"]),
            row["batch_name"],
        )
        if row.get("inchikey"):
            by_inchikey[(row["inchikey"],) + base] = int(row["cp_result_id"])
        if row.get("compound_name"):
            by_name[(row["compound_name"],) + base] = int(row["cp_result_id"])
    return by_inchikey, by_name


def _normalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={k: v for k, v in OBS_ALIAS_MAP.items() if k in df.columns}).copy()
    if "dose_unit" not in renamed.columns:
        renamed["dose_unit"] = DEFAULT_DOSE_UNIT
    if "timepoint_h" not in renamed.columns:
        renamed["timepoint_h"] = DEFAULT_TIMEPOINT_H
    return renamed


def attach_feature_frame(
    conn,
    table_name: str,
    features_df: pd.DataFrame,
    feature_source: str,
) -> int:
    """Attach profile/image features to consensus CP results."""
    if features_df is None or features_df.empty:
        return 0

    df = _normalize_feature_frame(features_df)
    feature_cols = [
        c for c in df.columns
        if c not in {
            "cp_result_id", "compound_name", "canonical_smiles", "inchikey",
            "batch_name", "cell_line_name", "dose", "dose_unit", "timepoint_h",
            "feature_source", "storage_uri",
        }
    ]

    by_inchikey, by_name = _load_result_key_map(conn)
    inserted = 0

    for row in df.to_dict(orient="records"):
        cp_result_id = row.get("cp_result_id")
        if cp_result_id is None:
            base = (
                row.get("cell_line_name"),
                float(row["dose"]) if row.get("dose") is not None and pd.notna(row.get("dose")) else None,
                row.get("dose_unit", DEFAULT_DOSE_UNIT),
                float(row.get("timepoint_h", DEFAULT_TIMEPOINT_H)),
                row.get("batch_name"),
            )
            if row.get("inchikey"):
                cp_result_id = by_inchikey.get((row["inchikey"],) + base)
            if cp_result_id is None and row.get("compound_name"):
                cp_result_id = by_name.get((row["compound_name"],) + base)
        if cp_result_id is None:
            continue

        feature_json = {
            key: float(row[key])
            for key in feature_cols
            if pd.notna(row.get(key))
        }
        conn.execute(
            f"""INSERT OR REPLACE INTO {table_name}
            (cp_result_id, feature_source, storage_uri, feature_json, n_features)
            VALUES (?, ?, ?, ?, ?)""",
            (
                int(cp_result_id),
                row.get("feature_source", feature_source),
                row.get("storage_uri"),
                json.dumps(feature_json, sort_keys=True),
                len(feature_json),
            ),
        )
        inserted += 1

    conn.commit()
    return inserted


def attach_orthogonal_evidence(conn, evidence_df: pd.DataFrame | None) -> int:
    """Attach orthogonal evidence rows to CP perturbation results."""
    if evidence_df is None or evidence_df.empty:
        return 0

    df = _normalize_feature_frame(evidence_df)
    by_inchikey, by_name = _load_result_key_map(conn)
    inserted = 0
    for row in df.to_dict(orient="records"):
        cp_result_id = row.get("cp_result_id")
        if cp_result_id is None:
            base = (
                row.get("cell_line_name"),
                float(row["dose"]) if row.get("dose") is not None and pd.notna(row.get("dose")) else None,
                row.get("dose_unit", DEFAULT_DOSE_UNIT),
                float(row.get("timepoint_h", DEFAULT_TIMEPOINT_H)),
                row.get("batch_name"),
            )
            if row.get("inchikey"):
                cp_result_id = by_inchikey.get((row["inchikey"],) + base)
            if cp_result_id is None and row.get("compound_name"):
                cp_result_id = by_name.get((row["compound_name"],) + base)
        if cp_result_id is None:
            continue
        conn.execute(
            """INSERT INTO cp_orthogonal_evidence
            (cp_result_id, evidence_domain, evidence_label, source_name,
             source_record_id, match_key, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                int(cp_result_id),
                row.get("evidence_domain", "unknown"),
                row.get("evidence_label", "unknown"),
                row.get("source_name"),
                row.get("source_record_id"),
                row.get("match_key"),
                row.get("notes"),
            ),
        )
        inserted += 1

    conn.execute(
        """UPDATE cp_perturbation_results
        SET has_orthogonal_evidence = 1
        WHERE cp_result_id IN (SELECT DISTINCT cp_result_id FROM cp_orthogonal_evidence)"""
    )
    conn.commit()
    return inserted


def ingest_jump_tables(
    conn,
    observations: pd.DataFrame,
    profile_features: pd.DataFrame | None = None,
    image_features: pd.DataFrame | None = None,
    orthogonal_evidence: pd.DataFrame | None = None,
    dataset_name: str = "cpg0016-jump",
    dataset_version: str = "1.0",
    annotation_mode: str = "annotated",
    source_url: str | None = None,
) -> dict[str, int]:
    """Ingest JUMP assembled observations plus optional feature/evidence tables."""
    obs = normalize_observations(observations)
    dataset_id = _ensure_dataset_version(
        conn,
        dataset_name,
        dataset_version,
        annotation_mode=annotation_mode,
        source_url=source_url,
        row_count=len(obs),
    )

    for row in obs.itertuples(index=False):
        s = pd.Series(row._asdict())
        compound_id = _resolve_or_insert_compound(conn, s)
        cell_line_id = _ensure_cell_line(conn, s)
        assay_context_id = _ensure_assay_context(conn, s, cell_line_id)
        batch_id = _ensure_batch(conn, dataset_id, s["batch_name"])
        plate_id = _ensure_plate(conn, batch_id, s["plate_name"])

        conn.execute(
            """INSERT OR REPLACE INTO cp_observations
            (compound_id, plate_id, assay_context_id, well_id, site_id, replicate_id,
             dose, dose_unit, timepoint_h, control_type, dmso_distance,
             replicate_reproducibility, viability_ratio, qc_pass, image_uri,
             source_record_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                compound_id,
                plate_id,
                assay_context_id,
                s["well_id"],
                s.get("site_id"),
                s.get("replicate_id"),
                s.get("dose"),
                s.get("dose_unit", DEFAULT_DOSE_UNIT),
                float(s.get("timepoint_h", DEFAULT_TIMEPOINT_H)),
                s.get("control_type", "perturbation"),
                s.get("dmso_distance"),
                s.get("replicate_reproducibility"),
                s.get("viability_ratio"),
                int(s.get("qc_pass", 1)),
                s.get("image_uri"),
                s.get("source_record_id"),
            ),
        )
    conn.commit()

    n_results = refresh_cp_perturbation_results(conn)
    n_profile = attach_feature_frame(conn, "cp_profile_features", profile_features, "jump_profile")
    n_image = attach_feature_frame(conn, "cp_image_features", image_features, "jump_deep_feature")
    n_orth = attach_orthogonal_evidence(conn, orthogonal_evidence)
    return {
        "n_observations": int(len(obs)),
        "n_results": int(n_results),
        "n_profile_features": int(n_profile),
        "n_image_features": int(n_image),
        "n_orthogonal_evidence": int(n_orth),
    }
