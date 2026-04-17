"""Dataset helpers for the Cell Painting LLM benchmark."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CP_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "negbiodb_cp.db"
FEWSHOT_SEEDS = [42, 43, 44]
MAX_PER_COMPOUND = 10

OUTCOME_TO_L1_LETTER = {
    "inactive": "A",
    "weak_phenotype": "B",
    "strong_phenotype": "C",
    "toxic_or_artifact": "D",
}
L1_LETTER_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_L1_LETTER.items()}

JSONL_SCHEMA_FIELDS = [
    "question_id",
    "task",
    "split",
    "difficulty",
    "context_text",
    "gold_answer",
    "gold_category",
    "metadata",
]


def _format_float(value, fmt: str = ".3f", default: str = "NA") -> str:
    if value is None or pd.isna(value):
        return default
    return format(float(value), fmt)


def _format_dose(value, dose_unit: str) -> str:
    if value is None or pd.isna(value):
        return f"NA {dose_unit}"
    return f"{format(float(value), 'g')} {dose_unit}"


def load_cp_candidate_pool(
    db_path: Path,
    min_confidence: str | None = None,
    outcome_label: str | None = None,
    extra_where: str = "",
    allow_proxy_smoke: bool = False,
) -> pd.DataFrame:
    """Load CP benchmark candidates from the consensus result table."""
    from negbiodb_cp.cp_db import ensure_cp_production_ready, get_connection

    where_parts = ["1=1"]
    if min_confidence:
        rank = {"gold": 1, "silver": 2, "bronze": 3, "copper": 4}
        allowed = [k for k, v in rank.items() if v <= rank[min_confidence]]
        vals = ", ".join(f"'{v}'" for v in allowed)
        where_parts.append(f"r.confidence_tier IN ({vals})")
    if outcome_label:
        where_parts.append(f"r.outcome_label = '{outcome_label}'")
    if extra_where:
        where_parts.append(extra_where)
    where_clause = " AND ".join(where_parts)

    conn = get_connection(db_path)
    try:
        ensure_cp_production_ready(conn, allow_proxy_smoke=allow_proxy_smoke)
        df = pd.read_sql_query(
            f"""
            SELECT
                r.cp_result_id,
                r.compound_id,
                r.cell_line_id,
                r.batch_id,
                r.dose,
                r.dose_unit,
                r.timepoint_h,
                r.num_valid_observations,
                r.dmso_distance_mean,
                r.replicate_reproducibility,
                r.viability_ratio,
                r.outcome_label,
                r.confidence_tier,
                r.has_orthogonal_evidence,
                c.compound_name,
                c.canonical_smiles,
                c.inchikey,
                cl.cell_line_name,
                cl.tissue,
                cl.disease,
                b.batch_name
            FROM cp_perturbation_results r
            JOIN compounds c ON r.compound_id = c.compound_id
            JOIN cp_cell_lines cl ON r.cell_line_id = cl.cell_line_id
            JOIN cp_batches b ON r.batch_id = b.batch_id
            WHERE {where_clause}
            ORDER BY r.cp_result_id
            """,
            conn,
        )
    finally:
        conn.close()
    return df


def load_cp_annotation_summary(db_path: Path) -> dict:
    """Load CP dataset annotation summary from a DB path."""
    from negbiodb_cp.cp_db import get_connection, get_cp_annotation_summary

    conn = get_connection(db_path)
    try:
        return get_cp_annotation_summary(conn)
    finally:
        conn.close()


def apply_max_per_compound(
    df: pd.DataFrame,
    max_per_compound: int = MAX_PER_COMPOUND,
    rng: np.random.RandomState | None = None,
) -> pd.DataFrame:
    """Cap rows per compound to avoid over-representation."""
    if rng is None:
        rng = np.random.RandomState(42)
    kept = []
    for _, group in df.groupby("compound_id"):
        if len(group) <= max_per_compound:
            kept.append(group)
        else:
            idx = rng.choice(len(group), size=max_per_compound, replace=False)
            kept.append(group.iloc[idx])
    if not kept:
        return df.iloc[0:0].copy()
    return pd.concat(kept, ignore_index=True)


def difficulty_from_tier(tier: str) -> str:
    return {
        "gold": "easy",
        "silver": "easy",
        "bronze": "medium",
        "copper": "hard",
    }.get(str(tier), "medium")


def assign_splits(
    df: pd.DataFrame,
    fewshot_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> pd.DataFrame:
    """Assign fewshot/val/test splits after shuffling."""
    rng = np.random.RandomState(seed)
    total = fewshot_size + val_size + test_size
    if len(df) < total:
        test_size = max(0, len(df) - fewshot_size - val_size)
    idx = rng.permutation(len(df))
    df = df.iloc[idx].reset_index(drop=True).copy()
    splits = ["test"] * len(df)
    for i in range(min(fewshot_size, len(df))):
        splits[i] = "fewshot"
    for i in range(fewshot_size, min(fewshot_size + val_size, len(df))):
        splits[i] = "val"
    df["split"] = splits
    return df


def construct_evidence_description(record: dict | pd.Series) -> str:
    """Build evidence text for a CP consensus perturbation result."""
    compound = record.get("compound_name") or record.get("inchikey") or "Unknown compound"
    cell_line = record.get("cell_line_name", "Unknown cell line")
    dose = record.get("dose")
    dose_unit = record.get("dose_unit", "uM")
    batch = record.get("batch_name", "unknown batch")
    dmso_distance = record.get("dmso_distance_mean")
    reproducibility = record.get("replicate_reproducibility")
    viability = record.get("viability_ratio")
    observations = record.get("num_valid_observations")

    return (
        f"Cell Painting perturbation summary for {compound} in {cell_line}.\n"
        f"Batch: {batch}. Dose: {_format_dose(dose, dose_unit)}. "
        f"Timepoint: {_format_float(record.get('timepoint_h', 48.0), 'g', '48')} h.\n"
        f"Assay-valid observations: {int(observations) if pd.notna(observations) else 0}.\n"
        f"Distance to matched DMSO centroid: {_format_float(dmso_distance)}.\n"
        f"Replicate reproducibility: {_format_float(reproducibility)}.\n"
        f"Viability/count proxy ratio vs control: {_format_float(viability)}."
    )


def construct_l1_context(record: dict | pd.Series) -> str:
    compound = record.get("compound_name") or record.get("inchikey") or "Unknown compound"
    cell_line = record.get("cell_line_name", "Unknown cell line")
    dose = record.get("dose")
    dose_unit = record.get("dose_unit", "uM")
    batch = record.get("batch_name", "unknown batch")
    observations = record.get("num_valid_observations")
    tier = record.get("confidence_tier", "unknown")
    return (
        f"Cell Painting perturbation summary for {compound} in {cell_line}.\n"
        f"Batch: {batch}. Dose: {_format_dose(dose, dose_unit)}. "
        f"Timepoint: {_format_float(record.get('timepoint_h', 48.0), 'g', '48')} h.\n"
        f"Assay-valid observations: {int(observations) if pd.notna(observations) else 0}. "
        f"Confidence tier: {tier}.\n\n"
        "Question: Which outcome class best matches this perturbation?\n"
        "A) inactive\n"
        "B) weak_phenotype\n"
        "C) strong_phenotype\n"
        "D) toxic_or_artifact"
    )


def construct_l2_context(record: dict | pd.Series) -> str:
    return (
        "Cell Painting Structured Report\n"
        f"Compound: {record.get('compound_name') or record.get('inchikey')}\n"
        f"Cell line: {record.get('cell_line_name')}\n"
        f"Batch: {record.get('batch_name')}\n"
        f"Dose: {_format_dose(record.get('dose'), record.get('dose_unit') or 'uM')}\n"
        f"Timepoint_h: {_format_float(record.get('timepoint_h', 48.0), 'g', '48')}\n"
        f"DMSO distance: {_format_float(record.get('dmso_distance_mean'))}\n"
        f"Replicate reproducibility: {_format_float(record.get('replicate_reproducibility'))}\n"
        f"QC viability ratio: {_format_float(record.get('viability_ratio'))}"
    )


def construct_l3_context(record: dict | pd.Series) -> str:
    orth = "yes" if int(record.get("has_orthogonal_evidence", 0)) else "no"
    return (
        f"{construct_evidence_description(record)}\n"
        f"Orthogonal evidence recorded: {orth}.\n"
        "Based on the assay metrics above, classify the perturbation outcome and explain your reasoning. "
        "Do not invent a mechanism that is not supported by the assay summary."
    )


def construct_l4_context(record: dict | pd.Series) -> str:
    compound = record.get("compound_name") or record.get("inchikey") or "Unknown compound"
    return (
        f"Compound: {compound}\n"
        f"Cell line: {record.get('cell_line_name')}\n"
        f"Dose: {_format_dose(record.get('dose'), record.get('dose_unit') or 'uM')}\n"
        f"Timepoint: {_format_float(record.get('timepoint_h', 48.0), 'g', '48')} h"
    )


def write_jsonl(records: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def write_dataset_metadata(output_dir: Path, task: str, stats: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / f"{task.replace('-', '_')}_metadata.json"
    payload = {
        "task": task,
        "domain": "cp",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **stats,
    }
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return meta_path
