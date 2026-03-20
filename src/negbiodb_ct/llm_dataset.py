"""Dataset builder utilities for CT LLM benchmark (CT-L1 through CT-L4).

Shared constants, SQL helpers, sampling, splitting, and I/O functions
used by all four build_ct_l{1..4}_dataset.py scripts.

Mirrors the role that DTI scripts handled inline, but centralised per spec §11.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

MAX_PER_DRUG = 10  # Prevent single drug dominating any class

# Few-shot set seeds (3 independent sets for variance) — re-exported from prompts
FEWSHOT_SEEDS = [42, 43, 44]

# Base intervention filter applied in all SQL queries
BASE_INTERVENTION_FILTER = """
    i.intervention_type IN ('drug', 'biologic', 'combination')
    AND LOWER(i.intervention_name) NOT LIKE '%placebo%'
"""

# JSONL record fields (spec §8).  gold_answer contents per task:
#   CT-L1: "A"|"B"|"C"|"D"|"E"  (MCQ letter)
#   CT-L2: failure_category string, e.g. "efficacy"  (Phase 1 gold)
#   CT-L3: failure_category string (judge uses context, not gold_answer)
#   CT-L4: "tested"|"untested"
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

# ---------------------------------------------------------------------------
# Therapeutic area inference  (conditions.therapeutic_area is 0% populated)
# ---------------------------------------------------------------------------

THERAPEUTIC_AREA_KEYWORDS: dict[str, list[str]] = {
    "oncology": [
        "cancer", "tumor", "tumour", "carcinoma", "lymphoma", "leukemia",
        "melanoma", "sarcoma", "myeloma", "glioma", "neoplasm",
    ],
    "cardiology": [
        "heart", "cardiac", "coronary", "atrial", "hypertension",
        "myocardial", "arrhythmia", "angina",
    ],
    "neurology": [
        "alzheimer", "parkinson", "epilepsy", "seizure", "multiple sclerosis",
        "neuropathy", "stroke", "migraine", "dementia",
    ],
    "psychiatry": [
        "depression", "anxiety", "schizophrenia", "bipolar", "ptsd",
    ],
    "infectious": [
        "hiv", "hepatitis", "tuberculosis", "malaria", "covid", "influenza",
    ],
    "metabolic": [
        "diabetes", "obesity", "metabolic syndrome", "hyperlipidemia",
    ],
    "respiratory": [
        "asthma", "copd", "pulmonary", "lung fibrosis",
    ],
    "autoimmune": [
        "rheumatoid", "lupus", "crohn", "colitis", "psoriasis",
    ],
}


def infer_therapeutic_area(condition_name: str) -> str:
    """Keyword heuristic: match condition_name against THERAPEUTIC_AREA_KEYWORDS.

    Returns first match or 'other'.  Used for L3 diversity and L3/L4 context.
    """
    if not condition_name:
        return "other"
    lower = condition_name.lower()
    for area, keywords in THERAPEUTIC_AREA_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return area
    return "other"


# ---------------------------------------------------------------------------
# Code-name detection (L4)
# ---------------------------------------------------------------------------

_CODE_NAME_RE = re.compile(r"^[A-Z]{2,5}-\d+", re.IGNORECASE)


def is_code_name(name: str) -> bool:
    """Return True if name looks like a drug code (e.g., BMS-123456, ABT-737)."""
    return bool(_CODE_NAME_RE.match(name.strip()))


# ---------------------------------------------------------------------------
# DB query helpers
# ---------------------------------------------------------------------------


def load_candidate_pool(
    db_path: Path,
    tier_filter: str | None = None,
    extra_where: str = "",
) -> pd.DataFrame:
    """Load candidate records with standard intervention/condition/trial JOINs.

    Parameters
    ----------
    db_path : Path to CT database
    tier_filter : e.g. "!= 'copper'" or "= 'bronze'" or "IN ('gold', 'silver')"
    extra_where : additional SQL WHERE clauses (AND-joined)

    Returns
    -------
    DataFrame with trial_failure_results + intervention + condition + trial columns.
    """
    from negbiodb_ct.ct_db import get_connection

    where_parts = [BASE_INTERVENTION_FILTER.strip()]
    if tier_filter:
        where_parts.append(f"tfr.confidence_tier {tier_filter}")
    if extra_where:
        where_parts.append(extra_where)
    where_clause = " AND ".join(where_parts)

    sql = f"""
    SELECT
        tfr.result_id, tfr.failure_category, tfr.failure_detail,
        tfr.confidence_tier, tfr.result_interpretation,
        tfr.p_value_primary, tfr.effect_size, tfr.effect_size_type,
        tfr.ci_lower, tfr.ci_upper, tfr.primary_endpoint_met,
        tfr.serious_adverse_events, tfr.highest_phase_reached,
        tfr.arm_description,
        i.intervention_name, i.canonical_smiles, i.molecular_type,
        i.intervention_type, i.chembl_id, i.intervention_id,
        c.condition_name, c.mesh_id, c.condition_id,
        ct.trial_phase, ct.blinding, ct.control_type,
        ct.enrollment_actual, ct.sponsor_type, ct.randomized,
        ct.source_trial_id, ct.why_stopped, ct.has_results,
        ct.completion_date
    FROM trial_failure_results tfr
    JOIN interventions i ON tfr.intervention_id = i.intervention_id
    JOIN conditions c ON tfr.condition_id = c.condition_id
    LEFT JOIN clinical_trials ct ON tfr.trial_id = ct.trial_id
    WHERE {where_clause}
    """

    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()

    logger.info("Loaded %d candidate records (tier_filter=%s)", len(df), tier_filter)
    return df


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def apply_max_per_drug(
    df: pd.DataFrame,
    max_per_drug: int = MAX_PER_DRUG,
    rng: np.random.RandomState | None = None,
) -> pd.DataFrame:
    """Cap records per intervention to prevent single drug dominating.

    Random sample within each intervention_id if over the cap.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    groups = df.groupby("intervention_id")
    kept = []
    for _, group in groups:
        if len(group) <= max_per_drug:
            kept.append(group)
        else:
            idx = rng.choice(len(group), size=max_per_drug, replace=False)
            kept.append(group.iloc[idx])
    result = pd.concat(kept, ignore_index=True)
    n_dropped = len(df) - len(result)
    if n_dropped > 0:
        logger.info(
            "apply_max_per_drug: kept %d, dropped %d (cap=%d)",
            len(result), n_dropped, max_per_drug,
        )
    return result


def assign_splits(
    df: pd.DataFrame,
    fewshot_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> pd.DataFrame:
    """Assign fewshot/val/test splits.

    Shuffles df and assigns first fewshot_size as 'fewshot', next val_size
    as 'val', remainder (up to test_size) as 'test'.  Returns df with 'split' column.
    """
    rng = np.random.RandomState(seed)
    total_needed = fewshot_size + val_size + test_size
    if len(df) < total_needed:
        logger.warning(
            "Dataset (%d) smaller than requested splits (%d). "
            "Adjusting test_size to %d.",
            len(df), total_needed, len(df) - fewshot_size - val_size,
        )
        test_size = max(0, len(df) - fewshot_size - val_size)

    idx = rng.permutation(len(df))
    df = df.iloc[idx].reset_index(drop=True)

    split_col = ["test"] * len(df)
    for i in range(min(fewshot_size, len(df))):
        split_col[i] = "fewshot"
    for i in range(fewshot_size, min(fewshot_size + val_size, len(df))):
        split_col[i] = "val"
    # Remaining up to fewshot_size + val_size + test_size are 'test'
    # Anything beyond is dropped
    df = df.iloc[: fewshot_size + val_size + test_size].copy()
    df["split"] = split_col[: len(df)]

    logger.info(
        "Splits: fewshot=%d, val=%d, test=%d",
        (df["split"] == "fewshot").sum(),
        (df["split"] == "val").sum(),
        (df["split"] == "test").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_jsonl(records: list[dict], output_path: Path) -> None:
    """Write records as JSONL (one JSON per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    logger.info("Wrote %d records to %s", len(records), output_path)


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file into list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_dataset_metadata(
    output_dir: Path,
    task: str,
    stats: dict,
) -> None:
    """Write metadata.json with dataset statistics, creation date, tier distributions."""
    meta = {
        "task": task,
        "created": datetime.now(timezone.utc).isoformat(),
        **stats,
    }
    meta_path = output_dir / f"{task.replace('-', '_')}_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Metadata saved to %s", meta_path)
