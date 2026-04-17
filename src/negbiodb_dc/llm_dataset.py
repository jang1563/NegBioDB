"""LLM benchmark dataset builder for DC domain (Drug Combination Synergy).

Shared utilities for building DC-L1 through DC-L4 JSONL datasets:
  - load_dc_candidate_pool: SQL query for drug-pair records
  - construct_l1_context: Classification context
  - construct_l2_context: Mechanism extraction context
  - construct_l3_context: Rich pharmacological context for reasoning
  - construct_l4_context: Minimal identification context
  - apply_max_per_drug: Cap records per drug (prevent dominance)
  - assign_splits, write_jsonl, write_dataset_metadata
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Classification mappings ─────────────────────────────────────────

# DC-L1: 4-way classification (maps DB consensus_class → L1 letter)
# Note: L1 gold_answer is determined by ZIP threshold, not consensus_class.
# DB CHECK allows: synergistic, additive, antagonistic, context_dependent.
L1_CLASS_MAP = {
    "synergistic": "B",
    "additive": "B",
    "antagonistic": "C",
    "context_dependent": "B",
}

# Synergy class descriptions
SYNERGY_DESCRIPTIONS = {
    "strongly_synergistic": "Strong synergy: drugs amplify each other's effects (ZIP > 10)",
    "synergistic": "Moderate synergy: drugs work better together (5 < ZIP ≤ 10)",
    "additive": "Additive: combined effect equals sum of individual effects (-5 ≤ ZIP ≤ 5)",
    "antagonistic": "Antagonistic: drugs interfere with each other (-10 ≤ ZIP < -5)",
    "strongly_antagonistic": "Strong antagonism: drugs strongly counteract each other (ZIP < -10)",
}

# L4 temporal group definitions
L4_GROUPS = {
    "classic_combos": "Well-known combination (in DrugComb + ALMANAC)",
    "recent_combos": "Recently tested combination (post-2020 DrugComb)",
    "untested_plausible": "Plausible but untested (same therapeutic area)",
    "untested_rare": "Unlikely tested (different therapeutic areas)",
}

# Few-shot seed sets
FEWSHOT_SEEDS = [42, 43, 44]

MAX_PER_DRUG = 15  # Cap records per individual drug to prevent dominance


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


# ── Candidate pool loading ──────────────────────────────────────────


def load_dc_candidate_pool(
    conn,
    min_confidence: str = "bronze",
    consensus_class: str | None = None,
) -> pd.DataFrame:
    """Load drug-pair records for dataset construction.

    Returns DataFrame with columns:
        pair_id, compound_a_id, compound_b_id, drug_a_name, drug_b_name,
        smiles_a, smiles_b, num_cell_lines, num_sources, num_measurements,
        median_zip, median_bliss, antagonism_fraction, synergy_fraction,
        consensus_class, confidence_tier, num_shared_targets, target_jaccard,
        compound_a_degree, compound_b_degree, drug_a_targets, drug_b_targets
    """
    tier_filter = {
        "gold": "('gold')",
        "silver": "('gold', 'silver')",
        "bronze": "('gold', 'silver', 'bronze')",
        "copper": "('gold', 'silver', 'bronze', 'copper')",
    }
    tier_sql = tier_filter.get(min_confidence, "('gold', 'silver', 'bronze')")

    class_clause = ""
    params: tuple = ()
    if consensus_class:
        class_clause = "AND ddp.consensus_class = ?"
        params = (consensus_class,)

    query = f"""
    SELECT
        ddp.pair_id, ddp.compound_a_id, ddp.compound_b_id,
        ca.drug_name AS drug_a_name, cb.drug_name AS drug_b_name,
        ca.canonical_smiles AS smiles_a, cb.canonical_smiles AS smiles_b,
        ca.known_targets AS drug_a_targets, cb.known_targets AS drug_b_targets,
        ca.atc_code AS atc_a, cb.atc_code AS atc_b,
        ddp.num_cell_lines, ddp.num_sources, ddp.num_measurements,
        ddp.median_zip, ddp.median_bliss,
        ddp.antagonism_fraction, ddp.synergy_fraction,
        ddp.consensus_class, ddp.best_confidence AS confidence_tier,
        ddp.num_shared_targets, ddp.target_jaccard,
        ddp.compound_a_degree, ddp.compound_b_degree
    FROM drug_drug_pairs ddp
    JOIN compounds ca ON ddp.compound_a_id = ca.compound_id
    JOIN compounds cb ON ddp.compound_b_id = cb.compound_id
    WHERE ddp.best_confidence IN {tier_sql}
    {class_clause}
    """

    df = pd.read_sql_query(query, conn, params=params)
    logger.info("Loaded %d candidate records (min_confidence=%s)", len(df), min_confidence)
    return df


# ── Context construction ────────────────────────────────────────────


def construct_l1_context(row: pd.Series) -> str:
    """Build context for DC-L1 4-way classification task."""
    parts = [
        f"Drug A: {row['drug_a_name']}",
        f"Drug B: {row['drug_b_name']}",
    ]

    if row.get("drug_a_targets"):
        parts.append(f"Drug A targets: {row['drug_a_targets']}")
    if row.get("drug_b_targets"):
        parts.append(f"Drug B targets: {row['drug_b_targets']}")

    if row.get("num_shared_targets") and row["num_shared_targets"] > 0:
        parts.append(f"Shared targets: {row['num_shared_targets']} (Jaccard: {row.get('target_jaccard', 0):.2f})")

    if row.get("num_cell_lines"):
        parts.append(f"Tested in {row['num_cell_lines']} cell line(s)")

    if row.get("num_sources") and row["num_sources"] > 1:
        parts.append(f"Data from {row['num_sources']} independent sources")

    return "\n".join(parts)


def construct_l2_context(row: pd.Series) -> str:
    """Build context for DC-L2 mechanism extraction task.

    Simulates a pharmacology report describing the drug combination.
    """
    parts = [
        f"Drug Combination Report: {row['drug_a_name']} + {row['drug_b_name']}",
        "",
        f"Drug A ({row['drug_a_name']}):",
    ]

    if row.get("drug_a_targets"):
        parts.append(f"  Known targets: {row['drug_a_targets']}")
    if row.get("smiles_a"):
        parts.append(f"  Structure: {row['smiles_a']}")
    if row.get("atc_a"):
        parts.append(f"  ATC code: {row['atc_a']}")

    parts.append(f"\nDrug B ({row['drug_b_name']}):")
    if row.get("drug_b_targets"):
        parts.append(f"  Known targets: {row['drug_b_targets']}")
    if row.get("smiles_b"):
        parts.append(f"  Structure: {row['smiles_b']}")
    if row.get("atc_b"):
        parts.append(f"  ATC code: {row['atc_b']}")

    parts.append("\nCombination Data:")
    if row.get("num_shared_targets") is not None:
        parts.append(f"  Shared targets: {row['num_shared_targets']}")
        parts.append(f"  Target overlap (Jaccard): {row.get('target_jaccard', 0):.3f}")
    if row.get("num_cell_lines"):
        parts.append(f"  Tested in {row['num_cell_lines']} cell line(s)")
    if row.get("antagonism_fraction") is not None:
        parts.append(f"  Antagonism fraction: {row['antagonism_fraction']:.2f}")
        parts.append(f"  Synergy fraction: {row.get('synergy_fraction', 0):.2f}")

    return "\n".join(parts)


def construct_l3_context(row: pd.Series) -> str:
    """Build rich context for DC-L3 antagonism reasoning task.

    Includes pharmacological details for explaining WHY the combination fails.
    """
    parts = [
        f"Drug Combination: {row['drug_a_name']} + {row['drug_b_name']}",
        f"Outcome: {SYNERGY_DESCRIPTIONS.get(row.get('consensus_class', ''), 'Unknown')}",
        "",
        f"Drug A — {row['drug_a_name']}:",
    ]

    if row.get("drug_a_targets"):
        parts.append(f"  Molecular targets: {row['drug_a_targets']}")
    if row.get("smiles_a"):
        parts.append(f"  SMILES: {row['smiles_a']}")
    if row.get("atc_a"):
        parts.append(f"  ATC classification: {row['atc_a']}")

    parts.append(f"\nDrug B — {row['drug_b_name']}:")
    if row.get("drug_b_targets"):
        parts.append(f"  Molecular targets: {row['drug_b_targets']}")
    if row.get("smiles_b"):
        parts.append(f"  SMILES: {row['smiles_b']}")
    if row.get("atc_b"):
        parts.append(f"  ATC classification: {row['atc_b']}")

    parts.append("\nCombination Evidence:")
    if row.get("num_shared_targets") is not None:
        parts.append(f"  Shared molecular targets: {row['num_shared_targets']}")
        parts.append(f"  Target overlap (Jaccard index): {row.get('target_jaccard', 0):.3f}")
    if row.get("median_zip") is not None:
        parts.append(f"  Median ZIP synergy score: {row['median_zip']:.1f}")
    if row.get("median_bliss") is not None:
        parts.append(f"  Median Bliss score: {row['median_bliss']:.1f}")
    if row.get("num_cell_lines"):
        parts.append(f"  Cell lines tested: {row['num_cell_lines']}")
    if row.get("num_sources") and row["num_sources"] > 1:
        parts.append(f"  Independent data sources: {row['num_sources']}")
    if row.get("antagonism_fraction") is not None:
        parts.append(f"  Cell lines showing antagonism: {row['antagonism_fraction']:.0%}")

    return "\n".join(parts)


def construct_l4_context(row: pd.Series) -> str:
    """Build minimal context for DC-L4 tested/untested discrimination."""
    parts = [
        f"Drug A: {row['drug_a_name']}",
        f"Drug B: {row['drug_b_name']}",
    ]
    if row.get("drug_a_targets"):
        parts.append(f"Drug A targets: {row['drug_a_targets']}")
    if row.get("drug_b_targets"):
        parts.append(f"Drug B targets: {row['drug_b_targets']}")
    return "\n".join(parts)


# ── Sampling utilities ──────────────────────────────────────────────


def apply_max_per_drug(
    df: pd.DataFrame,
    max_per_drug: int = MAX_PER_DRUG,
    rng: np.random.RandomState | None = None,
) -> pd.DataFrame:
    """Cap records per individual drug to prevent dominance."""
    if rng is None:
        rng = np.random.RandomState(42)

    # Count appearances per drug (as either A or B)
    drug_counts: dict[str, int] = {}
    keep_mask = []

    shuffled = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    for _, row in shuffled.iterrows():
        da = row["drug_a_name"]
        db = row["drug_b_name"]
        ca = drug_counts.get(da, 0)
        cb = drug_counts.get(db, 0)

        if ca < max_per_drug and cb < max_per_drug:
            keep_mask.append(True)
            drug_counts[da] = ca + 1
            drug_counts[db] = cb + 1
        else:
            keep_mask.append(False)

    result = shuffled[keep_mask].copy()
    logger.info("Max-per-drug filter: %d → %d records", len(df), len(result))
    return result


def assign_splits(
    df: pd.DataFrame,
    fewshot_size: int = 30,
    val_size: int = 0,
    test_size: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign split labels: fewshot → val → test."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))

    splits = ["test"] * len(df)
    for i, pos in enumerate(idx):
        if i < fewshot_size:
            splits[pos] = "fewshot"
        elif val_size > 0 and i < fewshot_size + val_size:
            splits[pos] = "val"

    df = df.copy()
    df["split"] = splits
    return df


def write_jsonl(records: list[dict], path: Path) -> int:
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(_json_safe(rec), ensure_ascii=False, allow_nan=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)
    return len(records)


def write_dataset_metadata(
    path: Path,
    task: str,
    n_records: int,
    split_counts: dict[str, int],
    **kwargs,
) -> None:
    """Write dataset metadata JSON."""
    meta = {
        "task": task,
        "domain": "dc",
        "n_records": int(n_records),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        **{k: (int(v) if hasattr(v, "item") else v) for k, v in kwargs.items()},
    }
    with open(path, "w") as f:
        json.dump(_json_safe(meta), f, indent=2, allow_nan=False)
