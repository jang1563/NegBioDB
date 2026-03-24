"""LLM benchmark dataset builder for GE domain.

Shared utilities for building GE-L1 through GE-L4 JSONL datasets:
  - load_ge_candidate_pool: SQL query for gene-cell_line records
  - construct_evidence_description: Template-based evidence text
  - construct_l3_context: Rich gene function + cell line context
  - construct_l4_context: Minimal identification context
  - apply_max_per_gene: Cap records per gene
  - assign_splits, write_jsonl, write_dataset_metadata
"""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_ge_candidate_pool(
    conn,
    min_confidence: str = "bronze",
    source_db: str | None = None,
) -> pd.DataFrame:
    """Load gene-cell_line records for dataset construction.

    Returns DataFrame with columns:
        gene_id, cell_line_id, entrez_id, gene_symbol,
        model_id, ccle_name, lineage, primary_disease,
        gene_effect_score, dependency_probability,
        evidence_type, confidence_tier, source_db,
        is_common_essential, is_reference_nonessential,
        gene_degree, cell_line_degree
    """
    tier_filter = {
        "gold": "('gold')",
        "silver": "('gold', 'silver')",
        "bronze": "('gold', 'silver', 'bronze')",
    }
    tier_sql = tier_filter.get(min_confidence, "('gold', 'silver', 'bronze')")

    source_clause = ""
    params: tuple = ()
    if source_db:
        source_clause = "AND nr.source_db = ?"
        params = (source_db,)

    query = f"""
    SELECT
        nr.gene_id, nr.cell_line_id,
        g.entrez_id, g.gene_symbol, g.is_common_essential, g.is_reference_nonessential,
        c.model_id, c.ccle_name, c.lineage, c.primary_disease,
        nr.gene_effect_score, nr.dependency_probability,
        nr.evidence_type, nr.confidence_tier, nr.source_db,
        p.gene_degree, p.cell_line_degree,
        p.num_screens, p.num_sources
    FROM ge_negative_results nr
    JOIN genes g ON nr.gene_id = g.gene_id
    JOIN cell_lines c ON nr.cell_line_id = c.cell_line_id
    LEFT JOIN gene_cell_pairs p ON nr.gene_id = p.gene_id AND nr.cell_line_id = p.cell_line_id
    WHERE nr.confidence_tier IN {tier_sql}
    {source_clause}
    """

    df = pd.read_sql_query(query, conn, params=params)
    logger.info("Loaded %d candidate records (min_confidence=%s)", len(df), min_confidence)
    return df


def construct_evidence_description(row: pd.Series) -> str:
    """Generate template-based evidence description for a gene-cell_line pair.

    Used for GE-L1 and GE-L2 tasks (since DepMap publications don't reference
    individual gene-cell_line pairs).
    """
    gene = row.get("gene_symbol", "UNKNOWN")
    cell_line = row.get("ccle_name") or row.get("model_id", "UNKNOWN")
    lineage = row.get("lineage", "unknown lineage")
    disease = row.get("primary_disease", "")
    effect = row.get("gene_effect_score")
    dep_prob = row.get("dependency_probability")
    evidence = row.get("evidence_type", "")
    source = row.get("source_db", "")

    parts = [f"Gene: {gene}"]
    parts.append(f"Cell line: {cell_line} ({lineage})")
    if disease:
        parts.append(f"Disease: {disease}")

    if "crispr" in evidence:
        parts.append("Screen: CRISPR-Cas9 (Chronos algorithm)")
    elif "rnai" in evidence:
        parts.append("Screen: RNAi (DEMETER2 algorithm)")
    else:
        parts.append(f"Evidence: {evidence}")

    if effect is not None and not pd.isna(effect):
        parts.append(f"Gene effect score: {effect:.3f}")
    if dep_prob is not None and not pd.isna(dep_prob):
        parts.append(f"Dependency probability: {dep_prob:.3f}")

    if source:
        parts.append(f"Source: {source}")

    tier = row.get("confidence_tier", "")
    if tier:
        parts.append(f"Confidence: {tier}")

    return "\n".join(parts)


def construct_l3_context(
    row: pd.Series,
    gene_description: str | None = None,
) -> str:
    """Build rich context for GE-L3 reasoning task."""
    gene = row.get("gene_symbol", "UNKNOWN")
    cell_line = row.get("ccle_name") or row.get("model_id", "UNKNOWN")
    lineage = row.get("lineage", "unknown lineage")
    disease = row.get("primary_disease", "")

    parts = []
    parts.append(f"Gene: {gene}")
    if gene_description:
        parts.append(f"Function: {gene_description}")

    parts.append(f"Cell line: {cell_line}")
    parts.append(f"Lineage: {lineage}")
    if disease:
        parts.append(f"Disease: {disease}")

    effect = row.get("gene_effect_score")
    dep_prob = row.get("dependency_probability")
    if effect is not None and not pd.isna(effect):
        parts.append(f"CRISPR gene effect (Chronos): {effect:.3f}")
    if dep_prob is not None and not pd.isna(dep_prob):
        parts.append(f"Dependency probability: {dep_prob:.3f}")

    is_ce = row.get("is_common_essential", 0)
    is_rne = row.get("is_reference_nonessential", 0)
    if is_rne:
        parts.append("This gene is in the reference non-essential gene set (Hart et al.)")
    if is_ce:
        parts.append("Note: This gene is classified as common essential in other cell types")

    degree = row.get("gene_degree")
    if degree and not pd.isna(degree):
        parts.append(f"Non-essential in {int(degree)} cell lines in DepMap")

    return "\n".join(parts)


def construct_l4_context(row: pd.Series) -> str:
    """Build minimal context for GE-L4 tested/untested discrimination."""
    gene = row.get("gene_symbol", "UNKNOWN")
    cell_line = row.get("ccle_name") or row.get("model_id", "UNKNOWN")
    lineage = row.get("lineage", "")

    parts = [f"Gene: {gene}", f"Cell line: {cell_line}"]
    if lineage:
        parts.append(f"Lineage: {lineage}")

    return "\n".join(parts)


def apply_max_per_gene(
    df: pd.DataFrame,
    max_per_gene: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Cap records per gene to prevent single gene dominating the dataset."""
    rng = np.random.RandomState(seed)
    groups = []

    for gene_id, group in df.groupby("gene_id"):
        if len(group) <= max_per_gene:
            groups.append(group)
        else:
            idx = rng.choice(len(group), max_per_gene, replace=False)
            groups.append(group.iloc[idx])

    result = pd.concat(groups, ignore_index=True)
    logger.info("After max_per_gene=%d: %d → %d records", max_per_gene, len(df), len(result))
    return result


def assign_splits(
    df: pd.DataFrame,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test splits to DataFrame rows."""
    if ratios is None:
        ratios = {"train": 0.7, "val": 0.1, "test": 0.2}

    rng = np.random.RandomState(seed)
    n = len(df)
    perm = rng.permutation(n)

    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    splits = np.empty(n, dtype="U5")
    splits[perm[:n_train]] = "train"
    splits[perm[n_train:n_train + n_val]] = "val"
    splits[perm[n_train + n_val:]] = "test"

    df = df.copy()
    df["split"] = splits
    return df


def write_jsonl(records: list[dict], output_path: Path) -> int:
    """Write records to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info("Wrote %d records to %s", len(records), output_path)
    return len(records)


def write_dataset_metadata(
    output_dir: Path,
    task: str,
    stats: dict,
) -> None:
    """Write dataset metadata JSON file.

    Args:
        output_dir: Directory to write the metadata file.
        task: Task name (e.g., "ge-l1").
        stats: Dictionary of dataset statistics to write.
    """
    meta = {
        "task": task,
        "domain": "ge",
        **stats,
    }
    meta_path = output_dir / f"{task}_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Wrote metadata to %s", meta_path)
