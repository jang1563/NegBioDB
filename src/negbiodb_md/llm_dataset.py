"""LLM benchmark dataset builder for MD domain (Metabolite-Disease Non-Association).

Builds MD-L1 through MD-L4 JSONL datasets:
  L1: 4-way MCQ — "Which metabolite is NOT a biomarker for disease Y?"
      (3 distractors = significant metabolites from the same study; 1 correct = non-significant)
  L2: Structured field extraction from study metadata
      (metabolite, disease, fold_change, platform, outcome)
  L3: Free-text reasoning about metabolite non-association
      (LLM-as-judge on 4 rubric axes; gold_reasoning auto-generated from metadata)
  L4: Discrimination — real non-association (MetaboLights/NMDR) vs synthetic
      (random metabolite-disease pairing never tested in any study)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EXPORT_DIR = _PROJECT_ROOT / "exports" / "md_llm"

FEWSHOT_SEEDS = [42, 43, 44]
MAX_PER_DISEASE = 20   # Cap records per disease to prevent imbalance

# L3 rubric axes (4-axis, 0-5 per axis)
L3_RUBRIC_AXES = [
    "metabolite_biology",    # Correct metabolite biochemistry
    "disease_mechanism",     # Correct disease pathophysiology
    "study_context",         # References study design limitations (n, platform, biofluid)
    "alternative_hypothesis", # Proposes testable alternatives
]


def _json_safe(value):
    """Convert numpy scalars / NaN to JSON-serializable types."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


# ── Candidate pool ────────────────────────────────────────────────────────────

def load_md_candidate_pool(
    conn,
    min_tier: str = "bronze",
    is_significant: int | None = None,
) -> pd.DataFrame:
    """Load MD results for dataset construction.

    Args:
        conn:           sqlite3 connection to negbiodb_md.db
        min_tier:       Minimum confidence tier for negatives (gold/silver/bronze/copper)
        is_significant: If 0/1, filter by significance; if None, return all

    Returns DataFrame with columns:
        result_id, pair_id, metabolite_id, metabolite_name, inchikey,
        metabolite_class, canonical_smiles, disease_id, disease_name,
        disease_category, mondo_id, study_id, study_source, external_id,
        title, description, biofluid, platform, n_disease, n_control,
        p_value, fdr, fold_change, log2_fc, is_significant, tier,
        consensus, best_tier, metabolite_degree, disease_degree
    """
    tier_order = {"gold": 1, "silver": 2, "bronze": 3, "copper": 4}
    tier_val = tier_order.get(min_tier, 3)
    tier_filter = " AND ".join(
        f"r.tier != '{t}'" for t, v in tier_order.items() if v > tier_val
    ) if tier_val < 4 else "1=1"

    sig_clause = ""
    if is_significant is not None:
        sig_clause = f"AND r.is_significant = {int(is_significant)}"

    rows = conn.execute(
        f"""SELECT
            r.result_id,
            p.pair_id,
            m.metabolite_id,
            m.name AS metabolite_name,
            m.inchikey,
            m.metabolite_class,
            m.canonical_smiles,
            d.disease_id,
            d.name AS disease_name,
            d.disease_category,
            d.mondo_id,
            s.study_id,
            s.source AS study_source,
            s.external_id,
            s.title,
            s.description,
            s.biofluid,
            s.platform,
            s.n_disease,
            s.n_control,
            r.p_value,
            r.fdr,
            r.fold_change,
            r.log2_fc,
            r.is_significant,
            r.tier,
            p.consensus,
            p.best_tier,
            p.metabolite_degree,
            p.disease_degree
        FROM md_biomarker_results r
        JOIN md_metabolites m ON r.metabolite_id = m.metabolite_id
        JOIN md_diseases d ON r.disease_id = d.disease_id
        JOIN md_studies s ON r.study_id = s.study_id
        JOIN md_metabolite_disease_pairs p
            ON p.metabolite_id = r.metabolite_id AND p.disease_id = r.disease_id
        WHERE ({tier_filter} OR r.is_significant = 1)
          {sig_clause}
        ORDER BY r.result_id"""
    ).fetchall()

    cols = [
        "result_id", "pair_id", "metabolite_id", "metabolite_name", "inchikey",
        "metabolite_class", "canonical_smiles", "disease_id", "disease_name",
        "disease_category", "mondo_id", "study_id", "study_source", "external_id",
        "title", "description", "biofluid", "platform", "n_disease", "n_control",
        "p_value", "fdr", "fold_change", "log2_fc", "is_significant", "tier",
        "consensus", "best_tier", "metabolite_degree", "disease_degree",
    ]
    return pd.DataFrame(rows, columns=cols)


# ── L1: 4-way MCQ ─────────────────────────────────────────────────────────────

def build_l1_context(row: pd.Series) -> str:
    """Build context string for L1 (no ground-truth info included)."""
    lines = [
        f"Disease: {row['disease_name']}",
        f"Study platform: {(row.get('platform') or 'unspecified').upper().replace('_', '-')}",
        f"Sample biofluid: {row.get('biofluid') or 'unspecified'}",
    ]
    if row.get("disease_category"):
        lines.append(f"Disease category: {row['disease_category']}")
    if row.get("n_disease"):
        lines.append(f"Disease group n: {int(row['n_disease'])}")
    return "\n".join(lines)


def build_l1_dataset(
    conn,
    n_records: int = 800,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> Path:
    """Build MD-L1 JSONL dataset.

    For each question:
      - Correct answer: a metabolite from the study that was NOT significant (is_significant=0)
      - 3 distractors: significant metabolites (is_significant=1) from the SAME study

    Returns path to output JSONL file.
    """
    if output_dir is None:
        output_dir = DEFAULT_EXPORT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    df = load_md_candidate_pool(conn, min_tier="bronze")

    # Build study-level lookup
    negatives = df[df["is_significant"] == 0].copy()
    positives = df[df["is_significant"] == 1].copy()

    pos_by_study: dict[int, list] = {}
    for _, row in positives.iterrows():
        pos_by_study.setdefault(row["study_id"], []).append(row)

    records = []
    disease_counts: dict[int, int] = {}

    for _, neg_row in negatives.sample(frac=1, random_state=seed).iterrows():
        disease_id = neg_row["disease_id"]
        if disease_counts.get(disease_id, 0) >= MAX_PER_DISEASE:
            continue

        study_id = neg_row["study_id"]
        study_positives = pos_by_study.get(study_id, [])
        if len(study_positives) < 3:
            continue  # Need at least 3 distractors from same study

        distractors = rng.sample(study_positives, 3)
        choices = [neg_row["metabolite_name"]] + [d["metabolite_name"] for d in distractors]
        rng.shuffle(choices)
        correct_letter = chr(ord("A") + choices.index(neg_row["metabolite_name"]))

        context = build_l1_context(neg_row)
        record = {
            "record_id": f"md_l1_{neg_row['result_id']}",
            "task": "md_l1",
            "context": context,
            "question": f"Which of the following metabolites is NOT a significant biomarker for {neg_row['disease_name']} in this study?",
            "choices": {chr(ord("A") + i): c for i, c in enumerate(choices)},
            "gold_answer": correct_letter,
            "metadata": _json_safe({
                "result_id": int(neg_row["result_id"]),
                "metabolite_id": int(neg_row["metabolite_id"]),
                "disease_id": int(disease_id),
                "study_id": int(study_id),
                "study_source": neg_row["study_source"],
                "p_value": neg_row["p_value"],
                "fdr": neg_row["fdr"],
                "tier": neg_row["tier"],
                "disease_category": neg_row["disease_category"],
            }),
        }
        records.append(record)
        disease_counts[disease_id] = disease_counts.get(disease_id, 0) + 1

        if len(records) >= n_records:
            break

    out_path = output_dir / "md_l1.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info("MD-L1: %d records → %s", len(records), out_path)
    return out_path


# ── L2: Structured field extraction ──────────────────────────────────────────

def build_l2_dataset(
    conn,
    n_records: int = 400,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> Path:
    """Build MD-L2 JSONL dataset.

    Source: Study metadata (title + description + result row data).
    Task: Extract metabolite, disease, fold_change, platform, outcome fields.
    Gold: exact values from the database.
    """
    if output_dir is None:
        output_dir = DEFAULT_EXPORT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_md_candidate_pool(conn, min_tier="bronze")
    # Use both positives and negatives for L2 (field extraction regardless of outcome)
    df = df[df["description"].notna() & (df["description"].str.len() > 50)]
    if len(df) > n_records:
        df = df.sample(n=n_records, random_state=seed)

    records = []
    for _, row in df.iterrows():
        context = (
            f"Study: {row['title']}\n"
            f"Source: {(row.get('study_source') or '').upper()} {row.get('external_id', '')}\n"
            f"Description: {(row.get('description') or '')[:300]}\n"
            f"Metabolite: {row['metabolite_name']}\n"
            f"Fold change: {row.get('fold_change')}\n"
            f"p-value: {row.get('p_value')}"
        )
        outcome = "significant" if row["is_significant"] else "not_significant"
        gold_fields = {
            "metabolite": row["metabolite_name"],
            "disease": row["disease_name"],
            "fold_change": row.get("fold_change"),
            "platform": row.get("platform"),
            "biofluid": row.get("biofluid"),
            "outcome": outcome,
        }
        record = {
            "record_id": f"md_l2_{row['result_id']}",
            "task": "md_l2",
            "context": context,
            "gold_fields": _json_safe(gold_fields),
            "metadata": _json_safe({
                "result_id": int(row["result_id"]),
                "metabolite_id": int(row["metabolite_id"]),
                "disease_id": int(row["disease_id"]),
                "study_id": int(row["study_id"]),
                "tier": row.get("tier"),
                "is_significant": int(row["is_significant"]),
            }),
        }
        records.append(record)

    out_path = output_dir / "md_l2.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info("MD-L2: %d records → %s", len(records), out_path)
    return out_path


# ── L3: Free-text reasoning ───────────────────────────────────────────────────

def _generate_gold_reasoning(row: pd.Series) -> str:
    """Auto-generate gold reasoning from study metadata.

    Covers 4 rubric axes: metabolite_biology, disease_mechanism,
    study_context, alternative_hypothesis.
    This is an initial template — top-50 records reviewed manually.
    """
    met_class = row.get("metabolite_class") or "metabolite"
    disease = row["disease_name"]
    platform = (row.get("platform") or "unspecified").replace("_", "-").upper()
    biofluid = row.get("biofluid") or "unspecified sample"
    pval = row.get("p_value")
    n = row.get("n_disease")
    pval_str = f"(p={pval:.3f})" if pval is not None else ""

    n_str = ""
    if n:
        n_str = f" In this study, the disease group included only {int(n)} participants, which may have limited statistical power."

    return (
        f"The {met_class} {row['metabolite_name']} was not found to be a significant "
        f"biomarker for {disease} in this {platform} metabolomics study "
        f"analyzing {biofluid} {pval_str}. "
        f"From a biochemical perspective, {row['metabolite_name']} is a {met_class} "
        f"and its metabolic pathway may not be directly disrupted in {disease}. "
        f"The disease mechanism of {disease} primarily involves pathways that may not "
        f"substantially alter circulating levels of this compound in {biofluid}."
        f"{n_str} "
        f"Alternative hypotheses include tissue-specific effects not captured in {biofluid}, "
        f"population heterogeneity, or the need for longitudinal rather than cross-sectional "
        f"sampling to detect metabolic changes."
    )


def build_l3_dataset(
    conn,
    n_records: int = 200,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> Path:
    """Build MD-L3 JSONL dataset with gold reasoning."""
    if output_dir is None:
        output_dir = DEFAULT_EXPORT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_md_candidate_pool(conn, min_tier="silver", is_significant=0)
    if len(df) > n_records:
        df = df.sample(n=n_records, random_state=seed)

    records = []
    for _, row in df.iterrows():
        context = (
            f"Metabolite: {row['metabolite_name']}"
            + (f" (class: {row['metabolite_class']})" if row.get("metabolite_class") else "")
            + f"\nDisease: {row['disease_name']}"
            + (f" (category: {row['disease_category']})" if row.get("disease_category") else "")
            + f"\nStudy: {row.get('title') or row.get('external_id', '')}"
            + f"\nPlatform: {(row.get('platform') or 'unspecified').upper().replace('_', '-')}"
            + f"\nBiofluid: {row.get('biofluid') or 'unspecified'}"
            + (f"\np-value: {row['p_value']:.4f}" if row.get("p_value") is not None else "")
            + (f"\nFDR: {row['fdr']:.4f}" if row.get("fdr") is not None else "")
            + (f"\nDisease group n: {int(row['n_disease'])}" if row.get("n_disease") else "")
        )
        gold_reasoning = _generate_gold_reasoning(row)
        record = {
            "record_id": f"md_l3_{row['result_id']}",
            "task": "md_l3",
            "context": context,
            "question": (
                f"Explain why {row['metabolite_name']} was not found to be a significant "
                f"biomarker for {row['disease_name']} in this study. "
                f"Consider the metabolite's biology, the disease mechanism, "
                f"study design limitations, and alternative hypotheses."
            ),
            "gold_reasoning": gold_reasoning,
            "rubric_axes": L3_RUBRIC_AXES,
            "metadata": _json_safe({
                "result_id": int(row["result_id"]),
                "metabolite_id": int(row["metabolite_id"]),
                "disease_id": int(row["disease_id"]),
                "study_id": int(row["study_id"]),
                "tier": row.get("tier"),
                "p_value": row.get("p_value"),
                "disease_category": row.get("disease_category"),
            }),
        }
        records.append(record)

    out_path = output_dir / "md_l3.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info("MD-L3: %d records → %s", len(records), out_path)
    return out_path


# ── L4: Discrimination ────────────────────────────────────────────────────────

def build_l4_dataset(
    conn,
    n_records: int = 200,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> Path:
    """Build MD-L4 JSONL dataset: real vs synthetic non-association.

    Real negatives: from MetaboLights/NMDR (actually measured, not significant)
    Synthetic negatives: random (metabolite, disease) pairs with NO entry in
        md_biomarker_results — i.e., never tested
    """
    if output_dir is None:
        output_dir = DEFAULT_EXPORT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    df_neg = load_md_candidate_pool(conn, min_tier="silver", is_significant=0)

    # Get all tested (metabolite_id, disease_id) pairs — for synthetic exclusion
    tested_pairs: set[tuple[int, int]] = set(
        (int(r[0]), int(r[1])) for r in conn.execute(
            "SELECT metabolite_id, disease_id FROM md_biomarker_results"
        ).fetchall()
    )

    all_metabolites = conn.execute(
        "SELECT metabolite_id, name FROM md_metabolites WHERE canonical_smiles IS NOT NULL"
    ).fetchall()
    all_diseases = conn.execute(
        "SELECT disease_id, name FROM md_diseases"
    ).fetchall()

    if not all_metabolites or not all_diseases:
        logger.warning("MD-L4: insufficient metabolites/diseases for synthetic pairs")
        return output_dir / "md_l4.jsonl"

    # Sample real negatives
    n_real = min(n_records // 2, len(df_neg))
    df_real = df_neg.sample(n=n_real, random_state=seed)

    # Generate synthetic pairs (never tested)
    synthetics = []
    attempts = 0
    max_attempts = n_records * 100
    while len(synthetics) < n_records // 2 and attempts < max_attempts:
        met = rng.choice(all_metabolites)
        dis = rng.choice(all_diseases)
        if (met[0], dis[0]) not in tested_pairs:
            synthetics.append({"metabolite_id": met[0], "metabolite_name": met[1],
                                "disease_id": dis[0], "disease_name": dis[1]})
            tested_pairs.add((met[0], dis[0]))  # prevent duplicates
        attempts += 1

    records = []

    # Real records (label = 1 = real negative)
    for _, row in df_real.iterrows():
        context = (
            f"Metabolite: {row['metabolite_name']}"
            + (f" (class: {row.get('metabolite_class') or 'unknown'})")
            + f"\nDisease: {row['disease_name']}"
            + f"\nStudy: {row.get('study_source', '').upper()} {row.get('external_id', '')}"
            + f"\nPlatform: {(row.get('platform') or 'unspecified').upper().replace('_', '-')}"
        )
        records.append({
            "record_id": f"md_l4_real_{row['result_id']}",
            "task": "md_l4",
            "context": context,
            "label": 1,  # 1 = real (actually measured, not significant)
            "label_text": "real",
            "metadata": _json_safe({
                "result_id": int(row["result_id"]),
                "metabolite_id": int(row["metabolite_id"]),
                "disease_id": int(row["disease_id"]),
                "study_source": row.get("study_source"),
                "tier": row.get("tier"),
            }),
        })

    # Synthetic records (label = 0 = synthetic / never tested)
    for i, syn in enumerate(synthetics):
        context = (
            f"Metabolite: {syn['metabolite_name']}"
            f"\nDisease: {syn['disease_name']}"
        )
        records.append({
            "record_id": f"md_l4_synth_{i}",
            "task": "md_l4",
            "context": context,
            "label": 0,  # 0 = synthetic (never tested)
            "label_text": "synthetic",
            "metadata": {
                "metabolite_id": syn["metabolite_id"],
                "disease_id": syn["disease_id"],
            },
        })

    rng.shuffle(records)

    out_path = output_dir / "md_l4.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info(
        "MD-L4: %d records (%d real, %d synthetic) → %s",
        len(records), n_real, len(synthetics), out_path,
    )
    return out_path
