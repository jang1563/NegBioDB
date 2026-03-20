#!/usr/bin/env python3
"""Build CT-L3 reasoning dataset for LLM benchmark.

Generates 200 records from gold tier (with silver fallback cascade),
Phase II-III, safety+efficacy. Requires ChEMBL resolution.
Supports biologics (no SMILES, molecular context).

Output: exports/ct_llm/ct_l3_dataset.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "exports" / "ct_llm"

N_TOTAL = 200
MIN_CHEMBL_COVERAGE_PCT = 25
MAX_ONCOLOGY_FRAC = 0.40
BIOLOGIC_TYPES = {"monoclonal_antibody", "antibody_drug_conjugate", "peptide", "other_biologic"}


def load_intervention_targets(db_path: Path) -> dict[int, list[dict]]:
    """Load intervention_targets: intervention_id → list of {uniprot, gene_symbol}."""
    from negbiodb_ct.ct_db import get_connection

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT intervention_id, uniprot_accession, gene_symbol "
            "FROM intervention_targets"
        ).fetchall()
    finally:
        conn.close()

    targets: dict[int, list[dict]] = {}
    for iid, uniprot, gene in rows:
        targets.setdefault(iid, []).append({
            "uniprot": uniprot,
            "gene_symbol": gene,
        })
    return targets


def format_l3_context(row: pd.Series, targets: list[dict] | None) -> str:
    """Generate CT-L3 reasoning context (gold tier, full quantitative)."""
    from negbiodb_ct.llm_dataset import infer_therapeutic_area

    lines = [f"Drug: {row.get('intervention_name', 'Unknown')}"]

    mol_type = row.get("molecular_type")
    if mol_type:
        lines.append(f"Drug type: {mol_type}")

    smiles = row.get("canonical_smiles")
    if smiles and pd.notna(smiles) and mol_type == "small_molecule":
        lines.append(f"SMILES: {smiles}")

    if targets:
        gene_list = [t["gene_symbol"] for t in targets if t.get("gene_symbol")]
        if gene_list:
            lines.append(f"Known targets: {', '.join(gene_list[:5])}")

    condition = row.get("condition_name", "Unknown")
    lines.append(f"Condition: {condition}")

    ta = infer_therapeutic_area(condition)
    lines.append(f"Therapeutic area: {ta}")

    phase = row.get("trial_phase") or row.get("highest_phase_reached")
    if phase:
        lines.append(f"Phase: {phase}")

    design = row.get("control_type")
    if design:
        lines.append(f"Design: {design}")
    blinding = row.get("blinding")
    if blinding and pd.notna(blinding):
        lines.append(f"Blinding: {blinding}")
    enrollment = row.get("enrollment_actual")
    if enrollment and pd.notna(enrollment):
        lines.append(f"Enrollment: {int(enrollment)}")

    endpoint_met = row.get("primary_endpoint_met")
    if endpoint_met and pd.notna(endpoint_met):
        lines.append(f"Primary endpoint met: {endpoint_met}")
    p_val = row.get("p_value_primary")
    if p_val and pd.notna(p_val):
        lines.append(f"p-value: {p_val}")
    effect = row.get("effect_size")
    if effect and pd.notna(effect):
        etype = row.get("effect_size_type", "")
        lines.append(f"Effect size ({etype}): {effect}" if etype else f"Effect size: {effect}")
    ci_lo, ci_hi = row.get("ci_lower"), row.get("ci_upper")
    if ci_lo and pd.notna(ci_lo) and ci_hi and pd.notna(ci_hi):
        lines.append(f"95% CI: [{ci_lo}, {ci_hi}]")

    saes = row.get("serious_adverse_events")
    if saes and pd.notna(saes):
        lines.append(f"Serious adverse events: {saes}")

    interp = row.get("result_interpretation")
    if interp and pd.notna(interp):
        lines.append(f"Interpretation: {interp}")

    return "\n".join(lines)


def _build_gold_reasoning(row: pd.Series, targets: list[dict]) -> str:
    """Build a brief evidence-based reasoning summary for few-shot examples.

    This is a structured factual summary, not expert-level reasoning.
    It demonstrates the expected response format for few-shot prompting.
    """
    parts = []
    drug = row.get("intervention_name", "The drug")
    condition = row.get("condition_name", "the condition")
    category = row.get("failure_category", "unknown")

    # Opening
    parts.append(
        f"{drug} failed in a clinical trial for {condition}, "
        f"classified as a {category} failure."
    )

    # Evidence
    p_val = row.get("p_value_primary")
    effect = row.get("effect_size")
    endpoint_met = row.get("primary_endpoint_met")
    if p_val and pd.notna(p_val):
        parts.append(f"The primary endpoint p-value was {p_val}.")
    if endpoint_met and pd.notna(endpoint_met):
        parts.append(f"Primary endpoint met: {endpoint_met}.")
    if effect and pd.notna(effect):
        etype = row.get("effect_size_type", "")
        if etype:
            parts.append(f"The effect size ({etype}) was {effect}.")
        else:
            parts.append(f"The effect size was {effect}.")

    # Targets
    if targets:
        gene_list = [t["gene_symbol"] for t in targets if t.get("gene_symbol")]
        if gene_list:
            parts.append(
                f"The drug targets {', '.join(gene_list[:3])}, "
                f"which may have been insufficient for this indication."
            )

    # Safety
    saes = row.get("serious_adverse_events")
    if saes and pd.notna(saes) and category == "safety":
        parts.append(f"Serious adverse events were reported: {saes}.")

    # Interpretation
    interp = row.get("result_interpretation")
    if interp and pd.notna(interp):
        parts.append(f"The trial interpretation noted: {interp}.")

    return " ".join(parts)


def build_l3_dataset(db_path: Path, seed: int) -> list[dict]:
    """Build CT-L3 dataset from DB with relaxation cascade."""
    from negbiodb_ct.llm_dataset import (
        apply_max_per_drug,
        infer_therapeutic_area,
        load_candidate_pool,
    )

    rng = np.random.RandomState(seed)
    int_targets = load_intervention_targets(db_path)

    # Base filter: gold tier, safety+efficacy, Phase II-III, ChEMBL resolved
    extra_where = (
        "tfr.failure_category IN ('safety', 'efficacy') "
        "AND ct.trial_phase IN ('phase_2', 'phase_2_3', 'phase_3') "
        "AND ct.has_results = 1 "
        "AND i.chembl_id IS NOT NULL "
        "AND (i.canonical_smiles IS NOT NULL OR i.molecular_type != 'small_molecule')"
    )
    pool = load_candidate_pool(db_path, tier_filter="= 'gold'", extra_where=extra_where)
    logger.info("L3 gold pool (strict): %d records", len(pool))

    # Relaxation cascade if pool too small
    relaxation_log = []
    if len(pool) < N_TOTAL:
        # Relax 1: allow silver tier (keep has_results requirement)
        extra_silver = (
            "tfr.failure_category IN ('safety', 'efficacy') "
            "AND ct.trial_phase IN ('phase_2', 'phase_2_3', 'phase_3') "
            "AND ct.has_results = 1 "
            "AND i.chembl_id IS NOT NULL "
            "AND (i.canonical_smiles IS NOT NULL OR i.molecular_type != 'small_molecule')"
        )
        pool_r1 = load_candidate_pool(
            db_path, tier_filter="IN ('gold', 'silver')", extra_where=extra_silver
        )
        relaxation_log.append(f"R1: allow silver → {len(pool_r1)} records")
        pool = pool_r1

    if len(pool) < N_TOTAL:
        # Relax 2: drop has_results requirement
        extra_r2 = (
            "tfr.failure_category IN ('safety', 'efficacy') "
            "AND ct.trial_phase IN ('phase_2', 'phase_2_3', 'phase_3') "
            "AND i.chembl_id IS NOT NULL "
            "AND (i.canonical_smiles IS NOT NULL OR i.molecular_type != 'small_molecule')"
        )
        pool_r2 = load_candidate_pool(
            db_path, tier_filter="IN ('gold', 'silver')", extra_where=extra_r2
        )
        relaxation_log.append(f"R2: drop has_results → {len(pool_r2)} records")
        pool = pool_r2

    if len(pool) < N_TOTAL:
        # Relax 3: drop SMILES requirement for biologics
        extra_r3 = (
            "tfr.failure_category IN ('safety', 'efficacy') "
            "AND ct.trial_phase IN ('phase_2', 'phase_2_3', 'phase_3') "
            "AND i.chembl_id IS NOT NULL"
        )
        pool_r3 = load_candidate_pool(
            db_path, tier_filter="IN ('gold', 'silver')", extra_where=extra_r3
        )
        relaxation_log.append(f"R3: drop SMILES req → {len(pool_r3)} records")
        pool = pool_r3

    if relaxation_log:
        logger.info("Relaxation cascade applied:\n  %s", "\n  ".join(relaxation_log))

    if len(pool) == 0:
        logger.error("No L3 candidates found even after relaxation!")
        return []

    # Apply max-per-drug
    pool = apply_max_per_drug(pool, rng=rng)

    # Diversity constraints
    pool["therapeutic_area"] = pool["condition_name"].apply(
        lambda x: infer_therapeutic_area(x) if pd.notna(x) else "other"
    )
    pool["is_biologic"] = pool["molecular_type"].isin(BIOLOGIC_TYPES)

    # Oncology cap
    oncology = pool[pool["therapeutic_area"] == "oncology"]
    non_oncology = pool[pool["therapeutic_area"] != "oncology"]
    max_oncology = int(N_TOTAL * MAX_ONCOLOGY_FRAC)
    if len(oncology) > max_oncology:
        oncology = oncology.sample(max_oncology, random_state=rng.randint(0, 2**31))
    pool = pd.concat([oncology, non_oncology], ignore_index=True)

    # Sample
    n_target = min(N_TOTAL, len(pool))

    # Balance safety/efficacy ~50/50
    safety_pool = pool[pool["failure_category"] == "safety"]
    efficacy_pool = pool[pool["failure_category"] == "efficacy"]
    n_safety = min(n_target // 2, len(safety_pool))
    n_efficacy = min(n_target - n_safety, len(efficacy_pool))
    if n_efficacy < n_target - n_safety:
        n_safety = min(n_target - n_efficacy, len(safety_pool))

    sampled_safety = safety_pool.sample(n_safety, random_state=rng.randint(0, 2**31))
    sampled_efficacy = efficacy_pool.sample(n_efficacy, random_state=rng.randint(0, 2**31))
    sampled = pd.concat([sampled_safety, sampled_efficacy], ignore_index=True)

    # Check biologic fraction
    n_biologic = sampled["is_biologic"].sum()
    logger.info(
        "Sampled %d (safety=%d, efficacy=%d, biologic=%d/%.0f%%)",
        len(sampled), n_safety, n_efficacy,
        n_biologic, 100 * n_biologic / max(len(sampled), 1),
    )

    # Build records
    records = []
    for _, row in sampled.iterrows():
        iid = row.get("intervention_id")
        targets = int_targets.get(int(iid), []) if pd.notna(iid) else []
        context_text = format_l3_context(row, targets)
        gold_reasoning = _build_gold_reasoning(row, targets)
        records.append({
            "question_id": None,
            "task": "CT-L3",
            "gold_answer": row["failure_category"],
            "gold_reasoning": gold_reasoning,
            "gold_category": row["failure_category"],
            "difficulty": None,  # L3 uses judge scoring, no difficulty
            "context_text": context_text,
            "metadata": {
                "result_id": int(row["result_id"]),
                "source_trial_id": row.get("source_trial_id"),
                "intervention_name": row.get("intervention_name"),
                "condition_name": row.get("condition_name"),
                "confidence_tier": row.get("confidence_tier"),
                "therapeutic_area": row.get("therapeutic_area"),
                "molecular_type": row.get("molecular_type"),
                "chembl_id": row.get("chembl_id"),
                "n_targets": len(targets),
            },
        })

    return records


def main(argv: list[str] | None = None) -> int:
    from negbiodb_ct.llm_dataset import (
        assign_splits,
        write_dataset_metadata,
        write_jsonl,
    )

    parser = argparse.ArgumentParser(description="Build CT-L3 reasoning dataset")
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_ct.db")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.db_path.exists():
        logger.error("CT database not found: %s", args.db_path)
        return 1

    records = build_l3_dataset(args.db_path, args.seed)
    if not records:
        logger.error("No records generated!")
        return 1

    df = pd.DataFrame(records)
    df = assign_splits(df, fewshot_size=20, val_size=20, test_size=160, seed=args.seed)

    output_records = []
    for i, (_, row) in enumerate(df.iterrows()):
        rec = row.to_dict()
        rec["question_id"] = f"CTL3-{i:04d}"
        output_records.append(rec)

    output_path = args.output_dir / "ct_l3_dataset.jsonl"
    write_jsonl(output_records, output_path)

    from collections import Counter
    splits = Counter(r["split"] for r in output_records)
    cats = Counter(r["gold_category"] for r in output_records)

    logger.info("=== CT-L3 Dataset Summary ===")
    logger.info("Total: %d", len(output_records))
    logger.info("Categories: %s", dict(cats))
    logger.info("Splits: %s", dict(splits))

    write_dataset_metadata(args.output_dir, "ct-l3", {
        "total": len(output_records),
        "categories": dict(cats),
        "splits": dict(splits),
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
