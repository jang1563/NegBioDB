#!/usr/bin/env python3
"""Build L3 Reasoning Pilot dataset for LLM benchmark.

Generates 50 well-known inactive DTI pairs with reasoning rubrics.
Target diversity: kinases 20, GPCRs 10, proteases 10, other 10

The LLM must explain WHY the compound is inactive against the target.
Evaluation: LLM-as-Judge with 4-dimension rubric.

Split: 5 few-shot + 5 val + 40 test

Output: exports/llm_benchmarks/l3_reasoning_pilot.jsonl
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEGBIODB_PATH = PROJECT_ROOT / "data" / "negbiodb.db"
NAMES_PATH = PROJECT_ROOT / "exports" / "compound_names.parquet"
OUTPUT_PATH = PROJECT_ROOT / "exports" / "llm_benchmarks" / "l3_reasoning_pilot.jsonl"

# Family allocation
FAMILY_ALLOCATION = {
    "kinase": 20,
    "GPCR": 10,
    "protease": 10,
    "other": 10,
}


def load_compound_names() -> dict:
    df = pd.read_parquet(NAMES_PATH)
    return {
        int(row["compound_id"]): row["pref_name"]
        for _, row in df.iterrows()
        if pd.notna(row["pref_name"])
    }


def select_reasoning_pairs(
    db_path: Path, names: dict, seed: int
) -> list[dict]:
    """Select 50 well-known inactive DTI pairs with target diversity.

    Prioritizes: multi-assay evidence, named compounds, diverse targets.
    """
    conn = sqlite3.connect(str(db_path))
    rng = random.Random(seed)

    # Select high-evidence pairs (multi-assay, named compounds)
    rows = conn.execute(
        """
        SELECT ctp.compound_id, ctp.target_id,
               c.canonical_smiles, c.inchikey,
               t.uniprot_accession, t.gene_symbol, t.target_family,
               ctp.num_assays, ctp.num_sources, ctp.earliest_year,
               ctp.median_pchembl
        FROM compound_target_pairs ctp
        JOIN compounds c ON ctp.compound_id = c.compound_id
        JOIN targets t ON ctp.target_id = t.target_id
        WHERE ctp.best_confidence = 'silver'
          AND c.chembl_id IS NOT NULL
          AND ctp.num_assays >= 2
        ORDER BY RANDOM()
        LIMIT 5000
        """,
    ).fetchall()
    conn.close()

    cols = [
        "compound_id", "target_id", "smiles", "inchikey",
        "uniprot", "gene_symbol", "family", "num_assays",
        "num_sources", "earliest_year", "median_pchembl",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["compound_name"] = df["compound_id"].map(names)
    named = df[df["compound_name"].notna()].copy()

    # L-5: Prefer targets with gene symbols for interpretability
    named = named.sort_values("gene_symbol", na_position="last")
    print(f"  Named high-evidence pairs: {len(named)}")
    print(f"  With gene symbol: {named['gene_symbol'].notna().sum()}")

    # M-2: Use FAMILY_ALLOCATION for family-stratified sampling
    # Classify into allocation buckets
    def classify_family(fam):
        if fam and fam.lower() == "kinase":
            return "kinase"
        if fam and fam.lower() in ("gpcr", "g protein-coupled receptor"):
            return "GPCR"
        if fam and fam.lower() in ("protease", "peptidase"):
            return "protease"
        return "other"

    named["family_bucket"] = named["family"].apply(classify_family)

    # One pair per target, stratified by family
    unique_targets = named.drop_duplicates("target_id")
    all_selected = []
    for bucket, n_target in FAMILY_ALLOCATION.items():
        pool = unique_targets[unique_targets["family_bucket"] == bucket]
        # Prefer targets with gene symbols
        pool = pool.sort_values("gene_symbol", na_position="last")
        n_sample = min(n_target, len(pool))
        sampled = pool.head(n_sample * 3).sample(
            min(n_sample, len(pool)), random_state=seed
        )
        for _, row in sampled.iterrows():
            all_selected.append(
                {
                    "class": "reasoning",
                    "compound_name": row["compound_name"],
                    "compound_smiles": row["smiles"],
                    "compound_inchikey": row["inchikey"],
                    "target_uniprot": row["uniprot"],
                    "target_gene": row["gene_symbol"],
                    "target_family": row["family"] or "protein",
                    "family_bucket": bucket,
                    "num_assays": int(row["num_assays"]),
                    "num_sources": int(row["num_sources"]),
                    "evidence_quality": "silver",
                }
            )
        print(f"  {bucket}: {n_sample}/{n_target} selected")

    # Fill remaining if any bucket was short
    remaining = 50 - len(all_selected)
    if remaining > 0:
        used_targets = {r["target_uniprot"] for r in all_selected}
        leftover = unique_targets[~unique_targets["uniprot"].isin(used_targets)]
        leftover = leftover.sort_values("gene_symbol", na_position="last")
        extra = leftover.head(remaining)
        for _, row in extra.iterrows():
            all_selected.append(
                {
                    "class": "reasoning",
                    "compound_name": row["compound_name"],
                    "compound_smiles": row["smiles"],
                    "compound_inchikey": row["inchikey"],
                    "target_uniprot": row["uniprot"],
                    "target_gene": row["gene_symbol"],
                    "target_family": row["family"] or "protein",
                    "family_bucket": classify_family(row["family"]),
                    "num_assays": int(row["num_assays"]),
                    "num_sources": int(row["num_sources"]),
                    "evidence_quality": "silver",
                }
            )
        print(f"  Backfill: {len(extra)} extra pairs")

    rng.shuffle(all_selected)
    return all_selected[:50]


def generate_context_text(record: dict) -> str:
    """Generate L3 reasoning prompt."""
    name = record.get("compound_name", "Unknown")
    smiles = record.get("compound_smiles", "")
    gene = record.get("target_gene")
    uniprot = record.get("target_uniprot", "Unknown")
    family = record.get("target_family") or "protein"
    target_str = f"{gene} ({uniprot}), {family}" if gene else f"{uniprot}, {family}"

    lines = [
        f"Compound: {name}",
        f"SMILES: {smiles}",
        f"Target: {target_str}",
        "",
        "This compound has been experimentally confirmed as INACTIVE against this target.",
        "",
        "Explain the likely molecular and pharmacological reasons for this inactivity.",
        "Consider: binding site compatibility, selectivity profile, structural features,",
        "mechanism of action, and any known SAR (structure-activity relationship) data.",
    ]
    return "\n".join(lines)


def split_dataset(records: list[dict], seed: int) -> list[dict]:
    """5 fewshot + 5 val + 40 test."""
    rng = random.Random(seed)
    rng.shuffle(records)
    for i, rec in enumerate(records):
        if i < 5:
            rec["split"] = "fewshot"
        elif i < 10:
            rec["split"] = "val"
        else:
            rec["split"] = "test"
    return records


def main():
    parser = argparse.ArgumentParser(description="Build L3 reasoning pilot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading data sources...")
    names = load_compound_names()
    print(f"  Compound names: {len(names)}")

    print("\nSelecting reasoning pairs by family...")
    records = select_reasoning_pairs(NEGBIODB_PATH, names, args.seed)
    print(f"\nTotal: {len(records)}")

    for rec in records:
        rec["context_text"] = generate_context_text(rec)

    records = split_dataset(records, args.seed)
    for i, rec in enumerate(records):
        rec["question_id"] = f"L3-{i:04d}"

    # Summary
    from collections import Counter
    families = Counter(r["target_family"] for r in records)
    splits = Counter(r["split"] for r in records)
    print(f"Families: {dict(families)}")
    print(f"Splits: {dict(splits)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
