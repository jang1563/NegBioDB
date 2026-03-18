#!/usr/bin/env python3
"""Build L4 Tested-vs-Untested dataset for LLM benchmark.

Generates 500 compound-target pairs:
  250 tested pairs (from NegBioDB, confirmed inactive)
    - 125 pre-2023 (earliest_year < 2023)
    - 125 post-2024 (earliest_year >= 2024)
  250 untested pairs
    - 125 trick pairs: well-known drug × well-known target, but untested
    - 125 drug × Tdark target: known drug × understudied target

Anti-contamination:
  - Pre-2023 vs post-2024 accuracy comparison (>15% gap → memorization flag)
  - Evidence citation requirement (LLM must provide assay ID / DOI)
  - All "untested" pairs verified against NegBioDB + ChEMBL positives

Split: 50 few-shot + 50 val + 400 test

Output: exports/llm_benchmarks/l4_tested_untested.jsonl
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEGBIODB_PATH = PROJECT_ROOT / "data" / "negbiodb.db"
POSITIVES_PATH = PROJECT_ROOT / "exports" / "chembl_positives_pchembl6.parquet"
NAMES_PATH = PROJECT_ROOT / "exports" / "compound_names.parquet"
OUTPUT_PATH = PROJECT_ROOT / "exports" / "llm_benchmarks" / "l4_tested_untested.jsonl"


def load_compound_names() -> dict:
    """compound_id -> pref_name."""
    df = pd.read_parquet(NAMES_PATH)
    return {
        int(row["compound_id"]): row["pref_name"]
        for _, row in df.iterrows()
        if pd.notna(row["pref_name"])
    }


def load_target_info(db_path: Path) -> dict:
    """target_id -> {uniprot, gene_symbol, family, dev_level}."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT target_id, uniprot_accession, gene_symbol, target_family, "
        "development_level FROM targets"
    ).fetchall()
    conn.close()
    return {
        r[0]: {
            "uniprot": r[1],
            "gene_symbol": r[2],
            "family": r[3],
            "dev_level": r[4],
        }
        for r in rows
    }


def load_tested_set(db_path: Path) -> set[tuple[str, str]]:
    """Load all tested (inchikey_connectivity, uniprot) pairs."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT DISTINCT c.inchikey_connectivity, t.uniprot_accession
        FROM compound_target_pairs ctp
        JOIN compounds c ON ctp.compound_id = c.compound_id
        JOIN targets t ON ctp.target_id = t.target_id
        """
    ).fetchall()
    conn.close()
    return {(r[0], r[1]) for r in rows}


def load_positive_set(positives_path: Path) -> set[tuple[str, str]]:
    """Load ChEMBL positive (inchikey, uniprot) pairs."""
    df = pd.read_parquet(positives_path, columns=["inchikey", "uniprot_id"])
    return {(row["inchikey"], row["uniprot_id"]) for _, row in df.iterrows()}


# ── Tested pairs ─────────────────────────────────────────────────────────────


def select_tested_pairs(
    db_path: Path,
    names: dict,
    target_info: dict,
    n_pre: int,
    n_post: int,
    seed: int,
) -> list[dict]:
    """Select tested pairs with temporal stratification."""
    conn = sqlite3.connect(str(db_path))

    def query_temporal(year_clause: str, limit: int) -> list:
        return conn.execute(
            f"""
            SELECT ctp.compound_id, ctp.target_id,
                   c.canonical_smiles, c.inchikey, c.inchikey_connectivity,
                   t.uniprot_accession, t.gene_symbol, t.target_family,
                   ctp.num_assays, ctp.num_sources, ctp.earliest_year,
                   ctp.best_confidence
            FROM compound_target_pairs ctp
            JOIN compounds c ON ctp.compound_id = c.compound_id
            JOIN targets t ON ctp.target_id = t.target_id
            WHERE ctp.best_confidence = 'silver'
              AND c.chembl_id IS NOT NULL
              AND {year_clause}
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    pre_rows = query_temporal("ctp.earliest_year < 2023", n_pre * 10)
    post_rows = query_temporal("ctp.earliest_year >= 2024", n_post * 10)
    conn.close()

    cols = [
        "compound_id", "target_id", "smiles", "inchikey", "inchikey_conn",
        "uniprot", "gene_symbol", "family", "num_assays", "num_sources",
        "earliest_year", "confidence",
    ]

    results = []
    for rows, n, temporal in [(pre_rows, n_pre, "pre_2023"), (post_rows, n_post, "post_2024")]:
        df = pd.DataFrame(rows, columns=cols)
        df["compound_name"] = df["compound_id"].map(names)
        named = df[df["compound_name"].notna()].copy()

        # Diversify targets, prefer those with gene symbols
        unique = named.drop_duplicates("target_id")
        with_gene = unique[unique["gene_symbol"].notna()]
        without_gene = unique[unique["gene_symbol"].isna()]
        # Prioritize targets with gene symbols
        prioritized = pd.concat([with_gene, without_gene])
        sampled = prioritized.head(n * 3)
        if len(sampled) < n:
            extra = named[~named.index.isin(sampled.index)]
            sampled = pd.concat([sampled, extra])
        sampled = sampled.sample(min(n, len(sampled)), random_state=seed)

        for _, row in sampled.iterrows():
            results.append(
                {
                    "class": "tested",
                    "correct_answer": "tested",
                    "temporal_group": temporal,
                    "compound_name": row["compound_name"],
                    "compound_smiles": row["smiles"],
                    "compound_inchikey": row["inchikey"],
                    "target_uniprot": row["uniprot"],
                    "target_gene": row["gene_symbol"],
                    "target_family": row["family"],
                    "num_assays": int(row["num_assays"]),
                    "num_sources": int(row["num_sources"]),
                    "earliest_year": int(row["earliest_year"]),
                    "evidence_quality": row["confidence"],
                    "source_db": "NegBioDB",
                }
            )

    random.Random(seed).shuffle(results)
    return results


# ── Untested pairs ───────────────────────────────────────────────────────────


def select_untested_pairs(
    db_path: Path,
    names: dict,
    target_info: dict,
    tested_set: set,
    positive_set: set,
    n_trick: int,
    n_tdark: int,
    seed: int,
) -> list[dict]:
    """Select untested pairs (trick + Tdark)."""
    conn = sqlite3.connect(str(db_path))
    rng = random.Random(seed)

    # Get well-known compounds (high degree, named)
    well_known = conn.execute(
        """
        SELECT c.compound_id, c.canonical_smiles, c.inchikey, c.inchikey_connectivity
        FROM compounds c
        WHERE c.chembl_id IS NOT NULL
        ORDER BY (
            SELECT MAX(ctp.compound_degree)
            FROM compound_target_pairs ctp
            WHERE ctp.compound_id = c.compound_id
        ) DESC
        LIMIT 2000
        """
    ).fetchall()

    # Get well-known targets (high degree)
    well_known_targets = conn.execute(
        """
        SELECT t.target_id, t.uniprot_accession, t.gene_symbol, t.target_family,
               t.development_level
        FROM targets t
        ORDER BY (
            SELECT MAX(ctp.target_degree)
            FROM compound_target_pairs ctp
            WHERE ctp.target_id = t.target_id
        ) DESC
        LIMIT 500
        """
    ).fetchall()

    # Get understudied targets (low degree, few tested compounds)
    tdark_targets = conn.execute(
        """
        SELECT t.target_id, t.uniprot_accession, t.gene_symbol, t.target_family,
               t.development_level
        FROM targets t
        WHERE (
            SELECT MAX(ctp.target_degree) FROM compound_target_pairs ctp
            WHERE ctp.target_id = t.target_id
        ) <= 10
        """
    ).fetchall()

    conn.close()

    # Filter named compounds
    named_compounds = [
        (cid, smi, ik, ikc)
        for cid, smi, ik, ikc in well_known
        if cid in names
    ]

    print(f"  Untested: {len(named_compounds)} well-known named compounds")
    print(f"  Untested: {len(well_known_targets)} well-known targets")
    print(f"  Untested: {len(tdark_targets)} understudied targets (degree ≤ 10)")

    # ── Trick pairs: well-known drug × well-known target, but untested ──
    trick_pairs = []
    rng.shuffle(named_compounds)
    for cid, smi, ik, ikc in named_compounds:
        if len(trick_pairs) >= n_trick:
            break
        rng.shuffle(well_known_targets)
        for tid, uniprot, gene, family, dev in well_known_targets[:20]:
            # Check if this pair is untested (not in tested set or positive set)
            ik_14 = ikc if ikc else ik[:14] if ik else None
            if ik_14 and (ik_14, uniprot) not in tested_set and (ik, uniprot) not in positive_set:
                trick_pairs.append(
                    {
                        "class": "untested",
                        "correct_answer": "untested",
                        "untested_type": "trick",
                        "compound_name": names[cid],
                        "compound_smiles": smi,
                        "compound_inchikey": ik,
                        "target_uniprot": uniprot,
                        "target_gene": gene,
                        "target_family": family,
                        "target_dev_level": dev,
                        "source_db": None,
                    }
                )
                break

    # ── Drug × understudied target ──
    tdark_pairs = []
    for cid, smi, ik, ikc in named_compounds:
        if len(tdark_pairs) >= n_tdark:
            break
        rng.shuffle(tdark_targets)
        for tid, uniprot, gene, family, dev in tdark_targets[:10]:
            ik_14 = ikc if ikc else ik[:14] if ik else None
            if ik_14 and (ik_14, uniprot) not in tested_set and (ik, uniprot) not in positive_set:
                tdark_pairs.append(
                    {
                        "class": "untested",
                        "correct_answer": "untested",
                        "untested_type": "tdark",
                        "compound_name": names[cid],
                        "compound_smiles": smi,
                        "compound_inchikey": ik,
                        "target_uniprot": uniprot,
                        "target_gene": gene,
                        "target_family": family,
                        "target_dev_level": dev,
                        "source_db": None,
                    }
                )
                break

    print(f"  Trick pairs: {len(trick_pairs)}")
    print(f"  Tdark pairs: {len(tdark_pairs)}")

    all_untested = trick_pairs + tdark_pairs
    rng.shuffle(all_untested)
    return all_untested


# ── Context generation ────────────────────────────────────────────────────────


def generate_context_text(record: dict) -> str:
    """Generate the L4 prompt context."""
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
        "Has this compound-target pair been experimentally tested for interaction?",
        "If tested, provide the source (database, assay ID, or publication).",
        "",
        "Answer: tested / untested",
    ]
    return "\n".join(lines)


# ── Split ─────────────────────────────────────────────────────────────────────


def split_dataset(records: list[dict], seed: int) -> list[dict]:
    """50 fewshot + 50 val + 400 test (balanced tested/untested in each)."""
    rng = random.Random(seed)

    by_class = {"tested": [], "untested": []}
    for rec in records:
        by_class[rec["class"]].append(rec)

    for cls_records in by_class.values():
        rng.shuffle(cls_records)

    for cls, cls_records in by_class.items():
        for i, rec in enumerate(cls_records):
            if i < 25:
                rec["split"] = "fewshot"
            elif i < 50:
                rec["split"] = "val"
            else:
                rec["split"] = "test"

    all_records = by_class["tested"] + by_class["untested"]
    rng.shuffle(all_records)
    return all_records


def main():
    parser = argparse.ArgumentParser(description="Build L4 tested/untested dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    print("Loading data sources...")
    names = load_compound_names()
    print(f"  Compound names: {len(names)}")

    target_info = load_target_info(NEGBIODB_PATH)
    print(f"  Targets: {len(target_info)}")

    print("Loading tested/positive sets for verification...")
    tested_set = load_tested_set(NEGBIODB_PATH)
    print(f"  Tested pairs: {len(tested_set)}")

    positive_set = load_positive_set(POSITIVES_PATH)
    print(f"  Positive pairs: {len(positive_set)}")

    # Select tested pairs (temporal split)
    print("\nSelecting tested pairs...")
    tested = select_tested_pairs(NEGBIODB_PATH, names, target_info, 125, 125, seed)
    print(f"  Selected: {len(tested)}")
    pre_count = sum(1 for r in tested if r.get("temporal_group") == "pre_2023")
    post_count = sum(1 for r in tested if r.get("temporal_group") == "post_2024")
    print(f"  Pre-2023: {pre_count}, Post-2024: {post_count}")

    # Select untested pairs
    print("\nSelecting untested pairs...")
    untested = select_untested_pairs(
        NEGBIODB_PATH, names, target_info, tested_set, positive_set, 125, 125, seed
    )
    print(f"  Selected: {len(untested)}")

    # Assemble
    all_records = tested + untested
    total = len(all_records)
    print(f"\nTotal records: {total}")

    # Generate context
    for rec in all_records:
        rec["context_text"] = generate_context_text(rec)

    # Split
    all_records = split_dataset(all_records, seed)

    # Add IDs
    for i, rec in enumerate(all_records):
        rec["question_id"] = f"L4-{i:04d}"

    # Verify: no untested pair should be in tested_set or positive_set
    n_leaks = 0
    for rec in all_records:
        if rec["class"] == "untested":
            ik = rec.get("compound_inchikey", "")
            uni = rec.get("target_uniprot", "")
            ik14 = ik[:14] if ik else ""
            if (ik14, uni) in tested_set or (ik, uni) in positive_set:
                n_leaks += 1
    print(f"\nVerification: {n_leaks} leaked untested pairs (should be 0)")

    # Summary
    from collections import Counter
    print("\n=== Dataset Summary ===")
    print(f"Classes: {Counter(r['class'] for r in all_records)}")
    print(f"Splits: {Counter(r['split'] for r in all_records)}")
    if tested:
        print(f"Temporal: pre_2023={pre_count}, post_2024={post_count}")
    if untested:
        ut_types = Counter(r.get("untested_type") for r in all_records if r["class"] == "untested")
        print(f"Untested types: {ut_types}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  {total} pairs")


if __name__ == "__main__":
    main()
