#!/usr/bin/env python3
"""Build GE-L1 MCQ dataset for LLM benchmark.

Generates 1,200 four-way MCQ records across 4 essentiality classes:
  A) Common essential   (300) — Required for viability in nearly all cell types
  B) Selective essential (300) — Required specifically in this lineage/context
  C) Non-essential       (300) — Knockout has no significant effect on viability
  D) Unknown/Untested    (300) — Not tested in this cell line

Difficulty: easy(40%), medium(35%), hard(25%)
Split: 240 fewshot (60/class) + 240 val (60/class) + 720 test (180/class)

Essential pairs (classes A and B) are sourced from the raw CRISPRGeneEffect.csv
and CRISPRGeneDependency.csv files because the database only stores non-essential pairs.

Output: exports/ge_llm/ge_l1_dataset.jsonl

Usage:
    PYTHONPATH=src python scripts_depmap/build_ge_l1_dataset.py --data-dir data/depmap_raw
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "exports" / "ge_llm"

N_PER_CLASS = 300

FRAC_EASY = 0.40
FRAC_MEDIUM = 0.35
FRAC_HARD = 0.25

CLASS_LABELS = {
    "A": "common_essential",
    "B": "selective_essential",
    "C": "non_essential",
    "D": "unknown_untested",
}

_GENE_COL_RE = re.compile(r"^(.+?)\s*\((\d+)\)$")


def _load_essential_from_raw(
    data_dir: Path,
    conn,
    n: int,
    rng: np.random.RandomState,
    *,
    common_essential_only: bool,
) -> pd.DataFrame:
    """Load essential gene-cell_line pairs from raw CSV files.

    The database only stores non-essential pairs, so essential pairs
    (dep_prob > 0.5 AND gene_effect < -1.0) must be sourced from the
    raw CRISPRGeneEffect.csv and CRISPRGeneDependency.csv files.
    """
    effect_file = data_dir / "CRISPRGeneEffect.csv"
    dep_file = data_dir / "CRISPRGeneDependency.csv"

    if not effect_file.exists() or not dep_file.exists():
        logger.error("Raw CSV files not found in %s", data_dir)
        return pd.DataFrame()

    # Load gene metadata from DB
    genes_df = pd.read_sql_query("""
        SELECT gene_id, entrez_id, gene_symbol, description,
               is_common_essential, is_reference_nonessential
        FROM genes
    """, conn)
    entrez_to_gene = dict(zip(genes_df["entrez_id"], genes_df.index))
    genes_by_entrez = genes_df.set_index("entrez_id")

    # Load cell line metadata from DB
    cl_df = pd.read_sql_query("""
        SELECT cell_line_id, model_id, ccle_name, lineage, primary_disease
        FROM cell_lines
    """, conn)
    model_to_cl = dict(zip(cl_df["model_id"], cl_df.index))
    cl_by_model = cl_df.set_index("model_id")

    # Filter genes by common_essential flag
    if common_essential_only:
        valid_entrez = set(genes_df[genes_df["is_common_essential"] == 1]["entrez_id"])
        label = "common essential"
    else:
        valid_entrez = set(genes_df[genes_df["is_common_essential"] == 0]["entrez_id"])
        label = "selective essential"

    logger.info("Scanning raw CSVs for %s pairs (%d candidate genes)...", label, len(valid_entrez))

    # Read headers to get gene column mapping
    effect_header = pd.read_csv(effect_file, nrows=0)
    dep_header = pd.read_csv(dep_file, nrows=0)

    gene_cols_effect = []
    col_to_entrez = {}
    for col in effect_header.columns[1:]:  # skip first col (ModelID)
        m = _GENE_COL_RE.match(col.strip())
        if m:
            entrez = int(m.group(2))
            if entrez in valid_entrez:
                gene_cols_effect.append(col)
                col_to_entrez[col] = entrez

    # Find matching columns in dep file
    dep_cols = set(dep_header.columns)
    gene_cols_both = [c for c in gene_cols_effect if c in dep_cols]
    logger.info("Found %d candidate gene columns in both files", len(gene_cols_both))

    if not gene_cols_both:
        return pd.DataFrame()

    # Read both files with subset of columns, indexed by ModelID.
    # NOTE: The two CSV files have DIFFERENT row orderings, so we must
    # join by ModelID — not zip rows together.
    usecols_effect = [effect_header.columns[0]] + gene_cols_both
    usecols_dep = [dep_header.columns[0]] + gene_cols_both

    logger.info("Reading effect file (%d columns)...", len(usecols_effect))
    eff_df = pd.read_csv(effect_file, usecols=usecols_effect, index_col=0)
    logger.info("Reading dependency file (%d columns)...", len(usecols_dep))
    dep_df = pd.read_csv(dep_file, usecols=usecols_dep, index_col=0)

    # Intersect cell lines present in both files and in DB
    common_models = set(eff_df.index) & set(dep_df.index) & set(cl_by_model.index)
    logger.info("Cell lines in both files and DB: %d", len(common_models))

    essential_records = []
    target_n = n * 3  # oversample to allow for filtering

    for model_id in common_models:
        if len(essential_records) >= target_n:
            break

        cl_row = cl_by_model.loc[model_id]
        eff_row = eff_df.loc[model_id]
        dep_row = dep_df.loc[model_id]

        for col in gene_cols_both:
            effect = eff_row[col]
            dep_prob = dep_row[col]

            if pd.isna(effect) or pd.isna(dep_prob):
                continue

            # Essential: dep_prob > 0.5 AND gene_effect < -1.0
            if dep_prob > 0.5 and effect < -1.0:
                entrez = col_to_entrez[col]
                g_row = genes_by_entrez.loc[entrez]
                essential_records.append({
                    "gene_id": int(g_row["gene_id"]),
                    "gene_symbol": g_row["gene_symbol"],
                    "entrez_id": entrez,
                    "description": g_row["description"],
                    "is_common_essential": int(g_row["is_common_essential"]),
                    "is_reference_nonessential": int(g_row["is_reference_nonessential"]),
                    "cell_line_id": int(cl_row["cell_line_id"]),
                    "model_id": model_id,
                    "ccle_name": cl_row["ccle_name"],
                    "lineage": cl_row["lineage"],
                    "primary_disease": cl_row["primary_disease"],
                    "gene_effect_score": float(effect),
                    "dependency_probability": float(dep_prob),
                })

    logger.info("Found %d %s pairs from raw CSVs", len(essential_records), label)

    if not essential_records:
        return pd.DataFrame()

    df = pd.DataFrame(essential_records)
    if len(df) >= n:
        return df.sample(n=n, random_state=rng).reset_index(drop=True)
    logger.warning("%s: only %d available, need %d", label.capitalize(), len(df), n)
    return df.reset_index(drop=True)


def load_common_essential(
    conn, n: int, rng: np.random.RandomState, data_dir: Path,
) -> pd.DataFrame:
    """Load common essential gene-cell_line pairs (class A) from raw CSVs."""
    return _load_essential_from_raw(data_dir, conn, n, rng, common_essential_only=True)


def load_selective_essential(
    conn, n: int, rng: np.random.RandomState, data_dir: Path,
) -> pd.DataFrame:
    """Load selective essential gene-cell_line pairs (class B) from raw CSVs."""
    return _load_essential_from_raw(data_dir, conn, n, rng, common_essential_only=False)


def load_non_essential(conn, n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Load non-essential gene-cell_line pairs (class C)."""
    from negbiodb_depmap.llm_dataset import load_ge_candidate_pool

    df = load_ge_candidate_pool(conn, min_confidence="silver")
    df = df[df["dependency_probability"].fillna(1) < 0.3].copy()
    df = df[df["gene_effect_score"].fillna(-999) > -0.3].copy()

    if len(df) >= n:
        return df.sample(n=n, random_state=rng).reset_index(drop=True)
    logger.warning("Non-essential: only %d available, need %d", len(df), n)
    return df.reset_index(drop=True)


def load_unknown_untested(conn, n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Generate unknown/untested gene-cell_line pairs (class D).

    Pairs genes and cell lines that are in the DB but NOT paired together.
    """
    genes = pd.read_sql_query("""
        SELECT gene_id, gene_symbol, entrez_id, description,
               is_common_essential, is_reference_nonessential
        FROM genes ORDER BY RANDOM() LIMIT 500
    """, conn)

    cell_lines = pd.read_sql_query("""
        SELECT cell_line_id, model_id, ccle_name, lineage, primary_disease
        FROM cell_lines ORDER BY RANDOM() LIMIT 100
    """, conn)

    # Use gene_cell_pairs (aggregated) instead of scanning all 28M+ negative results
    tested = set()
    rows = conn.execute(
        "SELECT gene_id, cell_line_id FROM gene_cell_pairs"
    ).fetchall()
    for r in rows:
        tested.add((r[0], r[1]))

    records = []
    for _ in range(n * 10):
        if len(records) >= n:
            break
        gi = rng.randint(len(genes))
        ci = rng.randint(len(cell_lines))
        gid = int(genes.iloc[gi]["gene_id"])
        clid = int(cell_lines.iloc[ci]["cell_line_id"])
        if (gid, clid) not in tested:
            rec = {**genes.iloc[gi].to_dict(), **cell_lines.iloc[ci].to_dict()}
            rec["gene_effect_score"] = None
            rec["dependency_probability"] = None
            records.append(rec)
            tested.add((gid, clid))

    return pd.DataFrame(records[:n])


def format_l1_context(row: pd.Series, difficulty: str) -> str:
    """Format context for GE-L1 MCQ."""
    gene = row.get("gene_symbol", "UNKNOWN")
    cell_line = row.get("ccle_name") or row.get("model_id", "UNKNOWN")
    lineage = row.get("lineage", "unknown lineage")
    disease = row.get("primary_disease", "")

    parts = [f"Gene: {gene}", f"Cell line: {cell_line} ({lineage})"]

    if difficulty == "easy":
        desc = row.get("description")
        if desc and isinstance(desc, str):
            parts.append(f"Gene function: {desc[:200]}")
        if disease:
            parts.append(f"Disease: {disease}")
        effect = row.get("gene_effect_score")
        dep = row.get("dependency_probability")
        if effect is not None and not pd.isna(effect):
            parts.append(f"Chronos gene effect: {effect:.3f}")
        if dep is not None and not pd.isna(dep):
            parts.append(f"Dependency probability: {dep:.3f}")
    elif difficulty == "medium":
        if disease:
            parts.append(f"Disease: {disease}")

    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build GE-L1 MCQ dataset.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_depmap.db")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "depmap_raw",
                        help="Directory with raw CRISPRGeneEffect.csv and CRISPRGeneDependency.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "ge_l1_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from negbiodb_depmap.depmap_db import get_connection
    from negbiodb_depmap.llm_dataset import (
        apply_max_per_gene,
        assign_splits,
        write_dataset_metadata,
        write_jsonl,
    )

    rng = np.random.RandomState(args.seed)
    conn = get_connection(args.db)

    try:
        # Load each class
        class_a = load_common_essential(conn, N_PER_CLASS, rng, args.data_dir)
        class_a["gold_answer"] = "A"
        class_a["gold_category"] = CLASS_LABELS["A"]

        class_b = load_selective_essential(conn, N_PER_CLASS, rng, args.data_dir)
        class_b["gold_answer"] = "B"
        class_b["gold_category"] = CLASS_LABELS["B"]

        class_c = load_non_essential(conn, N_PER_CLASS, rng)
        class_c["gold_answer"] = "C"
        class_c["gold_category"] = CLASS_LABELS["C"]

        class_d = load_unknown_untested(conn, N_PER_CLASS, rng)
        class_d["gold_answer"] = "D"
        class_d["gold_category"] = CLASS_LABELS["D"]
    finally:
        conn.close()

    combined = pd.concat([class_a, class_b, class_c, class_d], ignore_index=True)
    combined = apply_max_per_gene(combined, max_per_gene=10)
    logger.info("Combined: %d records", len(combined))

    # Assign difficulty
    n_total = len(combined)
    difficulties = (
        ["easy"] * int(n_total * FRAC_EASY)
        + ["medium"] * int(n_total * FRAC_MEDIUM)
        + ["hard"] * (n_total - int(n_total * FRAC_EASY) - int(n_total * FRAC_MEDIUM))
    )
    rng.shuffle(difficulties)
    combined["difficulty"] = difficulties[:len(combined)]

    # Assign splits (class-stratified)
    split_parts = []
    for letter in sorted(CLASS_LABELS.keys()):
        class_df = combined[combined["gold_answer"] == letter].copy()
        class_df = assign_splits(class_df, ratios={"train": 0.2, "val": 0.2, "test": 0.6})
        # Rename train to fewshot
        class_df["split"] = class_df["split"].replace({"train": "fewshot"})
        split_parts.append(class_df)
    combined = pd.concat(split_parts, ignore_index=True)

    # Build JSONL records
    records = []
    for i, (_, row) in enumerate(combined.iterrows()):
        difficulty = row["difficulty"]
        context = format_l1_context(row, difficulty)
        rec = {
            "question_id": f"GEL1-{i:04d}",
            "task": "ge-l1",
            "split": row["split"],
            "difficulty": difficulty,
            "context_text": context,
            "gold_answer": row["gold_answer"],
            "gold_category": row["gold_category"],
            "metadata": {
                "gene_symbol": row.get("gene_symbol"),
                "model_id": row.get("model_id"),
                "lineage": row.get("lineage"),
                "is_common_essential": int(row.get("is_common_essential", 0)) if pd.notna(row.get("is_common_essential")) else None,
            },
        }
        records.append(rec)

    write_jsonl(records, args.output)

    stats = {
        "n_total": len(records),
        "n_per_class": {v: N_PER_CLASS for v in CLASS_LABELS.values()},
        "difficulty_distribution": dict(combined["difficulty"].value_counts()),
        "split_distribution": dict(combined["split"].value_counts()),
        "seed": args.seed,
    }
    write_dataset_metadata(args.output.parent, "ge-l1", stats)

    logger.info("GE-L1 dataset built: %d records", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
