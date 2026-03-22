#!/usr/bin/env python3
"""Analyze PPI-L4 contamination gap vs protein popularity (study depth).

Addresses expert panel review findings R4-4 and R1-5:
  Is the L4 temporal contamination gap (pre-2015 vs post-2020) driven by
  genuine memorization, or confounded with protein study depth / popularity?

Approach:
  1. Load L4 dataset with gene symbols and temporal groups
  2. Map gene symbols → network degree (from PPI DB protein_protein_pairs)
  3. Load predictions from each model
  4. Stratify accuracy by temporal_group × degree_bin (high/low median split)
  5. If pre-2015 advantage persists in both degree bins → true contamination
     If advantage only in high-degree → popularity confound

Output: Markdown table to stdout and results/ppi_llm/contamination_vs_popularity.md

Usage:
    PYTHONPATH=src python scripts_ppi/analyze_ppi_contamination.py \\
        --db-path data/negbiodb_ppi.db \\
        --results-dir results/ppi_llm/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "negbiodb_ppi.db"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "ppi_llm"
DATASET_PATH = PROJECT_ROOT / "exports" / "ppi_llm" / "ppi_l4_dataset.jsonl"


def load_l4_dataset(dataset_path: Path) -> list[dict]:
    """Load L4 dataset records."""
    records = []
    with open(dataset_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_protein_degrees(db_path: Path) -> dict[str, float]:
    """Load gene_symbol → network degree from PPI pairs parquet (fast).

    Falls back to DB query if parquet is unavailable.
    """
    parquet_path = db_path.parent.parent / "exports" / "ppi" / "negbiodb_ppi_pairs.parquet"
    if parquet_path.exists():
        import pandas as pd

        df = pd.read_parquet(
            parquet_path,
            columns=["gene_symbol_1", "protein1_degree"],
        )
        df = df.dropna(subset=["gene_symbol_1", "protein1_degree"])
        degrees = df.groupby("gene_symbol_1")["protein1_degree"].max().to_dict()
        return degrees

    # Fallback: slow DB query (~160s on 2.2M rows)
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("""
            SELECT p.gene_symbol, MAX(pp.protein1_degree)
            FROM proteins p
            JOIN protein_protein_pairs pp ON p.protein_id = pp.protein1_id
            WHERE p.gene_symbol IS NOT NULL AND pp.protein1_degree IS NOT NULL
            GROUP BY p.gene_symbol
        """).fetchall()
    finally:
        conn.close()

    return {row[0]: row[1] for row in rows}


def load_predictions(results_dir: Path) -> dict[str, dict[str, str]]:
    """Load predictions for all L4 runs.

    Returns:
        Dict mapping run_name → {question_id: prediction_text}
    """
    all_preds = {}
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith("ppi-l4_"):
            continue
        pred_file = run_dir / "predictions.jsonl"
        if not pred_file.exists():
            continue

        preds = {}
        with open(pred_file) as f:
            for line in f:
                rec = json.loads(line)
                qid = rec.get("question_id")
                pred = rec.get("prediction", "")
                if qid:
                    preds[qid] = pred
        all_preds[run_dir.name] = preds

    return all_preds


def parse_l4_prediction(text: str) -> str | None:
    """Parse L4 prediction to 'tested' or 'untested'."""
    if not text:
        return None
    low = text.strip().lower()
    if "tested" in low and "untested" not in low and "not tested" not in low:
        return "tested"
    if "untested" in low or "not tested" in low:
        return "untested"
    return None


def parse_run_name(name: str) -> tuple[str, str] | None:
    """Extract (model, config) from run name."""
    m = re.match(r"ppi-l4_(.+?)_(zero-shot|3-shot)(?:_fs\d+)?$", name)
    if m:
        return m.group(1), m.group(2)
    return None


def analyze(
    dataset: list[dict],
    degrees: dict[str, float],
    all_preds: dict[str, dict[str, str]],
) -> str:
    """Run the contamination vs popularity analysis."""
    lines = ["# Contamination vs Protein Popularity Analysis (PPI-L4)", ""]
    lines.append("**Question:** Is the temporal contamination gap driven by "
                 "memorization or protein popularity?")
    lines.append("")

    # Enrich dataset with degree info
    tested_records = [r for r in dataset if r.get("temporal_group") in ("pre_2015", "post_2020")]

    pair_degrees = []
    for rec in tested_records:
        meta = rec.get("metadata", {})
        g1 = meta.get("gene_symbol_1", "")
        g2 = meta.get("gene_symbol_2", "")
        d1 = degrees.get(g1, 0)
        d2 = degrees.get(g2, 0)
        pair_degrees.append((d1 + d2) / 2.0)

    if not pair_degrees:
        return "\n".join(lines + ["No tested records with degree data found."])

    median_deg = float(np.median(pair_degrees))
    lines.append(f"**Median pair degree:** {median_deg:.1f}")
    lines.append(f"**Tested records with temporal group:** {len(tested_records)}")
    lines.append("")

    # Per-model analysis
    header = "| Model | Config | Pre-2015 High | Pre-2015 Low | Post-2020 High | Post-2020 Low | Gap High | Gap Low |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines.extend([header, sep])

    for run_name, preds in sorted(all_preds.items()):
        parsed = parse_run_name(run_name)
        if not parsed:
            continue
        model, config = parsed

        # Compute accuracy in 4 cells: temporal_group × degree_bin
        cells = {}
        for group in ["pre_2015", "post_2020"]:
            for deg_bin in ["high", "low"]:
                cells[(group, deg_bin)] = {"correct": 0, "total": 0}

        for rec, avg_deg in zip(tested_records, pair_degrees):
            qid = rec["question_id"]
            pred_text = preds.get(qid)
            if pred_text is None:
                continue

            parsed_pred = parse_l4_prediction(pred_text)
            if parsed_pred is None:
                continue

            group = rec["temporal_group"]
            deg_bin = "high" if avg_deg >= median_deg else "low"
            cells[(group, deg_bin)]["total"] += 1
            if parsed_pred == rec["gold_answer"]:
                cells[(group, deg_bin)]["correct"] += 1

        def acc(g: str, d: str) -> float | None:
            c = cells[(g, d)]
            return c["correct"] / c["total"] if c["total"] > 0 else None

        pre_h = acc("pre_2015", "high")
        pre_l = acc("pre_2015", "low")
        post_h = acc("post_2020", "high")
        post_l = acc("post_2020", "low")

        gap_h = (pre_h - post_h) if pre_h is not None and post_h is not None else None
        gap_l = (pre_l - post_l) if pre_l is not None and post_l is not None else None

        def fmt(v: float | None) -> str:
            return f"{v:.3f}" if v is not None else "—"

        lines.append(
            f"| {model} | {config} | {fmt(pre_h)} | {fmt(pre_l)} | "
            f"{fmt(post_h)} | {fmt(post_l)} | {fmt(gap_h)} | {fmt(gap_l)} |"
        )

    # Model-averaged summary (aggregate 3-shot fs0/fs1/fs2 → mean)
    from collections import defaultdict
    model_gaps: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run_name, preds in sorted(all_preds.items()):
        parsed = parse_run_name(run_name)
        if not parsed:
            continue
        model, config = parsed
        cells_local: dict[tuple[str, str], dict[str, int]] = {}
        for group in ["pre_2015", "post_2020"]:
            for deg_bin in ["high", "low"]:
                cells_local[(group, deg_bin)] = {"correct": 0, "total": 0}
        for rec, avg_deg in zip(tested_records, pair_degrees):
            qid = rec["question_id"]
            pred_text = preds.get(qid)
            if pred_text is None:
                continue
            parsed_pred = parse_l4_prediction(pred_text)
            if parsed_pred is None:
                continue
            group = rec["temporal_group"]
            deg_bin = "high" if avg_deg >= median_deg else "low"
            cells_local[(group, deg_bin)]["total"] += 1
            if parsed_pred == rec["gold_answer"]:
                cells_local[(group, deg_bin)]["correct"] += 1
        def acc_local(g: str, d: str) -> float | None:
            c = cells_local[(g, d)]
            return c["correct"] / c["total"] if c["total"] > 0 else None
        gh = acc_local("pre_2015", "high")
        gl = acc_local("pre_2015", "low")
        ph = acc_local("post_2020", "high")
        pl = acc_local("post_2020", "low")
        if gh is not None and ph is not None:
            model_gaps[model]["gap_high"].append(gh - ph)
        if gl is not None and pl is not None:
            model_gaps[model]["gap_low"].append(gl - pl)

    lines.extend(["", "## Model-Averaged Summary", ""])
    lines.append("| Model | Avg Gap High | Avg Gap Low | Verdict |")
    lines.append("|---|---|---|---|")
    for model in sorted(model_gaps.keys()):
        gh_vals = model_gaps[model]["gap_high"]
        gl_vals = model_gaps[model]["gap_low"]
        avg_gh = float(np.mean(gh_vals)) if gh_vals else None
        avg_gl = float(np.mean(gl_vals)) if gl_vals else None
        if avg_gh is not None and avg_gl is not None:
            if avg_gh > 0.15 and avg_gl > 0.15:
                if avg_gl >= avg_gh:
                    verdict = "True contamination (stronger for obscure)"
                else:
                    verdict = "True contamination"
            elif avg_gh > 0.15:
                verdict = "Popularity confound"
            else:
                verdict = "No significant gap"
        else:
            verdict = "Insufficient data"
        fmt_gh = f"{avg_gh:.3f}" if avg_gh is not None else "—"
        fmt_gl = f"{avg_gl:.3f}" if avg_gl is not None else "—"
        lines.append(f"| {model} | {fmt_gh} | {fmt_gl} | {verdict} |")

    # Interpretation
    lines.extend(["", "## Interpretation Guide", ""])
    lines.append("- **Gap High > 0.15 AND Gap Low > 0.15:** True contamination "
                 "(memorization persists regardless of protein popularity)")
    lines.append("- **Gap High > 0.15 BUT Gap Low ≈ 0:** Popularity confound "
                 "(well-studied proteins drive the gap)")
    lines.append("- **Gap Low > Gap High:** Contamination stronger for "
                 "obscure proteins (pure memorization signal)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="PPI-L4 contamination vs popularity")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    args = parser.parse_args()

    print("Loading L4 dataset...")
    dataset = load_l4_dataset(args.dataset)
    print(f"  {len(dataset)} records")

    print("Loading protein degrees from DB...")
    degrees = load_protein_degrees(args.db_path)
    print(f"  {len(degrees)} proteins with degree data")

    print("Loading predictions...")
    all_preds = load_predictions(args.results_dir)
    print(f"  {len(all_preds)} L4 runs found")

    result = analyze(dataset, degrees, all_preds)
    print(f"\n{result}")

    out_path = args.results_dir / "contamination_vs_popularity.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(result)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
