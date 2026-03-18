#!/usr/bin/env python3
"""Build L1 MCQ dataset for LLM benchmark.

Generates 2,000 multiple-choice questions across 4 classes:
  A) Active    (400) — confirmed active, ChEMBL positives pChEMBL ≥ 6
  B) Inactive  (800) — confirmed inactive, NegBioDB silver tier
  C) Inconclusive (400) — ambiguous evidence (bronze tier, borderline)
  D) Conditionally active (400) — cross-target selectivity compounds

Difficulty: Easy 40% / Medium 35% / Hard 25%
Split: 200 few-shot (50/class) + 200 val (50/class) + 1,600 test

Output: exports/llm_benchmarks/l1_mcq.jsonl
"""

import argparse
import json
import random
import sqlite3
import time
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEGBIODB_PATH = PROJECT_ROOT / "data" / "negbiodb.db"
CHEMBL_PATH = PROJECT_ROOT / "data" / "chembl" / "chembl_36.db"
POSITIVES_PATH = PROJECT_ROOT / "exports" / "chembl_positives_pchembl6.parquet"
M1_PATH = PROJECT_ROOT / "exports" / "negbiodb_m1_balanced.parquet"
NAMES_PATH = PROJECT_ROOT / "exports" / "compound_names.parquet"
OUTPUT_PATH = PROJECT_ROOT / "exports" / "llm_benchmarks" / "l1_mcq.jsonl"

# Class sizes
N_ACTIVE = 400
N_INACTIVE = 800
N_INCONCLUSIVE = 400
N_CONDITIONAL = 400

# L-7: Max times a compound can appear within a single class.
# Set to 12 to accommodate DAVIS kinase panel (68 compounds × 375 targets).
# Prevents extreme dominance while allowing natural assay panel structure.
MAX_PER_COMPOUND = 12

# Difficulty proportions
FRAC_EASY = 0.40
FRAC_MEDIUM = 0.35
FRAC_HARD = 0.25


def load_compound_names() -> dict:
    """Load compound name cache: compound_id -> pref_name."""
    df = pd.read_parquet(NAMES_PATH)
    # Build lookup by compound_id
    id_names = {}
    for _, row in df.iterrows():
        if pd.notna(row["pref_name"]):
            id_names[int(row["compound_id"])] = row["pref_name"]
    return id_names


def load_target_info(db_path: Path) -> dict:
    """Load target info: target_id -> {uniprot, gene_symbol, family}."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT target_id, uniprot_accession, gene_symbol, target_family "
        "FROM targets"
    ).fetchall()
    conn.close()
    return {
        r[0]: {"uniprot": r[1], "gene_symbol": r[2], "family": r[3]}
        for r in rows
    }


def load_compound_ids(db_path: Path) -> dict:
    """Load compound lookup: inchikey -> compound_id and chembl_id -> compound_id."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT compound_id, inchikey, chembl_id, pubchem_cid FROM compounds"
    ).fetchall()
    conn.close()
    ik_map = {}
    chembl_map = {}
    for cid, ik, chembl, pcid in rows:
        if ik:
            ik_map[ik] = cid
        if chembl:
            chembl_map[chembl] = cid
    return ik_map, chembl_map


def fetch_pubchem_names(cids: list[int]) -> dict[int, str]:
    """Fetch names for PubChem CIDs not in cache."""
    if not cids:
        return {}
    all_names = {}
    batch_size = 100
    for i in range(0, len(cids), batch_size):
        batch = cids[i : i + batch_size]
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/synonyms/JSON"
        data = f"cid={','.join(str(c) for c in batch)}".encode()
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            for entry in result.get("InformationList", {}).get("Information", []):
                cid = entry.get("CID")
                synonyms = entry.get("Synonym", [])
                if cid and synonyms and not synonyms[0].isdigit():
                    all_names[cid] = synonyms[0]
        except Exception:
            pass
        time.sleep(0.3)
    return all_names


# ── Class A: Active ──────────────────────────────────────────────────────────


def select_active(positives_df: pd.DataFrame, names: dict,
                  ik_to_cid: dict, n: int, seed: int) -> list[dict]:
    """Select n active compound-target pairs from ChEMBL positives."""
    rng = random.Random(seed)
    df = positives_df.copy()

    # Map inchikey -> compound_id for name lookup
    df["compound_id"] = df["inchikey"].map(ik_to_cid)

    # Filter to compounds with names (for better MCQ quality)
    df["compound_name"] = df["compound_id"].map(names)
    named = df[df["compound_name"].notna()].copy()
    print(f"  Active: {len(named)}/{len(df)} have names")

    # Stratify by difficulty (based on pchembl)
    easy = named[named["pchembl_value"] > 7.5].copy()
    medium = named[
        (named["pchembl_value"] > 6.5) & (named["pchembl_value"] <= 7.5)
    ].copy()
    hard = named[named["pchembl_value"] <= 6.5].copy()

    n_easy = int(n * FRAC_EASY)
    n_medium = int(n * FRAC_MEDIUM)
    n_hard = n - n_easy - n_medium

    # Sample from each difficulty band
    selected = []
    for subset, count, diff in [
        (easy, n_easy, "easy"),
        (medium, n_medium, "medium"),
        (hard, n_hard, "hard"),
    ]:
        # Diversify: max 1 per UniProt to spread across targets
        by_target = subset.groupby("uniprot_id")
        pool = []
        for _, group in by_target:
            pool.append(group.sample(1, random_state=seed))
        if pool:
            pool_df = pd.concat(pool)
            if len(pool_df) < count:
                # Need more — allow duplicates per target
                extra = subset[~subset.index.isin(pool_df.index)]
                pool_df = pd.concat([pool_df, extra]).head(count * 3)
            sampled = pool_df.sample(min(count, len(pool_df)), random_state=seed)
        else:
            sampled = subset.sample(min(count, len(subset)), random_state=seed)

        for _, row in sampled.iterrows():
            selected.append(
                {
                    "class": "active",
                    "correct_answer": "A",
                    "difficulty": diff,
                    "compound_name": row["compound_name"],
                    "compound_smiles": row["smiles"],
                    "compound_inchikey": row["inchikey"],
                    "target_uniprot": row["uniprot_id"],
                    "activity_type": row["activity_type"],
                    "activity_value_nm": float(row["activity_value_nm"]),
                    "pchembl_value": float(row["pchembl_value"]),
                    "publication_year": (
                        int(row["publication_year"])
                        if pd.notna(row.get("publication_year"))
                        else None
                    ),
                    "evidence_quality": "gold",
                    "source_db": "ChEMBL",
                }
            )

    rng.shuffle(selected)
    return selected[:n]


# ── Class B: Inactive ────────────────────────────────────────────────────────


def select_inactive(
    db_path: Path, names: dict, target_info: dict, n: int, seed: int
) -> list[dict]:
    """Select n inactive pairs from NegBioDB silver tier."""
    conn = sqlite3.connect(str(db_path))

    # Query silver-tier pairs for compounds with chembl_id (ensures names)
    rows = conn.execute(
        """
        SELECT ctp.compound_id, ctp.target_id,
               c.canonical_smiles, c.inchikey,
               ctp.num_assays, ctp.num_sources, ctp.median_pchembl,
               ctp.min_activity_value, ctp.max_activity_value,
               ctp.earliest_year, ctp.best_confidence
        FROM compound_target_pairs ctp
        JOIN compounds c ON ctp.compound_id = c.compound_id
        WHERE ctp.best_confidence = 'silver'
          AND c.chembl_id IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n * 10,),
    ).fetchall()
    conn.close()

    cols = [
        "compound_id", "target_id", "smiles", "inchikey",
        "num_assays", "num_sources", "median_pchembl",
        "min_activity_value", "max_activity_value",
        "earliest_year", "best_confidence",
    ]
    df = pd.DataFrame(rows, columns=cols)

    # Add names and target info
    df["compound_name"] = df["compound_id"].map(names)
    named = df[df["compound_name"].notna()].copy()
    print(f"  Inactive: {len(named)}/{len(df)} have names")

    # Add target gene symbols
    named["gene_symbol"] = named["target_id"].map(
        lambda tid: target_info.get(tid, {}).get("gene_symbol")
    )
    named["target_family"] = named["target_id"].map(
        lambda tid: target_info.get(tid, {}).get("family")
    )
    named["target_uniprot"] = named["target_id"].map(
        lambda tid: target_info.get(tid, {}).get("uniprot")
    )

    # Difficulty based on evidence strength
    # Easy: many assays, clear inactive (low pchembl or null)
    # Medium: fewer assays, moderate evidence
    # Hard: single assay, near threshold
    named["difficulty"] = "medium"
    named.loc[named["num_assays"] >= 3, "difficulty"] = "easy"
    named.loc[named["num_sources"] >= 2, "difficulty"] = "easy"
    named.loc[
        (named["num_assays"] == 1) & (named["num_sources"] == 1), "difficulty"
    ] = "hard"

    n_easy = int(n * FRAC_EASY)
    n_medium = int(n * FRAC_MEDIUM)
    n_hard = n - n_easy - n_medium

    selected = []
    seen_compounds = set()
    for diff, count in [("easy", n_easy), ("medium", n_medium), ("hard", n_hard)]:
        pool = named[named["difficulty"] == diff]
        # Diversify compounds
        pool = pool[~pool["compound_id"].isin(seen_compounds)]
        sampled = pool.sample(min(count, len(pool)), random_state=seed)
        seen_compounds.update(sampled["compound_id"])

        for _, row in sampled.iterrows():
            activity_desc = _format_activity(
                row["min_activity_value"], row["max_activity_value"],
                row["median_pchembl"]
            )
            selected.append(
                {
                    "class": "inactive",
                    "correct_answer": "B",
                    "difficulty": diff,
                    "compound_name": row["compound_name"],
                    "compound_smiles": row["smiles"],
                    "compound_inchikey": row["inchikey"],
                    "target_uniprot": row["target_uniprot"],
                    "target_gene": row["gene_symbol"],
                    "target_family": row["target_family"],
                    "num_assays": int(row["num_assays"]),
                    "num_sources": int(row["num_sources"]),
                    "activity_description": activity_desc,
                    "pchembl_value": (
                        float(row["median_pchembl"])
                        if pd.notna(row["median_pchembl"])
                        else None
                    ),
                    "publication_year": (
                        int(row["earliest_year"])
                        if pd.notna(row["earliest_year"])
                        else None
                    ),
                    "evidence_quality": "silver",
                    "source_db": "NegBioDB",
                }
            )

    random.Random(seed).shuffle(selected)
    return selected[:n]


def _format_activity(min_val, max_val, median_pchembl):
    """Format activity description for inactive pairs."""
    parts = []
    if pd.notna(min_val):
        if min_val >= 10000:
            parts.append("No significant activity at 10 µM")
        elif min_val >= 1000:
            parts.append(f"Weak activity (>{min_val:.0f} nM)")
        else:
            # min_val < 1000 nM in an inactive pair means inconsistent assay results
            parts.append(
                f"Best measurement: {min_val:.0f} nM "
                f"(inconsistent across assays; classified inactive at 10 µM threshold)"
            )
    if pd.notna(median_pchembl) and median_pchembl > 0:
        parts.append(f"pChEMBL: {median_pchembl:.1f}")
    if not parts:
        parts.append("Inactive (below detection threshold)")
    return "; ".join(parts)


# ── Class C: Inconclusive ────────────────────────────────────────────────────


def select_inconclusive(
    db_path: Path, names: dict, target_info: dict, n: int, seed: int
) -> list[dict]:
    """Select n inconclusive pairs (bronze tier + borderline silver)."""
    conn = sqlite3.connect(str(db_path))

    # Part 1: Bronze tier (DAVIS) — all are single-assay Kd at threshold
    # DAVIS compounds have no chembl_id, so fetch PubChem names on demand
    bronze_rows = conn.execute(
        """
        SELECT ctp.compound_id, ctp.target_id,
               c.canonical_smiles, c.inchikey, c.pubchem_cid,
               ctp.num_assays, ctp.num_sources, ctp.median_pchembl,
               ctp.min_activity_value, ctp.max_activity_value,
               ctp.earliest_year
        FROM compound_target_pairs ctp
        JOIN compounds c ON ctp.compound_id = c.compound_id
        WHERE ctp.best_confidence = 'bronze'
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n * 3,),
    ).fetchall()

    # Fetch PubChem names for DAVIS compounds missing from cache
    bronze_cids_needing_names = set()
    for row in bronze_rows:
        cid = row[0]  # compound_id
        pcid = row[4]  # pubchem_cid
        if cid not in names and pcid:
            bronze_cids_needing_names.add(int(pcid))

    if bronze_cids_needing_names:
        print(f"  Fetching PubChem names for {len(bronze_cids_needing_names)} DAVIS compounds...")
        pc_names = fetch_pubchem_names(list(bronze_cids_needing_names))
        # Map pubchem_cid -> compound_id for update
        pcid_to_compid = {
            int(row[4]): row[0]
            for row in bronze_rows
            if row[4]
        }
        for pcid, name in pc_names.items():
            if pcid in pcid_to_compid:
                names[pcid_to_compid[pcid]] = name

    # Part 2: Borderline silver — single assay, activity near threshold, named
    borderline_rows = conn.execute(
        """
        SELECT ctp.compound_id, ctp.target_id,
               c.canonical_smiles, c.inchikey, c.pubchem_cid,
               ctp.num_assays, ctp.num_sources, ctp.median_pchembl,
               ctp.min_activity_value, ctp.max_activity_value,
               ctp.earliest_year
        FROM compound_target_pairs ctp
        JOIN compounds c ON ctp.compound_id = c.compound_id
        WHERE ctp.best_confidence = 'silver'
          AND c.chembl_id IS NOT NULL
          AND ctp.num_assays = 1
          AND ctp.num_sources = 1
          AND ctp.min_activity_value BETWEEN 5000 AND 15000
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n * 5,),
    ).fetchall()
    conn.close()

    cols = [
        "compound_id", "target_id", "smiles", "inchikey", "pubchem_cid",
        "num_assays", "num_sources", "median_pchembl",
        "min_activity_value", "max_activity_value", "earliest_year",
    ]

    bronze_df = pd.DataFrame(bronze_rows, columns=cols)
    bronze_df["inconclusive_reason"] = "single_assay_bronze"

    borderline_df = pd.DataFrame(borderline_rows, columns=cols)
    borderline_df["inconclusive_reason"] = "borderline_threshold"

    df = pd.concat([bronze_df, borderline_df], ignore_index=True)

    # Add names
    df["compound_name"] = df["compound_id"].map(names)
    named = df[df["compound_name"].notna()].copy()
    print(f"  Inconclusive: {len(named)}/{len(df)} have names")

    # If not enough named, use unnamed with SMILES
    if len(named) < n:
        unnamed = df[df["compound_name"].isna()].copy()
        unnamed["compound_name"] = unnamed["smiles"].str[:50] + "..."
        named = pd.concat([named, unnamed])

    # Add target info
    named["gene_symbol"] = named["target_id"].map(
        lambda tid: target_info.get(tid, {}).get("gene_symbol")
    )
    named["target_family"] = named["target_id"].map(
        lambda tid: target_info.get(tid, {}).get("family")
    )
    named["target_uniprot"] = named["target_id"].map(
        lambda tid: target_info.get(tid, {}).get("uniprot")
    )

    # Difficulty: bronze = medium (single-assay, ambiguous evidence),
    # borderline threshold = hard (near 10 µM cutoff, requires nuanced judgment)
    # No "easy" tier: inconclusive cases inherently require careful interpretation
    named["difficulty"] = named["inconclusive_reason"].map(
        {"single_assay_bronze": "medium", "borderline_threshold": "hard"}
    )

    # Sample: 60% medium (bronze), 40% hard (borderline)
    n_medium = int(n * 0.6)
    n_hard = n - n_medium

    selected = []
    for diff, count in [("medium", n_medium), ("hard", n_hard)]:
        pool = named[named["difficulty"] == diff]
        sampled = pool.sample(min(count, len(pool)), random_state=seed)
        for _, row in sampled.iterrows():
            reason = (
                "Single Kd measurement at threshold (DAVIS kinase panel)"
                if row["inconclusive_reason"] == "single_assay_bronze"
                else "Borderline activity near 10 µM threshold, single assay"
            )
            selected.append(
                {
                    "class": "inconclusive",
                    "correct_answer": "C",
                    "difficulty": diff,
                    "compound_name": row["compound_name"],
                    "compound_smiles": row["smiles"],
                    "compound_inchikey": row["inchikey"],
                    "target_uniprot": row["target_uniprot"],
                    "target_gene": row["gene_symbol"],
                    "target_family": row["target_family"],
                    "num_assays": int(row["num_assays"]),
                    "num_sources": int(row["num_sources"]),
                    "activity_description": reason,
                    "pchembl_value": (
                        float(row["median_pchembl"])
                        if pd.notna(row["median_pchembl"])
                        else None
                    ),
                    "evidence_quality": (
                        "bronze"
                        if row["inconclusive_reason"] == "single_assay_bronze"
                        else "silver"
                    ),
                    "source_db": (
                        "DAVIS"
                        if row["inconclusive_reason"] == "single_assay_bronze"
                        else "NegBioDB"
                    ),
                }
            )

    random.Random(seed).shuffle(selected)
    return selected[:n]


# ── Class D: Conditional ─────────────────────────────────────────────────────


def select_conditional(
    m1_path: Path,
    positives_df: pd.DataFrame,
    names: dict,
    ik_to_cid: dict,
    target_info: dict,
    n: int,
    seed: int,
) -> list[dict]:
    """Select n conditional (cross-target selectivity) pairs.

    These are compounds active against some targets but inactive against others.
    """
    m1 = pd.read_parquet(m1_path, columns=["smiles", "inchikey", "uniprot_id", "Y"])

    # Find compounds that appear as both active and inactive
    compound_labels = m1.groupby("inchikey")["Y"].agg(["sum", "count"])
    compound_labels.columns = ["n_active", "n_total"]
    compound_labels["n_inactive"] = (
        compound_labels["n_total"] - compound_labels["n_active"]
    )

    # Cross-target selectivity: active in ≥1, inactive in ≥1
    cross_target = compound_labels[
        (compound_labels["n_active"] >= 1) & (compound_labels["n_inactive"] >= 1)
    ].index.tolist()

    print(f"  Conditional: {len(cross_target)} cross-target selectivity compounds")

    # Map to compound_ids for name lookup
    cross_iks = set(cross_target)
    m1_cross = m1[m1["inchikey"].isin(cross_iks)].copy()
    m1_cross["compound_id"] = m1_cross["inchikey"].map(ik_to_cid)
    m1_cross["compound_name"] = m1_cross["compound_id"].map(names)

    # Filter to named compounds
    named_iks = set(
        m1_cross.loc[m1_cross["compound_name"].notna(), "inchikey"].unique()
    )
    print(f"  Conditional: {len(named_iks)} with names")

    # For each named cross-target compound, get its active and inactive targets
    selected = []
    rng = random.Random(seed)
    shuffled_iks = list(named_iks)
    rng.shuffle(shuffled_iks)

    for ik in shuffled_iks:
        if len(selected) >= n:
            break

        comp_data = m1_cross[m1_cross["inchikey"] == ik]
        active_targets = comp_data[comp_data["Y"] == 1]["uniprot_id"].tolist()
        inactive_targets = comp_data[comp_data["Y"] == 0]["uniprot_id"].tolist()

        if not active_targets or not inactive_targets:
            continue

        compound_name = comp_data["compound_name"].iloc[0]
        smiles = comp_data["smiles"].iloc[0]
        compound_id = comp_data["compound_id"].iloc[0]

        # Pick one inactive target for the question
        inactive_t = rng.choice(inactive_targets)
        # Get active target names for context
        active_genes = []
        for at in active_targets[:3]:  # Show up to 3 active targets
            info = _find_target_by_uniprot(target_info, at)
            if info and info.get("gene_symbol"):
                active_genes.append(info["gene_symbol"])

        inactive_info = _find_target_by_uniprot(target_info, inactive_t)

        # Difficulty based on number of targets
        if len(active_targets) >= 5 and len(inactive_targets) >= 3:
            difficulty = "hard"
        elif len(active_targets) >= 2:
            difficulty = "medium"
        else:
            difficulty = "hard"  # few targets = harder to reason about

        active_context = (
            f"Known active against: {', '.join(active_genes)}"
            if active_genes
            else f"Active against {len(active_targets)} other target(s)"
        )

        selected.append(
            {
                "class": "conditional",
                "correct_answer": "D",
                "difficulty": difficulty,
                "compound_name": compound_name,
                "compound_smiles": smiles,
                "compound_inchikey": ik,
                "target_uniprot": inactive_t,
                "target_gene": (
                    inactive_info["gene_symbol"] if inactive_info else None
                ),
                "target_family": (
                    inactive_info["family"] if inactive_info else None
                ),
                "num_active_targets": len(active_targets),
                "num_inactive_targets": len(inactive_targets),
                "active_context": active_context,
                "evidence_quality": "silver",
                "source_db": "NegBioDB+ChEMBL",
            }
        )

    rng.shuffle(selected)
    return selected[:n]


def _find_target_by_uniprot(target_info: dict, uniprot: str) -> dict | None:
    """Find target info by UniProt accession."""
    for tid, info in target_info.items():
        if info["uniprot"] == uniprot:
            return info
    return None


# ── Assembly ─────────────────────────────────────────────────────────────────


def add_target_info_to_active(records: list[dict], target_info: dict):
    """Add target gene/family to active records."""
    for rec in records:
        info = _find_target_by_uniprot(target_info, rec["target_uniprot"])
        if info:
            rec["target_gene"] = info.get("gene_symbol")
            rec["target_family"] = info.get("family")
        else:
            rec["target_gene"] = None
            rec["target_family"] = None


def generate_context_text(record: dict) -> str:
    """Generate the MCQ prompt context from a record.

    Design intent: This is an assay data interpretation test. The prompt
    deliberately includes activity measurements (pChEMBL, assay counts,
    activity descriptions) because the task tests whether the LLM can
    correctly interpret bioactivity data, not whether it can guess from
    compound/target names alone.
    """
    name = record.get("compound_name", "Unknown")
    smiles = record.get("compound_smiles", "")
    gene = record.get("target_gene")
    uniprot = record.get("target_uniprot", "Unknown")
    family = record.get("target_family") or "protein"

    # Target display: "EGFR (P00533), kinase" or "P00533, protein" if no gene
    if gene:
        target_str = f"{gene} ({uniprot}), {family}"
    else:
        target_str = f"{uniprot}, {family}"

    lines = [
        f"Compound: {name}",
        f"SMILES: {smiles}",
        f"Target: {target_str}",
    ]

    cls = record["class"]
    if cls == "active":
        act_type = record.get("activity_type", "IC50")
        act_val = record.get("activity_value_nm", 0)
        pchembl = record.get("pchembl_value", 0)
        lines.append(f"Activity: {act_type} = {act_val:.1f} nM (pChEMBL {pchembl:.2f})")
    elif cls == "inactive":
        desc = record.get("activity_description", "Inactive")
        n_assays = record.get("num_assays", 1)
        n_src = record.get("num_sources", 1)
        lines.append(f"Result: {desc}")
        lines.append(f"Evidence: {n_assays} assay(s), {n_src} source(s)")
    elif cls == "inconclusive":
        desc = record.get("activity_description", "Inconclusive")
        lines.append(f"Result: {desc}")
        lines.append(f"Evidence: {record.get('num_assays', 1)} assay, single source")
    elif cls == "conditional":
        ctx = record.get("active_context", "")
        lines.append(f"Context: {ctx}")
        lines.append("Tested against this target: no significant activity at 10 µM")

    lines.append("")
    lines.append("What is the most likely interaction outcome for this compound-target pair?")
    lines.append("A) Active  B) Inactive  C) Inconclusive  D) Conditionally active")

    return "\n".join(lines)


def split_dataset(records: list[dict], seed: int) -> list[dict]:
    """Assign split: 200 fewshot (50/class), 200 val (50/class), rest test."""
    rng = random.Random(seed)

    by_class = {}
    for rec in records:
        by_class.setdefault(rec["class"], []).append(rec)

    for cls_records in by_class.values():
        rng.shuffle(cls_records)

    for cls, cls_records in by_class.items():
        for i, rec in enumerate(cls_records):
            if i < 50:
                rec["split"] = "fewshot"
            elif i < 100:
                rec["split"] = "val"
            else:
                rec["split"] = "test"

    all_records = []
    for cls_records in by_class.values():
        all_records.extend(cls_records)

    rng.shuffle(all_records)
    return all_records


def main():
    parser = argparse.ArgumentParser(description="Build L1 MCQ dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    print("Loading data sources...")
    names = load_compound_names()
    print(f"  Compound names: {len(names)}")

    target_info = load_target_info(NEGBIODB_PATH)
    print(f"  Targets: {len(target_info)}")

    ik_to_cid, chembl_to_cid = load_compound_ids(NEGBIODB_PATH)
    print(f"  InChIKey map: {len(ik_to_cid)}")

    positives = pd.read_parquet(POSITIVES_PATH)
    print(f"  ChEMBL positives: {len(positives)}")

    # ── Select each class ──
    print("\nSelecting Active (A)...")
    active = select_active(positives, names, ik_to_cid, N_ACTIVE, seed)
    add_target_info_to_active(active, target_info)
    print(f"  Selected: {len(active)}")

    print("\nSelecting Inactive (B)...")
    inactive = select_inactive(NEGBIODB_PATH, names, target_info, N_INACTIVE, seed)
    print(f"  Selected: {len(inactive)}")

    print("\nSelecting Inconclusive (C)...")
    inconclusive = select_inconclusive(
        NEGBIODB_PATH, names, target_info, N_INCONCLUSIVE, seed
    )
    print(f"  Selected: {len(inconclusive)}")

    print("\nSelecting Conditional (D)...")
    conditional = select_conditional(
        M1_PATH, positives, names, ik_to_cid, target_info, N_CONDITIONAL, seed
    )
    print(f"  Selected: {len(conditional)}")

    # ── C-2: Cross-class dedup + L-7: Per-class compound repetition cap ──
    # Remove compound-target pairs that appear in multiple classes.
    # Priority: active > inactive > inconclusive > conditional.
    # Compound cap is per-class (not global) because the same compound
    # appearing as active against target A and inconclusive against target B
    # is scientifically valid (selectivity) and IS what L1 tests.
    used_pairs = set()      # (inchikey[:14], uniprot) pairs already used

    def _dedup_class(records, class_name):
        """Filter records removing cross-class pair conflicts and applying per-class compound cap."""
        kept = []
        removed_pair = 0
        removed_cap = 0
        class_compound_counts = {}  # per-class compound cap
        for rec in records:
            ik = rec.get("compound_inchikey", "")
            uni = rec.get("target_uniprot", "")
            ik14 = ik[:14] if ik else ""
            pair = (ik14, uni)

            if pair in used_pairs:
                removed_pair += 1
                continue
            if ik14 and class_compound_counts.get(ik14, 0) >= MAX_PER_COMPOUND:
                removed_cap += 1
                continue

            kept.append(rec)
            used_pairs.add(pair)
            class_compound_counts[ik14] = class_compound_counts.get(ik14, 0) + 1

        if removed_pair or removed_cap:
            print(f"  {class_name}: removed {removed_pair} pair conflicts, "
                  f"{removed_cap} compound cap violations")
        return kept

    active = _dedup_class(active, "Active")
    inactive = _dedup_class(inactive, "Inactive")
    inconclusive = _dedup_class(inconclusive, "Inconclusive")
    conditional = _dedup_class(conditional, "Conditional")

    # ── Assemble ──
    all_records = active + inactive + inconclusive + conditional
    total = len(all_records)
    print(f"\nTotal records: {total} (after dedup)")

    # Generate context text
    for rec in all_records:
        rec["context_text"] = generate_context_text(rec)

    # Assign splits
    all_records = split_dataset(all_records, seed)

    # Add question IDs
    for i, rec in enumerate(all_records):
        rec["question_id"] = f"L1-{i:04d}"

    # ── Summary ──
    print("\n=== Dataset Summary ===")
    class_counts = {}
    diff_counts = {}
    split_counts = {}
    for rec in all_records:
        class_counts[rec["class"]] = class_counts.get(rec["class"], 0) + 1
        diff_counts[rec["difficulty"]] = diff_counts.get(rec["difficulty"], 0) + 1
        split_counts[rec["split"]] = split_counts.get(rec["split"], 0) + 1

    print(f"Classes: {class_counts}")
    print(f"Difficulty: {diff_counts}")
    print(f"Splits: {split_counts}")

    # ── Save ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  {total} questions")


if __name__ == "__main__":
    main()
