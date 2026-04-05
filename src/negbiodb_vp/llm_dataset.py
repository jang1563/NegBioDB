"""Dataset builder utilities for VP LLM benchmark (VP-L1 through VP-L4).

Shared constants, SQL helpers, sampling, splitting, and I/O functions
used by all four build_vp_l{1..4}_dataset.py scripts.

Mirrors src/negbiodb_ppi/llm_dataset.py structure from PPI domain.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

MAX_PER_GENE = 10  # Prevent single gene dominating any class

# Few-shot set seeds (3 independent sets for variance) — re-exported from prompts
FEWSHOT_SEEDS = [42, 43, 44]

# Default VP DB path
DEFAULT_VP_DB_PATH = Path(__file__).parent.parent.parent / "data" / "negbiodb_vp.db"

# Classification → L1 gold answer mapping
CLASSIFICATION_TO_L1_ANSWER: dict[str, str] = {
    "pathogenic": "A",
    "likely_pathogenic": "A",  # Grouped with pathogenic
    "likely_benign": "B",
    "uncertain_significance": "C",
    "benign": "D",
    "benign/likely_benign": "D",
}

# Consequence type descriptions for context construction
CONSEQUENCE_DESCRIPTIONS: dict[str, str] = {
    "missense": "missense variant (amino acid substitution)",
    "nonsense": "nonsense variant (premature stop codon)",
    "synonymous": "synonymous variant (silent, no amino acid change)",
    "frameshift": "frameshift variant (reading frame disruption)",
    "splice": "splice site variant (affects mRNA splicing)",
    "inframe_indel": "in-frame insertion/deletion",
    "intronic": "intronic variant (non-coding region)",
}

# ACMG benign criteria descriptions
ACMG_CRITERIA_DESCRIPTIONS: dict[str, str] = {
    "BA1": "Allele frequency >5% in a population database (standalone benign)",
    "BS1": "Allele frequency greater than expected for disorder",
    "BS2": "Observed in a healthy adult with full penetrance expected at an early age",
    "BS3": "Well-established functional studies show no damaging effect",
    "BS4": "Lack of segregation in affected family members",
    "BP1": "Missense variant in a gene where only truncating variants cause disease",
    "BP2": "Observed in trans with a pathogenic variant for a fully penetrant dominant disorder",
    "BP3": "In-frame insertion/deletion in a repetitive region",
    "BP4": "Multiple computational predictions suggest no impact on gene product",
    "BP5": "Variant found in a case with an alternate molecular basis for disease",
    "BP6": "Reputable source reports variant as benign (deprecated)",
    "BP7": "Synonymous variant with no predicted splice impact",
}


# JSONL record fields
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
# DB query helpers
# ---------------------------------------------------------------------------


def load_vp_candidate_pool(
    db_path: Path,
    tier_filter: str | None = None,
    classification_filter: str | None = None,
    extra_where: str = "",
    require_scores: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load candidate negative result records from VP database.

    Parameters
    ----------
    db_path : Path to VP database
    tier_filter : e.g. "IN ('gold', 'silver')" or "= 'gold'"
    classification_filter : e.g. "= 'benign'" or "IN ('benign', 'likely_benign')"
    extra_where : additional SQL WHERE clauses (AND-joined)
    require_scores : if True, only return variants with CADD score
    limit : if set, randomly sample this many rows at SQL level

    Returns
    -------
    DataFrame with negative_results + variant/gene/disease annotations joined.
    """
    from negbiodb_vp.vp_db import get_connection

    where_parts = ["1=1"]
    if tier_filter:
        where_parts.append(f"nr.confidence_tier {tier_filter}")
    if classification_filter:
        where_parts.append(f"nr.classification {classification_filter}")
    if extra_where:
        where_parts.append(extra_where)
    if require_scores:
        where_parts.append("v.cadd_phred IS NOT NULL")
    where_clause = " AND ".join(where_parts)

    sql = f"""
    SELECT
        nr.result_id, nr.source_db, nr.confidence_tier,
        nr.classification, nr.evidence_type,
        nr.submission_year, nr.has_conflict,
        nr.num_benign_criteria,
        v.variant_id, v.chromosome, v.position,
        v.ref_allele, v.alt_allele,
        v.variant_type, v.consequence_type,
        v.hgvs_coding, v.hgvs_protein,
        v.clinvar_variation_id,
        v.gnomad_af_global, v.gnomad_af_nfe, v.gnomad_af_afr,
        v.gnomad_af_eas, v.gnomad_af_sas,
        v.cadd_phred, v.revel_score,
        v.alphamissense_score, v.alphamissense_class,
        v.phylop_score, v.gerp_score,
        v.sift_score, v.polyphen2_score,
        g.gene_id, g.gene_symbol, g.hgnc_id,
        g.pli_score, g.loeuf_score, g.missense_z,
        g.clingen_validity, g.gene_moi,
        d.disease_id, d.canonical_name AS disease_name,
        d.medgen_cui, d.inheritance_pattern
    FROM vp_negative_results nr
    JOIN variants v ON nr.variant_id = v.variant_id
    LEFT JOIN genes g ON v.gene_id = g.gene_id
    LEFT JOIN diseases d ON nr.disease_id = d.disease_id
    WHERE {where_clause}
    {"ORDER BY RANDOM() LIMIT " + str(limit) if limit else ""}
    """

    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()

    logger.info(
        "Loaded %d candidate records (tier_filter=%s, classification_filter=%s)",
        len(df), tier_filter, classification_filter,
    )
    return df


def load_variant_annotations(
    db_path: Path,
    variant_ids: list[int] | None = None,
) -> dict[int, dict]:
    """Load variant annotations as a dict keyed by variant_id."""
    from negbiodb_vp.vp_db import get_connection

    conn = get_connection(db_path)
    try:
        if variant_ids:
            placeholders = ",".join("?" * len(variant_ids))
            sql = f"""
            SELECT v.variant_id, v.chromosome, v.position, v.ref_allele, v.alt_allele,
                   v.variant_type, v.consequence_type, v.hgvs_coding, v.hgvs_protein,
                   v.gnomad_af_global, v.cadd_phred, v.revel_score,
                   v.alphamissense_score, v.alphamissense_class,
                   g.gene_symbol, g.pli_score, g.loeuf_score, g.clingen_validity
            FROM variants v
            LEFT JOIN genes g ON v.gene_id = g.gene_id
            WHERE v.variant_id IN ({placeholders})
            """
            rows = conn.execute(sql, variant_ids).fetchall()
        else:
            sql = """
            SELECT v.variant_id, v.chromosome, v.position, v.ref_allele, v.alt_allele,
                   v.variant_type, v.consequence_type, v.hgvs_coding, v.hgvs_protein,
                   v.gnomad_af_global, v.cadd_phred, v.revel_score,
                   v.alphamissense_score, v.alphamissense_class,
                   g.gene_symbol, g.pli_score, g.loeuf_score, g.clingen_validity
            FROM variants v
            LEFT JOIN genes g ON v.gene_id = g.gene_id
            """
            rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    cols = [
        "variant_id", "chromosome", "position", "ref_allele", "alt_allele",
        "variant_type", "consequence_type", "hgvs_coding", "hgvs_protein",
        "gnomad_af_global", "cadd_phred", "revel_score",
        "alphamissense_score", "alphamissense_class",
        "gene_symbol", "pli_score", "loeuf_score", "clingen_validity",
    ]
    return {
        row[0]: dict(zip(cols, row))
        for row in rows
    }


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------


def construct_l1_context(record: dict | pd.Series) -> str:
    """Build variant context for L1 classification prompt.

    Includes gene, variant notation, consequence, computational scores,
    population frequency, gene constraint, and inheritance pattern.
    """
    gene = record.get("gene_symbol", "Unknown")
    hgvs_c = record.get("hgvs_coding", "")
    hgvs_p = record.get("hgvs_protein", "")
    consequence = record.get("consequence_type", "unknown")
    cons_desc = CONSEQUENCE_DESCRIPTIONS.get(consequence, consequence)
    disease = record.get("disease_name", "not specified")
    inheritance = record.get("inheritance_pattern") or record.get("gene_moi") or "unknown"

    # Variant notation
    chrom = record.get("chromosome", "?")
    pos = record.get("position", "?")
    ref = record.get("ref_allele", "?")
    alt = record.get("alt_allele", "?")

    parts = [
        f"Gene: {gene}",
        f"Variant: chr{chrom}:{pos} {ref}>{alt}",
    ]
    if hgvs_c:
        parts.append(f"HGVS coding: {hgvs_c}")
    if hgvs_p:
        parts.append(f"HGVS protein: {hgvs_p}")
    parts.append(f"Consequence: {cons_desc}")
    parts.append(f"Condition: {disease}")
    parts.append(f"Inheritance pattern: {inheritance}")

    # Population frequency
    af = record.get("gnomad_af_global")
    if af is not None:
        parts.append(f"gnomAD global allele frequency: {af:.6f}")

    # Computational scores
    scores = []
    cadd = record.get("cadd_phred")
    if cadd is not None:
        scores.append(f"CADD Phred={cadd:.1f}")
    revel = record.get("revel_score")
    if revel is not None:
        scores.append(f"REVEL={revel:.3f}")
    am = record.get("alphamissense_score")
    if am is not None:
        am_class = record.get("alphamissense_class", "")
        scores.append(f"AlphaMissense={am:.3f} ({am_class})")
    phylop = record.get("phylop_score")
    if phylop is not None:
        scores.append(f"PhyloP={phylop:.2f}")
    if scores:
        parts.append(f"Computational scores: {', '.join(scores)}")

    # Gene constraint
    constraint = []
    pli = record.get("pli_score")
    if pli is not None:
        constraint.append(f"pLI={pli:.3f}")
    loeuf = record.get("loeuf_score")
    if loeuf is not None:
        constraint.append(f"LOEUF={loeuf:.3f}")
    missense_z = record.get("missense_z")
    if missense_z is not None:
        constraint.append(f"missense_z={missense_z:.2f}")
    if constraint:
        parts.append(f"Gene constraint: {', '.join(constraint)}")

    return "\n".join(parts)


def construct_l2_context(record: dict | pd.Series) -> str:
    """Build template-generated clinical genetics report for L2 extraction.

    Simulates a clinical report with variant interpretation details,
    ACMG criteria, and evidence descriptions.
    """
    gene = record.get("gene_symbol", "Unknown")
    hgvs_c = record.get("hgvs_coding") or "c.?"
    hgvs_p = record.get("hgvs_protein") or ""
    consequence = record.get("consequence_type", "unknown")
    disease = record.get("disease_name", "not specified")
    classification = record.get("classification", "likely_benign")

    # Parse ACMG criteria
    acmg_raw = record.get("acmg_criteria") or "[]"
    try:
        acmg_list = json.loads(acmg_raw) if isinstance(acmg_raw, str) else acmg_raw
    except (json.JSONDecodeError, TypeError):
        acmg_list = []

    af = record.get("gnomad_af_global")
    cadd = record.get("cadd_phred")
    revel = record.get("revel_score")

    # Build report
    lines = [
        f"CLINICAL VARIANT INTERPRETATION REPORT",
        f"",
        f"Patient variant: {gene} {hgvs_c}",
    ]
    if hgvs_p:
        lines.append(f"Protein change: {hgvs_p}")
    lines.extend([
        f"Variant type: {consequence}",
        f"Condition evaluated: {disease}",
        f"",
        f"Classification: {classification.replace('_', ' ').title()}",
        f"Classification method: ACMG/AMP",
        f"",
    ])

    # Evidence section
    lines.append("EVIDENCE SUMMARY:")
    if af is not None:
        if af > 0.01:
            lines.append(
                f"- Population frequency: This variant has a global allele frequency "
                f"of {af:.4f} in gnomAD, exceeding the BA1 threshold of 5% or BS1 "
                f"threshold for the condition."
            )
        else:
            lines.append(
                f"- Population frequency: gnomAD global AF = {af:.6f}."
            )

    if cadd is not None:
        if cadd < 15:
            lines.append(
                f"- Computational prediction: CADD Phred score of {cadd:.1f} "
                f"suggests low deleteriousness."
            )
        else:
            lines.append(
                f"- Computational prediction: CADD Phred score of {cadd:.1f}."
            )

    if revel is not None and consequence == "missense":
        if revel < 0.5:
            lines.append(
                f"- REVEL score: {revel:.3f} (below pathogenic threshold of 0.644), "
                f"supporting benign classification."
            )
        else:
            lines.append(f"- REVEL score: {revel:.3f}.")

    if acmg_list:
        criteria_text = ", ".join(acmg_list)
        lines.append(f"")
        lines.append(f"ACMG criteria applied: {criteria_text}")
        for code in acmg_list:
            desc = ACMG_CRITERIA_DESCRIPTIONS.get(code, "")
            if desc:
                lines.append(f"  - {code}: {desc}")

    lines.append("")
    lines.append(
        f"This report discusses 1 variant in total. "
        f"Classification was performed using the ACMG/AMP framework."
    )

    return "\n".join(lines)


def construct_l3_context(record: dict | pd.Series) -> str:
    """Build rich context for L3 reasoning prompts."""
    gene = record.get("gene_symbol", "Unknown")
    hgvs_c = record.get("hgvs_coding", "")
    hgvs_p = record.get("hgvs_protein", "")
    consequence = record.get("consequence_type", "unknown")
    cons_desc = CONSEQUENCE_DESCRIPTIONS.get(consequence, consequence)
    disease = record.get("disease_name", "not specified")
    classification = record.get("classification", "benign")
    inheritance = record.get("inheritance_pattern") or record.get("gene_moi") or "unknown"

    # Variant location
    chrom = record.get("chromosome", "?")
    pos = record.get("position", "?")
    ref = record.get("ref_allele", "?")
    alt = record.get("alt_allele", "?")

    parts = [
        f"Variant: {gene} chr{chrom}:{pos} {ref}>{alt}",
    ]
    if hgvs_c:
        parts.append(f"  HGVS coding: {hgvs_c}")
    if hgvs_p:
        parts.append(f"  HGVS protein: {hgvs_p}")
    parts.extend([
        f"  Consequence: {cons_desc}",
        f"  Classification: {classification.replace('_', ' ')}",
        f"",
        f"Condition: {disease}",
        f"  Inheritance: {inheritance}",
        f"",
    ])

    # Population frequency section
    af_parts = []
    af_global = record.get("gnomad_af_global")
    if af_global is not None:
        af_parts.append(f"Global: {af_global:.6f}")
    for pop in ["nfe", "afr", "eas", "sas"]:
        af_pop = record.get(f"gnomad_af_{pop}")
        if af_pop is not None and af_pop > 0:
            af_parts.append(f"{pop.upper()}: {af_pop:.6f}")
    if af_parts:
        parts.append(f"Population frequencies (gnomAD): {', '.join(af_parts)}")

    # Computational scores
    score_parts = []
    cadd = record.get("cadd_phred")
    if cadd is not None:
        score_parts.append(f"CADD Phred={cadd:.1f}")
    revel = record.get("revel_score")
    if revel is not None:
        score_parts.append(f"REVEL={revel:.3f}")
    am = record.get("alphamissense_score")
    if am is not None:
        am_cls = record.get("alphamissense_class", "")
        score_parts.append(f"AlphaMissense={am:.3f} ({am_cls})")
    phylop = record.get("phylop_score")
    if phylop is not None:
        score_parts.append(f"PhyloP={phylop:.2f}")
    gerp = record.get("gerp_score")
    if gerp is not None:
        score_parts.append(f"GERP={gerp:.2f}")
    sift = record.get("sift_score")
    if sift is not None:
        score_parts.append(f"SIFT={sift:.3f}")
    polyphen = record.get("polyphen2_score")
    if polyphen is not None:
        score_parts.append(f"PolyPhen2={polyphen:.3f}")
    if score_parts:
        parts.append(f"Computational scores: {', '.join(score_parts)}")

    # Gene constraint
    constraint = []
    pli = record.get("pli_score")
    if pli is not None:
        constraint.append(f"pLI={pli:.3f}")
    loeuf = record.get("loeuf_score")
    if loeuf is not None:
        constraint.append(f"LOEUF={loeuf:.3f}")
    missense_z = record.get("missense_z")
    if missense_z is not None:
        constraint.append(f"missense_z={missense_z:.2f}")
    clingen = record.get("clingen_validity")
    if clingen:
        constraint.append(f"ClinGen validity={clingen}")
    if constraint:
        parts.append(f"Gene constraint: {', '.join(constraint)}")

    # ACMG criteria if available
    acmg_raw = record.get("acmg_criteria") or "[]"
    try:
        acmg_list = json.loads(acmg_raw) if isinstance(acmg_raw, str) else acmg_raw
    except (json.JSONDecodeError, TypeError):
        acmg_list = []
    if acmg_list:
        parts.append(f"ACMG criteria applied: {', '.join(acmg_list)}")

    return "\n".join(parts)


def construct_l4_context(record: dict | pd.Series) -> str:
    """Build minimal context for L4 discrimination prompts."""
    gene = record.get("gene_symbol", "")
    hgvs_c = record.get("hgvs_coding") or ""
    hgvs_p = record.get("hgvs_protein") or ""
    consequence = record.get("consequence_type", "unknown")
    disease = record.get("disease_name", "not specified")

    chrom = record.get("chromosome", "?")
    pos = record.get("position", "?")
    ref = record.get("ref_allele", "?")
    alt = record.get("alt_allele", "?")

    parts = [
        f"Gene: {gene}",
        f"Variant: chr{chrom}:{pos} {ref}>{alt}",
    ]
    if hgvs_c:
        parts.append(f"HGVS: {hgvs_c}")
    if hgvs_p:
        parts.append(f"Protein: {hgvs_p}")
    parts.extend([
        f"Consequence: {consequence}",
        f"Condition: {disease}",
        f"",
        f"Has this variant-disease pair been assessed for pathogenicity in "
        f"clinical variant databases?",
    ])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def apply_max_per_gene(
    df: pd.DataFrame,
    max_per_gene: int = MAX_PER_GENE,
    rng: np.random.RandomState | None = None,
) -> pd.DataFrame:
    """Cap records per gene to prevent single gene dominating.

    Checks gene_id column.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    if "gene_id" not in df.columns:
        return df

    # Count appearances of each gene
    gene_counts = df["gene_id"].value_counts()
    over_limit = set(gene_counts[gene_counts > max_per_gene].index)
    if not over_limit:
        return df

    # Shuffle and keep first max_per_gene per gene
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    current_counts: dict = {}
    keep_mask = np.ones(len(df), dtype=bool)

    for idx in range(len(df)):
        gid = df.iloc[idx].get("gene_id")
        if gid is not None and gid in over_limit:
            cnt = current_counts.get(gid, 0)
            if cnt >= max_per_gene:
                keep_mask[idx] = False
                continue
            current_counts[gid] = cnt + 1
        elif gid is not None:
            current_counts[gid] = current_counts.get(gid, 0) + 1

    result = df[keep_mask].reset_index(drop=True)
    n_dropped = len(df) - len(result)
    if n_dropped > 0:
        logger.info(
            "apply_max_per_gene: kept %d, dropped %d (cap=%d)",
            len(result), n_dropped, max_per_gene,
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
    as 'val', remainder (up to test_size) as 'test'.
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
    """Write metadata.json with dataset statistics."""
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
