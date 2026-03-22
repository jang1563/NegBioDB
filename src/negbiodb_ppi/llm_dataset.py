"""Dataset builder utilities for PPI LLM benchmark (PPI-L1 through PPI-L4).

Shared constants, SQL helpers, sampling, splitting, and I/O functions
used by all four build_ppi_l{1..4}_dataset.py scripts.

Mirrors src/negbiodb_ct/llm_dataset.py structure from CT domain.
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

MAX_PER_PROTEIN = 10  # Prevent single protein dominating any class

# Few-shot set seeds (3 independent sets for variance) — re-exported from prompts
FEWSHOT_SEEDS = [42, 43, 44]

# Default PPI DB path
DEFAULT_PPI_DB_PATH = Path(__file__).parent.parent.parent / "data" / "negbiodb_ppi.db"

# Source → evidence category mapping (for L1 gold answers)
SOURCE_TO_L1_CATEGORY: dict[str, str] = {
    "intact_gold": "A",    # Direct experimental — curated non-interactions
    "intact_silver": "A",  # Direct experimental — curated non-interactions
    "huri": "B",           # Systematic screen — Y2H
    "humap": "C",          # Computational inference — co-fractionation ML
    "string": "D",         # Database score absence — zero combined score
}

# Detection method descriptions for evidence construction
DETECTION_METHOD_DESCRIPTIONS: dict[str, str] = {
    "co-immunoprecipitation": "co-immunoprecipitation (co-IP) assay",
    "pull down": "affinity pulldown assay",
    "affinity chromatography technology": "affinity chromatography",
    "two hybrid": "yeast two-hybrid (Y2H) screening",
    "anti bait coimmunoprecipitation": "anti-bait co-immunoprecipitation",
    "surface plasmon resonance": "surface plasmon resonance (SPR)",
    "fluorescence resonance energy transfer": "FRET-based proximity assay",
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


def load_ppi_candidate_pool(
    db_path: Path,
    tier_filter: str | None = None,
    source_filter: str | None = None,
    extra_where: str = "",
    require_annotations: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load candidate non-interaction records from PPI database.

    Parameters
    ----------
    db_path : Path to PPI database
    tier_filter : e.g. "IN ('gold', 'silver')" or "= 'gold'"
    source_filter : e.g. "= 'intact'" or "IN ('intact', 'huri')"
    extra_where : additional SQL WHERE clauses (AND-joined)
    require_annotations : if True, only return proteins with function_description
    limit : if set, randomly sample this many rows at SQL level

    Returns
    -------
    DataFrame with negative_results + protein annotations joined.
    """
    from negbiodb_ppi.ppi_db import get_connection

    where_parts = ["1=1"]
    if tier_filter:
        where_parts.append(f"nr.confidence_tier {tier_filter}")
    if source_filter:
        where_parts.append(f"nr.source_db {source_filter}")
    if extra_where:
        where_parts.append(extra_where)
    if require_annotations:
        where_parts.append("p1.function_description IS NOT NULL")
        where_parts.append("p2.function_description IS NOT NULL")
    where_clause = " AND ".join(where_parts)

    sql = f"""
    SELECT
        nr.result_id, nr.source_db, nr.confidence_tier,
        nr.detection_method, e.pubmed_id, e.experiment_type,
        p1.protein_id AS protein_id_1, p1.uniprot_accession AS uniprot_1,
        p1.gene_symbol AS gene_symbol_1, p1.sequence_length AS seq_len_1,
        p1.subcellular_location AS location_1,
        p1.function_description AS function_1,
        p1.go_terms AS go_terms_1,
        p1.domain_annotations AS domains_1,
        p2.protein_id AS protein_id_2, p2.uniprot_accession AS uniprot_2,
        p2.gene_symbol AS gene_symbol_2, p2.sequence_length AS seq_len_2,
        p2.subcellular_location AS location_2,
        p2.function_description AS function_2,
        p2.go_terms AS go_terms_2,
        p2.domain_annotations AS domains_2
    FROM ppi_negative_results nr
    JOIN proteins p1 ON nr.protein1_id = p1.protein_id
    JOIN proteins p2 ON nr.protein2_id = p2.protein_id
    LEFT JOIN ppi_experiments e ON nr.experiment_id = e.experiment_id
    WHERE {where_clause}
    {"ORDER BY RANDOM() LIMIT " + str(limit) if limit else ""}
    """

    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()

    logger.info(
        "Loaded %d candidate records (tier_filter=%s, source_filter=%s)",
        len(df), tier_filter, source_filter,
    )
    return df


def load_protein_annotations(
    db_path: Path,
    protein_ids: list[int] | None = None,
) -> dict[int, dict]:
    """Load protein annotations as a dict keyed by protein_id.

    Returns dict: protein_id -> {uniprot_accession, gene_symbol, function_description,
    go_terms, domain_annotations, subcellular_location, sequence_length}.
    """
    from negbiodb_ppi.ppi_db import get_connection

    conn = get_connection(db_path)
    try:
        if protein_ids:
            placeholders = ",".join("?" * len(protein_ids))
            sql = f"""
            SELECT protein_id, uniprot_accession, gene_symbol, sequence_length,
                   subcellular_location, function_description, go_terms,
                   domain_annotations
            FROM proteins WHERE protein_id IN ({placeholders})
            """
            rows = conn.execute(sql, protein_ids).fetchall()
        else:
            sql = """
            SELECT protein_id, uniprot_accession, gene_symbol, sequence_length,
                   subcellular_location, function_description, go_terms,
                   domain_annotations
            FROM proteins
            """
            rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    cols = [
        "protein_id", "uniprot_accession", "gene_symbol", "sequence_length",
        "subcellular_location", "function_description", "go_terms",
        "domain_annotations",
    ]
    return {
        row[0]: dict(zip(cols, row))
        for row in rows
    }


def load_publication_abstracts(
    db_path: Path,
) -> dict[int, dict]:
    """Load PubMed abstracts keyed by PMID.

    Returns dict: pmid -> {title, abstract, publication_year, journal}.
    """
    from negbiodb_ppi.ppi_db import get_connection

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT pmid, title, abstract, publication_year, journal "
            "FROM ppi_publication_abstracts"
        ).fetchall()
    finally:
        conn.close()

    return {
        row[0]: {
            "title": row[1],
            "abstract": row[2],
            "publication_year": row[3],
            "journal": row[4],
        }
        for row in rows
    }


# ---------------------------------------------------------------------------
# Evidence description construction
# ---------------------------------------------------------------------------


def construct_evidence_description(
    record: dict | pd.Series,
    difficulty: str = "easy",
) -> str:
    """Generate a natural-language evidence description from DB fields.

    The description varies in explicitness based on difficulty:
    - easy: Clear method description → obvious mapping to L1 category
    - medium: Ambiguous wording that could fit 2 categories
    - hard: Vague or mixed evidence descriptions
    """
    source = record.get("source_db", "")
    method = record.get("detection_method", "")
    gene1 = record.get("gene_symbol_1", record.get("uniprot_1", "Protein_1"))
    gene2 = record.get("gene_symbol_2", record.get("uniprot_2", "Protein_2"))

    if source in ("intact",) and method:
        method_desc = DETECTION_METHOD_DESCRIPTIONS.get(method, method)
        if difficulty == "easy":
            return (
                f"A {method_desc} was performed to test for direct binding between "
                f"{gene1} and {gene2}. No physical interaction was detected under "
                f"the experimental conditions used."
            )
        elif difficulty == "medium":
            return (
                f"An interaction study examined whether {gene1} associates with "
                f"{gene2} using a binding assay. The results were negative, "
                f"suggesting no stable complex formation."
            )
        else:  # hard
            return (
                f"Experimental analysis of {gene1} and {gene2} found no evidence "
                f"of association in the conditions tested."
            )

    elif source == "huri":
        if difficulty == "easy":
            return (
                f"In a genome-wide yeast two-hybrid (Y2H) screen testing all "
                f"pairwise combinations of human ORFs, the pair {gene1}-{gene2} "
                f"did not produce a positive signal in any replicate."
            )
        elif difficulty == "medium":
            return (
                f"A large-scale binary interaction screen systematically tested "
                f"{gene1} against {gene2}. The pair was not detected among the "
                f"positive interactions."
            )
        else:
            return (
                f"Systematic testing of {gene1} and {gene2} in a high-throughput "
                f"assay did not identify this pair as interacting."
            )

    elif source == "humap":
        if difficulty == "easy":
            return (
                f"Machine learning analysis of co-fractionation mass spectrometry "
                f"data from multiple cell types predicted that {gene1} and {gene2} "
                f"do not co-exist in the same protein complex."
            )
        elif difficulty == "medium":
            return (
                f"Computational analysis of proteomics data across multiple "
                f"experiments found no evidence that {gene1} and {gene2} "
                f"co-fractionate or belong to the same complex."
            )
        else:
            return (
                f"Analysis of protein complex data indicates {gene1} and {gene2} "
                f"are unlikely to be found in the same molecular assembly."
            )

    elif source == "string":
        if difficulty == "easy":
            return (
                f"Across all available evidence channels (experimental, database, "
                f"text-mining, co-expression, co-occurrence, gene fusion, "
                f"neighborhood), the combined interaction score between {gene1} "
                f"and {gene2} is zero or negligible."
            )
        elif difficulty == "medium":
            return (
                f"An aggregated database integrating multiple evidence types found "
                f"no meaningful interaction signal between {gene1} and {gene2}."
            )
        else:
            return (
                f"No significant association between {gene1} and {gene2} was "
                f"found across available interaction evidence."
            )

    # Fallback
    return (
        f"Available evidence does not support a physical interaction between "
        f"{gene1} and {gene2}."
    )


def construct_l3_context(record: dict | pd.Series) -> str:
    """Build rich context for L3 reasoning prompts."""
    gene1 = record.get("gene_symbol_1", record.get("uniprot_1", "Protein_1"))
    uniprot1 = record.get("uniprot_1", "")
    seq_len1 = record.get("seq_len_1", "unknown")
    func1 = record.get("function_1", "Function not available")
    loc1 = record.get("location_1", "Location not available")
    domains1 = record.get("domains_1", "No domain annotations available")

    gene2 = record.get("gene_symbol_2", record.get("uniprot_2", "Protein_2"))
    uniprot2 = record.get("uniprot_2", "")
    seq_len2 = record.get("seq_len_2", "unknown")
    func2 = record.get("function_2", "Function not available")
    loc2 = record.get("location_2", "Location not available")
    domains2 = record.get("domains_2", "No domain annotations available")

    method = record.get("detection_method", "experimental")
    method_desc = DETECTION_METHOD_DESCRIPTIONS.get(method, method) if method else "experimental"

    return (
        f"Protein 1: {gene1} ({uniprot1}, {seq_len1} AA)\n"
        f"  Function: {func1}\n"
        f"  Location: {loc1}\n"
        f"  Domains: {domains1}\n\n"
        f"Protein 2: {gene2} ({uniprot2}, {seq_len2} AA)\n"
        f"  Function: {func2}\n"
        f"  Location: {loc2}\n"
        f"  Domains: {domains2}\n\n"
        f"Experimental evidence: {method_desc} assay confirmed no physical interaction."
    )


def construct_l4_context(record: dict | pd.Series) -> str:
    """Build minimal context for L4 discrimination prompts."""
    gene1 = record.get("gene_symbol_1", "")
    name1 = record.get("protein_name_1", gene1)
    gene2 = record.get("gene_symbol_2", "")
    name2 = record.get("protein_name_2", gene2)

    return (
        f"Protein 1: {gene1} ({name1})\n"
        f"Protein 2: {gene2} ({name2})\n"
        f"Organism: Homo sapiens\n\n"
        f"Has this protein pair been experimentally tested for physical interaction?"
    )


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def apply_max_per_protein(
    df: pd.DataFrame,
    max_per_protein: int = MAX_PER_PROTEIN,
    rng: np.random.RandomState | None = None,
) -> pd.DataFrame:
    """Cap records per protein to prevent single protein dominating.

    Checks both protein_id_1 and protein_id_2 columns.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Count appearances of each protein across both columns
    protein_counts: dict[int, int] = {}
    for _, row in df.iterrows():
        for col in ("protein_id_1", "protein_id_2"):
            pid = row.get(col)
            if pid is not None:
                protein_counts[pid] = protein_counts.get(pid, 0) + 1

    # Identify over-represented proteins
    over_limit = {pid for pid, cnt in protein_counts.items() if cnt > max_per_protein}
    if not over_limit:
        return df

    # Iteratively remove records from over-represented proteins
    keep_mask = np.ones(len(df), dtype=bool)
    indices = rng.permutation(len(df))

    current_counts: dict[int, int] = {}
    for idx in indices:
        row = df.iloc[idx]
        p1 = row.get("protein_id_1")
        p2 = row.get("protein_id_2")

        c1 = current_counts.get(p1, 0) if p1 is not None else 0
        c2 = current_counts.get(p2, 0) if p2 is not None else 0

        if c1 >= max_per_protein or c2 >= max_per_protein:
            keep_mask[idx] = False
            continue

        if p1 is not None:
            current_counts[p1] = c1 + 1
        if p2 is not None:
            current_counts[p2] = c2 + 1

    result = df[keep_mask].reset_index(drop=True)
    n_dropped = len(df) - len(result)
    if n_dropped > 0:
        logger.info(
            "apply_max_per_protein: kept %d, dropped %d (cap=%d)",
            len(result), n_dropped, max_per_protein,
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
