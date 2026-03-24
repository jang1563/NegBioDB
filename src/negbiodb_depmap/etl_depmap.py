"""DepMap CRISPR gene essentiality ETL — load CRISPRGeneEffect + GeneDependency.

Pipeline:
  1. Parse Model.csv → insert cell_lines
  2. Parse gene columns from CRISPRGeneEffect header → insert genes
  3. Load reference gene sets → mark is_common_essential / is_reference_nonessential
  4. Insert ge_screens record for this CRISPR release
  5. Read CRISPRGeneEffect + CRISPRGeneDependency in chunks:
     - Merge gene effect + dependency probability per (cell_line, gene)
     - Apply DB inclusion threshold: gene_effect > -0.8 OR dep_prob < 0.5
     - Compute confidence tier per pair
     - Batch-insert into ge_negative_results
  6. refresh_all_ge_pairs()

Data format:
  - CRISPRGeneEffect.csv: rows = ModelID (ACH-*), cols = "HUGO (EntrezID)"
  - CRISPRGeneDependency.csv: same layout, values are dependency probability 0-1
  - Model.csv: cell line metadata (50+ cols)

Score system (Chronos): 0 = no effect, -1 = median essential gene effect

License: CC BY 4.0
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────

# DB inclusion: broad capture of anything not clearly essential
DB_GENE_EFFECT_THRESHOLD = -0.8  # gene_effect > -0.8 included
DB_DEP_PROB_THRESHOLD = 0.5      # dep_prob < 0.5 included

# Confidence tiering thresholds
GOLD_GENE_EFFECT = -0.2    # gene_effect > -0.2
GOLD_DEP_PROB = 0.3         # dep_prob < 0.3
SILVER_GENE_EFFECT = -0.5   # gene_effect > -0.5
SILVER_DEP_PROB = 0.5       # dep_prob < 0.5

# ── Gene column parsing ──────────────────────────────────────────────────

_GENE_COL_RE = re.compile(r"^(.+?)\s*\((\d+)\)$")


def parse_gene_column(col_name: str) -> tuple[str, int] | None:
    """Parse 'HUGO (EntrezID)' column header → (symbol, entrez_id).

    Examples:
        'TP53 (7157)' → ('TP53', 7157)
        'BRAF (673)' → ('BRAF', 673)
    Returns None if column doesn't match expected format.
    """
    m = _GENE_COL_RE.match(col_name.strip())
    if m:
        return m.group(1).strip(), int(m.group(2))
    return None


# ── Cell line loading ─────────────────────────────────────────────────────


def load_cell_lines(
    conn,
    model_file: Path,
) -> dict[str, int]:
    """Load cell line metadata from Model.csv into cell_lines table.

    Args:
        conn: SQLite connection.
        model_file: Path to Model.csv.

    Returns:
        Dict mapping ModelID → cell_line_id.
    """
    df = pd.read_csv(model_file, low_memory=False)
    inserted = 0

    for _, row in df.iterrows():
        model_id = str(row.get("ModelID", "")).strip()
        if not model_id:
            continue

        ccle_name = row.get("CCLEName") or row.get("CellLineName") or None
        if isinstance(ccle_name, float):
            ccle_name = None
        elif ccle_name:
            ccle_name = str(ccle_name).strip()

        stripped = row.get("StrippedCellLineName") or None
        if isinstance(stripped, float):
            stripped = None
        elif stripped:
            stripped = str(stripped).strip()

        lineage = row.get("OncotreeLineage") or None
        if isinstance(lineage, float):
            lineage = None
        elif lineage:
            lineage = str(lineage).strip()

        disease = row.get("OncotreePrimaryDisease") or None
        if isinstance(disease, float):
            disease = None
        elif disease:
            disease = str(disease).strip()

        subtype = row.get("OncotreeSubtype") or None
        if isinstance(subtype, float):
            subtype = None
        elif subtype:
            subtype = str(subtype).strip()

        sex = row.get("Sex") or None
        if isinstance(sex, float):
            sex = None
        elif sex:
            sex = str(sex).strip()

        pom = row.get("PrimaryOrMetastasis") or None
        if isinstance(pom, float):
            pom = None
        elif pom:
            pom = str(pom).strip()

        site = row.get("SampleCollectionSite") or None
        if isinstance(site, float):
            site = None
        elif site:
            site = str(site).strip()

        conn.execute(
            """INSERT OR IGNORE INTO cell_lines
            (model_id, ccle_name, stripped_name, lineage, primary_disease,
             subtype, sex, primary_or_metastasis, sample_collection_site)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (model_id, ccle_name, stripped, lineage, disease,
             subtype, sex, pom, site),
        )
        inserted += 1

    conn.commit()
    logger.info("Loaded %d cell lines from Model.csv", inserted)

    # Build model_id → cell_line_id lookup
    rows = conn.execute(
        "SELECT model_id, cell_line_id FROM cell_lines"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


# ── Gene loading ──────────────────────────────────────────────────────────


def load_genes_from_header(
    conn,
    gene_effect_file: Path,
) -> dict[int, int]:
    """Parse gene columns from CRISPRGeneEffect header, insert genes.

    Returns:
        Dict mapping column_index → gene_id (0-based, excluding row-id column).
    """
    with open(gene_effect_file) as f:
        header = f.readline().rstrip("\n")

    cols = header.split(",")
    col_to_gene_id: dict[int, int] = {}

    for i, col_name in enumerate(cols[1:], start=1):  # skip ModelID column
        parsed = parse_gene_column(col_name)
        if parsed is None:
            logger.warning("Unparseable gene column %d: '%s'", i, col_name)
            continue

        symbol, entrez_id = parsed
        conn.execute(
            "INSERT OR IGNORE INTO genes (entrez_id, gene_symbol) VALUES (?, ?)",
            (entrez_id, symbol),
        )
        row = conn.execute(
            "SELECT gene_id FROM genes WHERE entrez_id = ?",
            (entrez_id,),
        ).fetchone()
        col_to_gene_id[i] = row[0]

    conn.commit()
    logger.info("Loaded %d genes from header", len(col_to_gene_id))
    return col_to_gene_id


def load_reference_gene_sets(
    conn,
    essential_file: Path | None = None,
    nonessential_file: Path | None = None,
) -> dict[str, int]:
    """Mark genes as common essential or reference nonessential.

    Files contain one gene symbol per line (no header).
    Returns counts dict.
    """
    stats = {"common_essential": 0, "reference_nonessential": 0}

    if essential_file and essential_file.exists():
        symbols = set()
        with open(essential_file) as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    # Handle "HUGO (EntrezID)" format or bare symbol
                    parsed = parse_gene_column(s)
                    if parsed:
                        symbols.add(parsed[0])
                    else:
                        symbols.add(s)

        for symbol in symbols:
            conn.execute(
                "UPDATE genes SET is_common_essential = 1 WHERE gene_symbol = ?",
                (symbol,),
            )
        conn.commit()
        stats["common_essential"] = len(symbols)
        logger.info("Marked %d genes as common essential", len(symbols))

    if nonessential_file and nonessential_file.exists():
        symbols = set()
        with open(nonessential_file) as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    parsed = parse_gene_column(s)
                    if parsed:
                        symbols.add(parsed[0])
                    else:
                        symbols.add(s)

        for symbol in symbols:
            conn.execute(
                "UPDATE genes SET is_reference_nonessential = 1 WHERE gene_symbol = ?",
                (symbol,),
            )
        conn.commit()
        stats["reference_nonessential"] = len(symbols)
        logger.info("Marked %d genes as reference nonessential", len(symbols))

    return stats


# ── Confidence tiering ────────────────────────────────────────────────────


def assign_tier(
    gene_effect: float,
    dep_prob: float | None,
    is_reference_nonessential: bool = False,
) -> str:
    """Assign confidence tier for a single gene-cell_line pair.

    Gold:   gene_effect > -0.2 AND dep_prob < 0.3 AND reference nonessential
    Silver: gene_effect > -0.5 AND dep_prob < 0.5
    Bronze: gene_effect > -0.8 (OR dep_prob < 0.5)
    """
    if dep_prob is None:
        dep_prob = 1.0  # conservative default

    if (gene_effect > GOLD_GENE_EFFECT
            and dep_prob < GOLD_DEP_PROB
            and is_reference_nonessential):
        return "gold"
    elif gene_effect > SILVER_GENE_EFFECT and dep_prob < SILVER_DEP_PROB:
        return "silver"
    else:
        return "bronze"


# ── Main ETL ──────────────────────────────────────────────────────────────


def load_depmap_crispr(
    db_path: Path,
    gene_effect_file: Path,
    dependency_file: Path,
    model_file: Path,
    essential_file: Path | None = None,
    nonessential_file: Path | None = None,
    depmap_release: str = "25Q3",
    chunk_size: int = 100,
    batch_size: int = 5000,
) -> dict:
    """Load DepMap CRISPR screen data into GE database.

    Args:
        db_path: Path to GE SQLite database.
        gene_effect_file: CRISPRGeneEffect.csv (Chronos scores).
        dependency_file: CRISPRGeneDependency.csv (dependency probabilities).
        model_file: Model.csv (cell line metadata).
        essential_file: AchillesCommonEssentialControls.csv (optional).
        nonessential_file: AchillesNonessentialControls.csv (optional).
        depmap_release: Release identifier (e.g., '25Q3').
        chunk_size: Number of rows to read at a time from CSV.
        batch_size: Commit every N inserts.

    Returns:
        Stats dict with counts.
    """
    from negbiodb_depmap.depmap_db import get_connection, run_ge_migrations

    run_ge_migrations(db_path)
    conn = get_connection(db_path)

    stats = {
        "cell_lines_loaded": 0,
        "genes_loaded": 0,
        "ref_common_essential": 0,
        "ref_nonessential": 0,
        "pairs_considered": 0,
        "pairs_skipped_nan": 0,
        "pairs_skipped_essential": 0,
        "pairs_skipped_unmapped_cell_line": 0,
        "pairs_skipped_unmapped_gene": 0,
        "pairs_inserted": 0,
        "tier_gold": 0,
        "tier_silver": 0,
        "tier_bronze": 0,
    }

    try:
        # Step 1: Load cell lines
        model_id_to_clid = load_cell_lines(conn, model_file)
        stats["cell_lines_loaded"] = len(model_id_to_clid)

        # Step 2: Load genes from header
        col_to_gene_id = load_genes_from_header(conn, gene_effect_file)
        stats["genes_loaded"] = len(col_to_gene_id)

        # Step 3: Load reference gene sets
        ref_stats = load_reference_gene_sets(conn, essential_file, nonessential_file)
        stats["ref_common_essential"] = ref_stats.get("common_essential", 0)
        stats["ref_nonessential"] = ref_stats.get("reference_nonessential", 0)

        # Build lookup for reference nonessential genes
        ref_ne_gene_ids = {
            row[0]
            for row in conn.execute(
                "SELECT gene_id FROM genes WHERE is_reference_nonessential = 1"
            ).fetchall()
        }

        # Step 4: Insert screen record
        conn.execute(
            """INSERT OR IGNORE INTO ge_screens
            (source_db, depmap_release, screen_type, algorithm)
            VALUES ('depmap', ?, 'crispr', 'Chronos')""",
            (depmap_release,),
        )
        conn.commit()
        screen_row = conn.execute(
            "SELECT screen_id FROM ge_screens WHERE source_db='depmap' AND depmap_release=? AND screen_type='crispr'",
            (depmap_release,),
        ).fetchone()
        screen_id = screen_row[0]

        # Step 5: Read gene effect + dependency in chunks
        # Parse gene columns from dependency file header to verify alignment
        ge_reader = pd.read_csv(gene_effect_file, index_col=0, chunksize=chunk_size)
        dep_df = pd.read_csv(dependency_file, index_col=0)

        insert_count = 0
        for chunk_idx, ge_chunk in enumerate(ge_reader):
            for model_id in ge_chunk.index:
                model_id_str = str(model_id).strip()
                cl_id = model_id_to_clid.get(model_id_str)
                if cl_id is None:
                    stats["pairs_skipped_unmapped_cell_line"] += ge_chunk.shape[1]
                    continue

                for col_name in ge_chunk.columns:
                    stats["pairs_considered"] += 1

                    gene_effect = ge_chunk.at[model_id, col_name]
                    if pd.isna(gene_effect):
                        stats["pairs_skipped_nan"] += 1
                        continue

                    gene_effect = float(gene_effect)

                    # Get dependency probability
                    dep_prob = None
                    if model_id in dep_df.index and col_name in dep_df.columns:
                        dp_val = dep_df.at[model_id, col_name]
                        if not pd.isna(dp_val):
                            dep_prob = float(dp_val)

                    # DB inclusion filter
                    include = (gene_effect > DB_GENE_EFFECT_THRESHOLD)
                    if dep_prob is not None:
                        include = include or (dep_prob < DB_DEP_PROB_THRESHOLD)
                    if not include:
                        stats["pairs_skipped_essential"] += 1
                        continue

                    # Map gene column to gene_id
                    parsed = parse_gene_column(col_name)
                    if parsed is None:
                        stats["pairs_skipped_unmapped_gene"] += 1
                        continue
                    _, entrez_id = parsed
                    row = conn.execute(
                        "SELECT gene_id FROM genes WHERE entrez_id = ?",
                        (entrez_id,),
                    ).fetchone()
                    if row is None:
                        stats["pairs_skipped_unmapped_gene"] += 1
                        continue
                    gene_id = row[0]

                    # Confidence tier
                    is_ref_ne = gene_id in ref_ne_gene_ids
                    tier = assign_tier(gene_effect, dep_prob, is_ref_ne)
                    stats[f"tier_{tier}"] += 1

                    source_record_id = f"{model_id_str}_{entrez_id}"
                    conn.execute(
                        """INSERT OR IGNORE INTO ge_negative_results
                        (gene_id, cell_line_id, screen_id,
                         gene_effect_score, dependency_probability,
                         evidence_type, confidence_tier,
                         source_db, source_record_id, extraction_method)
                        VALUES (?, ?, ?, ?, ?, 'crispr_nonessential', ?,
                                'depmap', ?, 'score_threshold')""",
                        (gene_id, cl_id, screen_id,
                         gene_effect, dep_prob, tier, source_record_id),
                    )
                    insert_count += 1

                    if insert_count % batch_size == 0:
                        conn.commit()

            conn.commit()
            logger.info("Processed chunk %d (%d rows)", chunk_idx, ge_chunk.shape[0])

        conn.commit()

        # Step 6: Final count
        actual_inserted = conn.execute(
            "SELECT COUNT(*) FROM ge_negative_results WHERE source_db = 'depmap'"
        ).fetchone()[0]
        stats["pairs_inserted"] = actual_inserted

        # Dataset version
        conn.execute(
            "DELETE FROM dataset_versions WHERE name = 'depmap_crispr' AND version = ?",
            (depmap_release,),
        )
        conn.execute(
            """INSERT INTO dataset_versions (name, version, source_url, row_count, notes)
            VALUES ('depmap_crispr', ?, 'https://depmap.org/portal/download/all/',
                    ?, 'DepMap CRISPR Chronos gene effect + dependency probability')""",
            (depmap_release, actual_inserted),
        )
        conn.commit()

        logger.info(
            "CRISPR ETL complete: %d negative results (gold=%d, silver=%d, bronze=%d)",
            actual_inserted, stats["tier_gold"], stats["tier_silver"], stats["tier_bronze"],
        )

    finally:
        conn.close()

    return stats
