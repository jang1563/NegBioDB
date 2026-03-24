"""DEMETER2 RNAi gene dependency ETL — load D2_combined_gene_dep_scores.csv.

Key differences from CRISPR ETL:
  - DEMETER2 uses CCLE names for cell lines, not ModelID (ACH-*)
  - Cell line mapping: ccle_name → ModelID via cell_lines table (loaded from Model.csv)
  - Fallback: stripped_name match for slightly different naming conventions
  - No dependency probability (DEMETER2 only provides gene dep scores)
  - Screen type: 'rnai', algorithm: 'DEMETER2'

After loading, runs concordance tier upgrade:
  Gene-cell_line pairs with BOTH crispr + rnai sources agreeing on non-essential
  get upgraded from bronze → silver.

Data format:
  - D2_combined_gene_dep_scores.csv: rows = "HUGO (EntrezID)", cols = CCLE names
  - Transposed relative to CRISPRGeneEffect (genes in rows, cell lines in cols)
  - DEMETER2 score: 0 = no effect, -1 = median essential gene effect

License: CC BY 4.0
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from negbiodb_depmap.etl_depmap import (
    DB_GENE_EFFECT_THRESHOLD,
    SILVER_GENE_EFFECT,
    parse_gene_column,
)

logger = logging.getLogger(__name__)

# RNAi-specific threshold (DEMETER2 scale is similar to Chronos)
RNAI_CONCORDANCE_THRESHOLD = -0.3  # gene must score > -0.3 in RNAi to count as concordant


def _build_ccle_to_clid(conn) -> tuple[dict[str, int], dict[str, int]]:
    """Build CCLE name → cell_line_id and stripped_name → cell_line_id lookups."""
    rows = conn.execute(
        "SELECT cell_line_id, ccle_name, stripped_name FROM cell_lines"
    ).fetchall()
    ccle_map = {}
    stripped_map = {}
    for clid, ccle, stripped in rows:
        if ccle:
            ccle_map[ccle] = clid
        if stripped:
            stripped_map[stripped.upper()] = clid
    return ccle_map, stripped_map


def _resolve_cell_line(
    name: str,
    ccle_map: dict[str, int],
    stripped_map: dict[str, int],
) -> int | None:
    """Resolve DEMETER2 cell line name to cell_line_id.

    Strategy:
      1. Direct match on ccle_name
      2. Stripped/uppercased match on stripped_name
    """
    # Direct CCLE name match
    if name in ccle_map:
        return ccle_map[name]

    # Stripped name match (uppercase, remove common suffixes)
    stripped = name.upper().replace("-", "").replace(" ", "").replace("_", "")
    if stripped in stripped_map:
        return stripped_map[stripped]

    return None


def load_demeter2(
    db_path: Path,
    rnai_file: Path,
    depmap_release: str = "DEMETER2_v6",
    chunk_size: int = 500,
    batch_size: int = 5000,
) -> dict:
    """Load DEMETER2 RNAi gene dependency scores into GE database.

    Args:
        db_path: Path to GE SQLite database.
        rnai_file: D2_combined_gene_dep_scores.csv.
        depmap_release: Release identifier (e.g., 'DEMETER2_v6').
        chunk_size: Number of gene rows to read at a time.
        batch_size: Commit every N inserts.

    Returns:
        Stats dict with counts.
    """
    from negbiodb_depmap.depmap_db import get_connection, run_ge_migrations

    run_ge_migrations(db_path)
    conn = get_connection(db_path)

    stats = {
        "genes_in_file": 0,
        "cell_lines_in_file": 0,
        "cell_lines_mapped": 0,
        "cell_lines_unmapped": 0,
        "pairs_considered": 0,
        "pairs_skipped_nan": 0,
        "pairs_skipped_essential": 0,
        "pairs_skipped_unmapped_gene": 0,
        "pairs_inserted": 0,
        "tier_silver": 0,
        "tier_bronze": 0,
        "concordance_upgrades": 0,
    }

    try:
        # Build cell line mapping
        ccle_map, stripped_map = _build_ccle_to_clid(conn)

        # Insert screen record
        conn.execute(
            """INSERT OR IGNORE INTO ge_screens
            (source_db, depmap_release, screen_type, algorithm)
            VALUES ('demeter2', ?, 'rnai', 'DEMETER2')""",
            (depmap_release,),
        )
        conn.commit()
        screen_row = conn.execute(
            "SELECT screen_id FROM ge_screens WHERE source_db='demeter2' AND depmap_release=? AND screen_type='rnai'",
            (depmap_release,),
        ).fetchone()
        screen_id = screen_row[0]

        # DEMETER2 format: genes in rows, cell lines in columns
        # Read header to map cell lines
        header_df = pd.read_csv(rnai_file, nrows=0, index_col=0)
        cell_line_names = list(header_df.columns)
        stats["cell_lines_in_file"] = len(cell_line_names)

        # Map cell line columns to cell_line_ids
        col_to_clid: dict[str, int] = {}
        unmapped_cls = []
        for name in cell_line_names:
            clid = _resolve_cell_line(name, ccle_map, stripped_map)
            if clid is not None:
                col_to_clid[name] = clid
            else:
                unmapped_cls.append(name)

        stats["cell_lines_mapped"] = len(col_to_clid)
        stats["cell_lines_unmapped"] = len(unmapped_cls)
        if unmapped_cls:
            logger.warning(
                "Unmapped DEMETER2 cell lines (%d): %s",
                len(unmapped_cls),
                unmapped_cls[:10],
            )

        # Read gene dep scores in chunks (genes in rows)
        reader = pd.read_csv(rnai_file, index_col=0, chunksize=chunk_size)
        insert_count = 0

        # Build gene lookup: entrez_id → gene_id
        gene_lookup: dict[int, int] = {}
        for row in conn.execute("SELECT gene_id, entrez_id FROM genes WHERE entrez_id IS NOT NULL"):
            gene_lookup[row[1]] = row[0]

        # Lookup for reference nonessential
        ref_ne_gene_ids = {
            row[0]
            for row in conn.execute(
                "SELECT gene_id FROM genes WHERE is_reference_nonessential = 1"
            ).fetchall()
        }

        for chunk_idx, chunk in enumerate(reader):
            for gene_label in chunk.index:
                stats["genes_in_file"] += 1
                parsed = parse_gene_column(str(gene_label))
                if parsed is None:
                    continue

                symbol, entrez_id = parsed

                # Insert gene if not exists
                if entrez_id not in gene_lookup:
                    conn.execute(
                        "INSERT OR IGNORE INTO genes (entrez_id, gene_symbol) VALUES (?, ?)",
                        (entrez_id, symbol),
                    )
                    row = conn.execute(
                        "SELECT gene_id FROM genes WHERE entrez_id = ?",
                        (entrez_id,),
                    ).fetchone()
                    if row:
                        gene_lookup[entrez_id] = row[0]
                    else:
                        continue

                gene_id = gene_lookup[entrez_id]

                for cl_name, cl_id in col_to_clid.items():
                    stats["pairs_considered"] += 1

                    score = chunk.at[gene_label, cl_name]
                    if pd.isna(score):
                        stats["pairs_skipped_nan"] += 1
                        continue

                    score = float(score)

                    # DB inclusion (RNAi has no dep_prob)
                    if score <= DB_GENE_EFFECT_THRESHOLD:
                        stats["pairs_skipped_essential"] += 1
                        continue

                    # Tier assignment (no dep_prob for RNAi)
                    is_ref_ne = gene_id in ref_ne_gene_ids
                    if score > SILVER_GENE_EFFECT and is_ref_ne:
                        tier = "silver"
                        stats["tier_silver"] += 1
                    else:
                        tier = "bronze"
                        stats["tier_bronze"] += 1

                    source_record_id = f"{cl_name}_{entrez_id}"
                    conn.execute(
                        """INSERT OR IGNORE INTO ge_negative_results
                        (gene_id, cell_line_id, screen_id,
                         gene_effect_score, dependency_probability,
                         evidence_type, confidence_tier,
                         source_db, source_record_id, extraction_method)
                        VALUES (?, ?, ?, ?, NULL, 'rnai_nonessential', ?,
                                'demeter2', ?, 'score_threshold')""",
                        (gene_id, cl_id, screen_id, score, tier, source_record_id),
                    )
                    insert_count += 1

                    if insert_count % batch_size == 0:
                        conn.commit()

            conn.commit()
            logger.info("RNAi chunk %d processed", chunk_idx)

        conn.commit()

        # Concordance upgrade: bronze→silver for pairs with BOTH CRISPR and RNAi
        upgraded = _upgrade_concordant_pairs(conn)
        stats["concordance_upgrades"] = upgraded

        # Final count
        actual_inserted = conn.execute(
            "SELECT COUNT(*) FROM ge_negative_results WHERE source_db = 'demeter2'"
        ).fetchone()[0]
        stats["pairs_inserted"] = actual_inserted

        # Dataset version
        conn.execute(
            "DELETE FROM dataset_versions WHERE name = 'demeter2_rnai' AND version = ?",
            (depmap_release,),
        )
        conn.execute(
            """INSERT INTO dataset_versions (name, version, source_url, row_count, notes)
            VALUES ('demeter2_rnai', ?,
                    'https://figshare.com/articles/dataset/DEMETER2_data/6025238',
                    ?, 'DEMETER2 RNAi gene dependency scores')""",
            (depmap_release, actual_inserted),
        )
        conn.commit()

        logger.info(
            "RNAi ETL complete: %d results, %d concordance upgrades",
            actual_inserted, upgraded,
        )

    finally:
        conn.close()

    return stats


def _upgrade_concordant_pairs(conn) -> int:
    """Upgrade bronze → silver for gene-cell_line pairs with CRISPR + RNAi concordance.

    A pair qualifies if:
      - It has a CRISPR result (source_db='depmap')
      - It has an RNAi result (source_db='demeter2')
      - Both indicate non-essential (above their respective thresholds)
      - Current tier is 'bronze'
    """
    # Find bronze CRISPR results that have concordant RNAi results
    result = conn.execute(
        """UPDATE ge_negative_results
        SET confidence_tier = 'silver',
            evidence_type = 'multi_screen_concordant',
            updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
        WHERE confidence_tier = 'bronze'
          AND source_db = 'depmap'
          AND result_id IN (
            SELECT cr.result_id
            FROM ge_negative_results cr
            JOIN ge_negative_results rn
              ON cr.gene_id = rn.gene_id
              AND cr.cell_line_id = rn.cell_line_id
            WHERE cr.source_db = 'depmap'
              AND rn.source_db = 'demeter2'
              AND cr.confidence_tier = 'bronze'
              AND rn.gene_effect_score > ?
          )""",
        (RNAI_CONCORDANCE_THRESHOLD,),
    )
    upgraded = result.rowcount
    conn.commit()
    logger.info("Upgraded %d bronze→silver via CRISPR+RNAi concordance", upgraded)
    return upgraded
