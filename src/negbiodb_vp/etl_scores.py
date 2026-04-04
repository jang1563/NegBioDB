"""Computational score ETL for NegBioDB VP domain.

Annotates variants with pre-computed scores from CADD, REVEL, AlphaMissense,
PhyloP, GERP, SIFT, and PolyPhen2.

All score files are large (CADD ~80 GB) and must be pre-processed on HPC.
This module reads pre-extracted TSV files with matched scores for our variants.

Input format: TSV with columns:
    chromosome, position, ref, alt, cadd_phred, revel_score,
    alphamissense_score, alphamissense_class, phylop_score,
    gerp_score, sift_score, polyphen2_score
"""

import csv
import gzip
import io
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

_SCORE_OUTPUT_COLUMNS = [
    "chromosome",
    "position",
    "ref",
    "alt",
    "cadd_phred",
    "revel_score",
    "alphamissense_score",
    "alphamissense_class",
    "phylop_score",
    "gerp_score",
    "sift_score",
    "polyphen2_score",
]


def annotate_scores(
    conn,
    scores_tsv: Path,
    batch_size: int = 5000,
) -> dict:
    """Annotate existing variants with computational scores.

    Updates cadd_phred, revel_score, alphamissense_score, alphamissense_class,
    phylop_score, gerp_score, sift_score, polyphen2_score on variants table.

    Returns stats dict.
    """
    stats = {"variants_annotated": 0, "variants_not_found": 0, "rows_parsed": 0}

    # Build variant locus lookup: (chr, pos, ref, alt) → variant_id
    variant_lookup = {}
    for row in conn.execute(
        "SELECT variant_id, chromosome, position, ref_allele, alt_allele FROM variants"
    ):
        key = (row[1], row[2], row[3], row[4])
        variant_lookup[key] = row[0]

    logger.info("Variant lookup built: %d variants", len(variant_lookup))

    with open(scores_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats["rows_parsed"] += 1
            chrom = row.get("chromosome", "").replace("chr", "")
            try:
                pos = int(row.get("position", 0))
            except (ValueError, TypeError):
                continue
            ref = row.get("ref", "")
            alt = row.get("alt", "")

            key = (chrom, pos, ref, alt)
            variant_id = variant_lookup.get(key)
            if variant_id is None:
                stats["variants_not_found"] += 1
                continue

            # Parse scores (all nullable)
            cadd = _safe_float(row.get("cadd_phred"))
            revel = _safe_float(row.get("revel_score"))
            am_score = _safe_float(row.get("alphamissense_score"))
            am_class = row.get("alphamissense_class", "").strip() or None
            if am_class and am_class not in (
                "likely_pathogenic", "ambiguous", "likely_benign"
            ):
                am_class = None
            phylop = _safe_float(row.get("phylop_score"))
            gerp = _safe_float(row.get("gerp_score"))
            sift = _safe_float(row.get("sift_score"))
            polyphen2 = _safe_float(row.get("polyphen2_score"))

            conn.execute(
                """UPDATE variants SET
                    cadd_phred = COALESCE(?, cadd_phred),
                    revel_score = COALESCE(?, revel_score),
                    alphamissense_score = COALESCE(?, alphamissense_score),
                    alphamissense_class = COALESCE(?, alphamissense_class),
                    phylop_score = COALESCE(?, phylop_score),
                    gerp_score = COALESCE(?, gerp_score),
                    sift_score = COALESCE(?, sift_score),
                    polyphen2_score = COALESCE(?, polyphen2_score),
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                WHERE variant_id = ?""",
                (cadd, revel, am_score, am_class, phylop, gerp, sift, polyphen2, variant_id),
            )
            stats["variants_annotated"] += 1

            if stats["variants_annotated"] % batch_size == 0:
                conn.commit()

    conn.commit()
    logger.info(
        "Scores: %d variants annotated, %d not found",
        stats["variants_annotated"],
        stats["variants_not_found"],
    )
    return stats


def _safe_float(value) -> float | None:
    """Safely convert to float, returning None for invalid values."""
    if value is None or value == "" or value == "NA" or value == "nan" or value == ".":
        return None
    try:
        v = float(value)
        if v != v:  # NaN
            return None
        return v
    except (ValueError, TypeError):
        return None
