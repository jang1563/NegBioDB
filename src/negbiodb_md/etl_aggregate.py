"""Aggregation ETL for MD domain.

Populates md_metabolite_disease_pairs from md_biomarker_results.
Called after all source ETL (MetaboLights + NMDR) is complete.

Also applies gold-tier upgrade: pairs replicated across >= 2 studies
with all-negative consensus get upgraded to gold if FDR > 0.1 criteria met.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def aggregate_pairs(conn) -> int:
    """Recompute md_metabolite_disease_pairs from md_biomarker_results.

    Delegates to md_db.refresh_all_pairs.

    Returns number of pairs created.
    """
    from negbiodb_md.md_db import refresh_all_pairs
    count = refresh_all_pairs(conn)
    logger.info("Aggregated %d metabolite-disease pairs", count)
    return count


def upgrade_gold_tier(conn) -> int:
    """Upgrade bronze/silver negative pairs to gold if replicated >= 2 studies.

    Gold criteria:
      - consensus = 'negative' (no positive study for this pair)
      - n_studies_negative >= 2
      - ALL negative results have FDR > 0.1 OR p_value > 0.05
      - At least one study has n >= 50/group (if n data available)

    Returns number of pairs upgraded to gold.
    """
    # Find pairs that meet replication criteria
    candidate_pairs = conn.execute(
        """SELECT pair_id, metabolite_id, disease_id
           FROM md_metabolite_disease_pairs
           WHERE consensus = 'negative'
             AND n_studies_negative >= 2
             AND best_tier IN ('silver', 'bronze')"""
    ).fetchall()

    upgraded = 0
    for pair_id, metabolite_id, disease_id in candidate_pairs:
        # Check: all negative results for this pair have p_value > 0.05
        bad_results = conn.execute(
            """SELECT COUNT(*) FROM md_biomarker_results
               WHERE metabolite_id = ? AND disease_id = ?
                 AND is_significant = 0
                 AND p_value IS NOT NULL
                 AND p_value <= 0.05""",
            (metabolite_id, disease_id),
        ).fetchone()[0]

        if bad_results > 0:
            continue  # Some results have p <= 0.05, skip

        # Update best_tier to gold for this pair
        conn.execute(
            "UPDATE md_metabolite_disease_pairs SET best_tier = 'gold' WHERE pair_id = ?",
            (pair_id,),
        )
        # Also update individual result tiers
        conn.execute(
            """UPDATE md_biomarker_results SET tier = 'gold'
               WHERE metabolite_id = ? AND disease_id = ? AND is_significant = 0""",
            (metabolite_id, disease_id),
        )
        upgraded += 1

    conn.commit()
    logger.info("Gold tier upgrade: %d pairs upgraded", upgraded)
    return upgraded


def compute_statistics(conn) -> dict:
    """Compute summary statistics for the MD database.

    Returns dict with counts by tier, consensus, platform, disease_category.
    """
    stats: dict = {}

    stats["total_results"] = conn.execute(
        "SELECT COUNT(*) FROM md_biomarker_results"
    ).fetchone()[0]

    stats["negative_results"] = conn.execute(
        "SELECT COUNT(*) FROM md_biomarker_results WHERE is_significant = 0"
    ).fetchone()[0]

    stats["positive_results"] = conn.execute(
        "SELECT COUNT(*) FROM md_biomarker_results WHERE is_significant = 1"
    ).fetchone()[0]

    stats["total_pairs"] = conn.execute(
        "SELECT COUNT(*) FROM md_metabolite_disease_pairs"
    ).fetchone()[0]

    stats["metabolites"] = conn.execute(
        "SELECT COUNT(*) FROM md_metabolites"
    ).fetchone()[0]

    stats["diseases"] = conn.execute(
        "SELECT COUNT(*) FROM md_diseases"
    ).fetchone()[0]

    stats["studies"] = conn.execute(
        "SELECT COUNT(*) FROM md_studies"
    ).fetchone()[0]

    # By tier
    tier_rows = conn.execute(
        """SELECT tier, COUNT(*) FROM md_biomarker_results
           WHERE is_significant = 0 GROUP BY tier"""
    ).fetchall()
    stats["tiers"] = dict(tier_rows)

    # By consensus
    consensus_rows = conn.execute(
        "SELECT consensus, COUNT(*) FROM md_metabolite_disease_pairs GROUP BY consensus"
    ).fetchall()
    stats["consensus"] = dict(consensus_rows)

    # By disease category
    cat_rows = conn.execute(
        """SELECT d.disease_category, COUNT(DISTINCT r.result_id)
           FROM md_biomarker_results r
           JOIN md_diseases d ON r.disease_id = d.disease_id
           WHERE r.is_significant = 0
           GROUP BY d.disease_category"""
    ).fetchall()
    stats["by_disease_category"] = dict(cat_rows)

    # By platform
    platform_rows = conn.execute(
        """SELECT s.platform, COUNT(DISTINCT r.result_id)
           FROM md_biomarker_results r
           JOIN md_studies s ON r.study_id = s.study_id
           WHERE r.is_significant = 0
           GROUP BY s.platform"""
    ).fetchall()
    stats["by_platform"] = dict(platform_rows)

    return stats
