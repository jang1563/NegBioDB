#!/usr/bin/env python3
"""Cross-domain gene enrichment analysis for VP domain.

Analyzes overlap between VP genes and other NegBioDB domains (GE, PPI, DTI).
Tests: Are benign variants enriched in non-essential genes?

Usage:
    PYTHONPATH=src python scripts_vp/analyze_vp_cross_domain.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VP cross-domain analysis.")
    parser.add_argument("--vp-db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_vp.db")
    parser.add_argument("--ge-db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_depmap.db")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "vp_cross_domain.json")
    args = parser.parse_args(argv)

    from negbiodb_vp.vp_db import get_connection

    # Load VP genes
    vp_conn = get_connection(args.vp_db)
    try:
        vp_genes = pd.read_sql_query("""
            SELECT g.gene_symbol, g.entrez_id, g.pli_score, g.loeuf_score,
                   COUNT(DISTINCT nr.result_id) as n_benign_results,
                   COUNT(DISTINCT nr.disease_id) as n_diseases
            FROM genes g
            JOIN variants v ON g.gene_id = v.gene_id
            JOIN vp_negative_results nr ON v.variant_id = nr.variant_id
            GROUP BY g.gene_symbol
            ORDER BY n_benign_results DESC
        """, vp_conn)

        # Cross-domain bridge table
        bridge = pd.read_sql_query("""
            SELECT gene_id, domain, external_id
            FROM vp_cross_domain_genes
        """, vp_conn)
    finally:
        vp_conn.close()

    logger.info("VP genes with benign variants: %d", len(vp_genes))

    results = {
        "n_vp_genes": len(vp_genes),
        "top_genes_by_benign_count": vp_genes.head(20)[["gene_symbol", "n_benign_results", "n_diseases"]].to_dict("records"),
    }

    # Analyze GE overlap if available
    if args.ge_db.exists():
        from negbiodb.db import get_connection as get_ge_connection

        ge_conn = get_ge_connection(args.ge_db)
        try:
            ge_genes = pd.read_sql_query("""
                SELECT gene_symbol, entrez_id, is_common_essential, is_reference_nonessential
                FROM genes
            """, ge_conn)
        finally:
            ge_conn.close()

        ge_symbols = set(ge_genes["gene_symbol"])
        vp_symbols = set(vp_genes["gene_symbol"])
        overlap = vp_symbols & ge_symbols

        logger.info("VP-GE gene overlap: %d / %d VP genes (%.1f%%)",
                     len(overlap), len(vp_symbols), 100 * len(overlap) / len(vp_symbols))

        # Enrichment: benign variants vs essentiality
        merged = vp_genes.merge(ge_genes, on="gene_symbol", how="inner")
        if len(merged) > 0:
            essential = merged[merged["is_common_essential"] == 1]
            nonessential = merged[merged["is_reference_nonessential"] == 1]

            results["ge_overlap"] = {
                "n_overlap": len(overlap),
                "pct_overlap": len(overlap) / len(vp_symbols),
                "n_essential_with_benign": len(essential),
                "n_nonessential_with_benign": len(nonessential),
                "mean_benign_essential": float(essential["n_benign_results"].mean()) if len(essential) else 0,
                "mean_benign_nonessential": float(nonessential["n_benign_results"].mean()) if len(nonessential) else 0,
            }
    else:
        logger.info("GE database not found, skipping GE overlap analysis")

    # Gene constraint distribution
    pli_values = vp_genes["pli_score"].dropna()
    loeuf_values = vp_genes["loeuf_score"].dropna()
    results["constraint"] = {
        "pli_mean": float(pli_values.mean()) if len(pli_values) else None,
        "pli_median": float(pli_values.median()) if len(pli_values) else None,
        "loeuf_mean": float(loeuf_values.mean()) if len(loeuf_values) else None,
        "loeuf_median": float(loeuf_values.median()) if len(loeuf_values) else None,
        "n_constrained_genes": int((pli_values > 0.9).sum()) if len(pli_values) else 0,
        "n_tolerant_genes": int((loeuf_values > 1.0).sum()) if len(loeuf_values) else 0,
    }

    # Cross-domain bridge summary
    if len(bridge) > 0:
        results["bridge"] = dict(bridge["domain"].value_counts())

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
