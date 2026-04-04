#!/usr/bin/env python3
"""Build STRING v12.0 gene interaction graph for VariantGNN.

Downloads and processes STRING protein links to create a gene-level graph.
Filters by combined_score > 700 and maps to gene symbols.

Output: data/vp/string_gene_graph.pkl
  Contains: {
    'edge_index': np.array (2, E),
    'node_features': np.array (N_genes, 5),  # pLI, LOEUF, missense_z, variant_count, mean_severity
    'gene_to_idx': dict,  # gene_symbol -> node index
    'idx_to_gene': dict,  # node index -> gene_symbol
  }

Usage:
    PYTHONPATH=src python scripts_vp/build_gene_graph.py \
        --string-file data/vp/9606.protein.links.v12.0.txt \
        --db data/negbiodb_vp.db
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# STRING combined_score threshold (700 = "high confidence")
MIN_COMBINED_SCORE = 700


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build gene interaction graph from STRING.")
    parser.add_argument("--string-file", type=Path, required=True,
                        help="STRING protein.links file (e.g., 9606.protein.links.v12.0.txt)")
    parser.add_argument("--string-info", type=Path, default=None,
                        help="STRING protein.info file for ENSP->gene symbol mapping")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_vp.db")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "vp" / "string_gene_graph.pkl")
    parser.add_argument("--min-score", type=int, default=MIN_COMBINED_SCORE)
    args = parser.parse_args(argv)

    from negbiodb_vp.vp_db import get_connection

    # Load STRING links
    logger.info("Loading STRING links from %s...", args.string_file)
    links_df = pd.read_csv(args.string_file, sep=" ")
    logger.info("Total links: %d", len(links_df))

    # Filter by score
    links_df = links_df[links_df["combined_score"] >= args.min_score]
    logger.info("After score filter (>=%d): %d", args.min_score, len(links_df))

    # Build ENSP -> gene symbol mapping
    if args.string_info and args.string_info.exists():
        info_df = pd.read_csv(args.string_info, sep="\t")
        ensp_to_gene = dict(zip(info_df["#string_protein_id"], info_df["preferred_name"]))
    else:
        # Fallback: try to map from DB genes
        logger.warning("No STRING info file. Using DB gene symbols for mapping.")
        ensp_to_gene = {}

    # Load gene info from DB
    conn = get_connection(args.db)
    try:
        genes_df = pd.read_sql_query("""
            SELECT gene_id, gene_symbol, pli_score, loeuf_score, missense_z
            FROM genes WHERE gene_symbol IS NOT NULL
        """, conn)

        # Variant count per gene
        var_counts = pd.read_sql_query("""
            SELECT g.gene_symbol, COUNT(*) as variant_count,
                   AVG(CASE v.consequence_type
                       WHEN 'missense' THEN 3.0
                       WHEN 'nonsense' THEN 5.0
                       WHEN 'frameshift' THEN 5.0
                       WHEN 'splice' THEN 4.0
                       WHEN 'synonymous' THEN 1.0
                       ELSE 2.0 END) as mean_severity
            FROM variants v
            JOIN genes g ON v.gene_id = g.gene_id
            GROUP BY g.gene_symbol
        """, conn)
    finally:
        conn.close()

    gene_info = genes_df.set_index("gene_symbol")
    var_info = var_counts.set_index("gene_symbol")

    # Build gene-level graph
    # Map ENSP to gene symbols
    all_genes = set(gene_info.index)
    edges = []

    for _, row in links_df.iterrows():
        p1 = row["protein1"]
        p2 = row["protein2"]
        g1 = ensp_to_gene.get(p1)
        g2 = ensp_to_gene.get(p2)
        if g1 and g2 and g1 in all_genes and g2 in all_genes and g1 != g2:
            edges.append((g1, g2))

    # Deduplicate (undirected)
    edge_set = set()
    for g1, g2 in edges:
        key = (min(g1, g2), max(g1, g2))
        edge_set.add(key)

    logger.info("Gene-level edges: %d", len(edge_set))

    # Build node list
    edge_genes = set()
    for g1, g2 in edge_set:
        edge_genes.add(g1)
        edge_genes.add(g2)

    gene_list = sorted(edge_genes)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    idx_to_gene = {i: g for g, i in gene_to_idx.items()}

    # Build edge_index
    src_nodes = []
    dst_nodes = []
    for g1, g2 in edge_set:
        i, j = gene_to_idx[g1], gene_to_idx[g2]
        src_nodes.extend([i, j])  # undirected: add both directions
        dst_nodes.extend([j, i])

    edge_index = np.array([src_nodes, dst_nodes], dtype=np.int64)

    # Build node features: [pLI, LOEUF, missense_z, variant_count, mean_severity]
    n_genes = len(gene_list)
    node_features = np.zeros((n_genes, 5), dtype=np.float32)

    for i, gene in enumerate(gene_list):
        if gene in gene_info.index:
            row = gene_info.loc[gene]
            node_features[i, 0] = row.get("pli_score") or 0.0
            node_features[i, 1] = row.get("loeuf_score") or 1.0
            node_features[i, 2] = row.get("missense_z") or 0.0
        if gene in var_info.index:
            vrow = var_info.loc[gene]
            node_features[i, 3] = vrow.get("variant_count") or 0.0
            node_features[i, 4] = vrow.get("mean_severity") or 2.0

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graph_data = {
        "edge_index": edge_index,
        "node_features": node_features,
        "gene_to_idx": gene_to_idx,
        "idx_to_gene": idx_to_gene,
    }
    with open(args.output, "wb") as f:
        pickle.dump(graph_data, f)

    logger.info("Gene graph saved: %d nodes, %d edges -> %s",
                n_genes, len(edge_set), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
