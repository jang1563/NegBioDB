#!/usr/bin/env python3
"""Pre-compute ESM2-650M embeddings for missense variants.

For each missense variant, creates a mutant protein sequence and extracts
the ESM2 embedding at the variant position.

Requires: fair-esm, torch, GPU recommended.

Output: data/vp/esm2_embeddings.parquet

Usage (on HPC with GPU):
    PYTHONPATH=src python scripts_vp/precompute_esm2.py \
        --sequences data/vp/protein_sequences.tsv \
        --db data/negbiodb_vp.db \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Standard amino acid mapping
AA_1TO3 = {
    'A': 'Ala', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu', 'F': 'Phe',
    'G': 'Gly', 'H': 'His', 'I': 'Ile', 'K': 'Lys', 'L': 'Leu',
    'M': 'Met', 'N': 'Asn', 'P': 'Pro', 'Q': 'Gln', 'R': 'Arg',
    'S': 'Ser', 'T': 'Thr', 'V': 'Val', 'W': 'Trp', 'Y': 'Tyr',
}
AA_3TO1 = {v: k for k, v in AA_1TO3.items()}

HGVS_PATTERN = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")


def parse_hgvs_protein(hgvs: str) -> tuple[str, int, str] | None:
    """Parse HGVS protein notation. Returns (ref_aa, position, alt_aa) or None."""
    m = HGVS_PATTERN.search(hgvs)
    if not m:
        return None
    ref_3 = m.group(1)
    pos = int(m.group(2))
    alt_3 = m.group(3)
    ref_1 = AA_3TO1.get(ref_3)
    alt_1 = AA_3TO1.get(alt_3)
    if ref_1 and alt_1:
        return ref_1, pos, alt_1
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-compute ESM2 embeddings.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_vp.db")
    parser.add_argument("--sequences", type=Path, default=PROJECT_ROOT / "data" / "vp" / "protein_sequences.tsv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "vp" / "esm2_embeddings.parquet")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length for ESM2")
    args = parser.parse_args(argv)

    try:
        import torch
        import esm
    except ImportError:
        logger.error("Requires: pip install fair-esm torch")
        return 1

    from negbiodb_vp.vp_db import get_connection

    # Load protein sequences
    logger.info("Loading protein sequences...")
    seq_df = pd.read_csv(args.sequences, sep="\t")
    gene_to_seq = dict(zip(seq_df["gene_symbol"], seq_df["sequence"]))
    logger.info("Loaded sequences for %d genes", len(gene_to_seq))

    # Load missense variants
    conn = get_connection(args.db)
    try:
        variants = pd.read_sql_query("""
            SELECT v.variant_id, v.hgvs_protein, g.gene_symbol
            FROM variants v
            JOIN genes g ON v.gene_id = g.gene_id
            WHERE v.consequence_type = 'missense'
            AND v.hgvs_protein IS NOT NULL
            AND g.gene_symbol IS NOT NULL
        """, conn)
    finally:
        conn.close()

    logger.info("Missense variants: %d", len(variants))

    # Load ESM2-650M
    logger.info("Loading ESM2-650M model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(args.device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    EMBED_DIM = 1280
    results = []

    for i, (_, row) in enumerate(variants.iterrows()):
        variant_id = int(row["variant_id"])
        gene = row["gene_symbol"]
        hgvs = row["hgvs_protein"]

        seq = gene_to_seq.get(gene)
        if not seq:
            # Zero embedding for missing sequences
            results.append({"variant_id": variant_id, **{f"esm2_{j}": 0.0 for j in range(EMBED_DIM)}})
            continue

        parsed = parse_hgvs_protein(hgvs)
        if not parsed:
            results.append({"variant_id": variant_id, **{f"esm2_{j}": 0.0 for j in range(EMBED_DIM)}})
            continue

        ref_aa, pos, alt_aa = parsed

        # Create mutant sequence
        if pos > len(seq) or seq[pos - 1] != ref_aa:
            # Position mismatch — use zero embedding
            results.append({"variant_id": variant_id, **{f"esm2_{j}": 0.0 for j in range(EMBED_DIM)}})
            continue

        mut_seq = seq[:pos - 1] + alt_aa + seq[pos:]

        # Truncate if needed
        if len(mut_seq) > args.max_length:
            # Center around variant position
            start = max(0, pos - args.max_length // 2)
            end = start + args.max_length
            if end > len(mut_seq):
                end = len(mut_seq)
                start = max(0, end - args.max_length)
            mut_seq = mut_seq[start:end]
            var_idx_in_seq = pos - 1 - start
        else:
            var_idx_in_seq = pos - 1

        # Run ESM2
        data = [("variant", mut_seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(args.device)

        with torch.no_grad():
            output = model(batch_tokens, repr_layers=[33])
            # Extract embedding at variant position (+1 for BOS token)
            embedding = output["representations"][33][0, var_idx_in_seq + 1, :].cpu().numpy()

        results.append({"variant_id": variant_id, **{f"esm2_{j}": float(embedding[j]) for j in range(EMBED_DIM)}})

        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d", i + 1, len(variants))

    # Save as parquet
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_parquet(args.output, index=False)
    logger.info("Saved %d embeddings to %s", len(result_df), args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
