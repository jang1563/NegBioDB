#!/usr/bin/env python3
"""Fetch MANE Select canonical protein sequences from UniProt for ESM2.

For each gene with missense variants, fetches the canonical protein sequence
from UniProt using the gene symbol. Creates mutant sequences by substituting
the variant amino acid.

Output: data/vp/protein_sequences.tsv

Usage:
    PYTHONPATH=src python scripts_vp/fetch_protein_sequences.py --db data/negbiodb_vp.db
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"


def fetch_sequence(gene_symbol: str) -> str | None:
    """Fetch canonical protein sequence from UniProt for a human gene."""
    query = f"gene_exact:{gene_symbol}+AND+organism_id:9606+AND+reviewed:true"
    url = f"{UNIPROT_API}?query={query}&format=fasta&size=1"

    try:
        with urlopen(url, timeout=30) as resp:
            data = resp.read().decode("utf-8")
    except (HTTPError, Exception) as e:
        logger.warning("Failed to fetch %s: %s", gene_symbol, e)
        return None

    if not data.strip():
        return None

    lines = data.strip().split("\n")
    seq_lines = [l for l in lines if not l.startswith(">")]
    return "".join(seq_lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch protein sequences from UniProt.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_vp.db")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "vp" / "protein_sequences.tsv")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    args = parser.parse_args(argv)

    from negbiodb_vp.vp_db import get_connection

    conn = get_connection(args.db)
    try:
        # Get unique genes with missense variants
        rows = conn.execute("""
            SELECT DISTINCT g.gene_symbol
            FROM genes g
            JOIN variants v ON g.gene_id = v.gene_id
            WHERE v.consequence_type = 'missense'
            AND g.gene_symbol IS NOT NULL
            ORDER BY g.gene_symbol
        """).fetchall()
    finally:
        conn.close()

    genes = [r[0] for r in rows]
    logger.info("Genes with missense variants: %d", len(genes))

    # Resume support
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fetched = set()
    if args.output.exists():
        with open(args.output) as f:
            for line in f:
                if line.startswith("gene_symbol"):
                    continue
                parts = line.strip().split("\t")
                if parts:
                    fetched.add(parts[0])
        logger.info("Already fetched: %d", len(fetched))

    remaining = [g for g in genes if g not in fetched]
    logger.info("Remaining to fetch: %d", len(remaining))

    with open(args.output, "a") as f:
        if not fetched:
            f.write("gene_symbol\tsequence_length\tsequence\n")

        for i, gene in enumerate(remaining):
            seq = fetch_sequence(gene)
            if seq:
                f.write(f"{gene}\t{len(seq)}\t{seq}\n")
                f.flush()

            if (i + 1) % 100 == 0:
                logger.info("Progress: %d/%d", i + 1, len(remaining))

            time.sleep(args.delay)

    logger.info("Done. Sequences saved to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
