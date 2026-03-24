#!/usr/bin/env python3
"""Fetch gene descriptions from NCBI gene_info and update genes table.

Downloads Homo_sapiens.gene_info.gz from NCBI FTP, extracts descriptions,
and updates the genes table in the GE database.

Usage:
    PYTHONPATH=src python scripts_depmap/fetch_gene_descriptions.py
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NCBI_GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
)


def fetch_gene_descriptions(data_dir: Path) -> dict[int, str]:
    """Download and parse NCBI gene_info for human gene descriptions.

    Returns {entrez_id: description}.
    """
    gene_info_gz = data_dir / "Homo_sapiens.gene_info.gz"

    if not gene_info_gz.exists():
        logger.info("Downloading NCBI gene_info...")
        resp = requests.get(NCBI_GENE_INFO_URL, stream=True, timeout=120)
        resp.raise_for_status()
        gene_info_gz.parent.mkdir(parents=True, exist_ok=True)
        with open(gene_info_gz, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        logger.info("Downloaded %.1f MB", gene_info_gz.stat().st_size / 1e6)
    else:
        logger.info("Using cached gene_info: %s", gene_info_gz)

    descriptions: dict[int, str] = {}
    with gzip.open(gene_info_gz, "rt") as f:
        header = f.readline()  # skip header
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            # Fields: tax_id, GeneID, Symbol, LocusTag, Synonyms, dbXrefs,
            #         chromosome, map_location, description, ...
            try:
                entrez_id = int(fields[1])
            except ValueError:
                continue
            desc = fields[8]
            if desc and desc != "-":
                descriptions[entrez_id] = desc

    logger.info("Parsed %d gene descriptions from NCBI", len(descriptions))
    return descriptions


def update_gene_descriptions(db_path: Path, descriptions: dict[int, str]) -> int:
    """Update genes table with descriptions."""
    from negbiodb_depmap.depmap_db import get_connection

    conn = get_connection(db_path)
    updated = 0
    batch = []

    for entrez_id, desc in descriptions.items():
        batch.append((desc, entrez_id))
        if len(batch) >= 5000:
            conn.executemany(
                "UPDATE genes SET description = ? WHERE entrez_id = ?", batch
            )
            updated += len(batch)
            batch = []

    if batch:
        conn.executemany(
            "UPDATE genes SET description = ? WHERE entrez_id = ?", batch
        )
        updated += len(batch)

    conn.commit()

    # Check how many were actually updated
    r = conn.execute(
        "SELECT COUNT(*) FROM genes WHERE description IS NOT NULL"
    ).fetchone()
    conn.close()

    logger.info("Updated %d genes, %d now have descriptions", updated, r[0])
    return r[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch gene descriptions from NCBI.")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_depmap.db")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "depmap_raw")
    args = parser.parse_args(argv)

    descriptions = fetch_gene_descriptions(args.data_dir)
    n_updated = update_gene_descriptions(args.db, descriptions)
    logger.info("Done: %d genes with descriptions", n_updated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
