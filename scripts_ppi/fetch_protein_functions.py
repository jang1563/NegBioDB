#!/usr/bin/env python3
"""Fetch UniProt function descriptions, GO terms, and domain annotations.

Populates proteins.function_description, proteins.go_terms, and
proteins.domain_annotations for the PPI LLM benchmark.

Usage:
    PYTHONPATH=src python scripts_ppi/fetch_protein_functions.py
    PYTHONPATH=src python scripts_ppi/fetch_protein_functions.py --batch-size 200 --db data/negbiodb_ppi.db
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

UNIPROT_API = "https://rest.uniprot.org/uniprotkb"
FIELDS = "cc_function,cc_subcellular_location,go,ft_domain"
BATCH_SIZE = 100  # URL length limit — 400 causes 400 Bad Request


def _parse_function(entry: dict) -> str | None:
    """Extract function description from UniProt JSON entry."""
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                return texts[0].get("value")
    return None


def _parse_go_terms(entry: dict) -> str | None:
    """Extract GO terms as semicolon-separated string."""
    refs = entry.get("uniProtKBCrossReferences", [])
    go_terms = []
    for ref in refs:
        if ref.get("database") == "GO":
            go_id = ref.get("id", "")
            props = ref.get("properties", [])
            term = None
            for p in props:
                if p.get("key") == "GoTerm":
                    term = p.get("value")
                    break
            if term:
                go_terms.append(f"{go_id}:{term}")
            else:
                go_terms.append(go_id)
    return "; ".join(go_terms) if go_terms else None


def _parse_domains(entry: dict) -> str | None:
    """Extract domain annotations from UniProt JSON entry."""
    features = entry.get("features", [])
    domains = []
    for feat in features:
        if feat.get("type") == "Domain":
            desc = feat.get("description", "")
            if desc:
                domains.append(desc)
    return "; ".join(sorted(set(domains))) if domains else None


def fetch_batch(accessions: list[str], session: requests.Session) -> dict[str, dict]:
    """Fetch annotations for a batch of UniProt accessions.

    Returns dict mapping accession -> {function, go_terms, domains}.
    """
    query = " OR ".join(f"accession:{acc}" for acc in accessions)
    params = {
        "query": query,
        "fields": FIELDS,
        "format": "json",
        "size": len(accessions),
    }

    for attempt in range(3):
        try:
            resp = session.get(UNIPROT_API + "/search", params=params, timeout=60)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                logger.warning("Rate limited, sleeping %ds", retry_after)
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < 2:
                logger.warning("Attempt %d failed: %s, retrying...", attempt + 1, e)
                time.sleep(2 ** attempt)
            else:
                raise

    data = resp.json()
    results = {}
    for entry in data.get("results", []):
        acc = entry.get("primaryAccession")
        if acc:
            results[acc] = {
                "function_description": _parse_function(entry),
                "go_terms": _parse_go_terms(entry),
                "domain_annotations": _parse_domains(entry),
            }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch UniProt protein annotations.")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "negbiodb_ppi.db")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)

    from negbiodb_ppi.ppi_db import get_connection

    conn = get_connection(args.db)

    # Get proteins needing annotations
    rows = conn.execute(
        "SELECT protein_id, uniprot_accession FROM proteins "
        "WHERE function_description IS NULL "
        "ORDER BY protein_id"
    ).fetchall()
    total = len(rows)
    logger.info("Proteins needing annotations: %d", total)

    if total == 0:
        logger.info("All proteins already annotated.")
        conn.close()
        return 0

    session = requests.Session()
    session.headers["User-Agent"] = "NegBioDB/1.0 (jak4013@med.cornell.edu)"

    updated = 0
    for i in range(0, total, args.batch_size):
        batch = rows[i:i + args.batch_size]
        accessions = [r[1] for r in batch]
        id_map = {r[1]: r[0] for r in batch}

        try:
            results = fetch_batch(accessions, session)
        except Exception as e:
            logger.error("Failed batch %d-%d: %s", i, i + len(batch), e)
            continue

        for acc, annotations in results.items():
            pid = id_map.get(acc)
            if pid is None:
                continue
            conn.execute(
                "UPDATE proteins SET function_description=?, go_terms=?, domain_annotations=? "
                "WHERE protein_id=?",
                (annotations["function_description"], annotations["go_terms"],
                 annotations["domain_annotations"], pid),
            )
            updated += 1

        conn.commit()
        logger.info(
            "Batch %d/%d: fetched %d/%d annotations (total updated: %d)",
            i // args.batch_size + 1,
            (total + args.batch_size - 1) // args.batch_size,
            len(results), len(batch), updated,
        )
        time.sleep(0.5)  # Be polite to UniProt

    conn.close()
    logger.info("Done. Updated %d/%d proteins.", updated, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
