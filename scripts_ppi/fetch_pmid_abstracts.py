#!/usr/bin/env python3
"""Fetch PubMed abstracts for IntAct publication PMIDs.

Stores title, abstract, publication year, and journal in
ppi_publication_abstracts table (created by migration 002).

Usage:
    PYTHONPATH=src python scripts_ppi/fetch_pmid_abstracts.py
    PYTHONPATH=src python scripts_ppi/fetch_pmid_abstracts.py --db data/negbiodb_ppi.db
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 50  # PubMed allows up to 200, but 50 is safer


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed efetch XML to extract article metadata."""
    root = ElementTree.fromstring(xml_text)
    articles = []

    for article_el in root.findall(".//PubmedArticle"):
        pmid_el = article_el.find(".//PMID")
        if pmid_el is None or pmid_el.text is None:
            continue
        pmid = int(pmid_el.text)

        # Title
        title_el = article_el.find(".//ArticleTitle")
        title = title_el.text if title_el is not None and title_el.text else None

        # Abstract
        abstract_parts = []
        for abs_el in article_el.findall(".//AbstractText"):
            label = abs_el.get("Label", "")
            text = "".join(abs_el.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts) if abstract_parts else None

        # Publication year
        pub_year = None
        for date_el in [
            article_el.find(".//ArticleDate"),
            article_el.find(".//PubDate"),
        ]:
            if date_el is not None:
                year_el = date_el.find("Year")
                if year_el is not None and year_el.text:
                    try:
                        pub_year = int(year_el.text)
                        break
                    except ValueError:
                        pass

        # Journal
        journal_el = article_el.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else None

        if abstract:
            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "publication_year": pub_year,
                "journal": journal,
            })

    return articles


def fetch_abstracts(pmids: list[int], session: requests.Session) -> list[dict]:
    """Fetch abstracts for a batch of PMIDs from PubMed."""
    params = {
        "db": "pubmed",
        "id": ",".join(str(p) for p in pmids),
        "rettype": "xml",
        "retmode": "xml",
    }

    for attempt in range(3):
        try:
            resp = session.get(EFETCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            return _parse_pubmed_xml(resp.text)
        except requests.RequestException as e:
            if attempt < 2:
                logger.warning("Attempt %d failed: %s, retrying...", attempt + 1, e)
                time.sleep(2 ** attempt)
            else:
                raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts for IntAct PMIDs.")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "negbiodb_ppi.db")
    args = parser.parse_args(argv)

    from negbiodb_ppi.ppi_db import get_connection

    conn = get_connection(args.db)

    # Get unique PMIDs from IntAct experiments
    rows = conn.execute(
        "SELECT DISTINCT pubmed_id FROM ppi_experiments "
        "WHERE source_db='intact' AND pubmed_id IS NOT NULL"
    ).fetchall()
    all_pmids = [int(r[0]) for r in rows if r[0] is not None]

    # Also include HuRI PMID
    huri_pmid = 32296183
    if huri_pmid not in all_pmids:
        all_pmids.append(huri_pmid)

    # Filter out already-fetched PMIDs
    existing = set(
        r[0] for r in conn.execute(
            "SELECT pmid FROM ppi_publication_abstracts"
        ).fetchall()
    )
    pmids = [p for p in all_pmids if p not in existing]
    logger.info("Total PMIDs: %d, already fetched: %d, to fetch: %d",
                len(all_pmids), len(existing), len(pmids))

    if not pmids:
        logger.info("All abstracts already fetched.")
        conn.close()
        return 0

    session = requests.Session()
    session.headers["User-Agent"] = "NegBioDB/1.0 (negbiodb@institution.edu)"

    total_fetched = 0
    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i:i + BATCH_SIZE]
        try:
            articles = fetch_abstracts(batch, session)
        except Exception as e:
            logger.error("Failed batch %d-%d: %s", i, i + len(batch), e)
            continue

        for article in articles:
            conn.execute(
                "INSERT OR IGNORE INTO ppi_publication_abstracts "
                "(pmid, title, abstract, publication_year, journal) "
                "VALUES (?, ?, ?, ?, ?)",
                (article["pmid"], article["title"], article["abstract"],
                 article["publication_year"], article["journal"]),
            )
            total_fetched += 1

        conn.commit()
        logger.info(
            "Batch %d/%d: fetched %d abstracts (total: %d)",
            i // BATCH_SIZE + 1,
            (len(pmids) + BATCH_SIZE - 1) // BATCH_SIZE,
            len(articles), total_fetched,
        )
        time.sleep(0.5)  # NCBI rate limit: 3 requests/sec without API key

    # Summary
    year_dist = conn.execute(
        "SELECT publication_year, COUNT(*) FROM ppi_publication_abstracts "
        "GROUP BY publication_year ORDER BY publication_year"
    ).fetchall()
    logger.info("Publication year distribution:")
    for year, count in year_dist:
        logger.info("  %s: %d", year, count)

    conn.close()
    logger.info("Done. Fetched %d abstracts.", total_fetched)
    return 0


if __name__ == "__main__":
    sys.exit(main())
