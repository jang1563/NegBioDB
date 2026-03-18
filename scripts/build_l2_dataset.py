#!/usr/bin/env python3
"""Build L2 Abstract Annotation dataset for LLM benchmark.

Step 1 (automated): Search PubMed for abstracts reporting negative DTI results.
Step 2 (manual): Human annotation of gold-standard structured extraction.

This script handles Step 1: abstract retrieval and candidate selection.

Search strategy:
  - PubMed E-utilities with queries for negative DTI reporting
  - Stratify: 40 explicit / 30 hedged / 30 implicit negative results
  - Output: candidate abstracts for human review

Output: exports/llm_benchmarks/l2_candidates.jsonl
        (Gold file created manually: l2_gold.jsonl)
"""

import argparse
import json
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "exports" / "llm_benchmarks" / "l2_candidates.jsonl"

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DELAY = 0.4  # seconds between API calls (NCBI recommends max 3/sec without key)


def search_pubmed(query: str, retmax: int = 100) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    params = (
        f"db=pubmed&term={urllib.request.quote(query)}"
        f"&retmax={retmax}&retmode=json&sort=relevance"
    )
    url = f"{PUBMED_BASE}/esearch.fcgi?{params}"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch abstract text for list of PMIDs using efetch."""
    if not pmids:
        return []

    # Batch fetch (up to 200 per request)
    results = []
    for i in range(0, len(pmids), 100):
        batch = pmids[i : i + 100]
        ids = ",".join(batch)
        url = f"{PUBMED_BASE}/efetch.fcgi?db=pubmed&id={ids}&rettype=xml"

        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                xml_text = resp.read()
        except Exception as e:
            print(f"  Fetch error for batch {i//100}: {e}")
            continue

        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//Abstract")

            if pmid_el is None or abstract_el is None:
                continue

            pmid = pmid_el.text
            title = title_el.text if title_el is not None else ""

            # Concatenate all AbstractText elements
            abstract_parts = []
            for at in abstract_el.findall("AbstractText"):
                label = at.get("Label", "")
                text = "".join(at.itertext()).strip()
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract_text = " ".join(abstract_parts)

            # Get year
            year_el = article.find(".//PubDate/Year")
            year = int(year_el.text) if year_el is not None else None

            if abstract_text and len(abstract_text) > 100:
                results.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract_text": abstract_text,
                        "year": year,
                    }
                )

        time.sleep(DELAY)

    return results


# Negative DTI reporting search queries by category
QUERIES = {
    "explicit": [
        # Explicit statements of inactivity
        '("did not inhibit" OR "no inhibition" OR "showed no activity") AND '
        '("drug target" OR "IC50" OR "binding assay") AND '
        '("selectivity" OR "specificity") AND 2020:2025[dp]',
        # HTS negative results
        '("inactive" OR "no effect") AND ("high-throughput screening" OR "HTS") '
        'AND ("kinase" OR "protease" OR "GPCR") AND 2018:2025[dp]',
    ],
    "hedged": [
        # Hedged/qualified negative results
        '("weak activity" OR "marginal" OR "modest inhibition") AND '
        '("IC50" OR "Ki" OR "Kd") AND ("selectivity panel" OR "kinase panel") '
        'AND 2019:2025[dp]',
        # Borderline results
        '("borderline" OR "insufficient" OR "below threshold") AND '
        '("drug discovery" OR "medicinal chemistry") AND '
        '("IC50 >" OR "Ki >") AND 2018:2025[dp]',
    ],
    "implicit": [
        # Implicit negatives (selectivity studies where some targets are inactive)
        '("selectivity profile" OR "kinome scan" OR "selectivity panel") AND '
        '("selective for" OR "selective inhibitor") AND '
        '("drug target interaction" OR "kinase inhibitor") AND 2019:2025[dp]',
        # SAR studies with inactive analogues
        '("structure-activity relationship" OR "SAR") AND '
        '("inactive analogue" OR "loss of activity" OR "no binding") '
        'AND 2018:2025[dp]',
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Search PubMed for L2 candidates")
    parser.add_argument(
        "--per-query", type=int, default=80, help="Max PMIDs per query"
    )
    args = parser.parse_args()

    all_abstracts = {}  # pmid -> record (dedup)

    for category, queries in QUERIES.items():
        print(f"\n=== Category: {category} ===")
        for q in queries:
            print(f"  Query: {q[:80]}...")
            pmids = search_pubmed(q, retmax=args.per_query)
            print(f"  Found: {len(pmids)} PMIDs")

            if pmids:
                abstracts = fetch_abstracts(pmids)
                for a in abstracts:
                    a["search_category"] = category
                    all_abstracts[a["pmid"]] = a
                print(f"  Fetched: {len(abstracts)} abstracts with text")

            time.sleep(DELAY)

    # Stratify: target 40 explicit / 30 hedged / 30 implicit
    by_cat = {}
    for rec in all_abstracts.values():
        by_cat.setdefault(rec["search_category"], []).append(rec)

    targets = {"explicit": 50, "hedged": 40, "implicit": 40}
    selected = []
    for cat, recs in by_cat.items():
        n = targets.get(cat, 30)
        selected.extend(recs[:n])

    print(f"\n=== Summary ===")
    print(f"Total unique abstracts: {len(all_abstracts)}")
    from collections import Counter

    cat_counts = Counter(r["search_category"] for r in selected)
    print(f"Selected: {len(selected)} ({dict(cat_counts)})")

    # Save candidates
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for i, rec in enumerate(selected):
            rec["candidate_id"] = f"L2-C{i:04d}"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"\nNext step: Human review + annotation → l2_gold.jsonl")


if __name__ == "__main__":
    main()
