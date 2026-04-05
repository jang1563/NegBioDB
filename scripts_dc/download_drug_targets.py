#!/usr/bin/env python3
"""Download drug-target mapping data from DGIdb v5 GraphQL API.

DGIdb v5 removed the old monthly TSV downloads. This script uses the
GraphQL API (https://dgidb.org/api/graphql) to paginate through all
drug-gene interactions and write interactions.tsv.

Output:
  - interactions.tsv — drug_name, gene_name, interaction_type, source_name

Usage:
    python scripts_dc/download_drug_targets.py [--output-dir data/dc/drug_targets] [--force]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DGIDB_API = "https://dgidb.org/api/graphql"

INTERACTIONS_QUERY = """
query Interactions($first: Int!, $after: String) {
  interactions(first: $first, after: $after) {
    nodes {
      drug { name }
      gene { name }
      interactionTypes { type }
      sources { fullName }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""


def fetch_interactions_page(after: str | None, page_size: int = 1000) -> dict:
    """Fetch one page of interactions from the DGIdb GraphQL API."""
    variables: dict = {"first": page_size}
    if after:
        variables["after"] = after

    resp = requests.post(
        DGIDB_API,
        json={"query": INTERACTIONS_QUERY, "variables": variables},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]["interactions"]


def download_interactions(output_path: Path, force: bool = False, page_size: int = 1000) -> int:
    """Download all DGIdb interactions via GraphQL API.

    Returns number of interactions written.
    """
    if output_path.exists() and not force:
        size_mb = output_path.stat().st_size / 1e6
        print(f"  Already exists ({size_mb:.1f} MB), skipping: {output_path.name}")
        return 0

    print(f"  Downloading interactions via DGIdb GraphQL API ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    page_num = 0
    cursor = None

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["drug_name", "gene_name", "interaction_type", "source_name"])

        while True:
            try:
                result = fetch_interactions_page(cursor, page_size)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    print("  Rate limited (429), waiting 10s ...")
                    time.sleep(10)
                    continue
                raise

            nodes = result["nodes"]
            for node in nodes:
                drug_name = node["drug"]["name"] if node["drug"] else None
                gene_name = node["gene"]["name"] if node["gene"] else None
                if not drug_name or not gene_name:
                    continue
                interaction_types = "; ".join(
                    it["type"] for it in node.get("interactionTypes", []) if it.get("type")
                )
                sources = "; ".join(
                    s["fullName"] for s in node.get("sources", []) if s.get("fullName")
                )
                writer.writerow([drug_name, gene_name, interaction_types, sources])
                total_written += 1

            page_num += 1
            if page_num % 10 == 0:
                print(f"    Page {page_num}: {total_written} interactions so far ...")

            page_info = result["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            cursor = page_info["endCursor"]
            time.sleep(0.2)  # 5 req/sec max

    size_mb = output_path.stat().st_size / 1e6
    if total_written == 0:
        output_path.unlink()
        raise IOError("DGIdb returned 0 interactions")
    print(f"    Done: {total_written} interactions, {size_mb:.1f} MB")
    return total_written


def main():
    parser = argparse.ArgumentParser(
        description="Download drug-target mappings from DGIdb v5 API"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "dc" / "drug_targets",
        help="Output directory (default: data/dc/drug_targets)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    print(f"DGIdb drug-target download → {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_interactions(args.output_dir / "interactions.tsv", args.force)
    except Exception as e:
        print(f"  ERROR downloading interactions.tsv: {e}", file=sys.stderr)
        return 1

    print("\nDrug-target files downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
