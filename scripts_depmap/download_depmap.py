#!/usr/bin/env python
"""Download DepMap data files for GE domain.

Downloads from DepMap portal API and Figshare:
  - CRISPRGeneEffect.csv (Chronos scores)
  - CRISPRGeneDependency.csv (dependency probability)
  - Model.csv (cell line metadata)
  - AchillesCommonEssentialControls.csv
  - AchillesNonessentialControls.csv
  - D2_combined_gene_dep_scores.csv (DEMETER2 RNAi)
  - Omics files (expression, CN, mutations)

Uses the DepMap download API to resolve file URLs dynamically.
"""

import argparse
import csv
import io
import sys
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "depmap_raw"

DEPMAP_FILES_API = "https://depmap.org/portal/api/download/files"

# Files to download from DepMap portal
DEPMAP_TARGET_FILES = [
    "CRISPRGeneEffect.csv",
    "CRISPRGeneDependency.csv",
    "Model.csv",
    "AchillesCommonEssentialControls.csv",
    "AchillesNonessentialControls.csv",
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "OmicsCNGene.csv",
    "OmicsSomaticMutationsMatrixDamaging.csv",
]

DEMETER2_URL = "https://ndownloader.figshare.com/files/13515395"  # D2_combined_gene_dep_scores.csv


def resolve_depmap_urls(target_files: list[str]) -> dict[str, str]:
    """Fetch signed download URLs from the DepMap portal API.

    The API returns a CSV with columns: release, release_date, filename, url, md5_hash.
    We take the first (most recent) URL for each target file.
    """
    print("Fetching download URLs from DepMap portal API...")
    resp = requests.get(DEPMAP_FILES_API, timeout=60)
    resp.raise_for_status()

    targets_remaining = set(target_files)
    urls: dict[str, str] = {}

    reader = csv.DictReader(io.StringIO(resp.text))
    for row in reader:
        fname = row.get("filename", "")
        if fname in targets_remaining:
            urls[fname] = row["url"]
            targets_remaining.discard(fname)
            if not targets_remaining:
                break

    if targets_remaining:
        print(f"  WARNING: Could not find URLs for: {targets_remaining}")

    print(f"  Resolved {len(urls)}/{len(target_files)} file URLs\n")
    return urls


def download_file(url: str, output_path: Path, force: bool = False) -> bool:
    """Download a file with progress reporting."""
    if output_path.exists() and not force:
        print(f"  SKIP (exists): {output_path.name}")
        return False

    print(f"  Downloading: {output_path.name}...")
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r    {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.0f}%)", end="")

    print(f"\n    Done: {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download DepMap data files")
    parser.add_argument("--output-dir", type=str, default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument("--skip-omics", action="store_true", help="Skip large omics files")
    parser.add_argument("--skip-rnai", action="store_true", help="Skip DEMETER2 RNAi file")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading DepMap files to {output_dir}\n")

    # Resolve URLs from DepMap API
    targets = [f for f in DEPMAP_TARGET_FILES
               if not (args.skip_omics and f.startswith("Omics"))]
    url_map = resolve_depmap_urls(targets)

    for filename in targets:
        if filename not in url_map:
            print(f"  SKIP (no URL): {filename}")
            continue
        try:
            download_file(url_map[filename], output_dir / filename, force=args.force)
        except Exception as e:
            print(f"  ERROR: {filename}: {e}")

    if not args.skip_rnai:
        try:
            download_file(DEMETER2_URL, output_dir / "D2_combined_gene_dep_scores.csv", force=args.force)
        except Exception as e:
            print(f"  ERROR: DEMETER2: {e}")

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
