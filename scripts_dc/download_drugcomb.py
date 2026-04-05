#!/usr/bin/env python3
"""Download DrugComb bulk data from Zenodo for DC domain ETL.

Downloads:
  - drugcomb_data_v1.4.csv (~2.0 GB) — Raw inhibition data
  - DrugComb_drug_identifiers.xlsx (~5.9 MB) — Drug name → PubChem CID/InChIKey/SMILES
  - DrugComb_cell_line_identifiers.xlsx (~161 kB) — Cell line → COSMIC ID, tissue

Source: https://zenodo.org/records/18449193

Usage:
    python scripts_dc/download_drugcomb.py [--output-dir data/dc/drugcomb] [--force]
"""

import argparse
import sys
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Zenodo record 18449193 — DrugComb v1.4 bulk data
ZENODO_BASE = "https://zenodo.org/records/18449193/files"

DRUGCOMB_FILES = {
    "drugcomb_data_v1.4.csv": f"{ZENODO_BASE}/drugcomb_data_v1.4.csv",
    "DrugComb_drug_identifiers.xlsx": f"{ZENODO_BASE}/DrugComb_drug_identifiers.xlsx",
    "DrugComb_cell_line_identifiers.xlsx": f"{ZENODO_BASE}/DrugComb_cell_line_identifiers.xlsx",
}


def download_file(url: str, output_path: Path, force: bool = False) -> bool:
    """Download a file with progress reporting. Returns True on success."""
    if output_path.exists() and not force:
        size_mb = output_path.stat().st_size / 1e6
        print(f"  Already exists ({size_mb:.1f} MB), skipping: {output_path.name}")
        return True

    print(f"  Downloading {output_path.name} ...")
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r    {pct:.0f}%", end="", flush=True)
    print()

    size_mb = output_path.stat().st_size / 1e6
    if size_mb == 0:
        output_path.unlink()
        raise IOError(f"Downloaded file is empty (0 bytes): {output_path.name}")
    print(f"    Done: {size_mb:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download DrugComb data from Zenodo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "dc" / "drugcomb",
        help="Output directory (default: data/dc/drugcomb)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    print(f"DrugComb download → {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in DRUGCOMB_FILES.items():
        try:
            download_file(url, args.output_dir / filename, args.force)
        except Exception as e:
            print(f"  ERROR downloading {filename}: {e}", file=sys.stderr)
            return 1

    print("\nAll DrugComb files downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
