#!/usr/bin/env python3
"""Download ClinVar tab-delimited files for VP domain ETL.

Downloads:
  - variant_summary.txt.gz (~435 MB)
  - submission_summary.txt.gz (~250 MB)
  - gene_condition_source_id (~5 MB)

Usage:
    python scripts_vp/download_clinvar.py [--output-dir data/vp/clinvar] [--force]
"""

import argparse
import sys
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLINVAR_FILES = {
    "variant_summary.txt.gz": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",
    "submission_summary.txt.gz": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/submission_summary.txt.gz",
    "gene_condition_source_id": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id",
}


def download_file(url: str, output_path: Path, force: bool = False) -> bool:
    """Download a file with progress reporting. Returns True on success."""
    if output_path.exists() and not force:
        size_mb = output_path.stat().st_size / 1e6
        print(f"  Already exists ({size_mb:.1f} MB), skipping: {output_path.name}")
        return True

    print(f"  Downloading {output_path.name} ...")
    resp = requests.get(url, stream=True, timeout=300)
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
    print(f"    Done: {size_mb:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download ClinVar files for VP ETL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "vp" / "clinvar",
        help="Output directory (default: data/vp/clinvar)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    print(f"ClinVar download → {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in CLINVAR_FILES.items():
        try:
            download_file(url, args.output_dir / filename, args.force)
        except Exception as e:
            print(f"  ERROR downloading {filename}: {e}", file=sys.stderr)
            return 1

    print("\nAll ClinVar files downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
