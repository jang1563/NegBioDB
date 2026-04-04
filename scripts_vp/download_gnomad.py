#!/usr/bin/env python3
"""Download gnomAD files for VP domain ETL.

Downloads:
  - Gene constraint metrics (~15 MB, local)

For sites VCF (cloud-only), run on HPC:
  gsutil -m cp gs://gcp-public-data--gnomad/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr*.vcf.bgz /scratch/

Usage:
    python scripts_vp/download_gnomad.py [--output-dir data/vp/gnomad] [--force]
"""

import argparse
import sys
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONSTRAINT_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/"
    "constraint/gnomad.v4.1.constraint_metrics.tsv"
)


def download_file(url: str, output_path: Path, force: bool = False) -> bool:
    if output_path.exists() and not force:
        size_mb = output_path.stat().st_size / 1e6
        print(f"  Already exists ({size_mb:.1f} MB): {output_path.name}")
        return True

    print(f"  Downloading {output_path.name} ...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                print(f"\r    {downloaded / total * 100:.0f}%", end="", flush=True)
    print()
    print(f"    Done: {output_path.stat().st_size / 1e6:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download gnomAD files for VP ETL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "vp" / "gnomad",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"gnomAD download → {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    download_file(CONSTRAINT_URL, args.output_dir / "gnomad.v4.1.constraint_metrics.tsv", args.force)
    print("\nNote: Sites VCF must be downloaded on HPC via gsutil. See docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
