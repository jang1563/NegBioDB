#!/usr/bin/env python3
"""Download computational score files for VP domain (HPC only).

These files are large and must be downloaded on HPC scratch:
  - CADD v1.7: ~80 GB (whole_genome_SNVs.tsv.gz)
  - REVEL v1.3: ~900 MB (revel-v1.3_all_chromosomes.zip)
  - AlphaMissense: ~550 MB (AlphaMissense_hg38.tsv.gz)

Usage (on HPC):
    python scripts_vp/download_scores.py --output-dir /scratch/users/jak4013/vp_scores

This script prints wget/curl commands for manual download on HPC.
"""

import argparse
import sys
from pathlib import Path


SCORE_URLS = {
    "CADD v1.7": "https://kircherlab.bihealth.org/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz",
    "REVEL v1.3": "https://zenodo.org/records/7072866/files/revel-v1.3_all_chromosomes.zip",
    "AlphaMissense": "https://zenodo.org/records/10813168/files/AlphaMissense_hg38.tsv.gz",
}


def main():
    parser = argparse.ArgumentParser(description="Download score files for VP ETL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory (HPC scratch)")
    args = parser.parse_args()

    print("=== VP Score Downloads (run on HPC) ===\n")
    print(f"Target directory: {args.output_dir}\n")

    for name, url in SCORE_URLS.items():
        filename = url.split("/")[-1]
        dest = args.output_dir / filename
        print(f"# {name}")
        print(f"wget -c -O {dest} '{url}'\n")

    print("After download, export VP targets and extract scores:")
    print("  PYTHONPATH=src python scripts_vp/export_score_targets.py --output /scratch/.../vp_score_targets.tsv")
    print("  PYTHONPATH=src python scripts_vp/extract_scores_hpc.py --variant-list /scratch/.../vp_score_targets.tsv --output /scratch/.../merged_scores.tsv --cadd <path> --revel <path> --alphamissense <path>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
