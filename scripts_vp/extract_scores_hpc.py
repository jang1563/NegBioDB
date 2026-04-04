#!/usr/bin/env python3
"""Extract VP computational scores on HPC into a merged TSV."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract VP computational scores on HPC")
    parser.add_argument("--variant-list", type=Path, required=True, help="TSV from export_score_targets.py")
    parser.add_argument("--output", type=Path, required=True, help="Merged scores TSV to write")
    parser.add_argument("--cadd", type=Path, help="Path to CADD TSV(.gz)")
    parser.add_argument("--revel", type=Path, help="Path to REVEL TSV/CSV(.zip)")
    parser.add_argument("--alphamissense", type=Path, help="Path to AlphaMissense TSV(.gz)")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from negbiodb_vp.score_extract import extract_scores_for_targets

    stats = extract_scores_for_targets(
        targets_tsv=args.variant_list,
        output_tsv=args.output,
        cadd_tsv=args.cadd,
        revel_tsv=args.revel,
        alphamissense_tsv=args.alphamissense,
    )

    print("\n=== VP Score Extraction Results ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v:,}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
