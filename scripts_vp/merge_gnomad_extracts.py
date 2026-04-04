#!/usr/bin/env python3
"""Merge per-chromosome gnomAD extraction shards into one TSV.

Usage:
    PYTHONPATH=src python scripts_vp/merge_gnomad_extracts.py \
        --input-glob "/scratch/negbiodb/vp_gnomad/variant_frequencies.chr*.tsv" \
        --output data/vp/gnomad/variant_frequencies.tsv
"""

import argparse
import glob
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-chromosome gnomAD extraction TSV shards"
    )
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob pattern matching shard TSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Merged TSV output path",
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    shard_paths = [Path(path) for path in glob.glob(args.input_glob)]
    if not shard_paths:
        raise SystemExit(f"No shard files matched: {args.input_glob}")

    from negbiodb_vp.etl_gnomad import merge_gnomad_tsv_shards

    stats = merge_gnomad_tsv_shards(shard_paths, args.output)
    print("\n=== gnomAD Shard Merge Results ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v:,}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
