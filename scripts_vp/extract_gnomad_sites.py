#!/usr/bin/env python3
"""Extract gnomAD sites VCF shards into TSVs for VP frequency/copper ETL.

Usage:
    PYTHONPATH=src python scripts_vp/extract_gnomad_sites.py \
        --db-path data/negbiodb_vp.db \
        --vcf /scratch/gnomad/gnomad.exomes.v4.1.sites.chr17.vcf.bgz \
        --frequencies-out data/vp/gnomad/variant_frequencies.chr17.tsv \
        --copper-out data/vp/gnomad/copper_variants.chr17.tsv

Run one chromosome per job on HPC, then merge the per-chromosome TSVs with
`scripts_vp/merge_gnomad_extracts.py` before calling `scripts_vp/load_gnomad.py`.
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Extract gnomAD VCF shards into TSVs for VP ETL"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=_PROJECT_ROOT / "data" / "negbiodb_vp.db",
    )
    parser.add_argument(
        "--vcf",
        type=Path,
        nargs="+",
        required=True,
        help="One or more gnomAD sites VCF(.bgz) shard paths",
    )
    parser.add_argument(
        "--frequencies-out",
        type=Path,
        help="Output TSV for existing ClinVar variants with gnomAD AFs",
    )
    parser.add_argument(
        "--copper-out",
        type=Path,
        help="Output TSV for common variants eligible for copper-tier loading",
    )
    parser.add_argument("--af-threshold", type=float, default=0.01)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from negbiodb_vp.etl_gnomad import export_gnomad_site_tsvs
    from negbiodb_vp.vp_db import get_connection, run_vp_migrations

    run_vp_migrations(args.db_path)
    conn = get_connection(args.db_path)
    try:
        stats = export_gnomad_site_tsvs(
            conn,
            args.vcf,
            frequencies_out=args.frequencies_out,
            copper_out=args.copper_out,
            af_threshold=args.af_threshold,
        )
        print("\n=== gnomAD Sites Extraction Results ===")
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v:,}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
