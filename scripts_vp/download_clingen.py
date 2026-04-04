#!/usr/bin/env python3
"""Download ClinGen gene-disease validity CSV.

Source: https://search.clinicalgenome.org/kb/gene-validity/download

Usage:
    python scripts_vp/download_clingen.py --output-dir data/vp
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLINGEN_URL = "https://search.clinicalgenome.org/kb/gene-validity/download"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download ClinGen gene-disease validity.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "vp")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "clingen_gene_disease_validity.csv"

    logger.info("Downloading ClinGen gene-disease validity...")
    logger.info("URL: %s", CLINGEN_URL)

    try:
        urlretrieve(CLINGEN_URL, output_file)
        logger.info("Downloaded to %s", output_file)
    except Exception as e:
        logger.error("Download failed: %s", e)
        logger.info("If download fails, manually download from: %s", CLINGEN_URL)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
