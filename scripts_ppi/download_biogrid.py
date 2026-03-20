#!/usr/bin/env python
"""Download BioGRID positive PPI reference data.

Downloads:
  - BIOGRID-ALL-LATEST.tab3.zip  (~200 MB)

License: MIT
"""

import argparse
import zipfile
from pathlib import Path

from negbiodb.download import download_file_http, load_config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Download BioGRID positive PPI reference"
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=None,
        help="Destination directory (default: from config)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extraction of zip file",
    )
    args = parser.parse_args()

    cfg = load_config()
    biogrid_cfg = cfg["ppi_domain"]["downloads"]["biogrid"]
    dest_dir = Path(args.dest_dir or (_PROJECT_ROOT / biogrid_cfg["dest_dir"]))

    zip_path = dest_dir / "BIOGRID-ALL-LATEST.tab3.zip"
    download_file_http(biogrid_cfg["url"], zip_path, desc="BioGRID")

    if not args.no_extract and zip_path.exists():
        print("Extracting BioGRID zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        print(f"Extracted to {dest_dir}")

    print(f"\nBioGRID data in {dest_dir}")


if __name__ == "__main__":
    main()
