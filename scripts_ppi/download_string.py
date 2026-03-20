#!/usr/bin/env python
"""Download STRING v12.0 human protein links and UniProt mapping.

Downloads:
  - 9606.protein.links.v12.0.txt.gz  (~200 MB compressed)
  - human.uniprot_2_string.2018.tsv.gz  (~300 KB)

License: CC BY 4.0
"""

import argparse
from pathlib import Path

from negbiodb.download import download_file_http, load_config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Download STRING v12.0 data for PPI negatives"
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=None,
        help="Destination directory (default: from config)",
    )
    args = parser.parse_args()

    cfg = load_config()
    string_cfg = cfg["ppi_domain"]["downloads"]["string"]
    dest_dir = Path(args.dest_dir or (_PROJECT_ROOT / string_cfg["dest_dir"]))

    # Protein links file (large)
    links_fname = string_cfg["links_url"].rsplit("/", 1)[-1]
    download_file_http(
        string_cfg["links_url"],
        dest_dir / links_fname,
        desc=links_fname,
    )

    # UniProt mapping file (small)
    mapping_fname = string_cfg["mapping_url"].rsplit("/", 1)[-1]
    download_file_http(
        string_cfg["mapping_url"],
        dest_dir / mapping_fname,
        desc=mapping_fname,
    )

    print(f"\nSTRING v12.0 data downloaded to {dest_dir}")


if __name__ == "__main__":
    main()
