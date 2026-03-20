#!/usr/bin/env python
"""Download hu.MAP 3.0 pair lists.

Downloads:
  - neg_train_ppis.txt  (negative training pairs)
  - neg_test_ppis.txt   (negative test pairs)
  - train_ppis.txt      (positive training pairs, reference)
  - test_ppis.txt       (positive test pairs, reference)

License: CC0 (Public Domain)
"""

import argparse
from pathlib import Path

from negbiodb.download import download_file_http, load_config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Download hu.MAP 3.0 pair data")
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=None,
        help="Destination directory (default: from config)",
    )
    args = parser.parse_args()

    cfg = load_config()
    humap_cfg = cfg["ppi_domain"]["downloads"]["humap"]
    dest_dir = Path(args.dest_dir or (_PROJECT_ROOT / humap_cfg["dest_dir"]))
    base_url = humap_cfg["base_url"].rstrip("/")

    files = [
        humap_cfg["neg_train"],
        humap_cfg["neg_test"],
        humap_cfg["pos_train"],
        humap_cfg["pos_test"],
    ]

    for fname in files:
        url = f"{base_url}/{fname}"
        download_file_http(url, dest_dir / fname, desc=fname)

    print(f"\nhu.MAP 3.0 data downloaded to {dest_dir}")


if __name__ == "__main__":
    main()
