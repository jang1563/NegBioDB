#!/usr/bin/env python
"""Download IntAct negative interaction file.

Downloads:
  - intact_negative.txt (PSI-MI TAB 2.7, ~4.9 MB, ~870 negative evidences)

License: CC BY 4.0
"""

import argparse
from pathlib import Path

from negbiodb.download import download_file_http, load_config, verify_file_exists

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Download IntAct negative interactions"
    )
    parser.add_argument(
        "--dest", type=str, default=None, help="Destination path (default: from config)"
    )
    args = parser.parse_args()

    cfg = load_config()
    intact_cfg = cfg["ppi_domain"]["downloads"]["intact"]
    dest = Path(args.dest or (_PROJECT_ROOT / intact_cfg["dest"]))

    download_file_http(intact_cfg["url"], dest, desc="intact_negative.txt")

    min_bytes = intact_cfg.get("min_size_bytes", 0)
    if min_bytes and not verify_file_exists(dest, min_bytes):
        raise RuntimeError(
            f"Downloaded file too small: {dest.stat().st_size} bytes "
            f"(expected >= {min_bytes})"
        )

    print(f"\nIntAct negative file: {dest}")


if __name__ == "__main__":
    main()
