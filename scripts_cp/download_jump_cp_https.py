#!/usr/bin/env python3
"""Manifest-driven HTTPS downloader for mirrored JUMP Cell Painting assets."""

from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET_BASE = "https://cellpainting-gallery.s3.amazonaws.com"
GITHUB_METADATA_BASE = "https://raw.githubusercontent.com/jump-cellpainting/datasets/main/metadata"
XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
GITHUB_METADATA_FILES = [
    "README.md",
    "plate.csv.gz",
    "well.csv.gz",
    "compound.csv.gz",
    "compound_source.csv.gz",
    "perturbation_control.csv",
]


def list_keys(prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        url = f"{BUCKET_BASE}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=120) as response:
            root = ET.fromstring(response.read())
        keys.extend(
            elem.text for elem in root.findall("s3:Contents/s3:Key", XML_NS) if elem.text
        )
        next_token = root.findtext("s3:NextContinuationToken", default=None, namespaces=XML_NS)
        if not next_token:
            break
        token = next_token
    return keys


def download_key(key: str, target_root: Path) -> Path:
    out_path = target_root / key
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    url = f"{BUCKET_BASE}/{key}"
    with urllib.request.urlopen(url, timeout=120) as response:
        out_path.write_bytes(response.read())
    return out_path


def download_url(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    with urllib.request.urlopen(url, timeout=120) as response:
        out_path.write_bytes(response.read())
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download JUMP Cell Painting metadata over HTTPS.")
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--dataset-prefix", default="cpg0016-jump")
    parser.add_argument("--include-raw-images", action="store_true")
    parser.add_argument("--skip-github-metadata", action="store_true")
    args = parser.parse_args(argv)

    prefixes = [
        f"{args.dataset_prefix}/workspace/structure/",
        f"{args.dataset_prefix}/workspace_dl/",
    ]
    if args.include_raw_images:
        prefixes.append(f"{args.dataset_prefix}/images/")

    downloaded = 0
    for prefix in prefixes:
        for key in list_keys(prefix):
            if key.endswith("/"):
                continue
            download_key(key, args.target_root)
            downloaded += 1

    metadata_downloaded = 0
    if not args.skip_github_metadata:
        metadata_root = args.target_root / "metadata"
        for filename in GITHUB_METADATA_FILES:
            download_url(f"{GITHUB_METADATA_BASE}/{filename}", metadata_root / filename)
            metadata_downloaded += 1

    print(
        f"Downloaded {downloaded} S3 files and {metadata_downloaded} GitHub metadata files "
        f"under {args.target_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
