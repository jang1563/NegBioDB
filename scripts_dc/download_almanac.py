#!/usr/bin/env python3
"""Download NCI-ALMANAC combination drug growth data for DC domain ETL.

Downloads:
  - ComboDrugGrowth_Nov2017.csv — 3x3 dose-response matrices
    311,604 combinations, 104 FDA-approved drugs, 60 NCI-60 cell lines

Source options (in priority order):
  1. NCI DTP CellMiner FTP mirror (if available)
  2. NCI Wiki attachment (may require NCI VPN/authentication — try first)

Note: NCI-ALMANAC is public domain (US government work, not copyrighted).

Note on data overlap: DrugComb v1.4 already aggregates NCI-ALMANAC data.
The separate ALMANAC download is useful for raw dose-response curves and
independent validation, but the DC domain can run with DrugComb alone.

If this script fails due to NCI wiki authentication (403 Forbidden), you can:
  1. Download manually from: https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC
  2. Skip this download — DrugComb v1.4 (Zenodo 18449193) already includes ALMANAC data
  3. Use the processed combo scores from NCI CellMiner (XLSX format, different schema):
     https://discover.nci.nih.gov/cellminer/download/processeddataset/DTP_NCI60_ALMANAC_COMBO_SCORE.zip

Usage:
    python scripts_dc/download_almanac.py [--output-dir data/dc/almanac] [--force]
"""

import argparse
import sys
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# NCI-ALMANAC combo drug growth data (raw dose-response, ~150 MB zipped)
# If this URL returns 403, the data is behind NCI authentication.
ALMANAC_URL = (
    "https://wiki.nci.nih.gov/download/attachments/338237347/"
    "ComboDrugGrowth_Nov2017.zip"
)


def download_file(url: str, output_path: Path, force: bool = False) -> bool:
    """Download a file with progress reporting. Returns True on success."""
    if output_path.exists() and not force:
        size_mb = output_path.stat().st_size / 1e6
        print(f"  Already exists ({size_mb:.1f} MB), skipping: {output_path.name}")
        return True

    print(f"  Downloading {output_path.name} ...")
    resp = requests.get(url, stream=True, timeout=600)
    if resp.status_code == 403:
        print(
            f"  ERROR: 403 Forbidden — NCI wiki requires authentication.\n"
            f"  Download manually from:\n"
            f"    https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC\n"
            f"  Or skip — DrugComb v1.4 already aggregates NCI-ALMANAC data.",
            file=sys.stderr,
        )
        raise requests.HTTPError(f"403 Forbidden: {url}", response=resp)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r    {pct:.0f}%", end="", flush=True)
    print()

    size_mb = output_path.stat().st_size / 1e6
    if size_mb == 0:
        output_path.unlink()
        raise IOError(f"Downloaded file is empty (0 bytes): {output_path.name}")
    print(f"    Done: {size_mb:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download NCI-ALMANAC data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "dc" / "almanac",
        help="Output directory (default: data/dc/almanac)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    print(f"NCI-ALMANAC download → {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_file(ALMANAC_URL, args.output_dir / "ComboDrugGrowth_Nov2017.zip", args.force)
    except Exception as e:
        print(f"  ERROR downloading NCI-ALMANAC: {e}", file=sys.stderr)
        print(
            "  NOTE: The DC domain can run without NCI-ALMANAC.\n"
            "  DrugComb v1.4 already includes aggregated ALMANAC data.\n"
            "  Proceed with: load_drugcomb.py, map_identifiers.py, map_cell_lines.py",
            file=sys.stderr,
        )
        return 1

    # Unzip if zip file exists
    zip_path = args.output_dir / "ComboDrugGrowth_Nov2017.zip"
    csv_path = args.output_dir / "ComboDrugGrowth_Nov2017.csv"
    if zip_path.exists() and (not csv_path.exists() or args.force):
        import zipfile

        print("  Extracting zip archive ...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(args.output_dir)
            print("    Done.")
        except zipfile.BadZipFile:
            print(
                f"  ERROR: {zip_path.name} is not a valid zip file. "
                "Delete and re-download with --force.",
                file=sys.stderr,
            )
            return 1

    print("\nNCI-ALMANAC files downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
