"""Download AACT pipe-delimited snapshot from CTTI.

Downloads ~2.23 GB ZIP and extracts only the 13 needed tables
into data/ct/aact/ as pipe-delimited .txt files.

AACT (Aggregate Analysis of ClinicalTrials.gov) is maintained by
Duke/CTTI. Data is public domain (US government data).

Source: https://aact.ctti-clinicaltrials.org/pipe_files
Format: ZIP containing pipe-delimited .txt files (one per table)

Usage:
    python scripts_ct/download_aact.py [--url URL]
"""

import argparse
import zipfile
from pathlib import Path

from negbiodb.download import (
    check_disk_space,
    download_file_http,
    load_config,
    verify_file_exists,
)


def extract_needed_tables(
    zip_path: Path,
    dest_dir: Path,
    table_names: list[str],
) -> dict[str, Path]:
    """Extract only the needed tables from AACT ZIP.

    AACT ZIP contains files named like 'studies.txt', 'interventions.txt'.
    Returns dict mapping table_name -> extracted file path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        for table in table_names:
            fname = f"{table}.txt"
            matches = [n for n in all_names if n.endswith(fname)]
            if not matches:
                print(f"  WARNING: {fname} not found in ZIP")
                continue

            # Use the first match (handles subdirectories in ZIP)
            member = matches[0]
            dest_file = dest_dir / fname

            if dest_file.exists() and dest_file.stat().st_size > 0:
                print(f"  Already extracted: {fname}")
            else:
                print(f"  Extracting: {member} -> {fname}")
                with zf.open(member) as src, open(dest_file, "wb") as dst:
                    dst.write(src.read())

            extracted[table] = dest_file
            size_mb = dest_file.stat().st_size / (1024**2)
            print(f"    {fname}: {size_mb:.1f} MB")

    return extracted


def main():
    parser = argparse.ArgumentParser(
        description="Download AACT pipe-delimited snapshot"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Direct URL to AACT ZIP (overrides scraping)",
    )
    args = parser.parse_args()

    cfg = load_config()
    dl = cfg["ct_domain"]["downloads"]["aact"]
    dest_dir = Path(dl["dest_dir"])
    required_disk_gb = dl["required_disk_gb"]
    tables = dl["tables"]

    print("=== AACT Pipe Files Download ===")
    print(f"Dest dir:  {dest_dir}")
    print(f"Tables:    {len(tables)}")

    check_disk_space(dest_dir, required_disk_gb)

    # Determine URL
    if args.url:
        url = args.url
    else:
        print(
            "\nAACT download URL changes monthly. Provide via --url flag."
            "\nGet the latest URL from: "
            "https://aact.ctti-clinicaltrials.org/pipe_files"
            "\n\nExample:"
            "\n  python scripts_ct/download_aact.py --url "
            '"https://ctti-aact.nyc3.digitaloceanspaces.com/..."'
        )
        return

    # Download ZIP
    zip_path = dest_dir / "aact_pipe_files.zip"
    download_file_http(url, zip_path, desc="AACT pipe files")

    if not verify_file_exists(zip_path, min_bytes=100_000_000):
        raise ValueError(f"AACT ZIP too small or missing: {zip_path}")

    # Extract needed tables
    print(f"\nExtracting {len(tables)} tables from ZIP...")
    extracted = extract_needed_tables(zip_path, dest_dir, tables)

    print(f"\n=== AACT Download Summary ===")
    print(f"ZIP size:        {zip_path.stat().st_size / (1024**3):.2f} GB")
    print(f"Tables extracted: {len(extracted)} / {len(tables)}")
    missing = set(tables) - set(extracted.keys())
    if missing:
        print(f"MISSING tables:  {', '.join(sorted(missing))}")
    print("\nAACT download complete.")


if __name__ == "__main__":
    main()
