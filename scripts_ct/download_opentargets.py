"""Download Open Targets clinical trial stop reasons from HuggingFace.

Downloads a single Parquet file (~1 MB) with curated (text, label) pairs.
Contains ~5K rows with 17-category stop reason taxonomy.

Source: Razuvayevskaya et al. 2024 (Nature Genetics)
License: Apache 2.0

Usage:
    python scripts_ct/download_opentargets.py
"""

from pathlib import Path

from negbiodb.download import download_file_http, load_config, verify_file_exists


def main():
    cfg = load_config()
    dl = cfg["ct_domain"]["downloads"]["opentargets"]
    dest = Path(dl["dest"])

    print("=== Open Targets Stop Reasons Download ===")
    print(f"Dest: {dest}")

    download_file_http(dl["url"], dest, desc="Open Targets stop reasons")

    if not verify_file_exists(dest, min_bytes=1000):
        raise ValueError(f"Open Targets file too small or missing: {dest}")

    size_kb = dest.stat().st_size / 1024
    print(f"Downloaded: {dest} ({size_kb:.0f} KB)")
    print("\nOpen Targets download complete.")


if __name__ == "__main__":
    main()
