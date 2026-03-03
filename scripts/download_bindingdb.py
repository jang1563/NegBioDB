"""Download BindingDB full dataset (TSV in ZIP archive).

Source: https://www.bindingdb.org/bind/downloads/
File: BindingDB_All_*.tsv (~525 MB zip -> ~2 GB TSV)
"""

import zipfile
from pathlib import Path

import requests

from negbiodb.download import (
    check_disk_space,
    download_file_http,
    load_config,
    verify_file_exists,
)


def main():
    cfg = load_config()
    dl = cfg["downloads"]["bindingdb"]
    url = dl["url"]
    dest_dir = Path(dl["dest_dir"])
    min_bytes = dl["min_size_bytes"]

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "bindingdb_all.zip"

    print("=== BindingDB Download ===")
    print(f"URL:  {url}")
    print(f"Dest: {dest_dir}")

    check_disk_space(dest_dir, required_gb=3.0)

    try:
        download_file_http(url, zip_path, desc="BindingDB ZIP")
    except (requests.RequestException, OSError) as e:
        print(f"\nERROR: Download failed: {e}")
        print(
            "BindingDB uses a servlet-based URL that may require manual download."
        )
        print(f"Please download manually from: https://www.bindingdb.org/bind/downloads/")
        print(f"Save the ZIP file as: {zip_path}")
        return

    # Extract TSV from ZIP
    print("Extracting TSV from ZIP...")
    tsv_path = None
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".tsv"):
                zf.extract(name, dest_dir)
                extracted = dest_dir / name
                tsv_path = dest_dir / "BindingDB_All.tsv"
                if extracted != tsv_path:
                    extracted.rename(tsv_path)
                break

    if tsv_path is None:
        print("WARNING: No TSV file found in ZIP archive")
        return

    # Remove ZIP to save space
    zip_path.unlink()
    print(f"Removed ZIP: {zip_path}")

    if not verify_file_exists(tsv_path, min_bytes=min_bytes):
        print(f"WARNING: TSV smaller than expected ({min_bytes / 1e6:.0f} MB)")

    print(f"Extracted: {tsv_path} ({tsv_path.stat().st_size / (1024**2):.1f} MB)")
    print("\nBindingDB download complete.")


if __name__ == "__main__":
    main()
