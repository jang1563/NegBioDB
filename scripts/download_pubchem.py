"""Download PubChem bioactivities data via HTTPS.

Target file: bioactivities.tsv.gz (~2.8 GB)
Source: https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/Extras/bioactivities.tsv.gz

Columns (space-separated names):
  AID, SID, SID Group, CID, Activity Outcome, Activity Name,
  Activity Qualifier, Activity Value, Activity Unit, Protein Accession,
  Gene ID, Target TaxID, PMID
"""

import gzip
from pathlib import Path

from negbiodb.download import (
    check_disk_space,
    download_file_http,
    load_config,
    verify_file_exists,
)


def main():
    cfg = load_config()
    dl = cfg["downloads"]["pubchem"]
    url = dl["url"]
    dest = Path(dl["dest"])
    min_bytes = dl["min_size_bytes"]

    print(f"=== PubChem Bioactivities Download ===")
    print(f"URL:  {url}")
    print(f"Dest: {dest}")

    check_disk_space(dest.parent, required_gb=4.0)

    download_file_http(url, dest, desc="PubChem bioactivities.tsv.gz")

    if not verify_file_exists(dest, min_bytes=min_bytes):
        print(f"WARNING: File smaller than expected ({min_bytes / 1e9:.1f} GB)")

    # Print first few lines to verify column names
    print("\n--- First 3 lines (header + 2 rows) ---")
    with gzip.open(dest, "rt") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(line.rstrip()[:200])

    print("\nPubChem download complete.")


if __name__ == "__main__":
    main()
