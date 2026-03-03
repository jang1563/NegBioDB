"""Download required PubChem files for NegBioDB ETL via HTTPS.

Files:
  - bioactivities.tsv.gz                 (~2.8 GB)
  - bioassays.tsv.gz                     (~100+ MB)
  - Aid2GeneidAccessionUniProt.gz        (~tens of MB)
  - Sid2CidSMILES.gz                     (~hundreds of MB)
"""

import gzip
from pathlib import Path

from negbiodb.download import (
    check_disk_space,
    download_file_http,
    load_config,
    verify_file_exists,
)


def _download_and_verify(url: str, dest: Path, min_bytes: int, desc: str) -> None:
    print(f"\n[{desc}]")
    print(f"URL:  {url}")
    print(f"Dest: {dest}")
    download_file_http(url, dest, desc=desc)
    if not verify_file_exists(dest, min_bytes=min_bytes):
        print(f"WARNING: {dest.name} smaller than expected ({min_bytes / 1e6:.1f} MB)")


def main():
    cfg = load_config()
    dl = cfg["downloads"]["pubchem"]

    files = [
        {
            "url": dl["url"],
            "dest": Path(dl["dest"]),
            "min_size": dl["min_size_bytes"],
            "desc": "PubChem bioactivities.tsv.gz",
        },
        {
            "url": dl["bioassays_url"],
            "dest": Path(dl["bioassays_dest"]),
            "min_size": dl["bioassays_min_size_bytes"],
            "desc": "PubChem bioassays.tsv.gz",
        },
        {
            "url": dl["aid_uniprot_url"],
            "dest": Path(dl["aid_uniprot_dest"]),
            "min_size": dl["aid_uniprot_min_size_bytes"],
            "desc": "PubChem Aid2GeneidAccessionUniProt.gz",
        },
        {
            "url": dl["sid_cid_smiles_url"],
            "dest": Path(dl["sid_cid_smiles_dest"]),
            "min_size": dl["sid_cid_smiles_min_size_bytes"],
            "desc": "PubChem Sid2CidSMILES.gz",
        },
    ]

    print(f"=== PubChem Bioactivities Download ===")
    print("Downloading all required PubChem ETL files.")
    check_disk_space(files[0]["dest"].parent, required_gb=6.0)

    for spec in files:
        _download_and_verify(
            url=spec["url"],
            dest=spec["dest"],
            min_bytes=spec["min_size"],
            desc=spec["desc"],
        )

    # Print first few lines to verify column names
    print("\n--- First 3 lines (header + 2 rows) ---")
    with gzip.open(files[0]["dest"], "rt") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(line.rstrip()[:200])

    print("\nPubChem download complete (all files).")


if __name__ == "__main__":
    main()
