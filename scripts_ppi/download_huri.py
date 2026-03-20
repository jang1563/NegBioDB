#!/usr/bin/env python
"""Download HuRI positive PPI data and ENSG→UniProt mapping.

Downloads:
  - HI-union.tsv  (all HuRI screens combined, ~64K PPIs, Ensembl gene IDs)
  - HUMAN_9606_idmapping.dat.gz from UniProt FTP (~34 MB)
    → filtered to ensg_to_uniprot.tsv (~1-2 MB)

ORFeome gene list must be obtained separately from interactome-atlas.org
or Luck et al. 2020 Supplementary Table S1.

License: CC BY 4.0 (HuRI), CC BY 4.0 (UniProt)
"""

import gzip
import argparse
from pathlib import Path

from negbiodb.download import download_file_http, load_config
from negbiodb_ppi.protein_mapper import validate_uniprot

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_ensg_mapping(idmapping_gz: Path, output_tsv: Path) -> int:
    """Filter HUMAN_9606_idmapping.dat.gz to ENSG→UniProt TSV.

    The idmapping.dat format is TAB-delimited with 3 columns:
      UniProtAC \\t IDtype \\t ID

    We keep rows where IDtype == 'Ensembl' (gene-level ENSG IDs).
    For one-to-many, keep the first occurrence (SwissProt before TrEMBL).

    Returns:
        Number of unique ENSG→UniProt mappings written.
    """
    seen = set()
    count = 0
    with gzip.open(idmapping_gz, "rt") as fin, open(output_tsv, "w") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            uniprot_acc, id_type, id_value = parts[0], parts[1], parts[2]
            if id_type != "Ensembl":
                continue
            # Strip version suffix (ENSG00000166913.14 → ENSG00000166913)
            # HuRI uses bare ENSG IDs without version numbers
            ensg_bare = id_value.split(".")[0]
            if ensg_bare in seen:
                continue
            if not validate_uniprot(uniprot_acc):
                continue
            seen.add(ensg_bare)
            fout.write(f"{ensg_bare}\t{uniprot_acc}\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Download HuRI PPI data")
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=None,
        help="Destination directory (default: from config.yaml)",
    )
    parser.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Disable SSL verification for interactome-atlas.org",
    )
    args = parser.parse_args()

    cfg = load_config()
    huri_cfg = cfg["ppi_domain"]["downloads"]["huri"]
    dest_dir = Path(args.dest_dir or (_PROJECT_ROOT / huri_cfg["dest_dir"]))

    verify_ssl = not args.no_ssl_verify

    # 1. Download HI-union (preferred for negative derivation)
    download_file_http(
        huri_cfg["hi_union_url"],
        dest_dir / "HI-union.tsv",
        desc="HI-union.tsv",
        verify=verify_ssl,
    )

    # 2. Download and filter ENSG→UniProt mapping
    output_tsv = dest_dir / "ensg_to_uniprot.tsv"
    if output_tsv.exists() and output_tsv.stat().st_size > 0:
        print(f"Already exists: {output_tsv}")
    else:
        idmapping_gz = dest_dir / "HUMAN_9606_idmapping.dat.gz"
        download_file_http(
            huri_cfg["idmapping_url"],
            idmapping_gz,
            desc="HUMAN_9606_idmapping.dat.gz",
            verify=verify_ssl,
        )

        print("Filtering ENSG→UniProt mappings...")
        count = build_ensg_mapping(idmapping_gz, output_tsv)
        print(f"  {count} ENSG→UniProt mappings written to {output_tsv}")

        # Clean up the large .dat.gz to save space
        idmapping_gz.unlink()
        print(f"  Removed {idmapping_gz.name}")

    print(f"\nHuRI data downloaded to {dest_dir}")
    print(
        "NOTE: ORFeome v9.1 gene list must be obtained separately from "
        "interactome-atlas.org or Luck et al. 2020 Supp Table S1."
    )


if __name__ == "__main__":
    main()
