"""Download DAVIS kinase binding dataset from GitHub.

Source: https://github.com/dingyan20/Davis-Dataset-for-DTA-Prediction
Files: drugs.csv (68 drugs), proteins.csv (442 kinases),
       drug_protein_affinity.csv (29,444 pairs with pKd values)

Note: pytdc has dependency conflicts (requires rdkit<2024.3),
so we download directly from the GitHub repository.
"""

from pathlib import Path

import pandas as pd

from negbiodb.download import download_file_http, load_config


def main():
    cfg = load_config()
    dl = cfg["downloads"]["davis"]
    base_url = dl["base_url"]
    files = dl["files"]
    dest_dir = Path(dl["dest_dir"])
    min_rows = dl["min_rows"]

    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=== DAVIS Dataset Download ===")
    print(f"Source: {base_url}")
    print(f"Dest:   {dest_dir}")

    # Download each CSV file
    for fname in files:
        url = f"{base_url}/{fname}"
        dest = dest_dir / fname
        download_file_http(url, dest, desc=fname)

    # Load and merge into a single DataFrame
    drugs = pd.read_csv(dest_dir / "drugs.csv")
    proteins = pd.read_csv(dest_dir / "proteins.csv")
    affinities = pd.read_csv(dest_dir / "drug_protein_affinity.csv")

    print(f"\nDrugs:      {len(drugs)} compounds")
    print(f"Proteins:   {len(proteins)} kinases")
    print(f"Affinities: {len(affinities)} pairs")

    # Merge into a combined dataset
    merged = affinities.merge(drugs, on="Drug_Index").merge(proteins, on="Protein_Index")
    merged.to_parquet(dest_dir / "davis_merged.parquet", index=False)
    print(f"Merged:     {len(merged)} rows -> davis_merged.parquet")

    if len(affinities) < min_rows:
        print(f"WARNING: Fewer rows than expected ({len(affinities)} < {min_rows})")

    # Basic statistics
    n_active = (affinities["Affinity"] > 5.0).sum()
    n_inactive = (affinities["Affinity"] <= 5.0).sum()
    print(f"\nActive (pKd > 5):   {n_active}")
    print(f"Inactive (pKd <= 5): {n_inactive}")
    print(f"Active ratio:        {n_active / len(affinities):.1%}")

    print("\nDAVIS download complete.")


if __name__ == "__main__":
    main()
