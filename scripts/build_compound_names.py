#!/usr/bin/env python3
"""Build compound name cache from ChEMBL + PubChem synonyms.

Output: exports/compound_names.parquet
Columns: compound_id, chembl_id, pubchem_cid, pref_name, name_source

Priority: ChEMBL pref_name > ChEMBL compound_records > PubChem synonym > None
"""

import argparse
import json
import sqlite3
import time
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEGBIODB_PATH = PROJECT_ROOT / "data" / "negbiodb.db"
CHEMBL_PATH = PROJECT_ROOT / "data" / "chembl" / "chembl_36.db"
OUTPUT_PATH = PROJECT_ROOT / "exports" / "compound_names.parquet"

PUBCHEM_BATCH_SIZE = 100  # CIDs per request (conservative)
PUBCHEM_DELAY = 0.25  # seconds between requests (PubChem allows 5/sec)


def load_chembl_names(chembl_db: Path, our_chembl_ids: set[str]) -> dict[str, tuple[str, str]]:
    """Load ChEMBL ID -> (name, source) from pref_name + compound_records.

    Priority: pref_name > shortest compound_records name.
    """
    conn = sqlite3.connect(str(chembl_db))

    # Phase 1a: pref_name (highest quality - curated drug names)
    pref_names = dict(
        conn.execute(
            "SELECT chembl_id, pref_name FROM molecule_dictionary "
            "WHERE pref_name IS NOT NULL"
        ).fetchall()
    )

    # Phase 1b: compound_records names (broader coverage)
    # Get molregno -> chembl_id for our compounds
    molregno_to_chembl = {}
    for chembl_id, molregno in conn.execute(
        "SELECT chembl_id, molregno FROM molecule_dictionary"
    ):
        if chembl_id in our_chembl_ids:
            molregno_to_chembl[molregno] = chembl_id

    # Batch query compound_records for shortest name per molregno
    cr_names = {}
    molregnos = list(molregno_to_chembl.keys())
    batch_size = 5000
    for i in range(0, len(molregnos), batch_size):
        batch = molregnos[i:i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT molregno, MIN(compound_name) "
            f"FROM compound_records "
            f"WHERE molregno IN ({placeholders}) "
            f"  AND compound_name IS NOT NULL "
            f"  AND compound_name != '' "
            f"  AND LENGTH(compound_name) < 200 "
            f"GROUP BY molregno",
            batch,
        ).fetchall()
        for molregno, name in rows:
            chembl_id = molregno_to_chembl[molregno]
            cr_names[chembl_id] = name

    conn.close()

    # Merge: pref_name > compound_records
    result = {}
    for chembl_id in our_chembl_ids:
        if chembl_id in pref_names:
            result[chembl_id] = (pref_names[chembl_id], "chembl_pref")
        elif chembl_id in cr_names:
            result[chembl_id] = (cr_names[chembl_id], "chembl_record")

    return result


def load_negbiodb_compounds(db_path: Path) -> pd.DataFrame:
    """Load compound_id, chembl_id, pubchem_cid from NegBioDB."""
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT compound_id, chembl_id, pubchem_cid FROM compounds", conn
    )
    conn.close()
    return df


def fetch_pubchem_names_batch(cids: list[int]) -> dict[int, str]:
    """Fetch preferred names for a batch of PubChem CIDs via PUG REST synonyms."""
    if not cids:
        return {}

    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/synonyms/JSON"
    data = f"cid={','.join(str(c) for c in cids)}".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"  PubChem batch error ({len(cids)} CIDs): {e}")
        return {}

    names = {}
    for entry in result.get("InformationList", {}).get("Information", []):
        cid = entry.get("CID")
        synonyms = entry.get("Synonym", [])
        if cid and synonyms:
            name = synonyms[0]
            if not name.isdigit() and len(name) < 200:
                names[cid] = name
    return names


def fetch_pubchem_names(
    cids: list[int], batch_size: int = PUBCHEM_BATCH_SIZE
) -> dict[int, str]:
    """Fetch names for all CIDs in batches."""
    all_names = {}
    total_batches = (len(cids) + batch_size - 1) // batch_size

    for i in range(0, len(cids), batch_size):
        batch = cids[i : i + batch_size]
        batch_num = i // batch_size + 1

        if batch_num % 50 == 0 or batch_num == 1:
            print(
                f"  PubChem batch {batch_num}/{total_batches} "
                f"({len(all_names)} names so far)"
            )

        names = fetch_pubchem_names_batch(batch)
        all_names.update(names)
        time.sleep(PUBCHEM_DELAY)

    return all_names


def main():
    parser = argparse.ArgumentParser(description="Build compound name cache")
    parser.add_argument(
        "--skip-pubchem",
        action="store_true",
        help="Skip PubChem API calls (ChEMBL only)",
    )
    parser.add_argument(
        "--pubchem-limit",
        type=int,
        default=0,
        help="Max PubChem CIDs to query (0=all)",
    )
    args = parser.parse_args()

    print("Loading NegBioDB compounds...")
    df = load_negbiodb_compounds(NEGBIODB_PATH)
    print(f"  {len(df)} compounds total")
    print(f"  {df['chembl_id'].notna().sum()} with chembl_id")
    print(f"  {df['pubchem_cid'].notna().sum()} with pubchem_cid")

    # Phase 1: ChEMBL (pref_name + compound_records)
    our_chembl_ids = set(df.loc[df["chembl_id"].notna(), "chembl_id"])
    print(f"\nPhase 1: ChEMBL name lookup ({len(our_chembl_ids)} compounds)...")
    chembl_names = load_chembl_names(CHEMBL_PATH, our_chembl_ids)
    n_pref = sum(1 for _, s in chembl_names.values() if s == "chembl_pref")
    n_rec = sum(1 for _, s in chembl_names.values() if s == "chembl_record")
    print(f"  pref_name: {n_pref}")
    print(f"  compound_records: {n_rec}")
    print(f"  Total: {len(chembl_names)}/{len(our_chembl_ids)}")

    df["pref_name"] = None
    df["name_source"] = None

    mask_chembl = df["chembl_id"].notna()
    for idx in df[mask_chembl].index:
        cid = df.at[idx, "chembl_id"]
        if cid in chembl_names:
            name, source = chembl_names[cid]
            df.at[idx, "pref_name"] = name
            df.at[idx, "name_source"] = source

    # Phase 2: PubChem synonyms (for compounds without ChEMBL names)
    if not args.skip_pubchem:
        need_name = df["pref_name"].isna() & df["pubchem_cid"].notna()
        cids_to_query = (
            df.loc[need_name, "pubchem_cid"].dropna().astype(int).tolist()
        )

        if args.pubchem_limit > 0:
            cids_to_query = cids_to_query[: args.pubchem_limit]

        print(f"\nPhase 2: PubChem synonym lookup ({len(cids_to_query)} CIDs)...")

        if cids_to_query:
            pubchem_names = fetch_pubchem_names(cids_to_query)
            print(f"  Retrieved {len(pubchem_names)} names from PubChem")

            for idx in df[need_name].index:
                cid = df.at[idx, "pubchem_cid"]
                if pd.notna(cid) and int(cid) in pubchem_names:
                    df.at[idx, "pref_name"] = pubchem_names[int(cid)]
                    df.at[idx, "name_source"] = "pubchem"
    else:
        print("\nPhase 2: Skipped (--skip-pubchem)")

    # Summary
    named = df["pref_name"].notna().sum()
    print(f"\n=== Summary ===")
    print(f"Total compounds: {len(df)}")
    print(f"With name:       {named} ({100 * named / len(df):.1f}%)")
    by_source = df["name_source"].value_counts()
    for source, count in by_source.items():
        print(f"  {source}: {count}")
    print(f"Without name:    {len(df) - named}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  Size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
