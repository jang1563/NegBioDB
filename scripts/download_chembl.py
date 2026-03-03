"""Download ChEMBL SQLite database via chembl_downloader.

Uses pystow cache to avoid re-downloading. Creates a symlink
in data/chembl/ pointing to the cached SQLite file.
"""

import sqlite3
from pathlib import Path

from negbiodb.download import load_config


def main():
    cfg = load_config()
    dest_dir = Path(cfg["downloads"]["chembl"]["dest_dir"])
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=== ChEMBL SQLite Download ===")

    import chembl_downloader

    version = chembl_downloader.latest()
    print(f"Latest ChEMBL version: {version}")

    print("Downloading/extracting SQLite (pystow cache)...")
    sqlite_path = chembl_downloader.download_extract_sqlite()
    print(f"Cached at: {sqlite_path}")

    # Create symlink in data/chembl/
    link_name = dest_dir / f"chembl_{version}.db"
    if link_name.exists() or link_name.is_symlink():
        link_name.unlink()
    link_name.symlink_to(sqlite_path)
    print(f"Symlinked: {link_name} -> {sqlite_path}")

    # Verify key tables exist
    conn = sqlite3.connect(str(sqlite_path))
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()

    expected = {"activities", "assays", "compound_structures", "target_dictionary"}
    missing = expected - tables
    if missing:
        print(f"WARNING: Missing expected tables: {missing}")
    else:
        print(f"Verified tables: {sorted(expected)}")

    print(f"\nChEMBL {version} download complete.")


if __name__ == "__main__":
    main()
