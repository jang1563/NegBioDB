"""ETL module for drug-target mappings from DGIdb.

Populates the drug_targets table linking compounds → gene targets.
This data powers the target_overlap features for drug combination analysis:
  - num_shared_targets: how many targets two drugs share
  - target_jaccard: Jaccard similarity over target sets

Source: DGIdb interactions.tsv (https://www.dgidb.org/downloads)
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from negbiodb_dc.dc_db import get_connection

logger = logging.getLogger(__name__)


def parse_dgidb_interactions(interactions_path: Path) -> list[dict]:
    """Parse DGIdb interactions TSV.

    Expected columns: drug_name, gene_name, interaction_types, sources
    (Column names may vary across DGIdb versions.)

    Returns:
        List of dicts with keys: drug_name, gene_symbol, source
    """
    logger.info("Reading DGIdb interactions: %s", interactions_path)
    df = pd.read_csv(interactions_path, sep="\t", low_memory=False)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "drug" in cl and "name" in cl:
            col_map[col] = "drug_name"
        elif "gene" in cl and "name" in cl:
            col_map[col] = "gene_symbol"
        elif cl in ("gene", "gene_symbol"):
            col_map[col] = "gene_symbol"
        elif "source" in cl and "db" not in cl:
            col_map[col] = "source_name"
    df = df.rename(columns=col_map)

    if "drug_name" not in df.columns or "gene_symbol" not in df.columns:
        raise ValueError(
            f"Cannot find drug_name/gene_symbol columns. Found: {list(df.columns)}"
        )

    records = []
    seen = set()
    for _, row in df.iterrows():
        drug = str(row["drug_name"]).strip()
        gene = str(row["gene_symbol"]).strip()
        if not drug or not gene or drug == "nan" or gene == "nan":
            continue

        key = (drug, gene)
        if key in seen:
            continue
        seen.add(key)

        records.append({
            "drug_name": drug,
            "gene_symbol": gene,
            "source": "dgidb",
        })

    logger.info("Parsed %d unique drug-target interactions", len(records))
    return records


def load_drug_targets(
    conn,
    interactions: list[dict],
    compound_cache: dict[str, int],
) -> dict[str, int]:
    """Load drug-target mappings into drug_targets table.

    Only loads targets for drugs that exist in the compounds table.

    Args:
        conn: Database connection.
        interactions: List of dicts from parse_dgidb_interactions.
        compound_cache: drug_name → compound_id mapping.

    Returns:
        Stats dict.
    """
    stats = {
        "targets_inserted": 0,
        "skipped_unknown_drug": 0,
        "skipped_duplicate": 0,
    }

    for rec in interactions:
        drug_name = rec["drug_name"]
        if drug_name not in compound_cache:
            stats["skipped_unknown_drug"] += 1
            continue

        compound_id = compound_cache[drug_name]
        gene_symbol = rec["gene_symbol"]
        source = rec.get("source", "dgidb")

        try:
            conn.execute(
                """INSERT INTO drug_targets
                (compound_id, gene_symbol, source)
                VALUES (?, ?, ?)""",
                (compound_id, gene_symbol, source),
            )
            stats["targets_inserted"] += 1
        except sqlite3.IntegrityError:
            stats["skipped_duplicate"] += 1

    conn.commit()
    logger.info(
        "Drug-target ETL: %d inserted, %d unknown drugs, %d duplicates",
        stats["targets_inserted"],
        stats["skipped_unknown_drug"],
        stats["skipped_duplicate"],
    )
    return stats


def run_drug_targets_etl(
    db_path: Path,
    data_dir: Path,
) -> dict[str, int]:
    """Run full drug-target ETL pipeline."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    interactions_path = data_dir / "interactions.tsv"
    if not interactions_path.exists():
        raise FileNotFoundError(
            f"DGIdb interactions file not found: {interactions_path}"
        )

    interactions = parse_dgidb_interactions(interactions_path)

    conn = get_connection(db_path)
    try:
        compound_cache = {
            row[1]: row[0]
            for row in conn.execute("SELECT compound_id, drug_name FROM compounds")
        }

        stats = load_drug_targets(conn, interactions, compound_cache)
        return stats
    finally:
        conn.close()
