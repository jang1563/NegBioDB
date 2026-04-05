#!/usr/bin/env python3
"""Resolve drug names to PubChem CID / InChIKey / SMILES via PubChemPy.

Updates the compounds table with chemical identifiers for drugs that
were auto-created from synergy data without full identifier mapping.

Usage:
    python scripts_dc/map_identifiers.py [--db-path data/negbiodb_dc.db] [--limit 100]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from negbiodb_dc.dc_db import DEFAULT_DC_DB_PATH, get_connection, run_dc_migrations

logger = logging.getLogger(__name__)


def resolve_pubchem(drug_name: str) -> dict | None:
    """Resolve drug name to PubChem identifiers.

    Returns dict with pubchem_cid, inchikey, canonical_smiles, or None.
    """
    try:
        import pubchempy as pcp
    except ImportError:
        logger.error("pubchempy not installed. pip install pubchempy")
        return None

    try:
        results = pcp.get_compounds(drug_name, "name")
        if not results:
            return None

        compound = results[0]
        return {
            "pubchem_cid": compound.cid,
            "inchikey": compound.inchikey,
            "canonical_smiles": compound.canonical_smiles,
        }
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            logger.error("PubChem rate limit (429) hit for '%s' — increase --delay", drug_name)
            raise  # Bubble up so caller can handle backoff
        logger.warning("HTTP error for '%s': %s", drug_name, e)
        return None
    except Exception as e:
        logger.warning("PubChem lookup failed for '%s': %s", drug_name, e)
        return None


def map_compound_identifiers(
    db_path: Path,
    limit: int | None = None,
    delay: float = 0.3,
) -> dict[str, int]:
    """Map compounds without PubChem CID to identifiers via PubChemPy.

    Args:
        db_path: Path to DC database.
        limit: Max number of compounds to resolve (for testing).
        delay: Seconds between PubChem API calls (rate limiting).

    Returns:
        Stats dict.
    """
    conn = get_connection(db_path)
    stats = {"resolved": 0, "failed": 0}

    try:
        query = """
            SELECT compound_id, drug_name FROM compounds
            WHERE pubchem_cid IS NULL
            ORDER BY compound_id
        """
        if limit:
            query += " LIMIT ?"
            rows = conn.execute(query, (limit,)).fetchall()
        else:
            rows = conn.execute(query).fetchall()
        total = len(rows)
        logger.info("Found %d compounds without PubChem CID", total)

        for i, (compound_id, drug_name) in enumerate(rows):
            if i > 0 and delay > 0:
                time.sleep(delay)

            result = resolve_pubchem(drug_name)
            if result:
                conn.execute(
                    """UPDATE compounds SET
                    pubchem_cid = ?, inchikey = ?, canonical_smiles = ?
                    WHERE compound_id = ?""",
                    (result["pubchem_cid"], result["inchikey"],
                     result["canonical_smiles"], compound_id),
                )
                stats["resolved"] += 1
            else:
                stats["failed"] += 1

            if (i + 1) % 50 == 0:
                conn.commit()
                logger.info("Progress: %d/%d resolved", i + 1, total)

        conn.commit()
        logger.info(
            "Identifier mapping: %d resolved, %d failed",
            stats["resolved"], stats["failed"],
        )
        return stats
    finally:
        conn.close()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Resolve drug names to PubChem identifiers"
    )
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DC_DB_PATH,
        help="Path to DC database",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max compounds to resolve (for testing)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Delay between API calls in seconds (default: 0.3)",
    )
    args = parser.parse_args()

    run_dc_migrations(args.db_path)
    stats = map_compound_identifiers(args.db_path, args.limit, args.delay)
    print(f"Results: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
