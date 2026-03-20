#!/usr/bin/env python
"""Fetch protein sequences from UniProt REST API and populate the PPI database.

Reads all proteins with NULL amino_acid_sequence from negbiodb_ppi.db,
fetches their sequences in batches from UniProt, and updates the DB.

Usage:
    PYTHONPATH=src python scripts_ppi/fetch_sequences.py [--db PATH] [--batch-size 500]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from negbiodb_ppi.ppi_db import DEFAULT_PPI_DB_PATH, get_connection  # noqa: E402

logger = logging.getLogger(__name__)

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, doubles each retry


def fetch_uniprot_batch(
    accessions: list[str],
    timeout: float = 60.0,
) -> dict[str, dict]:
    """Fetch protein metadata from UniProt REST API for a batch of accessions.

    Uses the /uniprotkb/search endpoint with OR-joined accession queries.
    Batch size should be kept ≤100 to avoid URL length limits.

    Args:
        accessions: List of UniProt accession strings (max ~100).
        timeout: Request timeout in seconds.

    Returns:
        Dict mapping accession -> {sequence, gene_symbol, subcellular_location}.
        Missing accessions (404/obsolete) are omitted from the result.
    """
    if not accessions:
        return {}

    # Build query: (accession:P12345 OR accession:Q9UHC1 OR ...)
    query = "(" + " OR ".join(f"accession:{acc}" for acc in accessions) + ")"

    params = {
        "query": query,
        "fields": "accession,sequence,gene_primary,cc_subcellular_location",
        "format": "json",
        "size": str(len(accessions)),
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                UNIPROT_SEARCH_URL,
                params=params,
                timeout=timeout,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            break
        except (requests.RequestException, requests.HTTPError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (2**attempt)
                logger.warning(
                    "UniProt batch attempt %d failed: %s. Retrying in %.1fs",
                    attempt + 1,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error("UniProt batch failed after %d attempts: %s", MAX_RETRIES, e)
                raise

    data = resp.json()
    results = data.get("results", [])

    parsed: dict[str, dict] = {}
    for entry in results:
        acc = entry.get("primaryAccession", "")
        if not acc:
            continue

        # Sequence
        seq_obj = entry.get("sequence", {})
        sequence = seq_obj.get("value", "")

        # Gene symbol
        gene_symbol = None
        genes = entry.get("genes", [])
        if genes:
            gene_name = genes[0].get("geneName", {})
            gene_symbol = gene_name.get("value")

        # Subcellular location
        subcellular = None
        comments = entry.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "SUBCELLULAR LOCATION":
                locs = comment.get("subcellularLocations", [])
                if locs:
                    loc_val = locs[0].get("location", {}).get("value")
                    if loc_val:
                        subcellular = loc_val
                        break

        parsed[acc] = {
            "sequence": sequence,
            "gene_symbol": gene_symbol,
            "subcellular_location": subcellular,
        }

    return parsed


def update_protein_sequences(
    db_path: Path,
    batch_size: int = 100,
    delay: float = 1.0,
    checkpoint_path: Path | None = None,
) -> dict:
    """Fetch and update all NULL sequences in the PPI database.

    Args:
        db_path: Path to negbiodb_ppi.db.
        batch_size: Number of accessions per API request (max 500).
        delay: Seconds to wait between batches.
        checkpoint_path: If provided, save/resume progress from this JSON file.

    Returns:
        Summary dict with fetched, failed, skipped, avg_seq_length.
    """
    conn = get_connection(db_path)

    # Get all proteins needing sequences
    rows = conn.execute(
        "SELECT uniprot_accession FROM proteins WHERE amino_acid_sequence IS NULL"
    ).fetchall()
    all_accs = [r[0] for r in rows]
    logger.info("Found %d proteins with NULL sequences", len(all_accs))

    # Resume from checkpoint if available
    done_accs: set[str] = set()
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            done_accs = set(json.load(f).get("completed", []))
        logger.info("Resuming from checkpoint: %d already done", len(done_accs))

    remaining = [a for a in all_accs if a not in done_accs]
    logger.info("Fetching %d remaining proteins in batches of %d", len(remaining), batch_size)

    fetched = 0
    failed = 0
    total_seq_len = 0
    completed_accs = list(done_accs)

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(remaining) + batch_size - 1) // batch_size

        try:
            results = fetch_uniprot_batch(batch)
        except Exception:
            logger.error("Batch %d/%d failed permanently, skipping", batch_num, total_batches)
            failed += len(batch)
            continue

        # Update DB
        for acc in batch:
            if acc in results:
                info = results[acc]
                seq = info["sequence"]
                if seq:
                    conn.execute(
                        """UPDATE proteins SET
                            amino_acid_sequence = ?,
                            sequence_length = ?,
                            gene_symbol = COALESCE(gene_symbol, ?),
                            subcellular_location = COALESCE(subcellular_location, ?),
                            updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                        WHERE uniprot_accession = ?""",
                        (
                            seq,
                            len(seq),
                            info["gene_symbol"],
                            info["subcellular_location"],
                            acc,
                        ),
                    )
                    fetched += 1
                    total_seq_len += len(seq)
                else:
                    failed += 1
                    logger.warning("Empty sequence for %s", acc)
            else:
                failed += 1
                logger.warning("Not found in UniProt: %s", acc)

            completed_accs.append(acc)

        conn.commit()

        # Save checkpoint
        if checkpoint_path:
            with open(checkpoint_path, "w") as f:
                json.dump({"completed": completed_accs}, f)

        logger.info(
            "Batch %d/%d: fetched=%d, failed=%d",
            batch_num,
            total_batches,
            fetched,
            failed,
        )

        if i + batch_size < len(remaining):
            time.sleep(delay)

    conn.close()

    avg_len = total_seq_len / fetched if fetched > 0 else 0
    summary = {
        "total": len(all_accs),
        "fetched": fetched,
        "failed": failed,
        "skipped": len(done_accs),
        "avg_seq_length": round(avg_len, 1),
    }
    logger.info("Summary: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch protein sequences from UniProt")
    parser.add_argument("--db", type=Path, default=DEFAULT_PPI_DB_PATH)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    summary = update_protein_sequences(
        db_path=args.db,
        batch_size=args.batch_size,
        delay=args.delay,
        checkpoint_path=args.checkpoint,
    )

    print(f"\nSequence fetch complete:")
    print(f"  Total proteins:    {summary['total']}")
    print(f"  Fetched:           {summary['fetched']}")
    print(f"  Failed:            {summary['failed']}")
    print(f"  Skipped (resumed): {summary['skipped']}")
    print(f"  Avg seq length:    {summary['avg_seq_length']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
