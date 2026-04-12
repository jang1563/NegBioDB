#!/usr/bin/env python3
"""Ingest MetaboLights studies into the MD database.

Usage:
    python scripts_md/02_ingest_metabolights.py [--limit N] [--db PATH]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest MetaboLights studies")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of qualifying studies to ingest")
    parser.add_argument("--db", type=str, default=None, help="Path to MD database")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-ingest studies already in DB")
    args = parser.parse_args()

    from negbiodb_md.md_db import get_md_connection
    from negbiodb_md.etl_hmdb import DEFAULT_CACHE_DB, get_cache_connection
    from negbiodb_md.etl_standardize import Standardizer
    from negbiodb_md.etl_metabolights import ingest_metabolights

    conn = get_md_connection(args.db)

    # Load HMDB cache if available (internal standardization only)
    hmdb_conn = None
    if DEFAULT_CACHE_DB.exists():
        hmdb_conn = get_cache_connection(DEFAULT_CACHE_DB)
        logger.info("HMDB internal cache loaded: %s", DEFAULT_CACHE_DB)
    else:
        logger.warning(
            "HMDB cache not found at %s — metabolite standardization will use PubChem only",
            DEFAULT_CACHE_DB,
        )

    standardizer = Standardizer(hmdb_cache_conn=hmdb_conn)
    n_inserted = ingest_metabolights(
        conn=conn,
        standardizer=standardizer,
        limit=args.limit,
        skip_existing=not args.no_skip,
    )
    logger.info("MetaboLights ingest complete: %d results inserted", n_inserted)

    if hmdb_conn:
        hmdb_conn.close()
    conn.close()


if __name__ == "__main__":
    main()
