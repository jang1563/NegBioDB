#!/usr/bin/env python3
"""Load JUMP Cell Painting tables into the CP domain database.

This script does not download large Cell Painting assets locally. It expects
assembled observation/profile/deep-feature tables that were prepared on HPC.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from negbiodb_cp.cp_db import DEFAULT_CP_DB_PATH, create_cp_database, get_connection
from negbiodb_cp.etl_jump import ingest_jump_tables, load_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load JUMP Cell Painting data into CP DB")
    parser.add_argument("--db", "--db-path", dest="db", type=Path, default=DEFAULT_CP_DB_PATH)
    parser.add_argument("--observations", type=Path, required=True,
                        help="Assembled observations table prepared on HPC")
    parser.add_argument("--profile-features", type=Path, default=None)
    parser.add_argument("--image-features", type=Path, default=None)
    parser.add_argument("--orthogonal-evidence", type=Path, default=None)
    parser.add_argument("--dataset-name", default="cpg0016-jump")
    parser.add_argument("--dataset-version", default="1.0")
    parser.add_argument(
        "--annotation-mode",
        choices=["annotated", "plate_proxy"],
        default="annotated",
    )
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args(argv)

    create_cp_database(args.db)

    observations = load_table(args.observations)
    profile_features = load_table(args.profile_features) if args.profile_features else None
    image_features = load_table(args.image_features) if args.image_features else None
    orthogonal_evidence = load_table(args.orthogonal_evidence) if args.orthogonal_evidence else None

    conn = get_connection(args.db)
    try:
        summary = ingest_jump_tables(
            conn,
            observations=observations,
            profile_features=profile_features,
            image_features=image_features,
            orthogonal_evidence=orthogonal_evidence,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            annotation_mode=args.annotation_mode,
            source_url=args.source_url,
        )
    finally:
        conn.close()

    logger.info("Loaded CP data: %s", summary)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
