#!/usr/bin/env python3
"""Generate CP ML exports and split assignments."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from negbiodb_cp.cp_db import (
    DEFAULT_CP_DB_PATH,
    ensure_cp_production_ready,
    get_connection,
)
from negbiodb_cp.export import (
    export_cp_feature_tables,
    export_cp_m1_dataset,
    export_cp_m2_dataset,
    generate_all_splits,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export CP ML datasets")
    parser.add_argument("--db", "--db-path", dest="db", type=Path, default=DEFAULT_CP_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "exports" / "cp_ml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-proxy-smoke", action="store_true")
    args = parser.parse_args(argv)

    if not args.db.exists():
        logger.error("CP database not found: %s", args.db)
        return 1

    conn = get_connection(args.db)
    try:
        annotation_summary = ensure_cp_production_ready(
            conn,
            allow_proxy_smoke=args.allow_proxy_smoke,
        )
        split_ids = generate_all_splits(conn, seed=args.seed)
        m1_path, n_m1 = export_cp_m1_dataset(conn, args.output_dir)
        m2_path, n_m2 = export_cp_m2_dataset(conn, args.output_dir)
        feature_paths = export_cp_feature_tables(conn, args.output_dir)
        meta = {
            "domain": "cp",
            "seed": args.seed,
            "split_ids": split_ids,
            "files": {
                "m1": str(m1_path),
                "m2": str(m2_path),
                **{name: str(path) for name, path in feature_paths.items()},
            },
            **annotation_summary,
        }
    finally:
        conn.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cp_ml_metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("Generated split ids: %s", split_ids)
    logger.info("Exported CP-M1: %s (%d rows)", m1_path, n_m1)
    logger.info("Exported CP-M2: %s (%d rows)", m2_path, n_m2)
    logger.info("Exported feature tables: %s", feature_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
