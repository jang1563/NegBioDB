#!/usr/bin/env python3
"""Map cell line names to DepMap model_id / COSMIC ID for cross-domain linking.

Updates the cell_lines table with depmap_model_id and tissue/cancer_type
from DepMap Model.csv file (same data used in GE domain).

Usage:
    python scripts_dc/map_cell_lines.py [--db-path data/negbiodb_dc.db] \
        [--depmap-model-csv data/depmap/Model.csv]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from negbiodb_dc.dc_db import DEFAULT_DC_DB_PATH, get_connection, run_dc_migrations

logger = logging.getLogger(__name__)


def load_depmap_models(model_csv: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load DepMap Model.csv and build name → metadata mappings.

    Returns:
        (lookup, lookup_stripped) where lookup maps upper-case name → info dict
        and lookup_stripped maps stripped name (no hyphens/spaces/underscores) → info dict.
    """
    df = pd.read_csv(model_csv, low_memory=False)

    # Build lookup by multiple name columns
    lookup: dict[str, dict] = {}
    name_cols = [c for c in df.columns if "name" in c.lower() or "alias" in c.lower()]
    name_cols = name_cols or ["CellLineName", "StrippedCellLineName"]

    for _, row in df.iterrows():
        model_id = str(row.get("ModelID", "")).strip()
        cosmic_id = row.get("COSMICID")
        tissue = str(row.get("OncotreeLineage", "")).strip() or None
        cancer_type = str(row.get("OncotreePrimaryDisease", "")).strip() or None
        primary_disease = str(row.get("PrimaryDisease", "")).strip() or None
        lineage = str(row.get("Lineage", "")).strip() or None
        lineage_subtype = str(row.get("LineageSubtype", "")).strip() or None

        if pd.notna(cosmic_id):
            try:
                cosmic_id = int(str(cosmic_id).strip())
            except (ValueError, TypeError):
                cosmic_id = None
        else:
            cosmic_id = None

        info = {
            "model_id": model_id if model_id else None,
            "cosmic_id": cosmic_id,
            "tissue": tissue,
            "cancer_type": cancer_type,
            "primary_disease": primary_disease,
            "lineage": lineage,
            "lineage_subtype": lineage_subtype,
        }

        # Index by all available name columns
        for col in name_cols:
            if col in row.index:
                name = str(row[col]).strip().upper()
                if name and name != "NAN":
                    lookup[name] = info

    # Pre-compute stripped name lookup (O(1) fallback instead of O(n²))
    lookup_stripped: dict[str, dict] = {
        k.replace("-", "").replace(" ", "").replace("_", ""): v
        for k, v in lookup.items()
    }

    logger.info("Loaded %d DepMap cell line entries", len(lookup))
    return lookup, lookup_stripped


def map_cell_lines_to_depmap(
    db_path: Path,
    depmap_lookup: dict[str, dict],
    depmap_lookup_stripped: dict[str, dict],
) -> dict[str, int]:
    """Map DC cell lines to DepMap model_id and metadata.

    Args:
        db_path: Path to DC database.
        depmap_lookup: cell_line_name (upper) → metadata dict.
        depmap_lookup_stripped: stripped name → metadata dict (pre-computed for O(1) fallback).

    Returns:
        Stats dict.
    """
    conn = get_connection(db_path)
    stats = {"matched": 0, "unmatched": 0}

    try:
        rows = conn.execute(
            "SELECT cell_line_id, cell_line_name FROM cell_lines"
        ).fetchall()

        for cell_line_id, cell_line_name in rows:
            name_upper = cell_line_name.strip().upper()

            # Try exact match first, then stripped (no hyphens/spaces/underscores)
            info = depmap_lookup.get(name_upper)
            if info is None:
                stripped = name_upper.replace("-", "").replace(" ", "").replace("_", "")
                info = depmap_lookup_stripped.get(stripped)

            if info:
                conn.execute(
                    """UPDATE cell_lines SET
                    depmap_model_id = COALESCE(depmap_model_id, ?),
                    cosmic_id = COALESCE(cosmic_id, ?),
                    tissue = COALESCE(tissue, ?),
                    cancer_type = COALESCE(cancer_type, ?),
                    primary_disease = COALESCE(primary_disease, ?),
                    lineage = COALESCE(lineage, ?),
                    lineage_subtype = COALESCE(lineage_subtype, ?)
                    WHERE cell_line_id = ?""",
                    (info["model_id"], info["cosmic_id"],
                     info["tissue"], info["cancer_type"],
                     info["primary_disease"], info["lineage"],
                     info["lineage_subtype"],
                     cell_line_id),
                )
                stats["matched"] += 1
            else:
                stats["unmatched"] += 1

        conn.commit()
        logger.info(
            "Cell line mapping: %d matched, %d unmatched",
            stats["matched"], stats["unmatched"],
        )
        return stats
    finally:
        conn.close()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Map cell lines to DepMap model IDs"
    )
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DC_DB_PATH,
        help="Path to DC database",
    )
    parser.add_argument(
        "--depmap-model-csv", type=Path,
        default=_PROJECT_ROOT / "data" / "depmap" / "Model.csv",
        help="Path to DepMap Model.csv",
    )
    args = parser.parse_args()

    if not args.depmap_model_csv.exists():
        print(f"ERROR: DepMap Model.csv not found: {args.depmap_model_csv}",
              file=sys.stderr)
        return 1

    run_dc_migrations(args.db_path)
    lookup, lookup_stripped = load_depmap_models(args.depmap_model_csv)
    stats = map_cell_lines_to_depmap(args.db_path, lookup, lookup_stripped)
    print(f"Results: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
