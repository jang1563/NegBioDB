"""PRISM drug sensitivity ETL — bridge tables linking GE cell lines to DTI compounds.

PRISM does NOT produce GE negative results. It populates prism_compounds and
prism_sensitivity tables for cross-domain analysis with DTI.

Data format:
  - Repurposing Hub sample sheet: compound metadata (broad_id, name, SMILES, etc.)
  - Primary screen matrix: log-fold change, cell lines × compounds
  - Secondary dose-response: AUC, IC50, EC50 values

Known limitations:
  - Not all broad_ids in the screen matrix have SMILES in the Hub sample sheet
  - ChEMBL ID mapping requires external lookup (InChIKey → ChEMBL)

License: CC BY 4.0
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_prism_compounds(
    db_path: Path,
    compound_meta_file: Path,
    batch_size: int = 2000,
) -> dict:
    """Load PRISM compound metadata into prism_compounds table.

    Args:
        db_path: Path to GE SQLite database.
        compound_meta_file: Repurposing Hub sample sheet CSV.
        batch_size: Commit every N inserts.

    Returns:
        Stats dict.
    """
    from negbiodb_depmap.depmap_db import get_connection, run_ge_migrations

    run_ge_migrations(db_path)
    conn = get_connection(db_path)

    stats = {
        "rows_in_file": 0,
        "compounds_inserted": 0,
        "compounds_with_smiles": 0,
        "compounds_with_inchikey": 0,
    }

    try:
        df = pd.read_csv(compound_meta_file, low_memory=False)
        stats["rows_in_file"] = len(df)

        insert_count = 0
        for _, row in df.iterrows():
            broad_id = row.get("broad_id") or row.get("IDs") or None
            if not broad_id or (isinstance(broad_id, float) and pd.isna(broad_id)):
                continue
            broad_id = str(broad_id).strip()

            name = row.get("name") or row.get("Name") or None
            if isinstance(name, float) and pd.isna(name):
                name = None
            elif name:
                name = str(name).strip()

            smiles = row.get("smiles") or row.get("SMILES") or None
            if isinstance(smiles, float) and pd.isna(smiles):
                smiles = None
            elif smiles:
                smiles = str(smiles).strip()

            inchikey = row.get("InChIKey") or row.get("inchikey") or None
            if isinstance(inchikey, float) and pd.isna(inchikey):
                inchikey = None
            elif inchikey:
                inchikey = str(inchikey).strip()

            moa = row.get("moa") or row.get("mechanism_of_action") or None
            if isinstance(moa, float) and pd.isna(moa):
                moa = None
            elif moa:
                moa = str(moa).strip()

            target = row.get("target") or row.get("target_name") or None
            if isinstance(target, float) and pd.isna(target):
                target = None
            elif target:
                target = str(target).strip()

            conn.execute(
                """INSERT OR IGNORE INTO prism_compounds
                (broad_id, name, smiles, inchikey, mechanism_of_action, target_name)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (broad_id, name, smiles, inchikey, moa, target),
            )

            if smiles:
                stats["compounds_with_smiles"] += 1
            if inchikey:
                stats["compounds_with_inchikey"] += 1

            insert_count += 1
            if insert_count % batch_size == 0:
                conn.commit()

        conn.commit()

        actual = conn.execute(
            "SELECT COUNT(*) FROM prism_compounds"
        ).fetchone()[0]
        stats["compounds_inserted"] = actual

        logger.info(
            "PRISM compounds loaded: %d total (%d SMILES, %d InChIKey)",
            actual, stats["compounds_with_smiles"], stats["compounds_with_inchikey"],
        )

    finally:
        conn.close()

    return stats


def load_prism_sensitivity(
    db_path: Path,
    primary_file: Path | None = None,
    secondary_file: Path | None = None,
    depmap_release: str = "PRISM_24Q2",
    batch_size: int = 5000,
) -> dict:
    """Load PRISM drug sensitivity data into prism_sensitivity table.

    Args:
        db_path: Path to GE SQLite database.
        primary_file: Primary screen matrix CSV (log-fold change).
        secondary_file: Secondary screen summary CSV (AUC, IC50).
        depmap_release: Release identifier.
        batch_size: Commit every N inserts.

    Returns:
        Stats dict.
    """
    from negbiodb_depmap.depmap_db import get_connection, run_ge_migrations

    run_ge_migrations(db_path)
    conn = get_connection(db_path)

    stats = {
        "primary_pairs": 0,
        "primary_skipped_no_compound": 0,
        "primary_skipped_no_cell_line": 0,
        "secondary_pairs": 0,
        "secondary_skipped_no_compound": 0,
        "secondary_skipped_no_cell_line": 0,
        "total_inserted": 0,
    }

    try:
        # Build lookups
        compound_lookup = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT broad_id, compound_id FROM prism_compounds"
            ).fetchall()
        }
        cl_lookup = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT model_id, cell_line_id FROM cell_lines"
            ).fetchall()
        }

        # Primary screen
        if primary_file and primary_file.exists():
            _load_primary_screen(
                conn, primary_file, compound_lookup, cl_lookup,
                depmap_release, batch_size, stats,
            )

        # Secondary screen
        if secondary_file and secondary_file.exists():
            _load_secondary_screen(
                conn, secondary_file, compound_lookup, cl_lookup,
                depmap_release, batch_size, stats,
            )

        actual = conn.execute(
            "SELECT COUNT(*) FROM prism_sensitivity"
        ).fetchone()[0]
        stats["total_inserted"] = actual

        # Dataset version
        conn.execute(
            "DELETE FROM dataset_versions WHERE name = 'prism_sensitivity' AND version = ?",
            (depmap_release,),
        )
        conn.execute(
            """INSERT INTO dataset_versions (name, version, source_url, row_count, notes)
            VALUES ('prism_sensitivity', ?, 'https://depmap.org/portal/download/all/',
                    ?, 'PRISM drug sensitivity (primary + secondary screens)')""",
            (depmap_release, actual),
        )
        conn.commit()

        logger.info(
            "PRISM sensitivity loaded: %d primary + %d secondary = %d total",
            stats["primary_pairs"], stats["secondary_pairs"], actual,
        )

    finally:
        conn.close()

    return stats


def _load_primary_screen(
    conn,
    primary_file: Path,
    compound_lookup: dict[str, int],
    cl_lookup: dict[str, int],
    depmap_release: str,
    batch_size: int,
    stats: dict,
) -> None:
    """Load primary screen matrix (cell lines × compounds, log-fold change)."""
    df = pd.read_csv(primary_file, index_col=0)
    insert_count = 0

    for model_id in df.index:
        model_id_str = str(model_id).strip()
        cl_id = cl_lookup.get(model_id_str)
        if cl_id is None:
            stats["primary_skipped_no_cell_line"] += df.shape[1]
            continue

        for col_name in df.columns:
            broad_id = str(col_name).strip()
            # Handle column formats: "BRD-K001::compound::2.5::HTS"
            if "::" in broad_id:
                broad_id = broad_id.split("::")[0]

            cpd_id = compound_lookup.get(broad_id)
            if cpd_id is None:
                stats["primary_skipped_no_compound"] += 1
                continue

            lfc = df.at[model_id, col_name]
            if pd.isna(lfc):
                continue

            conn.execute(
                """INSERT OR IGNORE INTO prism_sensitivity
                (compound_id, cell_line_id, screen_type, log_fold_change, depmap_release)
                VALUES (?, ?, 'primary', ?, ?)""",
                (cpd_id, cl_id, float(lfc), depmap_release),
            )
            stats["primary_pairs"] += 1
            insert_count += 1

            if insert_count % batch_size == 0:
                conn.commit()

    conn.commit()


def _load_secondary_screen(
    conn,
    secondary_file: Path,
    compound_lookup: dict[str, int],
    cl_lookup: dict[str, int],
    depmap_release: str,
    batch_size: int,
    stats: dict,
) -> None:
    """Load secondary screen dose-response data (row-oriented)."""
    df = pd.read_csv(secondary_file, low_memory=False)
    insert_count = 0

    for _, row in df.iterrows():
        broad_id = row.get("broad_id") or row.get("column_name") or None
        if not broad_id or (isinstance(broad_id, float) and pd.isna(broad_id)):
            continue
        broad_id = str(broad_id).strip()
        if "::" in broad_id:
            broad_id = broad_id.split("::")[0]

        cpd_id = compound_lookup.get(broad_id)
        if cpd_id is None:
            stats["secondary_skipped_no_compound"] += 1
            continue

        model_id = row.get("depmap_id") or row.get("row_name") or None
        if not model_id or (isinstance(model_id, float) and pd.isna(model_id)):
            continue
        model_id = str(model_id).strip()

        cl_id = cl_lookup.get(model_id)
        if cl_id is None:
            stats["secondary_skipped_no_cell_line"] += 1
            continue

        auc = row.get("auc")
        if isinstance(auc, float) and pd.isna(auc):
            auc = None
        elif auc is not None:
            auc = float(auc)

        ic50 = row.get("ic50")
        if isinstance(ic50, float) and pd.isna(ic50):
            ic50 = None
        elif ic50 is not None:
            ic50 = float(ic50)

        ec50 = row.get("ec50")
        if isinstance(ec50, float) and pd.isna(ec50):
            ec50 = None
        elif ec50 is not None:
            ec50 = float(ec50)

        conn.execute(
            """INSERT OR IGNORE INTO prism_sensitivity
            (compound_id, cell_line_id, screen_type, auc, ic50, ec50, depmap_release)
            VALUES (?, ?, 'secondary', ?, ?, ?, ?)""",
            (cpd_id, cl_id, auc, ic50, ec50, depmap_release),
        )
        stats["secondary_pairs"] += 1
        insert_count += 1

        if insert_count % batch_size == 0:
            conn.commit()

    conn.commit()
