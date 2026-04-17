"""Database helpers for the NegBioDB Cell Painting domain."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from negbiodb.db import get_connection, connect, get_applied_versions  # noqa: F401


def _resolve_project_root() -> Path:
    package_dir = Path(__file__).resolve().parent
    if package_dir.parent.name == "src":
        return package_dir.parent.parent
    return package_dir.parent


_PROJECT_ROOT = _resolve_project_root()
DEFAULT_CP_DB_PATH = _PROJECT_ROOT / "data" / "negbiodb_cp.db"
DEFAULT_CP_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations_cp"
ANNOTATION_MODES = ("annotated", "plate_proxy")


def run_cp_migrations(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> list[str]:
    """Apply pending CP-domain migrations."""
    import glob
    import os

    if db_path is None:
        db_path = DEFAULT_CP_DB_PATH
    if migrations_dir is None:
        migrations_dir = DEFAULT_CP_MIGRATIONS_DIR

    db_path = Path(db_path)
    migrations_dir = Path(migrations_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(db_path)
    try:
        applied = get_applied_versions(conn)
        migration_files = sorted(glob.glob(str(migrations_dir / "*.sql")))
        newly_applied = []

        for mf in migration_files:
            version = os.path.basename(mf).split("_")[0]
            if version not in applied:
                with open(mf) as f:
                    sql = f.read()
                conn.executescript(sql)
                newly_applied.append(version)

        return newly_applied
    finally:
        conn.close()


def create_cp_database(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> Path:
    """Create a CP database by running all migrations."""
    if db_path is None:
        db_path = DEFAULT_CP_DB_PATH

    db_path = Path(db_path)
    applied = run_cp_migrations(db_path, migrations_dir)

    if applied:
        print(f"Applied {len(applied)} CP migration(s): {', '.join(applied)}")
    else:
        print("CP database is up to date (no pending migrations).")

    return db_path


def get_cp_annotation_summary(conn: sqlite3.Connection) -> dict:
    """Return dataset/annotation-mode summary for CP benchmark rows."""
    rows = conn.execute(
        """
        SELECT DISTINCT
            dv.name,
            dv.version,
            COALESCE(dv.annotation_mode, 'annotated') AS annotation_mode
        FROM cp_perturbation_results r
        JOIN cp_batches b ON r.batch_id = b.batch_id
        LEFT JOIN dataset_versions dv ON b.dataset_id = dv.dataset_id
        ORDER BY dv.name, dv.version
        """
    ).fetchall()

    datasets = []
    modes = []
    for name, version, annotation_mode in rows:
        mode = annotation_mode or "annotated"
        datasets.append(
            {
                "name": name,
                "version": version,
                "annotation_mode": mode,
            }
        )
        if mode not in modes:
            modes.append(mode)

    return {
        "dataset_versions": datasets,
        "annotation_modes": modes,
        "production_ready": "plate_proxy" not in modes,
    }


def ensure_cp_production_ready(
    conn: sqlite3.Connection,
    *,
    allow_proxy_smoke: bool = False,
) -> dict:
    """Raise if a CP DB contains proxy-only rows and proxy mode is not allowed."""
    summary = get_cp_annotation_summary(conn)
    if not allow_proxy_smoke and "plate_proxy" in summary["annotation_modes"]:
        raise ValueError(
            "CP benchmark/export path is blocked for plate_proxy datasets. "
            "Re-run with --allow-proxy-smoke only for plumbing smoke validation."
        )
    return summary
