"""Database connection and migration runner for NegBioDB."""

import glob
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "negbiodb.db"
DEFAULT_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with NegBioDB-standard PRAGMAs.

    Sets WAL journal mode and enables foreign key enforcement.
    The caller is responsible for closing the connection.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connect(db_path: str | Path):
    """Context manager for NegBioDB database connections.

    Usage:
        with connect("data/negbiodb.db") as conn:
            conn.execute("SELECT ...")
    """
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_applied_versions(conn: sqlite3.Connection) -> set[str]:
    """Return the set of migration versions already applied."""
    try:
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        return {row[0] for row in rows}
    except sqlite3.OperationalError:
        return set()


def run_migrations(db_path: str | Path,
                   migrations_dir: str | Path | None = None) -> list[str]:
    """Apply pending SQL migrations to the database.

    Migrations are .sql files in migrations_dir, sorted by filename prefix.
    Version is extracted from filename: "001_initial_schema.sql" -> "001".
    Already-applied versions (in schema_migrations) are skipped.

    Returns:
        List of version strings applied in this run.
    """
    if migrations_dir is None:
        migrations_dir = DEFAULT_MIGRATIONS_DIR

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


def refresh_all_pairs(conn: sqlite3.Connection) -> int:
    """Refresh compound_target_pairs aggregation across ALL sources.

    Deletes all existing pairs and re-aggregates from negative_results,
    merging cross-source data with best confidence tier selection.
    """
    conn.execute("DELETE FROM compound_target_pairs")
    conn.execute(
        """INSERT INTO compound_target_pairs
        (compound_id, target_id, num_assays, num_sources,
         best_confidence, best_result_type, earliest_year,
         median_pchembl, min_activity_value, max_activity_value)
        SELECT
            compound_id,
            target_id,
            COUNT(DISTINCT COALESCE(assay_id, -1)),
            COUNT(DISTINCT source_db),
            CASE MIN(CASE confidence_tier
                WHEN 'gold' THEN 1 WHEN 'silver' THEN 2
                WHEN 'bronze' THEN 3 WHEN 'copper' THEN 4 END)
                WHEN 1 THEN 'gold' WHEN 2 THEN 'silver'
                WHEN 3 THEN 'bronze' WHEN 4 THEN 'copper' END,
            MIN(result_type),
            MIN(publication_year),
            AVG(pchembl_value),
            MIN(activity_value),
            MAX(activity_value)
        FROM negative_results
        GROUP BY compound_id, target_id"""
    )

    count = conn.execute("SELECT COUNT(*) FROM compound_target_pairs").fetchone()[0]
    return count


def create_database(db_path: str | Path | None = None,
                    migrations_dir: str | Path | None = None) -> Path:
    """Create a new NegBioDB database by running all migrations.

    Convenience wrapper around run_migrations with sensible defaults.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    db_path = Path(db_path)
    applied = run_migrations(db_path, migrations_dir)

    if applied:
        print(f"Applied {len(applied)} migration(s): {', '.join(applied)}")
    else:
        print("Database is up to date (no pending migrations).")

    return db_path
