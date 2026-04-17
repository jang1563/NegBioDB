"""Database helpers for the derived NegBioGraph layer."""

from __future__ import annotations

import glob
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def _resolve_project_root() -> Path:
    package_dir = Path(__file__).resolve().parent
    if package_dir.parent.name == "src":
        return package_dir.parent.parent
    return package_dir.parent


_PROJECT_ROOT = _resolve_project_root()
DEFAULT_GRAPH_DB_PATH = _PROJECT_ROOT / "data" / "negbiodb_graph.db"
DEFAULT_GRAPH_DUCKDB_PATH = _PROJECT_ROOT / "data" / "negbiodb_graph.duckdb"
DEFAULT_GRAPH_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations_graph"


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with NegBioGraph-standard PRAGMAs."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connect(db_path: str | Path):
    """Context manager for NegBioGraph database connections."""
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_applied_versions(conn: sqlite3.Connection) -> set[str]:
    """Return the set of applied graph migration versions."""
    try:
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        return {row[0] for row in rows}
    except sqlite3.OperationalError:
        return set()


def run_graph_migrations(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> list[str]:
    """Apply pending SQL migrations for the graph database."""
    if db_path is None:
        db_path = DEFAULT_GRAPH_DB_PATH
    if migrations_dir is None:
        migrations_dir = DEFAULT_GRAPH_MIGRATIONS_DIR

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
            if version in applied:
                continue
            with open(mf) as handle:
                sql = handle.read()
            conn.executescript(sql)
            newly_applied.append(version)
        return newly_applied
    finally:
        conn.close()


def create_graph_database(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> Path:
    """Create the graph database by running graph migrations."""
    if db_path is None:
        db_path = DEFAULT_GRAPH_DB_PATH
    db_path = Path(db_path)
    applied = run_graph_migrations(db_path, migrations_dir)
    if applied:
        print(f"Applied {len(applied)} graph migration(s): {', '.join(applied)}")
    else:
        print("Graph database is up to date (no pending migrations).")
    return db_path
