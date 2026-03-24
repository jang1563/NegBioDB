"""Database connection and migration runner for NegBioDB GE (Gene Essentiality) domain.

Reuses the Common Layer from negbiodb.db (get_connection, connect,
run_migrations) with GE-specific defaults.
"""

import glob
import os
from pathlib import Path

# Reuse Common Layer infrastructure
from negbiodb.db import get_connection, connect, get_applied_versions  # noqa: F401

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GE_DB_PATH = _PROJECT_ROOT / "data" / "negbiodb_depmap.db"
DEFAULT_GE_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations_depmap"


def run_ge_migrations(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> list[str]:
    """Apply pending GE-domain migrations to the database.

    Mirrors negbiodb.db.run_migrations but uses GE-specific defaults.
    """
    if db_path is None:
        db_path = DEFAULT_GE_DB_PATH
    if migrations_dir is None:
        migrations_dir = DEFAULT_GE_MIGRATIONS_DIR

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


def create_ge_database(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> Path:
    """Create a new GE database by running all GE migrations."""
    if db_path is None:
        db_path = DEFAULT_GE_DB_PATH

    db_path = Path(db_path)
    applied = run_ge_migrations(db_path, migrations_dir)

    if applied:
        print(f"Applied {len(applied)} GE migration(s): {', '.join(applied)}")
    else:
        print("GE database is up to date (no pending migrations).")

    return db_path


def refresh_all_ge_pairs(conn) -> int:
    """Refresh gene_cell_pairs aggregation from ge_negative_results.

    Deletes all existing pairs and re-aggregates, computing best confidence,
    best evidence type, score ranges, and degree counts.
    """
    conn.execute("DELETE FROM ge_split_assignments")
    conn.execute("DELETE FROM gene_cell_pairs")
    conn.execute(
        """INSERT INTO gene_cell_pairs
        (gene_id, cell_line_id, num_screens, num_sources,
         best_confidence, best_evidence_type,
         min_gene_effect, max_gene_effect, mean_gene_effect)
        SELECT
            gene_id,
            cell_line_id,
            COUNT(DISTINCT COALESCE(screen_id, -1)),
            COUNT(DISTINCT source_db),
            CASE MIN(CASE confidence_tier
                WHEN 'gold' THEN 1 WHEN 'silver' THEN 2
                WHEN 'bronze' THEN 3 END)
                WHEN 1 THEN 'gold' WHEN 2 THEN 'silver'
                WHEN 3 THEN 'bronze' END,
            CASE MIN(CASE evidence_type
                WHEN 'reference_nonessential' THEN 1
                WHEN 'multi_screen_concordant' THEN 2
                WHEN 'crispr_nonessential' THEN 3
                WHEN 'rnai_nonessential' THEN 4
                WHEN 'context_nonessential' THEN 5 END)
                WHEN 1 THEN 'reference_nonessential'
                WHEN 2 THEN 'multi_screen_concordant'
                WHEN 3 THEN 'crispr_nonessential'
                WHEN 4 THEN 'rnai_nonessential'
                WHEN 5 THEN 'context_nonessential' END,
            MIN(gene_effect_score),
            MAX(gene_effect_score),
            AVG(gene_effect_score)
        FROM ge_negative_results
        GROUP BY gene_id, cell_line_id"""
    )

    # Compute gene_degree: number of cell lines where this gene is non-essential
    conn.execute("DROP TABLE IF EXISTS _gdeg")
    conn.execute(
        """CREATE TEMP TABLE _gdeg (
            gene_id INTEGER PRIMARY KEY, deg INTEGER)"""
    )
    conn.execute(
        """INSERT INTO _gdeg
        SELECT gene_id, COUNT(DISTINCT cell_line_id)
        FROM gene_cell_pairs GROUP BY gene_id"""
    )
    conn.execute(
        """UPDATE gene_cell_pairs SET gene_degree = (
            SELECT deg FROM _gdeg d
            WHERE d.gene_id = gene_cell_pairs.gene_id
        )"""
    )
    conn.execute("DROP TABLE _gdeg")

    # Compute cell_line_degree: number of genes non-essential in this cell line
    conn.execute("DROP TABLE IF EXISTS _cldeg")
    conn.execute(
        """CREATE TEMP TABLE _cldeg (
            cell_line_id INTEGER PRIMARY KEY, deg INTEGER)"""
    )
    conn.execute(
        """INSERT INTO _cldeg
        SELECT cell_line_id, COUNT(DISTINCT gene_id)
        FROM gene_cell_pairs GROUP BY cell_line_id"""
    )
    conn.execute(
        """UPDATE gene_cell_pairs SET cell_line_degree = (
            SELECT deg FROM _cldeg d
            WHERE d.cell_line_id = gene_cell_pairs.cell_line_id
        )"""
    )
    conn.execute("DROP TABLE _cldeg")

    count = conn.execute(
        "SELECT COUNT(*) FROM gene_cell_pairs"
    ).fetchone()[0]
    return count
