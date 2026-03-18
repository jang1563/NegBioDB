"""Database connection and migration runner for NegBioDB Clinical Trial domain.

Reuses the Common Layer from negbiodb.db (get_connection, connect,
run_migrations) with CT-specific defaults.
"""

from pathlib import Path

# Reuse Common Layer infrastructure
from negbiodb.db import get_connection, connect, get_applied_versions  # noqa: F401

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CT_DB_PATH = _PROJECT_ROOT / "data" / "negbiodb_ct.db"
DEFAULT_CT_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations_ct"


def run_ct_migrations(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> list[str]:
    """Apply pending CT-domain migrations to the database.

    Mirrors negbiodb.db.run_migrations but uses CT-specific defaults.
    """
    import glob
    import os

    if db_path is None:
        db_path = DEFAULT_CT_DB_PATH
    if migrations_dir is None:
        migrations_dir = DEFAULT_CT_MIGRATIONS_DIR

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


def create_ct_database(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> Path:
    """Create a new CT database by running all CT migrations."""
    if db_path is None:
        db_path = DEFAULT_CT_DB_PATH

    db_path = Path(db_path)
    applied = run_ct_migrations(db_path, migrations_dir)

    if applied:
        print(f"Applied {len(applied)} CT migration(s): {', '.join(applied)}")
    else:
        print("CT database is up to date (no pending migrations).")

    return db_path


def refresh_all_ct_pairs(conn) -> int:
    """Refresh intervention_condition_pairs aggregation across ALL sources.

    Deletes all existing pairs and re-aggregates from trial_failure_results.
    """
    conn.execute("DELETE FROM intervention_condition_pairs")
    conn.execute(
        """INSERT INTO intervention_condition_pairs
        (intervention_id, condition_id, num_trials, num_sources,
         best_confidence, primary_failure_category, earliest_year,
         highest_phase_reached)
        SELECT
            intervention_id,
            condition_id,
            COUNT(DISTINCT trial_id),
            COUNT(DISTINCT source_db),
            CASE MIN(CASE confidence_tier
                WHEN 'gold' THEN 1 WHEN 'silver' THEN 2
                WHEN 'bronze' THEN 3 WHEN 'copper' THEN 4 END)
                WHEN 1 THEN 'gold' WHEN 2 THEN 'silver'
                WHEN 3 THEN 'bronze' WHEN 4 THEN 'copper' END,
            -- Most common failure category as primary
            (SELECT r2.failure_category
             FROM trial_failure_results r2
             WHERE r2.intervention_id = trial_failure_results.intervention_id
               AND r2.condition_id = trial_failure_results.condition_id
             GROUP BY r2.failure_category
             ORDER BY COUNT(*) DESC LIMIT 1),
            MIN(publication_year),
            -- CASE-based phase ordering (lexicographic MAX is wrong
            -- because 'not_applicable' > 'early_phase_1' alphabetically)
            CASE MAX(CASE highest_phase_reached
                WHEN 'phase_4' THEN 8
                WHEN 'phase_3' THEN 7
                WHEN 'phase_2_3' THEN 6
                WHEN 'phase_2' THEN 5
                WHEN 'phase_1_2' THEN 4
                WHEN 'phase_1' THEN 3
                WHEN 'early_phase_1' THEN 2
                WHEN 'not_applicable' THEN 1
                ELSE 0 END)
                WHEN 8 THEN 'phase_4'
                WHEN 7 THEN 'phase_3'
                WHEN 6 THEN 'phase_2_3'
                WHEN 5 THEN 'phase_2'
                WHEN 4 THEN 'phase_1_2'
                WHEN 3 THEN 'phase_1'
                WHEN 2 THEN 'early_phase_1'
                WHEN 1 THEN 'not_applicable'
                ELSE NULL END
        FROM trial_failure_results
        GROUP BY intervention_id, condition_id"""
    )

    # Compute intervention_degree
    conn.execute("DROP TABLE IF EXISTS _ideg")
    conn.execute(
        """CREATE TEMP TABLE _ideg (
            intervention_id INTEGER PRIMARY KEY, deg INTEGER)"""
    )
    conn.execute(
        """INSERT INTO _ideg
        SELECT intervention_id, COUNT(DISTINCT condition_id)
        FROM intervention_condition_pairs GROUP BY intervention_id"""
    )
    conn.execute(
        """UPDATE intervention_condition_pairs SET intervention_degree = (
            SELECT deg FROM _ideg d
            WHERE d.intervention_id = intervention_condition_pairs.intervention_id
        )"""
    )
    conn.execute("DROP TABLE _ideg")

    # Compute condition_degree
    conn.execute("DROP TABLE IF EXISTS _cdeg")
    conn.execute(
        """CREATE TEMP TABLE _cdeg (
            condition_id INTEGER PRIMARY KEY, deg INTEGER)"""
    )
    conn.execute(
        """INSERT INTO _cdeg
        SELECT condition_id, COUNT(DISTINCT intervention_id)
        FROM intervention_condition_pairs GROUP BY condition_id"""
    )
    conn.execute(
        """UPDATE intervention_condition_pairs SET condition_degree = (
            SELECT deg FROM _cdeg d
            WHERE d.condition_id = intervention_condition_pairs.condition_id
        )"""
    )
    conn.execute("DROP TABLE _cdeg")

    count = conn.execute(
        "SELECT COUNT(*) FROM intervention_condition_pairs"
    ).fetchone()[0]
    return count
