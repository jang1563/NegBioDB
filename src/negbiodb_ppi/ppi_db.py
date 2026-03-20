"""Database connection and migration runner for NegBioDB PPI domain.

Reuses the Common Layer from negbiodb.db (get_connection, connect,
run_migrations) with PPI-specific defaults.
"""

from pathlib import Path

# Reuse Common Layer infrastructure
from negbiodb.db import get_connection, connect, get_applied_versions  # noqa: F401

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PPI_DB_PATH = _PROJECT_ROOT / "data" / "negbiodb_ppi.db"
DEFAULT_PPI_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations_ppi"


def run_ppi_migrations(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> list[str]:
    """Apply pending PPI-domain migrations to the database.

    Mirrors negbiodb.db.run_migrations but uses PPI-specific defaults.
    """
    import glob
    import os

    if db_path is None:
        db_path = DEFAULT_PPI_DB_PATH
    if migrations_dir is None:
        migrations_dir = DEFAULT_PPI_MIGRATIONS_DIR

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


def create_ppi_database(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> Path:
    """Create a new PPI database by running all PPI migrations."""
    if db_path is None:
        db_path = DEFAULT_PPI_DB_PATH

    db_path = Path(db_path)
    applied = run_ppi_migrations(db_path, migrations_dir)

    if applied:
        print(f"Applied {len(applied)} PPI migration(s): {', '.join(applied)}")
    else:
        print("PPI database is up to date (no pending migrations).")

    return db_path


def refresh_all_ppi_pairs(conn) -> int:
    """Refresh protein_protein_pairs aggregation from ppi_negative_results.

    Deletes all existing pairs and re-aggregates, computing best confidence,
    best evidence type, degree counts, and score ranges.
    """
    conn.execute("DELETE FROM ppi_split_assignments")
    conn.execute("DELETE FROM protein_protein_pairs")
    conn.execute(
        """INSERT INTO protein_protein_pairs
        (protein1_id, protein2_id, num_experiments, num_sources,
         best_confidence, best_evidence_type, earliest_year,
         min_interaction_score, max_interaction_score)
        SELECT
            protein1_id,
            protein2_id,
            COUNT(DISTINCT COALESCE(experiment_id, -1)),
            COUNT(DISTINCT source_db),
            CASE MIN(CASE confidence_tier
                WHEN 'gold' THEN 1 WHEN 'silver' THEN 2
                WHEN 'bronze' THEN 3 WHEN 'copper' THEN 4 END)
                WHEN 1 THEN 'gold' WHEN 2 THEN 'silver'
                WHEN 3 THEN 'bronze' WHEN 4 THEN 'copper' END,
            CASE MIN(CASE evidence_type
                WHEN 'experimental_non_interaction' THEN 1
                WHEN 'literature_reported' THEN 2
                WHEN 'ml_predicted_negative' THEN 3
                WHEN 'low_score_negative' THEN 4
                WHEN 'compartment_separated' THEN 5 END)
                WHEN 1 THEN 'experimental_non_interaction'
                WHEN 2 THEN 'literature_reported'
                WHEN 3 THEN 'ml_predicted_negative'
                WHEN 4 THEN 'low_score_negative'
                WHEN 5 THEN 'compartment_separated' END,
            MIN(publication_year),
            MIN(interaction_score),
            MAX(interaction_score)
        FROM ppi_negative_results
        GROUP BY protein1_id, protein2_id"""
    )

    # Compute true network degree for each protein.
    # In PPI, both columns are proteins; a protein can appear on EITHER side
    # due to canonical ordering (protein1_id < protein2_id). Must union both.
    conn.execute("DROP TABLE IF EXISTS _pdeg")
    conn.execute(
        """CREATE TEMP TABLE _pdeg (
            protein_id INTEGER PRIMARY KEY, deg INTEGER)"""
    )
    conn.execute(
        """INSERT INTO _pdeg
        SELECT protein_id, COUNT(DISTINCT partner_id)
        FROM (
            SELECT protein1_id AS protein_id, protein2_id AS partner_id
            FROM protein_protein_pairs
            UNION ALL
            SELECT protein2_id AS protein_id, protein1_id AS partner_id
            FROM protein_protein_pairs
        )
        GROUP BY protein_id"""
    )
    conn.execute(
        """UPDATE protein_protein_pairs SET protein1_degree = (
            SELECT deg FROM _pdeg d
            WHERE d.protein_id = protein_protein_pairs.protein1_id
        )"""
    )
    conn.execute(
        """UPDATE protein_protein_pairs SET protein2_degree = (
            SELECT deg FROM _pdeg d
            WHERE d.protein_id = protein_protein_pairs.protein2_id
        )"""
    )
    conn.execute("DROP TABLE _pdeg")

    count = conn.execute(
        "SELECT COUNT(*) FROM protein_protein_pairs"
    ).fetchone()[0]
    return count
