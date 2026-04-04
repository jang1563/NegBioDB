"""Database connection and migration runner for NegBioDB VP (Variant Pathogenicity) domain.

Reuses the Common Layer from negbiodb.db (get_connection, connect,
run_migrations) with VP-specific defaults.
"""

import glob
import os
from pathlib import Path

# Reuse Common Layer infrastructure
from negbiodb.db import get_connection, connect, get_applied_versions  # noqa: F401

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_VP_DB_PATH = _PROJECT_ROOT / "data" / "negbiodb_vp.db"
DEFAULT_VP_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations_vp"


def run_vp_migrations(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> list[str]:
    """Apply pending VP-domain migrations to the database.

    Mirrors negbiodb.db.run_migrations but uses VP-specific defaults.
    """
    if db_path is None:
        db_path = DEFAULT_VP_DB_PATH
    if migrations_dir is None:
        migrations_dir = DEFAULT_VP_MIGRATIONS_DIR

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


def create_vp_database(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> Path:
    """Create a new VP database by running all VP migrations."""
    if db_path is None:
        db_path = DEFAULT_VP_DB_PATH

    db_path = Path(db_path)
    applied = run_vp_migrations(db_path, migrations_dir)

    if applied:
        print(f"Applied {len(applied)} VP migration(s): {', '.join(applied)}")
    else:
        print("VP database is up to date (no pending migrations).")

    return db_path


def refresh_all_vp_pairs(conn) -> int:
    """Refresh variant_disease_pairs aggregation from vp_negative_results.

    Deletes all existing pairs and split assignments, then re-aggregates
    computing best confidence, best evidence type, best classification,
    conflict flags, and degree counts.
    """
    conn.execute("DELETE FROM vp_split_assignments")
    conn.execute("DELETE FROM variant_disease_pairs")

    conn.execute(
        """INSERT INTO variant_disease_pairs
        (variant_id, disease_id, num_submissions, num_submitters,
         best_confidence, best_evidence_type, best_classification,
         earliest_year, has_conflict, max_population_af,
         num_benign_criteria)
        SELECT
            nr.variant_id,
            nr.disease_id,
            COUNT(*) AS num_submissions,
            COUNT(DISTINCT s.submitter_name) AS num_submitters,
            -- best_confidence: gold(1) > silver(2) > bronze(3) > copper(4)
            CASE MIN(CASE nr.confidence_tier
                WHEN 'gold' THEN 1 WHEN 'silver' THEN 2
                WHEN 'bronze' THEN 3 WHEN 'copper' THEN 4 END)
                WHEN 1 THEN 'gold' WHEN 2 THEN 'silver'
                WHEN 3 THEN 'bronze' WHEN 4 THEN 'copper' END,
            -- best_evidence_type: expert(1) > multi(2) > single(3) > population(4) > computational(5)
            CASE MIN(CASE nr.evidence_type
                WHEN 'expert_reviewed' THEN 1
                WHEN 'multi_submitter_concordant' THEN 2
                WHEN 'single_submitter' THEN 3
                WHEN 'population_frequency' THEN 4
                WHEN 'computational_only' THEN 5 END)
                WHEN 1 THEN 'expert_reviewed'
                WHEN 2 THEN 'multi_submitter_concordant'
                WHEN 3 THEN 'single_submitter'
                WHEN 4 THEN 'population_frequency'
                WHEN 5 THEN 'computational_only' END,
            -- best_classification: benign > likely_benign > benign/likely_benign
            CASE WHEN COUNT(CASE WHEN nr.classification = 'benign' THEN 1 END) > 0
                 THEN 'benign'
                 WHEN COUNT(CASE WHEN nr.classification = 'likely_benign' THEN 1 END) > 0
                 THEN 'likely_benign'
                 ELSE 'benign/likely_benign' END,
            MIN(nr.submission_year),
            MAX(nr.has_conflict),
            MAX(v.gnomad_af_global),
            MAX(nr.num_benign_criteria)
        FROM vp_negative_results nr
        LEFT JOIN vp_submissions s ON nr.submission_id = s.submission_id
        LEFT JOIN variants v ON nr.variant_id = v.variant_id
        GROUP BY nr.variant_id, nr.disease_id"""
    )

    # Compute variant_degree: number of diseases for each variant
    conn.execute("DROP TABLE IF EXISTS _vdeg")
    conn.execute(
        """CREATE TEMP TABLE _vdeg (
            variant_id INTEGER PRIMARY KEY, deg INTEGER)"""
    )
    conn.execute(
        """INSERT INTO _vdeg
        SELECT variant_id, COUNT(DISTINCT disease_id)
        FROM variant_disease_pairs GROUP BY variant_id"""
    )
    conn.execute(
        """UPDATE variant_disease_pairs SET variant_degree = (
            SELECT deg FROM _vdeg d
            WHERE d.variant_id = variant_disease_pairs.variant_id
        )"""
    )
    conn.execute("DROP TABLE _vdeg")

    # Compute disease_degree: number of variants for each disease
    conn.execute("DROP TABLE IF EXISTS _ddeg")
    conn.execute(
        """CREATE TEMP TABLE _ddeg (
            disease_id INTEGER PRIMARY KEY, deg INTEGER)"""
    )
    conn.execute(
        """INSERT INTO _ddeg
        SELECT disease_id, COUNT(DISTINCT variant_id)
        FROM variant_disease_pairs GROUP BY disease_id"""
    )
    conn.execute(
        """UPDATE variant_disease_pairs SET disease_degree = (
            SELECT deg FROM _ddeg d
            WHERE d.disease_id = variant_disease_pairs.disease_id
        )"""
    )
    conn.execute("DROP TABLE _ddeg")

    count = conn.execute(
        "SELECT COUNT(*) FROM variant_disease_pairs"
    ).fetchone()[0]
    return count
