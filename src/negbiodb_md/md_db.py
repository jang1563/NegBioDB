"""Database connection and migration runner for NegBioDB MD (Metabolomics-Disease) domain.

Reuses the Common Layer from negbiodb.db (get_connection, connect,
run_migrations) with MD-specific defaults.
"""

import glob
import os
from pathlib import Path

# Reuse Common Layer infrastructure
from negbiodb.db import get_connection, connect, get_applied_versions  # noqa: F401

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MD_DB_PATH = _PROJECT_ROOT / "data" / "negbiodb_md.db"
DEFAULT_MD_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations_md"

# Confidence tier ordering for SQL comparisons (gold=1 best)
TIER_ORDER = {"gold": 1, "silver": 2, "bronze": 3, "copper": 4}
TIER_SQL = "CASE tier WHEN 'gold' THEN 1 WHEN 'silver' THEN 2 WHEN 'bronze' THEN 3 WHEN 'copper' THEN 4 END"

# Significance thresholds for tier assignment
P_VALUE_THRESHOLD = 0.05
FDR_THRESHOLD = 0.1
N_GOLD = 50   # minimum n per group for gold tier
N_SILVER = 20  # minimum n per group for silver tier


def assign_tier(p_value: float | None, fdr: float | None, n_group: int | None) -> str:
    """Assign confidence tier to a negative result (is_significant=FALSE).

    Tier logic:
        gold:   FDR > 0.1, n >= 50/group (strong evidence of non-association)
        silver: p > 0.05, n >= 20/group
        bronze: p > 0.05, n < 20/group (or n unknown)
        copper: no statistics reported
    """
    if p_value is None and fdr is None:
        return "copper"
    p = p_value if p_value is not None else 1.0
    fdr_val = fdr if fdr is not None else 1.0
    n = n_group if n_group is not None else 0

    if fdr_val > FDR_THRESHOLD and n >= N_GOLD:
        return "gold"
    elif p > P_VALUE_THRESHOLD and n >= N_SILVER:
        return "silver"
    elif p > P_VALUE_THRESHOLD:
        return "bronze"
    else:
        # p <= threshold: technically significant — should not be calling assign_tier here
        return "bronze"


def run_md_migrations(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> list[str]:
    """Apply pending MD-domain migrations to the database.

    Mirrors negbiodb.db.run_migrations but uses MD-specific defaults.
    """
    if db_path is None:
        db_path = DEFAULT_MD_DB_PATH
    if migrations_dir is None:
        migrations_dir = DEFAULT_MD_MIGRATIONS_DIR

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


def create_md_database(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> Path:
    """Create a new MD database by running all MD migrations."""
    if db_path is None:
        db_path = DEFAULT_MD_DB_PATH

    db_path = Path(db_path)
    applied = run_md_migrations(db_path, migrations_dir)

    if applied:
        print(f"Applied {len(applied)} MD migration(s): {', '.join(applied)}")
    else:
        print("MD database is up to date (no pending migrations).")

    return db_path


def get_md_connection(db_path: str | Path | None = None):
    """Return a sqlite3 connection to the MD database, creating it if needed."""
    if db_path is None:
        db_path = DEFAULT_MD_DB_PATH
    db_path = Path(db_path)
    if not db_path.exists():
        create_md_database(db_path)
    return get_connection(db_path)


def refresh_all_pairs(conn) -> int:
    """Recompute md_metabolite_disease_pairs from md_biomarker_results.

    Deletes all existing pair rows and split assignments, then re-aggregates.
    Returns the number of pairs inserted.
    """
    conn.execute("DELETE FROM md_split_assignments")
    conn.execute("DELETE FROM md_metabolite_disease_pairs")

    conn.execute(
        """INSERT INTO md_metabolite_disease_pairs
            (metabolite_id, disease_id,
             n_studies_total, n_studies_negative, n_studies_positive,
             consensus, best_tier)
        SELECT
            r.metabolite_id,
            r.disease_id,
            COUNT(DISTINCT r.study_id) AS n_studies_total,
            COUNT(DISTINCT CASE WHEN r.is_significant = 0 THEN r.study_id END) AS n_studies_negative,
            COUNT(DISTINCT CASE WHEN r.is_significant = 1 THEN r.study_id END) AS n_studies_positive,
            CASE
                WHEN COUNT(DISTINCT CASE WHEN r.is_significant = 0 THEN r.study_id END) = 0
                     THEN 'positive'
                WHEN COUNT(DISTINCT CASE WHEN r.is_significant = 1 THEN r.study_id END) = 0
                     THEN 'negative'
                ELSE 'mixed'
            END AS consensus,
            -- best_tier: gold(1) > silver(2) > bronze(3) > copper(4)
            CASE MIN(CASE r.tier
                WHEN 'gold'   THEN 1
                WHEN 'silver' THEN 2
                WHEN 'bronze' THEN 3
                WHEN 'copper' THEN 4
                ELSE 5 END)
                WHEN 1 THEN 'gold'
                WHEN 2 THEN 'silver'
                WHEN 3 THEN 'bronze'
                WHEN 4 THEN 'copper'
                ELSE NULL
            END AS best_tier
        FROM md_biomarker_results r
        GROUP BY r.metabolite_id, r.disease_id"""
    )

    # Update degree columns
    conn.execute("DROP TABLE IF EXISTS _mdeg")
    conn.execute(
        "CREATE TEMP TABLE _mdeg (metabolite_id INTEGER PRIMARY KEY, deg INTEGER)"
    )
    conn.execute(
        """INSERT INTO _mdeg
        SELECT metabolite_id, COUNT(*) AS deg
        FROM md_metabolite_disease_pairs
        GROUP BY metabolite_id"""
    )
    conn.execute(
        """UPDATE md_metabolite_disease_pairs
        SET metabolite_degree = (
            SELECT deg FROM _mdeg d
            WHERE d.metabolite_id = md_metabolite_disease_pairs.metabolite_id
        )"""
    )
    conn.execute("DROP TABLE _mdeg")

    conn.execute("DROP TABLE IF EXISTS _ddeg")
    conn.execute(
        "CREATE TEMP TABLE _ddeg (disease_id INTEGER PRIMARY KEY, deg INTEGER)"
    )
    conn.execute(
        """INSERT INTO _ddeg
        SELECT disease_id, COUNT(*) AS deg
        FROM md_metabolite_disease_pairs
        GROUP BY disease_id"""
    )
    conn.execute(
        """UPDATE md_metabolite_disease_pairs
        SET disease_degree = (
            SELECT deg FROM _ddeg d
            WHERE d.disease_id = md_metabolite_disease_pairs.disease_id
        )"""
    )
    conn.execute("DROP TABLE _ddeg")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM md_metabolite_disease_pairs").fetchone()[0]
    return count
