"""Database connection and migration runner for NegBioDB DC (Drug Combination) domain.

Reuses the Common Layer from negbiodb.db (get_connection, connect,
run_migrations) with DC-specific defaults.
"""

import glob
import os
from pathlib import Path

# Reuse Common Layer infrastructure
from negbiodb.db import get_connection, connect, get_applied_versions  # noqa: F401

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DC_DB_PATH = _PROJECT_ROOT / "data" / "negbiodb_dc.db"
DEFAULT_DC_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations_dc"

# Synergy class thresholds (ZIP-based)
SYNERGY_THRESHOLDS = {
    "strongly_synergistic": (10, float("inf")),
    "synergistic": (5, 10),
    "additive": (-5, 5),
    "antagonistic": (-10, -5),
    "strongly_antagonistic": (float("-inf"), -10),
}


def classify_synergy(zip_score: float | None) -> str | None:
    """Classify a ZIP synergy score into a synergy class.

    Thresholds:
        ZIP > 10  → strongly_synergistic
        5 < ZIP ≤ 10 → synergistic
        -5 ≤ ZIP ≤ 5 → additive
        -10 ≤ ZIP < -5 → antagonistic
        ZIP < -10 → strongly_antagonistic
    """
    if zip_score is None:
        return None
    if zip_score > 10:
        return "strongly_synergistic"
    elif zip_score > 5:
        return "synergistic"
    elif zip_score >= -5:
        return "additive"
    elif zip_score >= -10:
        return "antagonistic"
    else:
        return "strongly_antagonistic"


def run_dc_migrations(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> list[str]:
    """Apply pending DC-domain migrations to the database.

    Mirrors negbiodb.db.run_migrations but uses DC-specific defaults.
    """
    if db_path is None:
        db_path = DEFAULT_DC_DB_PATH
    if migrations_dir is None:
        migrations_dir = DEFAULT_DC_MIGRATIONS_DIR

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


def create_dc_database(
    db_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> Path:
    """Create a new DC database by running all DC migrations."""
    if db_path is None:
        db_path = DEFAULT_DC_DB_PATH

    db_path = Path(db_path)
    applied = run_dc_migrations(db_path, migrations_dir)

    if applied:
        print(f"Applied {len(applied)} DC migration(s): {', '.join(applied)}")
    else:
        print("DC database is up to date (no pending migrations).")

    return db_path


def normalize_pair(compound_a_id: int, compound_b_id: int) -> tuple[int, int]:
    """Ensure symmetric pair ordering: compound_a_id < compound_b_id."""
    if compound_a_id == compound_b_id:
        raise ValueError("Drug pair cannot have the same compound for both A and B")
    return (min(compound_a_id, compound_b_id), max(compound_a_id, compound_b_id))


def refresh_all_drug_pairs(conn) -> int:
    """Refresh drug_drug_pairs and triples from dc_synergy_results.

    Deletes all existing pairs, triples, and split assignments,
    then re-aggregates from dc_synergy_results computing:
    - num_cell_lines, num_sources, num_measurements
    - median_zip, median_bliss (AVG as SQLite proxy)
    - antagonism_fraction, synergy_fraction
    - consensus_class (majority vote)
    - best_confidence tier
    - target overlap (from drug_targets table)
    - compound_a_degree, compound_b_degree
    """
    conn.execute("DELETE FROM dc_split_assignments")
    conn.execute("DELETE FROM drug_drug_cell_line_triples")
    conn.execute("DELETE FROM drug_drug_pairs")

    # Step 1: Aggregate dc_synergy_results → drug_drug_pairs
    conn.execute(
        """INSERT INTO drug_drug_pairs
        (compound_a_id, compound_b_id, num_cell_lines, num_sources,
         num_measurements, median_zip, median_bliss,
         antagonism_fraction, synergy_fraction, consensus_class,
         best_confidence)
        SELECT
            compound_a_id,
            compound_b_id,
            COUNT(DISTINCT cell_line_id) AS num_cell_lines,
            COUNT(DISTINCT source_db) AS num_sources,
            COUNT(*) AS num_measurements,
            AVG(zip_score) AS median_zip,
            AVG(bliss_score) AS median_bliss,
            -- antagonism_fraction: fraction of results that are antagonistic
            CAST(SUM(CASE WHEN synergy_class IN ('antagonistic', 'strongly_antagonistic')
                          THEN 1 ELSE 0 END) AS REAL) / COUNT(*),
            -- synergy_fraction: fraction of results that are synergistic
            CAST(SUM(CASE WHEN synergy_class IN ('synergistic', 'strongly_synergistic')
                          THEN 1 ELSE 0 END) AS REAL) / COUNT(*),
            -- consensus_class: majority vote, context_dependent if mixed
            CASE
                WHEN CAST(SUM(CASE WHEN synergy_class IN ('antagonistic', 'strongly_antagonistic')
                                   THEN 1 ELSE 0 END) AS REAL) / COUNT(*) > 0.5
                     THEN 'antagonistic'
                WHEN CAST(SUM(CASE WHEN synergy_class IN ('synergistic', 'strongly_synergistic')
                                   THEN 1 ELSE 0 END) AS REAL) / COUNT(*) > 0.5
                     THEN 'synergistic'
                WHEN CAST(SUM(CASE WHEN synergy_class = 'additive'
                                   THEN 1 ELSE 0 END) AS REAL) / COUNT(*) > 0.5
                     THEN 'additive'
                ELSE 'context_dependent'
            END,
            -- best_confidence: gold(1) > silver(2) > bronze(3) > copper(4)
            CASE MIN(CASE confidence_tier
                WHEN 'gold' THEN 1 WHEN 'silver' THEN 2
                WHEN 'bronze' THEN 3 WHEN 'copper' THEN 4 END)
                WHEN 1 THEN 'gold' WHEN 2 THEN 'silver'
                WHEN 3 THEN 'bronze' WHEN 4 THEN 'copper' END
        FROM dc_synergy_results
        GROUP BY compound_a_id, compound_b_id"""
    )

    # Step 2: Compute target overlap (from drug_targets table)
    conn.execute(
        """UPDATE drug_drug_pairs SET
            num_shared_targets = COALESCE((
                SELECT COUNT(DISTINCT dt_a.gene_symbol)
                FROM drug_targets dt_a
                INNER JOIN drug_targets dt_b
                    ON dt_a.gene_symbol = dt_b.gene_symbol
                WHERE dt_a.compound_id = drug_drug_pairs.compound_a_id
                  AND dt_b.compound_id = drug_drug_pairs.compound_b_id
            ), 0),
            target_jaccard = COALESCE((
                SELECT CAST(COUNT(DISTINCT shared.gene_symbol) AS REAL) /
                       NULLIF(COUNT(DISTINCT all_t.gene_symbol), 0)
                FROM (
                    SELECT gene_symbol FROM drug_targets
                    WHERE compound_id = drug_drug_pairs.compound_a_id
                    INTERSECT
                    SELECT gene_symbol FROM drug_targets
                    WHERE compound_id = drug_drug_pairs.compound_b_id
                ) shared,
                (
                    SELECT gene_symbol FROM drug_targets
                    WHERE compound_id = drug_drug_pairs.compound_a_id
                    UNION
                    SELECT gene_symbol FROM drug_targets
                    WHERE compound_id = drug_drug_pairs.compound_b_id
                ) all_t
            ), 0.0)"""
    )

    # Step 3: Populate drug_drug_cell_line_triples
    conn.execute(
        """INSERT INTO drug_drug_cell_line_triples
        (pair_id, cell_line_id, best_zip, best_bliss,
         num_measurements, synergy_class, confidence_tier)
        SELECT
            ddp.pair_id,
            sr.cell_line_id,
            AVG(sr.zip_score),
            AVG(sr.bliss_score),
            COUNT(*),
            -- synergy_class: based on avg ZIP for this triple
            CASE
                WHEN AVG(sr.zip_score) > 10 THEN 'strongly_synergistic'
                WHEN AVG(sr.zip_score) > 5 THEN 'synergistic'
                WHEN AVG(sr.zip_score) >= -5 THEN 'additive'
                WHEN AVG(sr.zip_score) >= -10 THEN 'antagonistic'
                WHEN AVG(sr.zip_score) < -10 THEN 'strongly_antagonistic'
                ELSE NULL
            END,
            -- best confidence for this triple
            CASE MIN(CASE sr.confidence_tier
                WHEN 'gold' THEN 1 WHEN 'silver' THEN 2
                WHEN 'bronze' THEN 3 WHEN 'copper' THEN 4 END)
                WHEN 1 THEN 'gold' WHEN 2 THEN 'silver'
                WHEN 3 THEN 'bronze' WHEN 4 THEN 'copper' END
        FROM dc_synergy_results sr
        INNER JOIN drug_drug_pairs ddp
            ON sr.compound_a_id = ddp.compound_a_id
           AND sr.compound_b_id = ddp.compound_b_id
        GROUP BY ddp.pair_id, sr.cell_line_id"""
    )

    # Step 4: Compute compound_a_degree (number of unique partners for each compound)
    conn.execute("DROP TABLE IF EXISTS _adeg")
    conn.execute(
        """CREATE TEMP TABLE _adeg (
            compound_id INTEGER PRIMARY KEY, deg INTEGER)"""
    )
    # A compound's degree = number of unique partners (as compound_a OR compound_b)
    conn.execute(
        """INSERT INTO _adeg
        SELECT compound_id, COUNT(*) AS deg FROM (
            SELECT compound_a_id AS compound_id, compound_b_id AS partner
            FROM drug_drug_pairs
            UNION ALL
            SELECT compound_b_id AS compound_id, compound_a_id AS partner
            FROM drug_drug_pairs
        ) GROUP BY compound_id"""
    )
    conn.execute(
        """UPDATE drug_drug_pairs SET compound_a_degree = (
            SELECT deg FROM _adeg d
            WHERE d.compound_id = drug_drug_pairs.compound_a_id
        )"""
    )
    conn.execute(
        """UPDATE drug_drug_pairs SET compound_b_degree = (
            SELECT deg FROM _adeg d
            WHERE d.compound_id = drug_drug_pairs.compound_b_id
        )"""
    )
    conn.execute("DROP TABLE _adeg")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM drug_drug_pairs").fetchone()[0]
    return count
