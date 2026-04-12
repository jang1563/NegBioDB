"""Tests for MD database migrations and helpers."""

from pathlib import Path
import pytest
from negbiodb_md.md_db import (
    create_md_database, get_md_connection, run_md_migrations,
    assign_tier, refresh_all_pairs,
)
from negbiodb.db import get_connection

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_md"


def test_run_md_migrations_creates_core_tables(tmp_path):
    db_path = tmp_path / "md.db"
    applied = run_md_migrations(db_path, _MIGRATIONS_DIR)
    assert applied == ["001"]

    conn = get_connection(db_path)
    try:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert "md_metabolites" in tables
    assert "md_diseases" in tables
    assert "md_studies" in tables
    assert "md_biomarker_results" in tables
    assert "md_metabolite_disease_pairs" in tables
    assert "md_split_definitions" in tables
    assert "md_split_assignments" in tables
    assert "schema_migrations" in tables
    assert "dataset_versions" in tables


def test_run_md_migrations_is_idempotent(tmp_path):
    db_path = tmp_path / "md.db"
    run_md_migrations(db_path, _MIGRATIONS_DIR)
    assert run_md_migrations(db_path, _MIGRATIONS_DIR) == []


def test_create_md_database_returns_path(tmp_path):
    db_path = tmp_path / "md.db"
    created = create_md_database(db_path, _MIGRATIONS_DIR)
    assert created == db_path
    assert Path(created).exists()


def test_get_md_connection_creates_db(tmp_path):
    db_path = tmp_path / "md_auto.db"
    conn = get_md_connection(db_path)
    assert conn is not None
    # Should be able to query
    row = conn.execute("SELECT COUNT(*) FROM md_metabolites").fetchone()
    assert row[0] == 0
    conn.close()


# ── assign_tier tests ────────────────────────────────────────────────────────

def test_assign_tier_gold():
    # FDR > 0.1, n >= 50 → gold
    assert assign_tier(p_value=0.3, fdr=0.25, n_group=60) == "gold"


def test_assign_tier_silver():
    # p > 0.05, n >= 20 → silver
    assert assign_tier(p_value=0.15, fdr=None, n_group=25) == "silver"


def test_assign_tier_bronze():
    # p > 0.05, n < 20 → bronze
    assert assign_tier(p_value=0.12, fdr=None, n_group=10) == "bronze"


def test_assign_tier_bronze_unknown_n():
    # p > 0.05, n unknown → bronze
    assert assign_tier(p_value=0.2, fdr=None, n_group=None) == "bronze"


def test_assign_tier_copper():
    # No statistics → copper
    assert assign_tier(p_value=None, fdr=None, n_group=None) == "copper"


def test_assign_tier_fdr_boundary():
    # FDR exactly at threshold (0.1 is NOT > 0.1 → silver, not gold)
    assert assign_tier(p_value=0.2, fdr=0.1, n_group=60) == "silver"


# ── refresh_all_pairs tests ──────────────────────────────────────────────────

def _seed_db(conn):
    """Insert minimal data for pair aggregation testing."""
    conn.execute(
        "INSERT INTO md_metabolites (name, inchikey) VALUES (?,?)",
        ("glucose", "WQZGKKKJIJFFOK-GASJEMHNSA-N"),
    )
    conn.execute(
        "INSERT INTO md_diseases (name, disease_category) VALUES (?,?)",
        ("type 2 diabetes mellitus", "metabolic"),
    )
    conn.execute(
        "INSERT INTO md_studies (source, external_id, platform, biofluid, comparison) VALUES (?,?,?,?,?)",
        ("metabolights", "MTBLS1", "lc_ms", "blood", "disease_vs_healthy"),
    )
    conn.execute(
        """INSERT INTO md_biomarker_results
           (metabolite_id, disease_id, study_id, p_value, fdr, is_significant, tier)
           VALUES (1,1,1, 0.3, 0.35, 0, 'silver')"""
    )
    conn.commit()


def test_refresh_all_pairs_creates_pair(tmp_path):
    db_path = tmp_path / "md.db"
    conn = get_md_connection(db_path)
    _seed_db(conn)
    n = refresh_all_pairs(conn)
    assert n == 1

    pair = conn.execute("SELECT * FROM md_metabolite_disease_pairs").fetchone()
    assert pair is not None
    conn.close()


def test_refresh_all_pairs_consensus_negative(tmp_path):
    db_path = tmp_path / "md.db"
    conn = get_md_connection(db_path)
    _seed_db(conn)
    refresh_all_pairs(conn)

    row = conn.execute(
        "SELECT consensus, best_tier FROM md_metabolite_disease_pairs"
    ).fetchone()
    assert row[0] == "negative"
    assert row[1] == "silver"
    conn.close()


def test_refresh_all_pairs_mixed_consensus(tmp_path):
    db_path = tmp_path / "md.db"
    conn = get_md_connection(db_path)
    _seed_db(conn)

    # Add a second study with significant result for same pair
    conn.execute(
        "INSERT INTO md_studies (source, external_id, platform, biofluid, comparison) VALUES (?,?,?,?,?)",
        ("nmdr", "ST000001", "nmr", "urine", "disease_vs_healthy"),
    )
    conn.execute(
        """INSERT INTO md_biomarker_results
           (metabolite_id, disease_id, study_id, p_value, fdr, is_significant)
           VALUES (1,1,2, 0.01, 0.03, 1)"""
    )
    conn.commit()
    refresh_all_pairs(conn)

    row = conn.execute(
        "SELECT consensus, n_studies_negative, n_studies_positive FROM md_metabolite_disease_pairs"
    ).fetchone()
    assert row[0] == "mixed"
    assert row[1] == 1
    assert row[2] == 1
    conn.close()
