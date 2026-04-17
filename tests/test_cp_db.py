"""Tests for CP database migrations and helpers."""

from pathlib import Path

from negbiodb_cp.cp_db import create_cp_database, get_connection, run_cp_migrations

from tests.cp_test_utils import MIGRATIONS_DIR


def test_run_cp_migrations_creates_core_tables(tmp_path):
    db_path = tmp_path / "cp.db"
    applied = run_cp_migrations(db_path, MIGRATIONS_DIR)
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

    assert "cp_perturbation_results" in tables
    assert "cp_split_assignments" in tables
    assert "cp_profile_features" in tables


def test_run_cp_migrations_is_idempotent(tmp_path):
    db_path = tmp_path / "cp.db"
    run_cp_migrations(db_path, MIGRATIONS_DIR)
    assert run_cp_migrations(db_path, MIGRATIONS_DIR) == []


def test_create_cp_database_returns_target_path(tmp_path):
    db_path = tmp_path / "cp.db"
    created = create_cp_database(db_path, MIGRATIONS_DIR)
    assert created == db_path
    assert Path(created).exists()
