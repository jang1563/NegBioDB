"""Tests for hu.MAP 3.0 negative PPI ETL."""

from pathlib import Path

import pytest

from negbiodb_ppi.etl_humap import parse_humap_pair_line, run_humap_etl
from negbiodb_ppi.ppi_db import get_connection, run_ppi_migrations

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_ppi"


@pytest.fixture
def ppi_db(tmp_path):
    db_path = tmp_path / "test_ppi.db"
    run_ppi_migrations(db_path, MIGRATIONS_DIR)
    return db_path


class TestParseHumapPairLine:
    def test_valid(self):
        assert parse_humap_pair_line("Q9UHC1\tP12345") == ("P12345", "Q9UHC1")

    def test_already_ordered(self):
        assert parse_humap_pair_line("P12345\tQ9UHC1") == ("P12345", "Q9UHC1")

    def test_invalid_three_columns(self):
        assert parse_humap_pair_line("A\tB\tC") is None

    def test_invalid_one_column(self):
        assert parse_humap_pair_line("P12345") is None

    def test_empty(self):
        assert parse_humap_pair_line("") is None

    def test_comment(self):
        assert parse_humap_pair_line("# comment") is None

    def test_whitespace(self):
        assert parse_humap_pair_line("  P12345 \t Q9UHC1  ") == ("P12345", "Q9UHC1")

    def test_self_interaction(self):
        assert parse_humap_pair_line("P12345\tP12345") is None

    def test_invalid_accession(self):
        assert parse_humap_pair_line("invalid\tP12345") is None


class TestRunHumapEtl:
    @pytest.fixture
    def humap_data_dir(self, tmp_path):
        data_dir = tmp_path / "humap"
        data_dir.mkdir()

        # Mock neg_train
        neg_train = data_dir / "neg_train.txt"
        neg_train.write_text("P00001\tP00002\nP00003\tP00004\n")

        # Mock neg_test
        neg_test = data_dir / "neg_test.txt"
        neg_test.write_text("P00005\tP00006\ninvalid\tP00007\n")

        return data_dir

    def test_basic_etl(self, ppi_db, humap_data_dir):
        stats = run_humap_etl(
            db_path=ppi_db,
            data_dir=humap_data_dir,
            neg_files=["neg_train.txt", "neg_test.txt"],
        )

        assert stats["lines_total"] == 4
        assert stats["lines_parsed"] == 3  # 1 invalid skipped
        assert stats["lines_skipped"] == 1
        assert stats["pairs_inserted"] == 3

    def test_all_silver_tier(self, ppi_db, humap_data_dir):
        run_humap_etl(
            db_path=ppi_db,
            data_dir=humap_data_dir,
            neg_files=["neg_train.txt", "neg_test.txt"],
        )

        conn = get_connection(ppi_db)
        try:
            tiers = conn.execute(
                "SELECT DISTINCT confidence_tier FROM ppi_negative_results"
            ).fetchall()
            assert tiers == [("silver",)]

            evidence = conn.execute(
                "SELECT DISTINCT evidence_type FROM ppi_negative_results"
            ).fetchall()
            assert evidence == [("ml_predicted_negative",)]
        finally:
            conn.close()

    def test_canonical_ordering(self, ppi_db, humap_data_dir):
        run_humap_etl(
            db_path=ppi_db,
            data_dir=humap_data_dir,
            neg_files=["neg_train.txt"],
        )

        conn = get_connection(ppi_db)
        try:
            rows = conn.execute(
                "SELECT protein1_id, protein2_id FROM ppi_negative_results"
            ).fetchall()
            for p1, p2 in rows:
                assert p1 < p2
        finally:
            conn.close()

    def test_proteins_inserted(self, ppi_db, humap_data_dir):
        run_humap_etl(
            db_path=ppi_db,
            data_dir=humap_data_dir,
            neg_files=["neg_train.txt", "neg_test.txt"],
        )

        conn = get_connection(ppi_db)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM proteins"
            ).fetchone()[0]
            assert count == 6  # P00001-P00006 (P00007 has invalid partner)
        finally:
            conn.close()

    def test_experiment_record(self, ppi_db, humap_data_dir):
        run_humap_etl(
            db_path=ppi_db,
            data_dir=humap_data_dir,
            neg_files=["neg_train.txt"],
        )

        conn = get_connection(ppi_db)
        try:
            exp = conn.execute(
                "SELECT source_db FROM ppi_experiments WHERE source_db = 'humap'"
            ).fetchone()
            assert exp is not None
        finally:
            conn.close()

    def test_dataset_version(self, ppi_db, humap_data_dir):
        run_humap_etl(
            db_path=ppi_db,
            data_dir=humap_data_dir,
            neg_files=["neg_train.txt"],
        )

        conn = get_connection(ppi_db)
        try:
            dv = conn.execute(
                "SELECT name, version FROM dataset_versions WHERE name = 'humap'"
            ).fetchone()
            assert dv[0] == "humap"
            assert dv[1] == "3.0"
        finally:
            conn.close()
