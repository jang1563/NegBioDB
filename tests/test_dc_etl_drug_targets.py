"""Tests for drug-target ETL module.

Tests DGIdb interactions parsing, drug-target loading, and dedup behavior.
"""

import sqlite3
import textwrap
from pathlib import Path

import pytest

from negbiodb_dc.dc_db import get_connection, run_dc_migrations
from negbiodb_dc.etl_drug_targets import (
    load_drug_targets,
    parse_dgidb_interactions,
    run_drug_targets_etl,
)

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_dc"


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_dc.db"
    run_dc_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def conn(tmp_db):
    c = get_connection(tmp_db)
    yield c
    c.close()


# ── DGIdb parsing ────────────────────────────────────────────────────


class TestParseDgidbInteractions:
    def test_basic_parsing(self, tmp_path):
        tsv_path = tmp_path / "interactions.tsv"
        tsv_path.write_text(textwrap.dedent("""\
            drug_name\tgene_name\tinteraction_types\tsource
            Imatinib\tABL1\tinhibitor\tChEMBL
            Imatinib\tKIT\tinhibitor\tDrugBank
            Gefitinib\tEGFR\tinhibitor\tChEMBL
        """))

        records = parse_dgidb_interactions(tsv_path)
        assert len(records) == 3
        assert records[0]["drug_name"] == "Imatinib"
        assert records[0]["gene_symbol"] == "ABL1"
        assert records[0]["source"] == "dgidb"

    def test_deduplication(self, tmp_path):
        tsv_path = tmp_path / "interactions.tsv"
        tsv_path.write_text(textwrap.dedent("""\
            drug_name\tgene_name\tinteraction_types
            Imatinib\tABL1\tinhibitor
            Imatinib\tABL1\tantagonist
        """))

        records = parse_dgidb_interactions(tsv_path)
        assert len(records) == 1  # Same (drug, gene) → deduped

    def test_nan_values_skipped(self, tmp_path):
        tsv_path = tmp_path / "interactions.tsv"
        tsv_path.write_text(textwrap.dedent("""\
            drug_name\tgene_name
            Imatinib\tABL1
            nan\tEGFR
            Gefitinib\tnan
            \tBRAF
        """))

        records = parse_dgidb_interactions(tsv_path)
        assert len(records) == 1
        assert records[0]["drug_name"] == "Imatinib"

    def test_missing_columns_raises(self, tmp_path):
        tsv_path = tmp_path / "bad.tsv"
        tsv_path.write_text("col_a\tcol_b\n1\t2\n")

        with pytest.raises(ValueError, match="Cannot find"):
            parse_dgidb_interactions(tsv_path)

    def test_empty_file_no_records(self, tmp_path):
        tsv_path = tmp_path / "interactions.tsv"
        tsv_path.write_text("drug_name\tgene_name\n")

        records = parse_dgidb_interactions(tsv_path)
        assert len(records) == 0


# ── Drug-target loading ──────────────────────────────────────────────


class TestLoadDrugTargets:
    def _seed_compounds(self, conn):
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('Imatinib')")
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('Gefitinib')")
        conn.commit()
        return {
            row[1]: row[0]
            for row in conn.execute("SELECT compound_id, drug_name FROM compounds")
        }

    def test_basic_insert(self, conn):
        cache = self._seed_compounds(conn)
        interactions = [
            {"drug_name": "Imatinib", "gene_symbol": "ABL1", "source": "dgidb"},
            {"drug_name": "Imatinib", "gene_symbol": "KIT", "source": "dgidb"},
            {"drug_name": "Gefitinib", "gene_symbol": "EGFR", "source": "dgidb"},
        ]

        stats = load_drug_targets(conn, interactions, cache)
        assert stats["targets_inserted"] == 3
        assert stats["skipped_unknown_drug"] == 0
        assert stats["skipped_duplicate"] == 0

    def test_unknown_drug_skipped(self, conn):
        cache = self._seed_compounds(conn)
        interactions = [
            {"drug_name": "UnknownDrug", "gene_symbol": "ABL1", "source": "dgidb"},
        ]

        stats = load_drug_targets(conn, interactions, cache)
        assert stats["skipped_unknown_drug"] == 1
        assert stats["targets_inserted"] == 0

    def test_duplicate_skipped(self, conn):
        cache = self._seed_compounds(conn)
        interactions = [
            {"drug_name": "Imatinib", "gene_symbol": "ABL1", "source": "dgidb"},
            {"drug_name": "Imatinib", "gene_symbol": "ABL1", "source": "dgidb"},
        ]

        stats = load_drug_targets(conn, interactions, cache)
        assert stats["targets_inserted"] == 1
        assert stats["skipped_duplicate"] == 1

    def test_integrity_error_caught(self, conn):
        """sqlite3.IntegrityError (not generic Exception) is caught for duplicates."""
        cache = self._seed_compounds(conn)
        # Insert first
        conn.execute(
            "INSERT INTO drug_targets (compound_id, gene_symbol, source) VALUES (?, ?, ?)",
            (cache["Imatinib"], "ABL1", "dgidb"),
        )
        conn.commit()

        interactions = [
            {"drug_name": "Imatinib", "gene_symbol": "ABL1", "source": "dgidb"},
        ]
        stats = load_drug_targets(conn, interactions, cache)
        assert stats["skipped_duplicate"] == 1

    def test_different_sources_allowed(self, conn):
        """Same (compound, gene) from different sources are separate entries."""
        cache = self._seed_compounds(conn)
        interactions = [
            {"drug_name": "Imatinib", "gene_symbol": "ABL1", "source": "dgidb"},
            {"drug_name": "Imatinib", "gene_symbol": "ABL1", "source": "chembl"},
        ]

        stats = load_drug_targets(conn, interactions, cache)
        assert stats["targets_inserted"] == 2

    def test_stored_values(self, conn):
        cache = self._seed_compounds(conn)
        interactions = [
            {"drug_name": "Imatinib", "gene_symbol": "ABL1", "source": "dgidb"},
        ]
        load_drug_targets(conn, interactions, cache)

        row = conn.execute(
            "SELECT compound_id, gene_symbol, source FROM drug_targets"
        ).fetchone()
        assert row[0] == cache["Imatinib"]
        assert row[1] == "ABL1"
        assert row[2] == "dgidb"


# ── End-to-end ETL ───────────────────────────────────────────────────


class TestRunDrugTargetsEtl:
    def test_missing_file_raises(self, tmp_path, tmp_db):
        with pytest.raises(FileNotFoundError, match="interactions"):
            run_drug_targets_etl(tmp_db, tmp_path)

    def test_end_to_end(self, tmp_path, tmp_db):
        # Seed compounds first
        conn = get_connection(tmp_db)
        conn.execute("INSERT INTO compounds (drug_name) VALUES ('Imatinib')")
        conn.commit()
        conn.close()

        # Create interactions file
        tsv_path = tmp_path / "interactions.tsv"
        tsv_path.write_text(textwrap.dedent("""\
            drug_name\tgene_name
            Imatinib\tABL1
            Imatinib\tKIT
            UnknownDrug\tEGFR
        """))

        stats = run_drug_targets_etl(tmp_db, tmp_path)
        assert stats["targets_inserted"] == 2
        assert stats["skipped_unknown_drug"] == 1
