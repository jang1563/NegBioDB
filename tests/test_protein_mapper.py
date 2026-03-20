"""Tests for PPI protein mapper: UniProt validation & canonical pair ordering."""

from pathlib import Path

import pytest

from negbiodb_ppi.ppi_db import get_connection, run_ppi_migrations
from negbiodb_ppi.protein_mapper import (
    canonical_pair,
    get_or_insert_protein,
    validate_uniprot,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_ppi"


@pytest.fixture
def ppi_db(tmp_path):
    """Create a temporary PPI database with all migrations applied."""
    db_path = tmp_path / "test_ppi.db"
    run_ppi_migrations(db_path, MIGRATIONS_DIR)
    return db_path


class TestValidateUniprot:
    """Test UniProt accession validation."""

    def test_valid_swissprot_p(self):
        assert validate_uniprot("P12345") == "P12345"

    def test_valid_swissprot_q(self):
        assert validate_uniprot("Q9UHC1") == "Q9UHC1"

    def test_valid_swissprot_o(self):
        assert validate_uniprot("O15553") == "O15553"

    def test_valid_trembl_short(self):
        assert validate_uniprot("A0A024") == "A0A024"

    def test_valid_trembl_long(self):
        assert validate_uniprot("A0A0K9P0T2") == "A0A0K9P0T2"

    def test_isoform_stripped(self):
        assert validate_uniprot("P12345-2") == "P12345"

    def test_isoform_long_stripped(self):
        assert validate_uniprot("Q9UHC1-14") == "Q9UHC1"

    def test_whitespace_stripped(self):
        assert validate_uniprot("  P12345  ") == "P12345"

    def test_invalid_numeric_only(self):
        assert validate_uniprot("12345") is None

    def test_invalid_too_short(self):
        assert validate_uniprot("P1") is None

    def test_invalid_all_alpha(self):
        assert validate_uniprot("ABCDEF") is None

    def test_invalid_empty(self):
        assert validate_uniprot("") is None

    def test_invalid_none(self):
        assert validate_uniprot(None) is None

    def test_invalid_lowercase(self):
        assert validate_uniprot("p12345") is None

    def test_invalid_special_chars(self):
        assert validate_uniprot("P123!5") is None


class TestCanonicalPair:
    """Test canonical pair ordering."""

    def test_already_ordered(self):
        assert canonical_pair("A00001", "B00002") == ("A00001", "B00002")

    def test_reversed(self):
        assert canonical_pair("B00002", "A00001") == ("A00001", "B00002")

    def test_same_accession(self):
        assert canonical_pair("P12345", "P12345") == ("P12345", "P12345")

    def test_real_accessions(self):
        assert canonical_pair("Q9UHC1", "P12345") == ("P12345", "Q9UHC1")


class TestGetOrInsertProtein:
    """Test protein insertion/retrieval."""

    def test_insert_new_protein(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            pid = get_or_insert_protein(conn, "P12345")
            conn.commit()
            assert pid == 1

            row = conn.execute(
                "SELECT uniprot_accession, organism FROM proteins "
                "WHERE protein_id = ?",
                (pid,),
            ).fetchone()
            assert row[0] == "P12345"
            assert row[1] == "Homo sapiens"
        finally:
            conn.close()

    def test_return_existing_protein(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            pid1 = get_or_insert_protein(conn, "P12345")
            conn.commit()
            pid2 = get_or_insert_protein(conn, "P12345")
            assert pid1 == pid2

            count = conn.execute(
                "SELECT COUNT(*) FROM proteins"
            ).fetchone()[0]
            assert count == 1
        finally:
            conn.close()

    def test_insert_with_gene_symbol(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            pid = get_or_insert_protein(conn, "Q9UHC1", gene_symbol="BRCA1")
            conn.commit()

            row = conn.execute(
                "SELECT gene_symbol FROM proteins WHERE protein_id = ?",
                (pid,),
            ).fetchone()
            assert row[0] == "BRCA1"
        finally:
            conn.close()

    def test_insert_with_sequence(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            seq = "MTEYKLVVV"
            pid = get_or_insert_protein(conn, "P01116", sequence=seq)
            conn.commit()

            row = conn.execute(
                "SELECT amino_acid_sequence, sequence_length FROM proteins "
                "WHERE protein_id = ?",
                (pid,),
            ).fetchone()
            assert row[0] == seq
            assert row[1] == len(seq)
        finally:
            conn.close()

    def test_insert_without_sequence_null_length(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            pid = get_or_insert_protein(conn, "P12345")
            conn.commit()

            row = conn.execute(
                "SELECT sequence_length FROM proteins WHERE protein_id = ?",
                (pid,),
            ).fetchone()
            assert row[0] is None
        finally:
            conn.close()

    def test_multiple_proteins(self, ppi_db):
        conn = get_connection(ppi_db)
        try:
            pid1 = get_or_insert_protein(conn, "P12345")
            pid2 = get_or_insert_protein(conn, "Q9UHC1")
            conn.commit()
            assert pid1 != pid2

            count = conn.execute(
                "SELECT COUNT(*) FROM proteins"
            ).fetchone()[0]
            assert count == 2
        finally:
            conn.close()
