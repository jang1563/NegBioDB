"""Tests for scripts_ppi/fetch_sequences.py — UniProt sequence fetching."""

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_ppi"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fetch_sequences import fetch_uniprot_batch, update_protein_sequences


@pytest.fixture
def ppi_db(tmp_path):
    """Create a minimal PPI database with 3 proteins (no sequences)."""
    db_path = tmp_path / "test_ppi.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE proteins (
            protein_id INTEGER PRIMARY KEY AUTOINCREMENT,
            uniprot_accession TEXT NOT NULL UNIQUE,
            uniprot_entry_name TEXT,
            gene_symbol TEXT,
            amino_acid_sequence TEXT,
            sequence_length INTEGER,
            organism TEXT DEFAULT 'Homo sapiens',
            taxonomy_id INTEGER DEFAULT 9606,
            subcellular_location TEXT,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )"""
    )
    conn.executemany(
        "INSERT INTO proteins (uniprot_accession) VALUES (?)",
        [("P12345",), ("Q9UHC1",), ("P99999",)],
    )
    conn.commit()
    conn.close()
    return db_path


def _mock_uniprot_response():
    """Build a mock UniProt JSON response."""
    return {
        "results": [
            {
                "primaryAccession": "P12345",
                "sequence": {"value": "MKTAYIAKQRQISFVKSHFSRQ"},
                "genes": [{"geneName": {"value": "BRCA1"}}],
                "comments": [
                    {
                        "commentType": "SUBCELLULAR LOCATION",
                        "subcellularLocations": [
                            {"location": {"value": "Nucleus"}}
                        ],
                    }
                ],
            },
            {
                "primaryAccession": "Q9UHC1",
                "sequence": {"value": "MAAPWRRGARL"},
                "genes": [{"geneName": {"value": "MLH1"}}],
                "comments": [],
            },
            # P99999 deliberately missing — simulates 404/obsolete
        ]
    }


class TestFetchUniprotBatch:
    @patch("fetch_sequences.requests.get")
    def test_basic_fetch(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _mock_uniprot_response()
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_uniprot_batch(["P12345", "Q9UHC1", "P99999"])

        assert "P12345" in result
        assert result["P12345"]["sequence"] == "MKTAYIAKQRQISFVKSHFSRQ"
        assert result["P12345"]["gene_symbol"] == "BRCA1"
        assert result["P12345"]["subcellular_location"] == "Nucleus"

        assert "Q9UHC1" in result
        assert result["Q9UHC1"]["gene_symbol"] == "MLH1"
        assert result["Q9UHC1"]["subcellular_location"] is None

        # P99999 not in response → not in result
        assert "P99999" not in result

    def test_empty_list(self):
        result = fetch_uniprot_batch([])
        assert result == {}

    @patch("fetch_sequences.requests.get")
    def test_retry_on_failure(self, mock_get):
        import requests as req

        mock_fail = MagicMock()
        mock_fail.raise_for_status.side_effect = req.HTTPError("503")

        mock_ok = MagicMock()
        mock_ok.json.return_value = {"results": []}
        mock_ok.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_fail, mock_ok]

        with patch("fetch_sequences.RETRY_BACKOFF", 0.01):
            result = fetch_uniprot_batch(["P12345"])
        assert result == {}
        assert mock_get.call_count == 2


class TestUpdateProteinSequences:
    @patch("fetch_sequences.fetch_uniprot_batch")
    def test_update_sequences(self, mock_fetch, ppi_db):
        mock_fetch.return_value = {
            "P12345": {
                "sequence": "MKTAYIAKQRQISFVKSHFSRQ",
                "gene_symbol": "BRCA1",
                "subcellular_location": "Nucleus",
            },
            "Q9UHC1": {
                "sequence": "MAAPWRRGARL",
                "gene_symbol": "MLH1",
                "subcellular_location": None,
            },
            # P99999 missing → fails
        }

        summary = update_protein_sequences(ppi_db, batch_size=500, delay=0)

        assert summary["total"] == 3
        assert summary["fetched"] == 2
        assert summary["failed"] == 1

        conn = sqlite3.connect(str(ppi_db))
        rows = conn.execute(
            "SELECT uniprot_accession, amino_acid_sequence, sequence_length, gene_symbol "
            "FROM proteins ORDER BY uniprot_accession"
        ).fetchall()
        conn.close()

        # P12345
        assert rows[0][1] == "MKTAYIAKQRQISFVKSHFSRQ"
        assert rows[0][2] == 22
        assert rows[0][3] == "BRCA1"

        # P99999 still NULL
        assert rows[1][1] is None

        # Q9UHC1
        assert rows[2][1] == "MAAPWRRGARL"
        assert rows[2][2] == 11

    @patch("fetch_sequences.fetch_uniprot_batch")
    def test_checkpoint_resume(self, mock_fetch, ppi_db, tmp_path):
        checkpoint = tmp_path / "checkpoint.json"

        # First run: fetch P12345 only (batch_size=1, limit 1 batch)
        mock_fetch.return_value = {
            "P12345": {
                "sequence": "MKTA",
                "gene_symbol": None,
                "subcellular_location": None,
            },
        }
        update_protein_sequences(
            ppi_db, batch_size=1, delay=0, checkpoint_path=checkpoint
        )
        assert checkpoint.exists()

        with open(checkpoint) as f:
            ckpt = json.load(f)
        # At least P12345 should be in completed
        assert "P12345" in ckpt["completed"]

    @patch("fetch_sequences.fetch_uniprot_batch")
    def test_preserves_existing_gene_symbol(self, mock_fetch, ppi_db):
        """COALESCE should not overwrite existing gene_symbol."""
        conn = sqlite3.connect(str(ppi_db))
        conn.execute(
            "UPDATE proteins SET gene_symbol = 'EXISTING' WHERE uniprot_accession = 'P12345'"
        )
        conn.commit()
        conn.close()

        mock_fetch.return_value = {
            "P12345": {
                "sequence": "MKTA",
                "gene_symbol": "NEW_GENE",
                "subcellular_location": None,
            },
            "Q9UHC1": {
                "sequence": "MAAP",
                "gene_symbol": "MLH1",
                "subcellular_location": None,
            },
            "P99999": {
                "sequence": "ABCD",
                "gene_symbol": None,
                "subcellular_location": None,
            },
        }

        update_protein_sequences(ppi_db, batch_size=500, delay=0)

        conn = sqlite3.connect(str(ppi_db))
        row = conn.execute(
            "SELECT gene_symbol FROM proteins WHERE uniprot_accession = 'P12345'"
        ).fetchone()
        conn.close()

        # COALESCE keeps existing value
        assert row[0] == "EXISTING"
