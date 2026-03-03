"""Tests for BindingDB ETL pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from negbiodb.db import connect, create_database
from negbiodb.etl_bindingdb import (
    _extract_inactive_rows_from_chunk,
    _parse_relation_value,
    run_bindingdb_etl,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


@pytest.fixture
def migrated_db(tmp_path):
    db_path = tmp_path / "test.db"
    create_database(db_path, MIGRATIONS_DIR)
    return db_path


class TestBindingDBHelpers:

    def test_parse_relation_value(self):
        assert _parse_relation_value(">10000") == (">", 10000.0)
        assert _parse_relation_value("<=500") == ("<=", 500.0)
        assert _parse_relation_value("12345") == ("=", 12345.0)
        assert _parse_relation_value(None) == ("=", None)

    def test_extract_inactive_rows_human_only(self):
        chunk = pd.DataFrame(
            {
                "Ligand SMILES": ["c1ccccc1", "c1ccccc1", "CCO", "CCN"],
                "UniProt (SwissProt) Primary ID of Target Chain": [
                    "P00533",
                    "P00533",
                    "P12345",
                    "P12345",
                ],
                "Target Source Organism According to Curator or DataSource": [
                    "Homo sapiens",
                    "Homo sapiens",
                    "Mus musculus",
                    "Homo sapiens",
                ],
                "Ki (nM)": [">10000", "500", ">15000", None],
                "Kd (nM)": [None, None, None, ">12000"],
                "BindingDB Reactant_set_id": [1, 2, 3, 4],
                "Publication Year": [2010, 2011, 2012, 2013],
            }
        )

        rows = _extract_inactive_rows_from_chunk(
            chunk, inactivity_threshold_nm=10000, human_only=True
        )
        assert len(rows) == 2
        assert {r["activity_type"] for r in rows} == {"Ki", "Kd"}
        assert all(r["species_tested"] == "Homo sapiens" for r in rows)

    def test_extract_inactive_rows_with_non_human_enabled(self):
        chunk = pd.DataFrame(
            {
                "Ligand SMILES": ["CCO"],
                "UniProt (SwissProt) Primary ID of Target Chain": ["P12345"],
                "Target Source Organism According to Curator or DataSource": [
                    "Mus musculus"
                ],
                "Ki (nM)": [">15000"],
                "BindingDB Reactant_set_id": [3],
                "Publication Year": [2012],
            }
        )

        rows = _extract_inactive_rows_from_chunk(
            chunk, inactivity_threshold_nm=10000, human_only=False
        )
        assert len(rows) == 1
        assert rows[0]["species_tested"] == "Mus musculus"

    def test_extract_inactive_rows_requires_organism_when_human_only(self):
        chunk = pd.DataFrame(
            {
                "Ligand SMILES": ["c1ccccc1"],
                "UniProt (SwissProt) Primary ID of Target Chain": ["P00533"],
                "Ki (nM)": [">10000"],
            }
        )
        rows = _extract_inactive_rows_from_chunk(
            chunk, inactivity_threshold_nm=10000, human_only=True
        )
        assert rows == []


class TestRunBindingDBETL:

    def test_run_bindingdb_etl_small_dataset(self, migrated_db, tmp_path):
        tsv_path = tmp_path / "BindingDB_All.tsv"
        pd.DataFrame(
            {
                "Ligand SMILES": ["c1ccccc1", "c1ccccc1", "CCO", "CCN"],
                "UniProt (SwissProt) Primary ID of Target Chain": [
                    "P00533",
                    "P00533",
                    "P12345",
                    "P12345",
                ],
                "Target Source Organism According to Curator or DataSource": [
                    "Homo sapiens",
                    "Homo sapiens",
                    "Homo sapiens",
                    "Homo sapiens",
                ],
                "Ki (nM)": [">10000", ">20000", None, "500"],
                "IC50 (nM)": [None, None, "15000", "500"],
                "BindingDB Reactant_set_id": [1, 2, 3, 4],
                "Publication Year": [2010, 2011, 2020, 2021],
            }
        ).to_csv(tsv_path, sep="\t", index=False)

        stats = run_bindingdb_etl(
            db_path=migrated_db,
            bindingdb_tsv_path=tsv_path,
            chunksize=2,
        )

        assert stats["rows_read"] == 4
        assert stats["rows_filtered_inactive"] == 3
        assert stats["results_inserted"] == 3

        with connect(migrated_db) as conn:
            n_results = conn.execute(
                "SELECT COUNT(*) FROM negative_results WHERE source_db='bindingdb'"
            ).fetchone()[0]
            assert n_results == 3

            n_pairs = conn.execute(
                "SELECT COUNT(*) FROM compound_target_pairs"
            ).fetchone()[0]
            assert n_pairs == 2

            species = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT species_tested FROM negative_results WHERE source_db='bindingdb'"
                ).fetchall()
            }
            assert species == {"Homo sapiens"}

    def test_run_bindingdb_etl_respects_threshold_and_human_toggle(
        self, migrated_db, tmp_path, monkeypatch
    ):
        import negbiodb.etl_bindingdb as mod

        tsv_path = tmp_path / "BindingDB_All.tsv"
        pd.DataFrame(
            {
                "Ligand SMILES": ["c1ccccc1", "CCO"],
                "UniProt (SwissProt) Primary ID of Target Chain": ["P00533", "P12345"],
                "Target Source Organism According to Curator or DataSource": [
                    "Homo sapiens",
                    "Mus musculus",
                ],
                "Ki (nM)": ["15000", "25000"],
                "BindingDB Reactant_set_id": [1, 2],
                "Publication Year": [2010, 2011],
            }
        ).to_csv(tsv_path, sep="\t", index=False)

        monkeypatch.setattr(
            mod,
            "load_config",
            lambda: {
                "inactivity_threshold_nm": 10000,
                "downloads": {"bindingdb": {"dest_dir": "unused"}},
                "bindingdb_etl": {
                    "chunksize": 100000,
                    "inactive_threshold_nm": 20000,
                    "human_only": False,
                },
            },
        )

        stats = run_bindingdb_etl(
            db_path=migrated_db,
            bindingdb_tsv_path=tsv_path,
            chunksize=10,
        )
        assert stats["results_inserted"] == 1

        with connect(migrated_db) as conn:
            row = conn.execute(
                "SELECT inactivity_threshold, species_tested FROM negative_results "
                "WHERE source_db='bindingdb'"
            ).fetchone()
            assert row == (20000.0, "Mus musculus")
