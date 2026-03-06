"""Tests for PubChem ETL pipeline."""

import gzip
import sqlite3
from pathlib import Path

import pytest

from negbiodb.db import connect, create_database
from negbiodb.etl_pubchem import (
    build_sid_lookup_db,
    load_aid_to_uniprot_map,
    load_confirmatory_aids,
    load_confirmatory_human_aids,
    run_pubchem_etl,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


@pytest.fixture
def migrated_db(tmp_path):
    db_path = tmp_path / "test.db"
    create_database(db_path, MIGRATIONS_DIR)
    return db_path


def _write_gz_tsv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join("" if v is None else str(v) for v in row) + "\n")
    return path


class TestPubChemHelpers:

    def test_load_confirmatory_aids(self, tmp_path):
        bioassays = _write_gz_tsv(
            tmp_path / "bioassays.tsv.gz",
            ["AID", "Assay Type", "Protein Accession"],
            [
                [1001, "confirmatory", "P00533"],
                [1002, "primary", "P00533"],
            ],
        )
        aids = load_confirmatory_aids(bioassays)
        assert aids == {1001}

    def test_load_confirmatory_aids_prefers_column_with_confirmatory_values(self, tmp_path):
        bioassays = _write_gz_tsv(
            tmp_path / "bioassays.tsv.gz",
            ["AID", "BioAssay Types", "Outcome Type", "Protein Accessions"],
            [
                [1001, None, "Confirmatory", "P00533"],
                [1002, None, "Primary", "P00533"],
            ],
        )
        aids = load_confirmatory_aids(bioassays)
        assert aids == {1001}

    def test_load_confirmatory_human_aids(self, tmp_path):
        bioassays = _write_gz_tsv(
            tmp_path / "bioassays.tsv.gz",
            ["AID", "Outcome Type", "Protein Accessions", "Target TaxIDs"],
            [
                [3001, "Confirmatory", "P00533", "9606;10090"],
                [3002, "Confirmatory", "P12931", "10090"],
                [3003, "Primary", "P99999", "9606"],
            ],
        )
        aids = load_confirmatory_human_aids(bioassays)
        assert aids == {3001}

    def test_load_aid_to_uniprot_map_keeps_first_duplicate(self, tmp_path):
        aid_map = _write_gz_tsv(
            tmp_path / "Aid2GeneidAccessionUniProt.gz",
            ["AID", "UniProt"],
            [
                [1001, "P00533"],
                [1001, "Q9Y6K9"],
                [1002, "P12931"],
            ],
        )
        mapping = load_aid_to_uniprot_map(aid_map)
        assert mapping[1001] == "P00533"
        assert mapping[1002] == "P12931"

    def test_load_aid_to_uniprot_map_parses_pipe_format(self, tmp_path):
        aid_map = _write_gz_tsv(
            tmp_path / "Aid2GeneidAccessionUniProt.gz",
            ["AID", "UniProt"],
            [[1001, "sp|P00533|EGFR_HUMAN"]],
        )
        mapping = load_aid_to_uniprot_map(aid_map)
        assert mapping[1001] == "P00533"

    def test_load_aid_to_uniprot_map_ignores_non_uniprot_tokens(self, tmp_path):
        aid_map = _write_gz_tsv(
            tmp_path / "Aid2GeneidAccessionUniProt.gz",
            ["AID", "UniProt"],
            [
                [1001, "1Y7V_A"],
                [1001, "P00533"],
            ],
        )
        mapping = load_aid_to_uniprot_map(aid_map)
        assert mapping[1001] == "P00533"

    def test_build_sid_lookup_db_from_headerless_file(self, tmp_path):
        sid_map = tmp_path / "Sid2CidSMILES.gz"
        with gzip.open(sid_map, "wt") as f:
            f.write("10\t241\tc1ccccc1\n")
            f.write("11\t242\tCCO\n")

        lookup_db = build_sid_lookup_db(sid_map, tmp_path / "sid_lookup.sqlite")
        conn = sqlite3.connect(str(lookup_db))
        try:
            rows = conn.execute(
                "SELECT sid, cid, smiles FROM sid_cid_map ORDER BY sid"
            ).fetchall()
        finally:
            conn.close()

        assert rows == [(10, 241, "c1ccccc1"), (11, 242, "CCO")]

    def test_build_sid_lookup_db_accepts_isomeric_smiles_header(self, tmp_path):
        sid_map = _write_gz_tsv(
            tmp_path / "Sid2CidSMILES.gz",
            ["SID", "CID", "Isomeric SMILES"],
            [
                [10, 241, "c1ccccc1"],
                [11, 242, "CCO"],
            ],
        )

        lookup_db = build_sid_lookup_db(sid_map, tmp_path / "sid_lookup.sqlite")
        conn = sqlite3.connect(str(lookup_db))
        try:
            rows = conn.execute(
                "SELECT sid, cid, smiles FROM sid_cid_map ORDER BY sid"
            ).fetchall()
        finally:
            conn.close()

        assert rows == [(10, 241, "c1ccccc1"), (11, 242, "CCO")]

    def test_build_sid_lookup_db_rebuilds_on_source_change(self, tmp_path):
        sid_map = tmp_path / "Sid2CidSMILES.gz"
        with gzip.open(sid_map, "wt") as f:
            f.write("10\t241\tc1ccccc1\n")
        lookup_db = tmp_path / "sid_lookup.sqlite"
        build_sid_lookup_db(sid_map, lookup_db)

        with gzip.open(sid_map, "wt") as f:
            f.write("10\t241\tc1ccccc1\n")
            f.write("11\t242\tCCO\n")
        build_sid_lookup_db(sid_map, lookup_db)

        conn = sqlite3.connect(str(lookup_db))
        try:
            rows = conn.execute(
                "SELECT sid, cid, smiles FROM sid_cid_map ORDER BY sid"
            ).fetchall()
        finally:
            conn.close()

        assert rows == [(10, 241, "c1ccccc1"), (11, 242, "CCO")]


class TestRunPubChemETL:

    def test_run_pubchem_etl_small_dataset(self, migrated_db, tmp_path):
        bioactivities = _write_gz_tsv(
            tmp_path / "bioactivities.tsv.gz",
            [
                "AID",
                "SID",
                "CID",
                "Activity Outcome",
                "Activity Name",
                "Activity Value",
                "Activity Unit",
                "Protein Accession",
                "Target TaxID",
            ],
            [
                [1001, 10, None, "Inactive", "IC50", 20000, "nM", "P00533", 9606],
                [1001, 11, None, "Active", "IC50", 25000, "nM", "P00533", 9606],
                [1002, 12, None, "Inactive", "Ki", 15000, "nM", None, 9606],
                [1003, 13, None, "Inactive", "IC50", 30000, "nM", "P99999", 10090],
            ],
        )
        bioassays = _write_gz_tsv(
            tmp_path / "bioassays.tsv.gz",
            ["AID", "Assay Type", "Protein Accession"],
            [
                [1001, "confirmatory", "P00533"],
                [1002, "confirmatory", "Q9H2X3"],
                [1003, "primary", "P99999"],
            ],
        )
        aid_map = _write_gz_tsv(
            tmp_path / "Aid2GeneidAccessionUniProt.gz",
            ["AID", "UniProt"],
            [
                [1002, "Q9H2X3"],
                [1003, "P99999"],
            ],
        )
        sid_map = tmp_path / "Sid2CidSMILES.gz"
        with gzip.open(sid_map, "wt") as f:
            f.write("10\t241\tc1ccccc1\n")
            f.write("12\t242\tCCO\n")
            f.write("13\t243\tCCN\n")

        stats = run_pubchem_etl(
            db_path=migrated_db,
            bioactivities_path=bioactivities,
            bioassays_path=bioassays,
            aid_uniprot_path=aid_map,
            sid_cid_smiles_path=sid_map,
            sid_lookup_db_path=tmp_path / "sid_lookup.sqlite",
            chunksize=2,
        )

        assert stats["rows_read"] == 4
        assert stats["rows_filtered_inactive_confirmatory"] == 2
        assert stats["rows_mapped_ready"] == 2
        assert stats["results_inserted"] == 2

        with connect(migrated_db) as conn:
            n_results = conn.execute(
                "SELECT COUNT(*) FROM negative_results WHERE source_db='pubchem'"
            ).fetchone()[0]
            assert n_results == 2

            assays = conn.execute(
                "SELECT COUNT(*) FROM assays WHERE source_db='pubchem'"
            ).fetchone()[0]
            assert assays == 2

            targets = {
                row[0]
                for row in conn.execute(
                    "SELECT uniprot_accession FROM targets"
                ).fetchall()
            }
            assert "P00533" in targets
            assert "Q9H2X3" in targets

            thresholds = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT inactivity_threshold FROM negative_results WHERE source_db='pubchem'"
                ).fetchall()
            }
            assert thresholds == {10000.0}

            species = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT species_tested FROM negative_results WHERE source_db='pubchem'"
                ).fetchall()
            }
            assert species == {"Homo sapiens"}

    def test_run_pubchem_etl_human_only_strict_filtering(self, migrated_db, tmp_path):
        bioactivities = _write_gz_tsv(
            tmp_path / "bioactivities.tsv.gz",
            [
                "AID",
                "SID",
                "CID",
                "Activity Outcome",
                "Activity Name",
                "Activity Value",
                "Activity Unit",
                "Protein Accession",
                "Target TaxID",
            ],
            [
                # Missing taxid, but AID is human-confirmatory in bioassays -> keep
                [2001, 21, None, "Inactive", "IC50", 20000, "nM", None, None],
                # Missing taxid, non-human AID in bioassays -> drop
                [2002, 22, None, "Inactive", "IC50", 20000, "nM", None, None],
                # Explicit non-human taxid should be dropped even if AID is human in bioassays
                [2003, 23, None, "Inactive", "IC50", 20000, "nM", None, 10090],
                # Explicit human taxid should be kept
                [2004, 24, None, "Inactive", "IC50", 20000, "nM", None, 9606],
            ],
        )
        bioassays = _write_gz_tsv(
            tmp_path / "bioassays.tsv.gz",
            ["AID", "Outcome Type", "Protein Accessions", "Target TaxIDs"],
            [
                [2001, "Confirmatory", "P20001", "9606"],
                [2002, "Confirmatory", "P20002", "10090"],
                [2003, "Confirmatory", "P20003", "9606"],
                [2004, "Confirmatory", "P20004", None],
            ],
        )
        aid_map = _write_gz_tsv(
            tmp_path / "Aid2GeneidAccessionUniProt.gz",
            ["AID", "UniProt"],
            [
                [2001, "P20001"],
                [2002, "P20002"],
                [2003, "P20003"],
                [2004, "P20004"],
            ],
        )
        sid_map = _write_gz_tsv(
            tmp_path / "Sid2CidSMILES.gz",
            ["SID", "CID", "SMILES"],
            [
                [21, 121, "CCO"],
                [22, 122, "CCN"],
                [23, 123, "CCC"],
                [24, 124, "CCCl"],
            ],
        )

        stats = run_pubchem_etl(
            db_path=migrated_db,
            bioactivities_path=bioactivities,
            bioassays_path=bioassays,
            aid_uniprot_path=aid_map,
            sid_cid_smiles_path=sid_map,
            sid_lookup_db_path=tmp_path / "sid_lookup.sqlite",
            chunksize=2,
        )

        assert stats["rows_read"] == 4
        assert stats["rows_filtered_inactive_confirmatory"] == 2
        assert stats["rows_mapped_ready"] == 2
        assert stats["results_inserted"] == 2

        with connect(migrated_db) as conn:
            rows = conn.execute(
                """
                SELECT source_record_id, species_tested
                FROM negative_results
                WHERE source_db='pubchem'
                ORDER BY source_record_id
                """
            ).fetchall()
            assert rows == [
                ("PUBCHEM:2001:21", "Homo sapiens"),
                ("PUBCHEM:2004:24", "Homo sapiens"),
            ]

    def test_run_pubchem_etl_uses_aid_map_when_direct_accession_is_non_uniprot(
        self, migrated_db, tmp_path
    ):
        bioactivities = _write_gz_tsv(
            tmp_path / "bioactivities.tsv.gz",
            [
                "AID",
                "SID",
                "CID",
                "Activity Outcome",
                "Activity Name",
                "Activity Value",
                "Activity Unit",
                "Protein Accession",
                "Target TaxID",
            ],
            [
                [3001, 31, None, "Inactive", "IC50", 20000, "nM", "1Y7V_A", 9606],
            ],
        )
        bioassays = _write_gz_tsv(
            tmp_path / "bioassays.tsv.gz",
            ["AID", "Outcome Type", "Protein Accessions", "Target TaxIDs"],
            [
                [3001, "Confirmatory", "P30001", "9606"],
            ],
        )
        aid_map = _write_gz_tsv(
            tmp_path / "Aid2GeneidAccessionUniProt.gz",
            ["AID", "UniProt"],
            [
                [3001, "P30001"],
            ],
        )
        sid_map = _write_gz_tsv(
            tmp_path / "Sid2CidSMILES.gz",
            ["SID", "CID", "SMILES"],
            [
                [31, 131, "CCO"],
            ],
        )

        stats = run_pubchem_etl(
            db_path=migrated_db,
            bioactivities_path=bioactivities,
            bioassays_path=bioassays,
            aid_uniprot_path=aid_map,
            sid_cid_smiles_path=sid_map,
            sid_lookup_db_path=tmp_path / "sid_lookup.sqlite",
            chunksize=10,
        )

        assert stats["rows_read"] == 1
        assert stats["rows_filtered_inactive_confirmatory"] == 1
        assert stats["rows_mapped_ready"] == 1
        assert stats["results_inserted"] == 1

        with connect(migrated_db) as conn:
            targets = {
                row[0]
                for row in conn.execute("SELECT uniprot_accession FROM targets").fetchall()
            }
            assert "P30001" in targets
            assert "1Y7V_A" not in targets
