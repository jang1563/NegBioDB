"""Tests for ChEMBL ETL pipeline."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from negbiodb.db import connect, create_database
from negbiodb.etl_chembl import (
    extract_chembl_inactives,
    find_chembl_db,
    insert_chembl_compounds,
    insert_chembl_negative_results,
    insert_chembl_targets,
    prepare_chembl_targets,
    refresh_all_pairs,
    standardize_chembl_compounds,
)
from negbiodb.standardize import standardize_smiles

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def migrated_db(tmp_path):
    """Create a fresh migrated database."""
    db_path = tmp_path / "test.db"
    create_database(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def sample_chembl_df():
    """Sample DataFrame mimicking ChEMBL extraction output."""
    return pd.DataFrame({
        "activity_id": [1001, 1002, 1003, 1004],
        "molregno": [100, 100, 200, 300],
        "chembl_compound_id": ["CHEMBL25", "CHEMBL25", "CHEMBL1234", "CHEMBL5678"],
        "canonical_smiles": [
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
            "CC(=O)Oc1ccccc1C(=O)O",  # same compound, different target
            "c1ccccc1",               # benzene
            "CC(=O)O",               # acetic acid
        ],
        "standard_inchi_key": [
            "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            "UHOVQNZJYSORNB-UHFFFAOYSA-N",
            "QTBSBXVTEAMEQO-UHFFFAOYSA-N",
        ],
        "pchembl_value": [4.0, 3.5, None, 4.2],
        "standard_type": ["IC50", "Ki", "IC50", "Kd"],
        "standard_value": [100000.0, 316000.0, 50000.0, 63000.0],
        "standard_relation": ["=", "=", ">", "="],
        "standard_units": ["nM", "nM", "nM", "nM"],
        "uniprot_accession": ["P00533", "P12931", "P00533", "P12931"],
        "chembl_target_id": ["CHEMBL203", "CHEMBL267", "CHEMBL203", "CHEMBL267"],
        "target_name": ["EGFR", "SRC", "EGFR", "SRC"],
        "organism": ["Homo sapiens", "Homo sapiens", "Homo sapiens", "Homo sapiens"],
        "protein_sequence": ["MRKLL" * 20, "MGSNK" * 20, "MRKLL" * 20, "MGSNK" * 20],
        "sequence_length": [100, 100, 100, 100],
        "assay_chembl_id": ["CHEMBL_A1", "CHEMBL_A2", "CHEMBL_A3", "CHEMBL_A4"],
        "publication_year": [2010, 2015, None, 2020],
    })


# ============================================================
# TestFindChEMBLDB
# ============================================================


class TestFindChEMBLDB:

    def test_finds_db(self, tmp_path):
        chembl_dir = tmp_path / "chembl"
        chembl_dir.mkdir()
        db_file = chembl_dir / "chembl_36.db"
        db_file.touch()
        result = find_chembl_db(chembl_dir)
        assert result == db_file

    def test_latest_version(self, tmp_path):
        chembl_dir = tmp_path / "chembl"
        chembl_dir.mkdir()
        (chembl_dir / "chembl_35.db").touch()
        (chembl_dir / "chembl_36.db").touch()
        result = find_chembl_db(chembl_dir)
        assert result.name == "chembl_36.db"

    def test_no_db_raises(self, tmp_path):
        chembl_dir = tmp_path / "chembl"
        chembl_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            find_chembl_db(chembl_dir)


# ============================================================
# TestStandardizeChEMBLCompounds
# ============================================================


class TestStandardizeChEMBLCompounds:

    def test_deduplicates_by_molregno(self, sample_chembl_df):
        compounds, mapping = standardize_chembl_compounds(sample_chembl_df)
        # 4 rows but only 3 unique molregnos (100, 200, 300)
        assert len(compounds) == 3
        assert len(mapping) == 3

    def test_returns_inchikey_mapping(self, sample_chembl_df):
        compounds, mapping = standardize_chembl_compounds(sample_chembl_df)
        assert 100 in mapping
        assert 200 in mapping
        assert 300 in mapping
        # All InChIKeys should be valid format
        for ik in mapping.values():
            assert len(ik) == 27
            assert ik.count("-") == 2

    def test_chembl_id_preserved(self, sample_chembl_df):
        compounds, _ = standardize_chembl_compounds(sample_chembl_df)
        chembl_ids = {c["chembl_id"] for c in compounds}
        assert "CHEMBL25" in chembl_ids
        assert "CHEMBL1234" in chembl_ids

    def test_invalid_smiles_skipped(self):
        df = pd.DataFrame({
            "molregno": [1, 2],
            "chembl_compound_id": ["CHEMBL1", "CHEMBL2"],
            "canonical_smiles": ["not_valid", "c1ccccc1"],
        })
        compounds, mapping = standardize_chembl_compounds(df)
        assert len(compounds) == 1
        assert 2 in mapping
        assert 1 not in mapping


# ============================================================
# TestPrepareChEMBLTargets
# ============================================================


class TestPrepareChEMBLTargets:

    def test_deduplicates_by_accession(self, sample_chembl_df):
        targets = prepare_chembl_targets(sample_chembl_df)
        # 4 rows but only 2 unique UniProt accessions
        assert len(targets) == 2

    def test_target_fields(self, sample_chembl_df):
        targets = prepare_chembl_targets(sample_chembl_df)
        accessions = {t["uniprot_accession"] for t in targets}
        assert "P00533" in accessions
        assert "P12931" in accessions
        for t in targets:
            assert "chembl_target_id" in t
            assert "amino_acid_sequence" in t
            assert "sequence_length" in t


# ============================================================
# TestInsertChEMBLCompounds
# ============================================================


class TestInsertChEMBLCompounds:

    def test_insert_new(self, migrated_db):
        compounds = [standardize_smiles("c1ccccc1")]
        compounds[0]["chembl_id"] = "CHEMBL277500"
        with connect(migrated_db) as conn:
            mapping = insert_chembl_compounds(conn, compounds)
            conn.commit()
        assert compounds[0]["inchikey"] in mapping

    def test_idempotent(self, migrated_db):
        compounds = [standardize_smiles("c1ccccc1")]
        compounds[0]["chembl_id"] = "CHEMBL277500"
        with connect(migrated_db) as conn:
            m1 = insert_chembl_compounds(conn, compounds)
            m2 = insert_chembl_compounds(conn, compounds)
            conn.commit()
        ik = compounds[0]["inchikey"]
        assert m1[ik] == m2[ik]

    def test_cross_db_dedup_with_davis(self, migrated_db):
        """Inserting same InChIKey from DAVIS and ChEMBL should not duplicate."""
        benzene = standardize_smiles("c1ccccc1")
        with connect(migrated_db) as conn:
            # Insert as if from DAVIS (with pubchem_cid)
            conn.execute(
                """INSERT INTO compounds
                (canonical_smiles, inchikey, inchikey_connectivity, inchi, pubchem_cid)
                VALUES (?, ?, ?, ?, ?)""",
                (benzene["canonical_smiles"], benzene["inchikey"],
                 benzene["inchikey_connectivity"], benzene["inchi"], 241),
            )
            davis_cid = conn.execute(
                "SELECT compound_id FROM compounds WHERE inchikey = ?",
                (benzene["inchikey"],),
            ).fetchone()[0]

            # Now insert same compound as if from ChEMBL
            chembl_compound = dict(benzene)
            chembl_compound["chembl_id"] = "CHEMBL277500"
            mapping = insert_chembl_compounds(conn, [chembl_compound])
            conn.commit()

            # Should map to same compound_id (INSERT OR IGNORE)
            assert mapping[benzene["inchikey"]] == davis_cid
            count = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
            assert count == 1


# ============================================================
# TestInsertChEMBLTargets
# ============================================================


class TestInsertChEMBLTargets:

    def test_insert_new(self, migrated_db):
        targets = [{
            "uniprot_accession": "P00533",
            "chembl_target_id": "CHEMBL203",
            "amino_acid_sequence": "MRKLL" * 20,
            "sequence_length": 100,
        }]
        with connect(migrated_db) as conn:
            mapping = insert_chembl_targets(conn, targets)
            conn.commit()
        assert "P00533" in mapping

    def test_idempotent(self, migrated_db):
        targets = [{
            "uniprot_accession": "P00533",
            "chembl_target_id": "CHEMBL203",
            "amino_acid_sequence": "MRKLL" * 20,
            "sequence_length": 100,
        }]
        with connect(migrated_db) as conn:
            m1 = insert_chembl_targets(conn, targets)
            m2 = insert_chembl_targets(conn, targets)
            conn.commit()
        assert m1["P00533"] == m2["P00533"]


# ============================================================
# TestInsertChEMBLNegativeResults
# ============================================================


class TestInsertChEMBLNegativeResults:

    def _setup_data(self, conn):
        """Insert minimal compound and target for testing."""
        conn.execute(
            """INSERT INTO compounds
            (canonical_smiles, inchikey, inchikey_connectivity, chembl_id)
            VALUES ('CC(=O)Oc1ccccc1C(=O)O',
                    'BSYNRYMUTXBXSQ-UHFFFAOYSA-N', 'BSYNRYMUTXBXSQ', 'CHEMBL25')"""
        )
        cid = conn.execute("SELECT compound_id FROM compounds").fetchone()[0]
        conn.execute(
            """INSERT INTO targets (uniprot_accession, chembl_target_id, sequence_length)
            VALUES ('P00533', 'CHEMBL203', 100)"""
        )
        tid = conn.execute("SELECT target_id FROM targets").fetchone()[0]
        return cid, tid

    def test_insert(self, migrated_db):
        with connect(migrated_db) as conn:
            cid, tid = self._setup_data(conn)

            df = pd.DataFrame({
                "activity_id": [1001],
                "molregno": [100],
                "uniprot_accession": ["P00533"],
                "pchembl_value": [4.0],
                "standard_type": ["IC50"],
                "standard_value": [100000.0],
                "standard_relation": ["="],
                "standard_units": ["nM"],
                "publication_year": [2010],
            })
            molregno_to_ik = {100: "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"}
            ik_to_cid = {"BSYNRYMUTXBXSQ-UHFFFAOYSA-N": cid}
            acc_to_tid = {"P00533": tid}

            inserted, skipped = insert_chembl_negative_results(
                conn, df, molregno_to_ik, ik_to_cid, acc_to_tid,
            )
            conn.commit()

        assert inserted == 1
        assert skipped == 0

    def test_confidence_silver(self, migrated_db):
        with connect(migrated_db) as conn:
            cid, tid = self._setup_data(conn)

            df = pd.DataFrame({
                "activity_id": [1001],
                "molregno": [100],
                "uniprot_accession": ["P00533"],
                "pchembl_value": [4.0],
                "standard_type": ["IC50"],
                "standard_value": [100000.0],
                "standard_relation": ["="],
                "standard_units": ["nM"],
                "publication_year": [2010],
            })
            insert_chembl_negative_results(
                conn, df,
                {100: "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
                {"BSYNRYMUTXBXSQ-UHFFFAOYSA-N": cid},
                {"P00533": tid},
            )
            conn.commit()
            row = conn.execute(
                "SELECT confidence_tier FROM negative_results"
            ).fetchone()
        assert row[0] == "silver"

    def test_right_censored(self, migrated_db):
        """Right-censored records should have activity_relation='>'."""
        with connect(migrated_db) as conn:
            cid, tid = self._setup_data(conn)

            df = pd.DataFrame({
                "activity_id": [1001],
                "molregno": [100],
                "uniprot_accession": ["P00533"],
                "pchembl_value": [None],
                "standard_type": ["IC50"],
                "standard_value": [50000.0],
                "standard_relation": [">"],
                "standard_units": ["nM"],
                "publication_year": [None],
            })
            insert_chembl_negative_results(
                conn, df,
                {100: "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
                {"BSYNRYMUTXBXSQ-UHFFFAOYSA-N": cid},
                {"P00533": tid},
            )
            conn.commit()
            row = conn.execute(
                "SELECT activity_relation, pchembl_value FROM negative_results"
            ).fetchone()
        assert row[0] == ">"
        assert row[1] is None

    def test_skips_unmapped(self, migrated_db):
        with connect(migrated_db) as conn:
            cid, tid = self._setup_data(conn)

            df = pd.DataFrame({
                "activity_id": [1001, 1002],
                "molregno": [100, 999],  # 999 not in mapping
                "uniprot_accession": ["P00533", "P00533"],
                "pchembl_value": [4.0, 4.0],
                "standard_type": ["IC50", "IC50"],
                "standard_value": [100000.0, 100000.0],
                "standard_relation": ["=", "="],
                "standard_units": ["nM", "nM"],
                "publication_year": [2010, 2010],
            })
            inserted, skipped = insert_chembl_negative_results(
                conn, df,
                {100: "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
                {"BSYNRYMUTXBXSQ-UHFFFAOYSA-N": cid},
                {"P00533": tid},
            )
            conn.commit()

        assert inserted == 1
        assert skipped == 1


# ============================================================
# TestRefreshAllPairs
# ============================================================


class TestRefreshAllPairs:

    def test_aggregates_across_sources(self, migrated_db):
        """Pairs from different sources should be correctly merged."""
        with connect(migrated_db) as conn:
            # Insert compound and target
            conn.execute(
                """INSERT INTO compounds
                (canonical_smiles, inchikey, inchikey_connectivity)
                VALUES ('c1ccccc1', 'UHOVQNZJYSORNB-UHFFFAOYSA-N', 'UHOVQNZJYSORNB')"""
            )
            conn.execute(
                """INSERT INTO targets (uniprot_accession, gene_symbol, sequence_length)
                VALUES ('Q2M2I8', 'AAK1', 100)"""
            )
            # Insert from DAVIS (bronze)
            conn.execute(
                """INSERT INTO negative_results
                (compound_id, target_id, result_type, confidence_tier,
                 activity_type, activity_value, activity_unit, pchembl_value,
                 inactivity_threshold, source_db, source_record_id, extraction_method,
                 publication_year)
                VALUES (1, 1, 'hard_negative', 'bronze',
                        'Kd', 10000.0, 'nM', 5.0,
                        10000.0, 'davis', 'DAVIS:0_0', 'database_direct', 2011)"""
            )
            # Insert from ChEMBL (silver) — same compound-target pair
            conn.execute(
                """INSERT INTO negative_results
                (compound_id, target_id, result_type, confidence_tier,
                 activity_type, activity_value, activity_unit, pchembl_value,
                 inactivity_threshold, source_db, source_record_id, extraction_method,
                 publication_year)
                VALUES (1, 1, 'hard_negative', 'silver',
                        'IC50', 50000.0, 'nM', 4.3,
                        10000.0, 'chembl', 'CHEMBL:12345', 'database_direct', 2015)"""
            )

            count = refresh_all_pairs(conn)
            conn.commit()

            # Should be 1 pair (merged)
            assert count == 1
            row = conn.execute(
                "SELECT num_sources, best_confidence FROM compound_target_pairs"
            ).fetchone()
            assert row[0] == 2  # two sources
            assert row[1] == "silver"  # silver > bronze

    def test_clears_old_pairs(self, migrated_db):
        """refresh_all_pairs should replace all existing pairs."""
        with connect(migrated_db) as conn:
            conn.execute(
                """INSERT INTO compounds
                (canonical_smiles, inchikey, inchikey_connectivity)
                VALUES ('c1ccccc1', 'UHOVQNZJYSORNB-UHFFFAOYSA-N', 'UHOVQNZJYSORNB')"""
            )
            conn.execute(
                """INSERT INTO targets (uniprot_accession, gene_symbol, sequence_length)
                VALUES ('Q2M2I8', 'AAK1', 100)"""
            )
            conn.execute(
                """INSERT INTO negative_results
                (compound_id, target_id, result_type, confidence_tier,
                 activity_type, activity_value, activity_unit,
                 inactivity_threshold, source_db, source_record_id, extraction_method)
                VALUES (1, 1, 'hard_negative', 'bronze',
                        'Kd', 10000.0, 'nM',
                        10000.0, 'davis', 'DAVIS:0_0', 'database_direct')"""
            )
            # First refresh
            count1 = refresh_all_pairs(conn)
            # Second refresh (should not double)
            count2 = refresh_all_pairs(conn)
            conn.commit()

            assert count1 == 1
            assert count2 == 1


# ============================================================
# TestExtractChEMBLInactives (with mock DB)
# ============================================================


class TestExtractChEMBLInactives:

    def _create_mock_chembl(self, tmp_path):
        """Create a minimal ChEMBL-like SQLite database for testing."""
        db_path = tmp_path / "mock_chembl.db"
        conn = sqlite3.connect(str(db_path))

        conn.executescript("""
            CREATE TABLE molecule_dictionary (molregno INTEGER PRIMARY KEY, chembl_id TEXT);
            CREATE TABLE compound_structures (molregno INTEGER PRIMARY KEY, canonical_smiles TEXT, standard_inchi_key TEXT);
            CREATE TABLE activities (
                activity_id INTEGER PRIMARY KEY, molregno INTEGER,
                assay_id INTEGER, doc_id INTEGER,
                pchembl_value REAL, standard_type TEXT,
                standard_value REAL, standard_relation TEXT,
                standard_units TEXT, data_validity_comment TEXT
            );
            CREATE TABLE assays (assay_id INTEGER PRIMARY KEY, tid INTEGER, chembl_id TEXT);
            CREATE TABLE target_dictionary (tid INTEGER PRIMARY KEY, chembl_id TEXT, pref_name TEXT, target_type TEXT, organism TEXT);
            CREATE TABLE target_components (tid INTEGER, component_id INTEGER);
            CREATE TABLE component_sequences (component_id INTEGER PRIMARY KEY, accession TEXT, sequence TEXT);
            CREATE TABLE docs (doc_id INTEGER PRIMARY KEY, year INTEGER);

            -- Insert test data
            INSERT INTO molecule_dictionary VALUES (1, 'CHEMBL25');
            INSERT INTO molecule_dictionary VALUES (2, 'CHEMBL1234');
            INSERT INTO compound_structures VALUES (1, 'CC(=O)Oc1ccccc1C(=O)O', 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N');
            INSERT INTO compound_structures VALUES (2, 'c1ccccc1', 'UHOVQNZJYSORNB-UHFFFAOYSA-N');

            INSERT INTO target_dictionary VALUES (1, 'CHEMBL203', 'EGFR', 'SINGLE PROTEIN', 'Homo sapiens');
            INSERT INTO target_dictionary VALUES (2, 'CHEMBL999', 'NonHuman', 'SINGLE PROTEIN', 'Mus musculus');
            INSERT INTO target_components VALUES (1, 1);
            INSERT INTO target_components VALUES (2, 2);
            INSERT INTO component_sequences VALUES (1, 'P00533', 'MRKLL');
            INSERT INTO component_sequences VALUES (2, 'P99999', 'MGSNK');

            INSERT INTO assays VALUES (1, 1, 'CHEMBL_A1');
            INSERT INTO assays VALUES (2, 2, 'CHEMBL_A2');

            INSERT INTO docs VALUES (1, 2010);

            -- Type 1: pChEMBL < 4.5 (should be extracted)
            INSERT INTO activities VALUES (1001, 1, 1, 1, 4.0, 'IC50', 100000.0, '=', 'nM', NULL);

            -- Borderline: pChEMBL = 4.8 (should NOT be extracted with borderline_lower=4.5)
            INSERT INTO activities VALUES (1002, 2, 1, 1, 4.8, 'Ki', 15000.0, '=', 'nM', NULL);

            -- Type 2: Right-censored (should be extracted)
            INSERT INTO activities VALUES (1003, 2, 1, 1, NULL, 'IC50', 50000.0, '>', 'nM', NULL);

            -- Non-human target (should NOT be extracted)
            INSERT INTO activities VALUES (1004, 1, 2, 1, 3.0, 'IC50', 1000000.0, '=', 'nM', NULL);

            -- Invalid data_validity_comment (should NOT be extracted)
            INSERT INTO activities VALUES (1005, 1, 1, 1, 3.0, 'IC50', 1000000.0, '=', 'nM', 'Outside typical range');
        """)

        conn.commit()
        conn.close()
        return db_path

    def test_extracts_type1_and_type2(self, tmp_path):
        mock_db = self._create_mock_chembl(tmp_path)
        cfg = {
            "borderline_exclusion": {"lower": 4.5, "upper": 5.5},
            "inactivity_threshold_nm": 10000,
        }
        df = extract_chembl_inactives(mock_db, cfg)

        # Should get activity 1001 (pChEMBL 4.0) and 1003 (right-censored >50000)
        assert len(df) == 2
        activity_ids = set(df["activity_id"].tolist())
        assert 1001 in activity_ids
        assert 1003 in activity_ids

    def test_excludes_borderline(self, tmp_path):
        mock_db = self._create_mock_chembl(tmp_path)
        cfg = {
            "borderline_exclusion": {"lower": 4.5, "upper": 5.5},
            "inactivity_threshold_nm": 10000,
        }
        df = extract_chembl_inactives(mock_db, cfg)

        # Activity 1002 (pChEMBL 4.8) should be excluded
        assert 1002 not in df["activity_id"].tolist()

    def test_excludes_non_human(self, tmp_path):
        mock_db = self._create_mock_chembl(tmp_path)
        cfg = {
            "borderline_exclusion": {"lower": 4.5, "upper": 5.5},
            "inactivity_threshold_nm": 10000,
        }
        df = extract_chembl_inactives(mock_db, cfg)

        # Activity 1004 (non-human target) should be excluded
        assert 1004 not in df["activity_id"].tolist()

    def test_excludes_invalid(self, tmp_path):
        mock_db = self._create_mock_chembl(tmp_path)
        cfg = {
            "borderline_exclusion": {"lower": 4.5, "upper": 5.5},
            "inactivity_threshold_nm": 10000,
        }
        df = extract_chembl_inactives(mock_db, cfg)

        # Activity 1005 (invalid data) should be excluded
        assert 1005 not in df["activity_id"].tolist()
