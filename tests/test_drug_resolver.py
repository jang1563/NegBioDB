"""Tests for drug name resolution cascade."""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from negbiodb_ct.ct_db import create_ct_database, get_connection
from negbiodb_ct.drug_resolver import (
    _map_mol_type,
    _pug_download_results,
    _pubchem_batch_cid_to_properties,
    _pubchem_name_lookup,
    _xml_escape,
    build_chembl_synonym_index,
    clean_drug_name,
    crossref_inchikey_to_chembl,
    insert_intervention_targets,
    is_non_drug_name,
    load_overrides,
    resolve_step1_chembl,
    resolve_step2_pubchem,
    resolve_step3_fuzzy,
    resolve_step4_overrides,
    update_interventions,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_ct"


@pytest.fixture
def ct_db(tmp_path):
    """Create a fresh CT database with all migrations applied."""
    db_path = tmp_path / "test_ct.db"
    create_ct_database(db_path, MIGRATIONS_DIR)
    return db_path


# ============================================================
# CLEAN DRUG NAME TESTS
# ============================================================


class TestCleanDrugName:
    def test_lowercase(self):
        assert clean_drug_name("Imatinib") == "imatinib"

    def test_strip_dosage_mg(self):
        assert clean_drug_name("Imatinib (400mg)") == "imatinib"

    def test_strip_dosage_ug(self):
        assert clean_drug_name("DrugX (50ug daily)") == "drugx"

    def test_strip_salt(self):
        assert clean_drug_name("Imatinib mesylate") == "imatinib"

    def test_strip_hydrochloride(self):
        assert clean_drug_name("Tamsulosin Hydrochloride") == "tamsulosin"

    def test_strip_trailing_paren(self):
        assert clean_drug_name("DrugA (formerly B123)") == "druga"

    def test_empty(self):
        assert clean_drug_name("") == ""

    def test_combined(self):
        assert clean_drug_name("Aspirin Sodium (500mg tablets)") == "aspirin"


# ============================================================
# STEP 1: CHEMBL SYNONYM INDEX TESTS
# ============================================================


class TestBuildChemblSynonymIndex:
    def test_missing_db(self, tmp_path):
        result = build_chembl_synonym_index(tmp_path / "nonexistent.db")
        assert result == {}


class TestResolveStep1:
    def test_exact_match(self):
        index = {"imatinib": "CHEMBL941", "aspirin": "CHEMBL25"}
        resolved = resolve_step1_chembl(["Imatinib", "Unknown Drug"], index)
        assert resolved == {"Imatinib": "CHEMBL941"}

    def test_no_match(self):
        index = {"imatinib": "CHEMBL941"}
        resolved = resolve_step1_chembl(["SomeNewDrug"], index)
        assert resolved == {}

    def test_salt_stripped(self):
        index = {"imatinib": "CHEMBL941"}
        resolved = resolve_step1_chembl(["Imatinib mesylate"], index)
        assert resolved == {"Imatinib mesylate": "CHEMBL941"}


# ============================================================
# STEP 3: FUZZY MATCH TESTS
# ============================================================


class TestResolveStep3Fuzzy:
    def test_close_match(self):
        index = {"imatinib": "CHEMBL941", "aspirin": "CHEMBL25"}
        # "imatinb" is close to "imatinib"
        resolved = resolve_step3_fuzzy(["Imatinb"], index, threshold=0.85)
        assert "Imatinb" in resolved
        assert resolved["Imatinb"] == "CHEMBL941"

    def test_no_close_match(self):
        index = {"imatinib": "CHEMBL941"}
        resolved = resolve_step3_fuzzy(["CompletelyDifferent"], index, threshold=0.90)
        assert resolved == {}

    def test_high_threshold(self):
        index = {"imatinib": "CHEMBL941"}
        # Very high threshold should reject approximate matches
        resolved = resolve_step3_fuzzy(["imatinb"], index, threshold=0.99)
        assert resolved == {}


# ============================================================
# STEP 4: OVERRIDES TESTS
# ============================================================


class TestLoadOverrides:
    def test_basic_load(self, tmp_path):
        csv = tmp_path / "overrides.csv"
        csv.write_text(
            "intervention_name,chembl_id,canonical_smiles,molecular_type\n"
            "trastuzumab,CHEMBL1201585,,monoclonal_antibody\n"
        )
        overrides = load_overrides(csv)
        assert "trastuzumab" in overrides
        assert overrides["trastuzumab"]["chembl_id"] == "CHEMBL1201585"

    def test_missing_file(self, tmp_path):
        overrides = load_overrides(tmp_path / "missing.csv")
        assert overrides == {}


class TestResolveStep4:
    def test_match(self):
        overrides = {
            "trastuzumab": {"chembl_id": "CHEMBL1201585", "molecular_type": "monoclonal_antibody"},
        }
        resolved = resolve_step4_overrides(["Trastuzumab"], overrides)
        assert "Trastuzumab" in resolved
        assert resolved["Trastuzumab"]["chembl_id"] == "CHEMBL1201585"


# ============================================================
# MOL TYPE MAPPING TESTS
# ============================================================


class TestMapMolType:
    def test_small_molecule(self):
        assert _map_mol_type("Small molecule") == "small_molecule"

    def test_antibody(self):
        assert _map_mol_type("Antibody") == "monoclonal_antibody"

    def test_protein(self):
        assert _map_mol_type("Protein") == "peptide"

    def test_none(self):
        assert _map_mol_type(None) == "unknown"

    def test_unknown(self):
        assert _map_mol_type("Unknown") == "unknown"


# ============================================================
# UPDATE INTERVENTIONS TESTS
# ============================================================


class TestUpdateInterventions:
    def test_basic_update(self, ct_db):
        conn = get_connection(ct_db)
        try:
            conn.execute(
                "INSERT INTO interventions (intervention_type, intervention_name) "
                "VALUES ('drug', 'TestDrug')"
            )
            conn.commit()

            resolutions = {
                1: {
                    "chembl_id": "CHEMBL941",
                    "canonical_smiles": "CC(=O)OC1=CC=CC=C1",
                    "molecular_type": "small_molecule",
                },
            }
            n = update_interventions(conn, resolutions)
            assert n == 1

            row = conn.execute(
                "SELECT chembl_id, canonical_smiles, molecular_type "
                "FROM interventions WHERE intervention_id = 1"
            ).fetchone()
            assert row[0] == "CHEMBL941"
            assert row[1] == "CC(=O)OC1=CC=CC=C1"
            assert row[2] == "small_molecule"
        finally:
            conn.close()


# ============================================================
# INSERT TARGETS TESTS
# ============================================================


class TestInsertInterventionTargets:
    def test_basic_insert(self, ct_db):
        conn = get_connection(ct_db)
        try:
            conn.execute(
                "INSERT INTO interventions (intervention_type, intervention_name) "
                "VALUES ('drug', 'TestDrug')"
            )
            conn.commit()

            targets = [{
                "chembl_id": "CHEMBL941",
                "uniprot_accession": "P00519",
                "gene_symbol": "ABL1",
                "action_type": "INHIBITOR",
            }]
            chembl_to_interv = {"CHEMBL941": [1]}
            n = insert_intervention_targets(conn, targets, chembl_to_interv)
            assert n == 1

            row = conn.execute(
                "SELECT uniprot_accession, source "
                "FROM intervention_targets WHERE intervention_id = 1"
            ).fetchone()
            assert row[0] == "P00519"
            assert row[1] == "chembl"
        finally:
            conn.close()

    def test_dedup(self, ct_db):
        conn = get_connection(ct_db)
        try:
            conn.execute(
                "INSERT INTO interventions (intervention_type, intervention_name) "
                "VALUES ('drug', 'TestDrug')"
            )
            conn.commit()

            targets = [
                {"chembl_id": "CHEMBL941", "uniprot_accession": "P00519",
                 "gene_symbol": "ABL1", "action_type": "INHIBITOR"},
                {"chembl_id": "CHEMBL941", "uniprot_accession": "P00519",
                 "gene_symbol": "ABL1", "action_type": "INHIBITOR"},
            ]
            chembl_to_interv = {"CHEMBL941": [1]}
            n = insert_intervention_targets(conn, targets, chembl_to_interv)
            assert n == 1  # deduped
        finally:
            conn.close()


# ============================================================
# NON-DRUG NAME FILTER TESTS
# ============================================================


class TestIsNonDrugName:
    def test_placebo(self):
        assert is_non_drug_name("Placebo") is True

    def test_placebo_oral_tablet(self):
        assert is_non_drug_name("Placebo Oral Tablet") is True

    def test_saline(self):
        assert is_non_drug_name("0.9% Saline") is True

    def test_sugar_pill(self):
        assert is_non_drug_name("sugar pill") is True

    def test_standard_of_care(self):
        assert is_non_drug_name("Standard of Care") is True

    def test_vehicle_cream(self):
        assert is_non_drug_name("Vehicle Cream") is True

    def test_blood_sample(self):
        assert is_non_drug_name("Blood sample") is True

    def test_sham(self):
        assert is_non_drug_name("Sham procedure") is True

    def test_real_drug(self):
        assert is_non_drug_name("Imatinib") is False

    def test_real_biologic(self):
        assert is_non_drug_name("Trastuzumab") is False

    def test_empty(self):
        assert is_non_drug_name("") is False

    def test_best_supportive_care(self):
        assert is_non_drug_name("Best supportive care plus placebo") is True


# ============================================================
# ENHANCED PUBCHEM API TESTS
# ============================================================


class TestPubchemNameLookup:
    def test_returns_dict_with_all_fields(self, monkeypatch):
        """Mock PubChem API to return CID, SMILES, InChIKey."""
        import urllib.request

        mock_response = json.dumps({
            "PropertyTable": {"Properties": [{
                "CID": 2244,
                "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            }]}
        }).encode()

        class MockResponse:
            def read(self):
                return mock_response
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: MockResponse())
        result = _pubchem_name_lookup("aspirin", rate_limit=100.0)
        assert result is not None
        assert result["cid"] == 2244
        assert "BSYNRYMUTXBXSQ" in result["inchikey"]
        assert result["smiles"] is not None

    def test_returns_none_on_404(self, monkeypatch):
        import urllib.request
        import urllib.error

        def mock_urlopen(*a, **kw):
            raise urllib.error.HTTPError(None, 404, "Not Found", {}, None)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        result = _pubchem_name_lookup("nonexistent_compound_xyz", rate_limit=100.0)
        assert result is None


class TestResolveStep2PubchemEnhanced:
    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        """Test that enhanced cache stores and restores dict entries."""
        import urllib.request

        mock_response = json.dumps({
            "PropertyTable": {"Properties": [{
                "CID": 5090,
                "CanonicalSMILES": "C1=CC=C(C=C1)C(=O)O",
                "InChIKey": "WPYMKLBDIGXBTP-UHFFFAOYSA-N",
            }]}
        }).encode()

        class MockResponse:
            def read(self):
                return mock_response
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: MockResponse())

        cache_path = tmp_path / "test_cache.json"
        result = resolve_step2_pubchem(["Benzoic Acid"], cache_path, rate_limit=100.0)
        assert "Benzoic Acid" in result
        assert result["Benzoic Acid"]["cid"] == 5090

        # Verify cache file
        with open(cache_path) as f:
            cache = json.load(f)
        assert "benzoic acid" in cache
        assert cache["benzoic acid"]["cid"] == 5090

    def test_backward_compat_int_cache(self, tmp_path, monkeypatch):
        """Old cache with int CIDs should still work."""
        cache_path = tmp_path / "old_cache.json"
        cache_path.write_text(json.dumps({"aspirin": 2244, "unknown_x": None}))

        # No API calls needed — all from cache
        result = resolve_step2_pubchem(
            ["Aspirin", "Unknown_X"], cache_path, rate_limit=100.0,
        )
        assert "Aspirin" in result
        assert result["Aspirin"]["cid"] == 2244
        assert "Unknown_X" not in result


# ============================================================
# INCHIKEY → CHEMBL CROSS-REFERENCE TESTS
# ============================================================


class TestCrossrefInchikeyToChembl:
    def test_missing_db(self, tmp_path):
        result = crossref_inchikey_to_chembl(
            ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N"],
            tmp_path / "nonexistent.db",
        )
        assert result == {}

    def test_empty_list(self, tmp_path):
        result = crossref_inchikey_to_chembl([], tmp_path / "nonexistent.db")
        assert result == {}

    def test_match_in_chembl(self, tmp_path):
        """Create a minimal ChEMBL-like DB with compound_structures."""
        db_path = tmp_path / "mini_chembl.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE molecule_dictionary "
            "(molregno INTEGER PRIMARY KEY, chembl_id TEXT)"
        )
        conn.execute(
            "CREATE TABLE compound_structures "
            "(molregno INTEGER PRIMARY KEY, standard_inchi_key TEXT)"
        )
        conn.execute("INSERT INTO molecule_dictionary VALUES (1, 'CHEMBL25')")
        conn.execute(
            "INSERT INTO compound_structures VALUES "
            "(1, 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N')"
        )
        conn.commit()
        conn.close()

        result = crossref_inchikey_to_chembl(
            ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "NONEXISTENT-KEY"],
            db_path,
        )
        assert result == {"BSYNRYMUTXBXSQ-UHFFFAOYSA-N": "CHEMBL25"}


# ============================================================
# XML ESCAPE TESTS
# ============================================================


class TestXmlEscape:
    def test_ampersand(self):
        assert _xml_escape("A&B") == "A&amp;B"

    def test_angle_brackets(self):
        assert _xml_escape("<tag>") == "&lt;tag&gt;"

    def test_quotes(self):
        assert _xml_escape('say "hello"') == 'say &quot;hello&quot;'

    def test_no_escape_needed(self):
        assert _xml_escape("imatinib") == "imatinib"

    def test_braces_pass_through(self):
        # Braces are NOT XML special chars — they pass through xml_escape.
        # But they must not break the template (we use .replace, not .format).
        assert _xml_escape("drug{1}") == "drug{1}"


# ============================================================
# PUG BATCH TSV DOWNLOAD TESTS
# ============================================================


class TestPugDownloadResults:
    def test_parse_tsv(self, monkeypatch):
        import urllib.request

        tsv_data = "aspirin\t2244\nimatinib\t123631\nunknown\t\n".encode()

        class MockResponse:
            def read(self):
                return tsv_data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: MockResponse())
        result = _pug_download_results("https://example.com/results.tsv")
        assert result == {"aspirin": 2244, "imatinib": 123631}

    def test_lowercase_keys(self, monkeypatch):
        import urllib.request

        tsv_data = "Aspirin\t2244\nIMATINIB\t123631\n".encode()

        class MockResponse:
            def read(self):
                return tsv_data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: MockResponse())
        result = _pug_download_results("https://example.com/results.tsv")
        assert "aspirin" in result
        assert "imatinib" in result

    def test_empty_response(self, monkeypatch):
        import urllib.request

        class MockResponse:
            def read(self):
                return b""
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: MockResponse())
        result = _pug_download_results("https://example.com/results.tsv")
        assert result == {}

    def test_ftp_to_https_conversion(self, monkeypatch):
        """Verify ftp:// URLs are converted to https://."""
        import urllib.request

        captured_urls = []

        class MockResponse:
            def read(self):
                return b""
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        def mock_urlopen(req, **kw):
            captured_urls.append(req.full_url)
            return MockResponse()

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        _pug_download_results("ftp://ftp.ncbi.nlm.nih.gov/pubchem/result.tsv")
        assert captured_urls[0].startswith("https://")


# ============================================================
# BATCH CID→PROPERTIES TESTS
# ============================================================


class TestPubchemBatchCidToProperties:
    def test_parses_response(self, monkeypatch):
        import urllib.request

        mock_data = json.dumps({
            "PropertyTable": {"Properties": [
                {"CID": 2244, "IsomericSMILES": "CC(=O)OC1=CC=CC=C1C(O)=O", "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
                {"CID": 5090, "CanonicalSMILES": "OC(=O)C1=CC=CC=C1", "InChIKey": "WPYMKLBDIGXBTP-UHFFFAOYSA-N"},
            ]}
        }).encode()

        class MockResponse:
            def read(self):
                return mock_data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: MockResponse())
        result = _pubchem_batch_cid_to_properties([2244, 5090])
        assert 2244 in result
        assert 5090 in result
        assert result[2244]["smiles"] == "CC(=O)OC1=CC=CC=C1C(O)=O"  # IsomericSMILES preferred
        assert result[2244]["inchikey"] == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

    def test_prefers_isomeric_over_canonical(self, monkeypatch):
        import urllib.request

        mock_data = json.dumps({
            "PropertyTable": {"Properties": [{
                "CID": 1,
                "IsomericSMILES": "ISO_SMILES",
                "CanonicalSMILES": "CAN_SMILES",
                "InChIKey": "TESTKEY",
            }]}
        }).encode()

        class MockResponse:
            def read(self):
                return mock_data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: MockResponse())
        result = _pubchem_batch_cid_to_properties([1])
        assert result[1]["smiles"] == "ISO_SMILES"

    def test_retries_on_http_error(self, monkeypatch):
        import urllib.request
        import urllib.error

        call_count = [0]

        def mock_urlopen(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 3:
                raise urllib.error.HTTPError(None, 503, "Service Unavailable", {}, None)
            mock_data = json.dumps({
                "PropertyTable": {"Properties": [
                    {"CID": 1, "IsomericSMILES": "C", "InChIKey": "KEY"},
                ]}
            }).encode()

            class MockResponse:
                def read(self):
                    return mock_data
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass

            return MockResponse()

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        monkeypatch.setattr("time.sleep", lambda _: None)
        result = _pubchem_batch_cid_to_properties([1], max_retries=3)
        assert 1 in result
        assert call_count[0] == 3


# ============================================================
# BATCH RESOLVE STEP 2 TESTS
# ============================================================


class TestResolveStep2BatchMode:
    def test_batch_mode_dispatches(self, tmp_path, monkeypatch):
        """With >100 names and use_batch=True, batch path should be used."""
        from negbiodb_ct import drug_resolver

        batch_called = [False]
        original_batch = drug_resolver._pubchem_batch_names_to_cids

        def mock_batch(names, batch_size=5000):
            batch_called[0] = True
            return {"drugname_001": 1}

        def mock_props(cids, batch_size=500, max_retries=3):
            return {1: {"smiles": "C", "inchikey": "KEY-A-B"}}

        monkeypatch.setattr(drug_resolver, "_pubchem_batch_names_to_cids", mock_batch)
        monkeypatch.setattr(drug_resolver, "_pubchem_batch_cid_to_properties", mock_props)

        names = [f"DrugName_{i:03d}" for i in range(150)]
        cache_path = tmp_path / "cache.json"
        result = resolve_step2_pubchem(names, cache_path, use_batch=True)
        assert batch_called[0] is True

    def test_single_mode_for_small_sets(self, tmp_path, monkeypatch):
        """With ≤100 names, single-name REST should be used."""
        import urllib.request

        mock_data = json.dumps({
            "PropertyTable": {"Properties": [{
                "CID": 2244, "IsomericSMILES": "C", "InChIKey": "KEY",
            }]}
        }).encode()

        class MockResponse:
            def read(self):
                return mock_data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: MockResponse())

        cache_path = tmp_path / "cache.json"
        result = resolve_step2_pubchem(
            ["aspirin"], cache_path, rate_limit=100.0, use_batch=False,
        )
        assert "aspirin" in result
