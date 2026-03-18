"""Tests for AACT ETL pipeline."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from negbiodb_ct.ct_db import create_ct_database, get_connection, refresh_all_ct_pairs
from negbiodb_ct.etl_aact import (
    BATCH_SIZE,
    normalize_phase,
    normalize_sponsor_type,
    normalize_intervention_type,
    parse_aact_date,
    prepare_interventions,
    prepare_conditions,
    prepare_trials,
    insert_interventions,
    insert_conditions,
    insert_trials,
    insert_trial_junctions,
    load_aact_table,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_ct"


@pytest.fixture
def ct_db(tmp_path):
    """Create a fresh CT database with all migrations applied."""
    db_path = tmp_path / "test_ct.db"
    create_ct_database(db_path, MIGRATIONS_DIR)
    return db_path


# ============================================================
# TRANSFORM TESTS
# ============================================================


class TestNormalizePhase:
    def test_phase_1(self):
        assert normalize_phase("Phase 1") == "phase_1"

    def test_phase_1_2(self):
        assert normalize_phase("Phase 1/Phase 2") == "phase_1_2"

    def test_phase_2(self):
        assert normalize_phase("Phase 2") == "phase_2"

    def test_phase_3(self):
        assert normalize_phase("Phase 3") == "phase_3"

    def test_phase_4(self):
        assert normalize_phase("Phase 4") == "phase_4"

    def test_early_phase_1(self):
        assert normalize_phase("Early Phase 1") == "early_phase_1"

    def test_not_applicable(self):
        assert normalize_phase("Not Applicable") == "not_applicable"

    def test_none(self):
        assert normalize_phase(None) is None

    def test_empty(self):
        assert normalize_phase("") is None

    def test_nan_float(self):
        assert normalize_phase(float("nan")) is None

    def test_unknown(self):
        assert normalize_phase("Phase 5") is None


class TestNormalizeSponsorType:
    def test_industry(self):
        assert normalize_sponsor_type("Industry") == "industry"

    def test_nih(self):
        assert normalize_sponsor_type("NIH") == "government"

    def test_us_fed(self):
        assert normalize_sponsor_type("U.S. Fed") == "government"

    def test_other(self):
        assert normalize_sponsor_type("Other") == "other"

    def test_none(self):
        assert normalize_sponsor_type(None) == "other"

    def test_unknown_value(self):
        assert normalize_sponsor_type("BigPharma Inc") == "other"


class TestNormalizeInterventionType:
    def test_drug(self):
        assert normalize_intervention_type("Drug") == "drug"

    def test_biological(self):
        assert normalize_intervention_type("Biological") == "biologic"

    def test_device(self):
        assert normalize_intervention_type("Device") == "device"

    def test_combination(self):
        assert normalize_intervention_type("Combination Product") == "combination"

    def test_dietary(self):
        assert normalize_intervention_type("Dietary Supplement") == "dietary"

    def test_none(self):
        assert normalize_intervention_type(None) == "other"

    def test_unknown(self):
        assert normalize_intervention_type("Quantum Therapy") == "other"


class TestParseAactDate:
    def test_month_year(self):
        assert parse_aact_date("January 2020") == "2020-01-01"

    def test_full_date(self):
        assert parse_aact_date("March 15, 2021") == "2021-03-15"

    def test_iso_format(self):
        assert parse_aact_date("2022-05-01") == "2022-05-01"

    def test_none(self):
        assert parse_aact_date(None) is None

    def test_empty(self):
        assert parse_aact_date("") is None

    def test_nan_float(self):
        assert parse_aact_date(float("nan")) is None

    def test_december(self):
        assert parse_aact_date("December 2023") == "2023-12-01"


# ============================================================
# PREPARE TESTS
# ============================================================


class TestPrepareInterventions:
    def test_deduplicates_by_name_and_type(self):
        df = pd.DataFrame({
            "nct_id": ["NCT001", "NCT002", "NCT003"],
            "intervention_type": ["Drug", "Drug", "Biological"],
            "name": ["Aspirin", "aspirin", "Aspirin"],
            "description": ["desc1", "desc2", "desc3"],
        })
        browse = pd.DataFrame(columns=["nct_id", "mesh_term"])
        result = prepare_interventions(df, browse)
        # "Aspirin" + Drug should dedup, but "Aspirin" + Biological is different
        assert len(result) == 2

    def test_skips_nan_names(self):
        df = pd.DataFrame({
            "nct_id": ["NCT001", "NCT002"],
            "intervention_type": ["Drug", "Drug"],
            "name": ["Aspirin", float("nan")],
            "description": ["d1", "d2"],
        })
        browse = pd.DataFrame(columns=["nct_id", "mesh_term"])
        result = prepare_interventions(df, browse)
        assert len(result) == 1

    def test_enriches_with_mesh(self):
        df = pd.DataFrame({
            "nct_id": ["NCT001"],
            "intervention_type": ["Drug"],
            "name": ["Aspirin"],
            "description": ["d1"],
        })
        browse = pd.DataFrame({
            "nct_id": ["NCT001"],
            "mesh_term": ["Aspirin"],
        })
        result = prepare_interventions(df, browse)
        assert result[0]["mesh_id"] == "Aspirin"


class TestPrepareConditions:
    def test_deduplicates_by_name(self):
        df = pd.DataFrame({
            "nct_id": ["NCT001", "NCT002"],
            "name": ["Diabetes", "diabetes"],
        })
        browse = pd.DataFrame(columns=["nct_id", "mesh_term"])
        result = prepare_conditions(df, browse)
        assert len(result) == 1


# ============================================================
# LOAD TESTS (with DB)
# ============================================================


class TestInsertInterventions:
    def test_basic_insert(self, ct_db):
        conn = get_connection(ct_db)
        try:
            items = [
                {"intervention_type": "drug", "intervention_name": "Imatinib", "mesh_id": None},
                {"intervention_type": "biologic", "intervention_name": "Trastuzumab", "mesh_id": "D00123"},
            ]
            name_to_id = insert_interventions(conn, items)
            assert len(name_to_id) == 2
            assert "imatinib" in name_to_id
            assert "trastuzumab" in name_to_id
        finally:
            conn.close()

    def test_dedup_same_name(self, ct_db):
        conn = get_connection(ct_db)
        try:
            items = [
                {"intervention_type": "drug", "intervention_name": "Aspirin", "mesh_id": None},
                {"intervention_type": "drug", "intervention_name": "Aspirin", "mesh_id": None},
            ]
            name_to_id = insert_interventions(conn, items)
            assert len(name_to_id) == 1
        finally:
            conn.close()


class TestInsertConditions:
    def test_basic_insert(self, ct_db):
        conn = get_connection(ct_db)
        try:
            items = [
                {"condition_name": "Diabetes", "mesh_id": None},
                {"condition_name": "Cancer", "mesh_id": "D009369"},
            ]
            name_to_id = insert_conditions(conn, items)
            assert len(name_to_id) == 2
            assert "diabetes" in name_to_id
        finally:
            conn.close()


class TestInsertTrials:
    def test_basic_insert(self, ct_db):
        conn = get_connection(ct_db)
        try:
            trials = [
                {
                    "source_db": "clinicaltrials_gov",
                    "source_trial_id": "NCT00000001",
                    "overall_status": "Completed",
                    "trial_phase": "phase_3",
                    "study_design": None,
                    "blinding": None,
                    "randomized": 1,
                    "enrollment_actual": 500,
                    "sponsor_type": "industry",
                    "sponsor_name": "TestPharma",
                    "start_date": "2020-01-01",
                    "primary_completion_date": "2022-06-01",
                    "completion_date": "2022-12-01",
                    "why_stopped": None,
                    "has_results": 1,
                },
            ]
            nct_to_id = insert_trials(conn, trials)
            assert len(nct_to_id) == 1
            assert "NCT00000001" in nct_to_id
        finally:
            conn.close()

    def test_unique_constraint_on_nct(self, ct_db):
        """Duplicate nct_id should be ignored (INSERT OR IGNORE)."""
        conn = get_connection(ct_db)
        try:
            trial = {
                "source_db": "clinicaltrials_gov",
                "source_trial_id": "NCT00000001",
                "overall_status": "Completed",
                "trial_phase": "phase_3",
                "study_design": None, "blinding": None,
                "randomized": 0, "enrollment_actual": None,
                "sponsor_type": "other", "sponsor_name": None,
                "start_date": None, "primary_completion_date": None,
                "completion_date": None, "why_stopped": None,
                "has_results": 0,
            }
            nct_to_id1 = insert_trials(conn, [trial])
            nct_to_id2 = insert_trials(conn, [trial])
            # Should get same ID back
            assert nct_to_id1["NCT00000001"] == nct_to_id2["NCT00000001"]
        finally:
            conn.close()


class TestInsertTrialJunctions:
    def test_links_created(self, ct_db):
        conn = get_connection(ct_db)
        try:
            # Set up data
            conn.execute(
                "INSERT INTO interventions (intervention_type, intervention_name) "
                "VALUES ('drug', 'DrugA')"
            )
            conn.execute(
                "INSERT INTO conditions (condition_name) VALUES ('DiseaseX')"
            )
            conn.execute(
                "INSERT INTO clinical_trials "
                "(source_db, source_trial_id, overall_status) "
                "VALUES ('clinicaltrials_gov', 'NCT001', 'Completed')"
            )
            conn.commit()

            raw_interv = pd.DataFrame({
                "nct_id": ["NCT001"],
                "name": ["DrugA"],
                "intervention_type": ["Drug"],
            })
            raw_cond = pd.DataFrame({
                "nct_id": ["NCT001"],
                "name": ["DiseaseX"],
            })

            n_ti, n_tc = insert_trial_junctions(
                conn, raw_interv, raw_cond,
                nct_to_trial_id={"NCT001": 1},
                name_to_intervention_id={"druga": 1},
                name_to_condition_id={"diseasex": 1},
            )
            assert n_ti == 1
            assert n_tc == 1
        finally:
            conn.close()


class TestLoadAactTable:
    def test_loads_pipe_delimited(self, tmp_path):
        """Test loading a pipe-delimited file."""
        test_file = tmp_path / "test_table.txt"
        test_file.write_text("col1|col2|col3\nA|B|C\nD|E|F\n")
        df = load_aact_table(tmp_path, "test_table")
        assert len(df) == 2
        assert list(df.columns) == ["col1", "col2", "col3"]

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_aact_table(tmp_path, "nonexistent")
