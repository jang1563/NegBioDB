"""Tests for outcome enrichment pipeline."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from negbiodb_ct.ct_db import create_ct_database, get_connection
from negbiodb_ct.etl_outcomes import (
    enrich_results_with_aact,
    enrich_results_with_shi_du,
    load_shi_du_efficacy,
    load_shi_du_safety,
    upgrade_confidence_tiers,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_ct"


@pytest.fixture
def ct_db(tmp_path):
    """Create a fresh CT database with all migrations applied."""
    db_path = tmp_path / "test_ct.db"
    create_ct_database(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def enrichment_db(ct_db):
    """CT database with sample data for enrichment testing."""
    conn = get_connection(ct_db)
    try:
        # Interventions
        conn.execute(
            "INSERT INTO interventions (intervention_type, intervention_name) "
            "VALUES ('drug', 'DrugA')"
        )
        # Conditions
        conn.execute("INSERT INTO conditions (condition_name) VALUES ('DiseaseX')")
        # Trials
        conn.execute(
            "INSERT INTO clinical_trials "
            "(source_db, source_trial_id, overall_status, trial_phase, has_results) "
            "VALUES ('clinicaltrials_gov', 'NCT001', 'Completed', 'phase_3', 1)"
        )
        conn.execute(
            "INSERT INTO clinical_trials "
            "(source_db, source_trial_id, overall_status, trial_phase, has_results) "
            "VALUES ('clinicaltrials_gov', 'NCT002', 'Terminated', 'phase_2', 0)"
        )
        # Junction tables
        conn.execute(
            "INSERT INTO trial_interventions (trial_id, intervention_id) VALUES (1, 1)"
        )
        conn.execute(
            "INSERT INTO trial_conditions (trial_id, condition_id) VALUES (1, 1)"
        )
        conn.execute(
            "INSERT INTO trial_interventions (trial_id, intervention_id) VALUES (2, 1)"
        )
        conn.execute(
            "INSERT INTO trial_conditions (trial_id, condition_id) VALUES (2, 1)"
        )
        # Publications (for tier upgrade test)
        conn.execute(
            "INSERT INTO trial_publications (trial_id, pubmed_id) VALUES (1, 12345678)"
        )
        # Failure results — one bronze, one silver
        conn.execute(
            "INSERT INTO trial_failure_results "
            "(intervention_id, condition_id, trial_id, "
            " failure_category, confidence_tier, highest_phase_reached, "
            " source_db, source_record_id, extraction_method) "
            "VALUES (1, 1, 1, 'efficacy', 'bronze', 'phase_3', "
            " 'clinicaltrials_gov', 'terminated:NCT001', 'nlp_classified')"
        )
        conn.execute(
            "INSERT INTO trial_failure_results "
            "(intervention_id, condition_id, trial_id, "
            " failure_category, confidence_tier, highest_phase_reached, "
            " source_db, source_record_id, extraction_method) "
            "VALUES (1, 1, 2, 'efficacy', 'silver', 'phase_2', "
            " 'clinicaltrials_gov', 'terminated:NCT002', 'nlp_classified')"
        )
        conn.commit()
    finally:
        conn.close()
    return ct_db


# ============================================================
# AACT ENRICHMENT TESTS
# ============================================================


class TestEnrichResultsWithAact:
    def test_updates_p_value(self, enrichment_db):
        conn = get_connection(enrichment_db)
        try:
            aact_df = pd.DataFrame({
                "nct_id": ["NCT001"],
                "p_value": [0.073],
                "ci_lower_limit": [-0.5],
                "ci_upper_limit": [0.1],
                "param_value": [-0.2],
                "param_type": ["Mean Difference"],
                "method": ["ANCOVA"],
            })
            n = enrich_results_with_aact(conn, aact_df)
            assert n >= 1

            row = conn.execute(
                "SELECT p_value_primary, primary_endpoint_met "
                "FROM trial_failure_results WHERE trial_id = 1"
            ).fetchone()
            assert abs(row[0] - 0.073) < 1e-6
            assert row[1] == 0  # p > 0.05
        finally:
            conn.close()

    def test_endpoint_met_when_p_low(self, enrichment_db):
        conn = get_connection(enrichment_db)
        try:
            aact_df = pd.DataFrame({
                "nct_id": ["NCT002"],
                "p_value": [0.003],
                "ci_lower_limit": [None],
                "ci_upper_limit": [None],
                "param_value": [None],
                "param_type": [None],
                "method": ["t-test"],
            })
            enrich_results_with_aact(conn, aact_df)

            row = conn.execute(
                "SELECT primary_endpoint_met FROM trial_failure_results "
                "WHERE trial_id = 2"
            ).fetchone()
            assert row[0] == 1  # p <= 0.05
        finally:
            conn.close()

    def test_empty_df(self, enrichment_db):
        conn = get_connection(enrichment_db)
        try:
            n = enrich_results_with_aact(conn, pd.DataFrame())
            assert n == 0
        finally:
            conn.close()


# ============================================================
# SHI & DU LOAD TESTS
# ============================================================


class TestLoadShiDuEfficacy:
    def test_basic_load(self, tmp_path):
        csv = tmp_path / "efficacy.csv"
        csv.write_text("nct_id,p_value,effect_size\nNCT001,0.05,1.2\nNCT002,0.8,-0.3\n")
        result = load_shi_du_efficacy(csv)
        assert len(result) == 2
        assert "p_value" in result.columns

    def test_missing_file(self, tmp_path):
        result = load_shi_du_efficacy(tmp_path / "nonexistent.csv")
        assert result.empty


class TestLoadShiDuSafety:
    def test_basic_load(self, tmp_path):
        csv = tmp_path / "safety.csv"
        csv.write_text(
            "NCT_ID,serious/other,affected,at_risk\n"
            "NCT001,Serious,5,100\n"
            "NCT001,Other,10,100\n"
            "NCT002,Serious,3,50\n"
        )
        result = load_shi_du_safety(csv)
        # NCT001 has 5 serious affected, NCT002 has 3
        assert len(result) == 2
        assert "sae_total" in result.columns
        nct1 = result[result["nct_id"] == "NCT001"]["sae_total"].iloc[0]
        assert nct1 == 5  # Only "Serious" rows counted

    def test_missing_file(self, tmp_path):
        result = load_shi_du_safety(tmp_path / "nonexistent.csv")
        assert result.empty


# ============================================================
# SHI & DU ENRICHMENT TESTS
# ============================================================


class TestEnrichResultsWithShiDu:
    def test_updates_safety_data(self, enrichment_db):
        conn = get_connection(enrichment_db)
        try:
            safety_df = pd.DataFrame({
                "nct_id": ["NCT001"],
                "sae_total": [10],
            })
            n = enrich_results_with_shi_du(conn, safety_df)
            assert n >= 1

            row = conn.execute(
                "SELECT serious_adverse_events "
                "FROM trial_failure_results WHERE trial_id = 1"
            ).fetchone()
            assert row[0] == 10
        finally:
            conn.close()

    def test_empty_df(self, enrichment_db):
        conn = get_connection(enrichment_db)
        try:
            n = enrich_results_with_shi_du(conn, pd.DataFrame())
            assert n == 0
        finally:
            conn.close()


# ============================================================
# TIER UPGRADE TESTS
# ============================================================


class TestUpgradeConfidenceTiers:
    def test_bronze_to_silver(self, enrichment_db):
        conn = get_connection(enrichment_db)
        try:
            # Use trial_id=2 (Phase 2, no PubMed) — won't cascade to gold
            conn.execute(
                "UPDATE trial_failure_results SET "
                "confidence_tier = 'bronze', p_value_primary = 0.12 "
                "WHERE trial_id = 2"
            )
            conn.commit()

            stats = upgrade_confidence_tiers(conn)
            assert stats["bronze_to_silver"] >= 1

            row = conn.execute(
                "SELECT confidence_tier FROM trial_failure_results WHERE trial_id = 2"
            ).fetchone()
            assert row[0] == "silver"
        finally:
            conn.close()

    def test_silver_to_gold(self, enrichment_db):
        conn = get_connection(enrichment_db)
        try:
            # Set up silver Phase III with PubMed
            # trial_id=1 is already Phase III with results and has a publication
            conn.execute(
                "UPDATE trial_failure_results SET "
                "confidence_tier = 'silver', highest_phase_reached = 'phase_3' "
                "WHERE trial_id = 1"
            )
            conn.commit()

            stats = upgrade_confidence_tiers(conn)
            assert stats["silver_to_gold"] == 1

            row = conn.execute(
                "SELECT confidence_tier FROM trial_failure_results WHERE trial_id = 1"
            ).fetchone()
            assert row[0] == "gold"
        finally:
            conn.close()

    def test_no_upgrade_without_evidence(self, enrichment_db):
        conn = get_connection(enrichment_db)
        try:
            stats = upgrade_confidence_tiers(conn)
            # Bronze result has no p-value → stays bronze
            assert stats["bronze_to_silver"] == 0
        finally:
            conn.close()
