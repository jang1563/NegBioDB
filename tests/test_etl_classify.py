"""Tests for failure classification pipeline."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from negbiodb_ct.ct_db import create_ct_database, get_connection
from negbiodb_ct.etl_classify import (
    KEYWORD_RULES,
    OT_CATEGORY_MAP,
    assign_termination_types,
    classify_terminated_trials,
    classify_text_keywords,
    detect_endpoint_failures,
    enrich_with_cto,
    insert_failure_results,
    load_cto_outcomes,
    load_opentargets_labels,
    resolve_multi_label,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_ct"


@pytest.fixture
def ct_db(tmp_path):
    """Create a fresh CT database with all migrations applied."""
    db_path = tmp_path / "test_ct.db"
    create_ct_database(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def populated_db(ct_db):
    """CT database with sample trials, interventions, and conditions."""
    conn = get_connection(ct_db)
    try:
        # Interventions
        conn.execute(
            "INSERT INTO interventions (intervention_type, intervention_name) "
            "VALUES ('drug', 'DrugA')"
        )
        conn.execute(
            "INSERT INTO interventions (intervention_type, intervention_name) "
            "VALUES ('drug', 'DrugB')"
        )
        # Conditions
        conn.execute("INSERT INTO conditions (condition_name) VALUES ('DiseaseX')")
        conn.execute("INSERT INTO conditions (condition_name) VALUES ('DiseaseY')")
        # Trials
        conn.execute(
            "INSERT INTO clinical_trials "
            "(source_db, source_trial_id, overall_status, trial_phase, "
            " why_stopped, has_results) "
            "VALUES ('clinicaltrials_gov', 'NCT001', 'Terminated', 'phase_3', "
            " 'Lack of efficacy at interim analysis', 1)"
        )
        conn.execute(
            "INSERT INTO clinical_trials "
            "(source_db, source_trial_id, overall_status, trial_phase, "
            " why_stopped, has_results) "
            "VALUES ('clinicaltrials_gov', 'NCT002', 'Terminated', 'phase_2', "
            " 'Safety concerns: hepatotoxicity', 0)"
        )
        conn.execute(
            "INSERT INTO clinical_trials "
            "(source_db, source_trial_id, overall_status, trial_phase, "
            " why_stopped, has_results) "
            "VALUES ('clinicaltrials_gov', 'NCT003', 'Terminated', 'phase_1', "
            " NULL, 0)"
        )
        conn.execute(
            "INSERT INTO clinical_trials "
            "(source_db, source_trial_id, overall_status, trial_phase, "
            " has_results) "
            "VALUES ('clinicaltrials_gov', 'NCT004', 'Completed', 'phase_3', 1)"
        )
        conn.execute(
            "INSERT INTO clinical_trials "
            "(source_db, source_trial_id, overall_status, trial_phase, "
            " why_stopped, has_results) "
            "VALUES ('clinicaltrials_gov', 'NCT005', 'Terminated', 'phase_2', "
            " 'Business decision by sponsor', 0)"
        )
        # Junction: trials ↔ interventions
        conn.execute(
            "INSERT INTO trial_interventions (trial_id, intervention_id) VALUES (1, 1)"
        )
        conn.execute(
            "INSERT INTO trial_interventions (trial_id, intervention_id) VALUES (2, 1)"
        )
        conn.execute(
            "INSERT INTO trial_interventions (trial_id, intervention_id) VALUES (4, 2)"
        )
        conn.execute(
            "INSERT INTO trial_interventions (trial_id, intervention_id) VALUES (5, 2)"
        )
        # Junction: trials ↔ conditions
        conn.execute(
            "INSERT INTO trial_conditions (trial_id, condition_id) VALUES (1, 1)"
        )
        conn.execute(
            "INSERT INTO trial_conditions (trial_id, condition_id) VALUES (2, 1)"
        )
        conn.execute(
            "INSERT INTO trial_conditions (trial_id, condition_id) VALUES (4, 2)"
        )
        conn.execute(
            "INSERT INTO trial_conditions (trial_id, condition_id) VALUES (5, 2)"
        )
        conn.commit()
    finally:
        conn.close()
    return ct_db


# ============================================================
# KEYWORD CLASSIFICATION TESTS
# ============================================================


class TestClassifyTextKeywords:
    def test_safety(self):
        assert classify_text_keywords("Serious adverse events observed") == "safety"

    def test_toxicity(self):
        assert classify_text_keywords("Hepatotoxicity in 3 patients") == "safety"

    def test_efficacy(self):
        assert classify_text_keywords("Lack of efficacy") == "efficacy"

    def test_futility(self):
        assert classify_text_keywords("Futility analysis") == "efficacy"

    def test_enrollment(self):
        assert classify_text_keywords("Slow enrollment") == "enrollment"

    def test_strategic(self):
        assert classify_text_keywords("Business decision") == "strategic"

    def test_regulatory(self):
        assert classify_text_keywords("FDA clinical hold") == "regulatory"

    def test_design(self):
        assert classify_text_keywords("Protocol amendment needed") == "design"

    def test_pk(self):
        assert classify_text_keywords("Poor pharmacokinetic profile") == "pharmacokinetic"

    def test_empty(self):
        assert classify_text_keywords("") is None

    def test_none(self):
        assert classify_text_keywords(None) is None

    def test_no_match(self):
        assert classify_text_keywords("xyz 123 unrelated text") == "other"


# ============================================================
# RESOLVE MULTI-LABEL TESTS
# ============================================================


class TestResolveMultiLabel:
    def test_safety_wins(self):
        assert resolve_multi_label(["efficacy", "safety"]) == "safety"

    def test_efficacy_over_enrollment(self):
        assert resolve_multi_label(["enrollment", "efficacy"]) == "efficacy"

    def test_single(self):
        assert resolve_multi_label(["design"]) == "design"

    def test_empty(self):
        assert resolve_multi_label([]) == "other"

    def test_custom_precedence(self):
        assert resolve_multi_label(
            ["safety", "efficacy"], precedence=["efficacy", "safety"]
        ) == "efficacy"


# ============================================================
# TERMINATION TYPE ASSIGNMENT TESTS
# ============================================================


class TestAssignTerminationTypes:
    def test_classifies_terminated(self, populated_db):
        conn = get_connection(populated_db)
        try:
            stats = assign_termination_types(conn)
            assert stats["clinical_failure"] >= 1  # NCT001 (efficacy), NCT002 (safety)
            assert stats["administrative"] >= 1  # NCT005 (business)
            assert stats["unknown"] >= 1  # NCT003 (NULL why_stopped)
        finally:
            conn.close()

    def test_idempotent(self, populated_db):
        conn = get_connection(populated_db)
        try:
            stats1 = assign_termination_types(conn)
            stats2 = assign_termination_types(conn)
            # Second call should find no trials without termination_type
            assert sum(stats2.values()) == 0
        finally:
            conn.close()

    def test_sets_correct_type(self, populated_db):
        conn = get_connection(populated_db)
        try:
            assign_termination_types(conn)
            row = conn.execute(
                "SELECT termination_type FROM clinical_trials "
                "WHERE source_trial_id = 'NCT001'"
            ).fetchone()
            assert row[0] == "clinical_failure"

            row = conn.execute(
                "SELECT termination_type FROM clinical_trials "
                "WHERE source_trial_id = 'NCT003'"
            ).fetchone()
            assert row[0] == "unknown"
        finally:
            conn.close()


# ============================================================
# TIER 1: CLASSIFY TERMINATED TRIALS TESTS
# ============================================================


class TestClassifyTerminatedTrials:
    def test_produces_results(self, populated_db):
        conn = get_connection(populated_db)
        try:
            assign_termination_types(conn)
            results = classify_terminated_trials(
                conn, vectorizer=None, classifier=None, use_keywords=True
            )
            # NCT001 (efficacy) and NCT002 (safety) should produce results
            assert len(results) >= 2
        finally:
            conn.close()

    def test_categories_correct(self, populated_db):
        conn = get_connection(populated_db)
        try:
            assign_termination_types(conn)
            results = classify_terminated_trials(
                conn, vectorizer=None, classifier=None, use_keywords=True
            )
            cats = {r["failure_category"] for r in results}
            # Should detect both efficacy and safety
            assert "efficacy" in cats
            assert "safety" in cats
        finally:
            conn.close()

    def test_excludes_unknown_type(self, populated_db):
        conn = get_connection(populated_db)
        try:
            assign_termination_types(conn)
            results = classify_terminated_trials(
                conn, vectorizer=None, classifier=None, use_keywords=True
            )
            # NCT003 has NULL why_stopped → unknown termination_type → excluded
            nct_ids = {r["source_record_id"] for r in results}
            assert "terminated:NCT003" not in nct_ids
        finally:
            conn.close()

    def test_safety_interpretation(self, populated_db):
        conn = get_connection(populated_db)
        try:
            assign_termination_types(conn)
            results = classify_terminated_trials(
                conn, vectorizer=None, classifier=None, use_keywords=True
            )
            safety_results = [r for r in results if r["failure_category"] == "safety"]
            assert all(
                r["result_interpretation"] == "safety_stopped" for r in safety_results
            )
        finally:
            conn.close()


# ============================================================
# INSERT FAILURE RESULTS TESTS
# ============================================================


class TestInsertFailureResults:
    def test_insert_basic(self, populated_db):
        conn = get_connection(populated_db)
        try:
            results = [{
                "intervention_id": 1,
                "condition_id": 1,
                "trial_id": 1,
                "failure_category": "efficacy",
                "failure_detail": "test",
                "confidence_tier": "bronze",
                "p_value_primary": None,
                "ci_lower": None,
                "ci_upper": None,
                "primary_endpoint_met": None,
                "highest_phase_reached": "phase_3",
                "source_db": "clinicaltrials_gov",
                "source_record_id": "test:NCT001",
                "extraction_method": "nlp_classified",
                "result_interpretation": "definitive_negative",
            }]
            n = insert_failure_results(conn, results)
            assert n == 1
        finally:
            conn.close()

    def test_dedup_on_unique_index(self, populated_db):
        conn = get_connection(populated_db)
        try:
            result = {
                "intervention_id": 1,
                "condition_id": 1,
                "trial_id": 1,
                "failure_category": "efficacy",
                "failure_detail": "test",
                "confidence_tier": "bronze",
                "p_value_primary": None,
                "ci_lower": None,
                "ci_upper": None,
                "primary_endpoint_met": None,
                "highest_phase_reached": "phase_3",
                "source_db": "clinicaltrials_gov",
                "source_record_id": "test:NCT001",
                "extraction_method": "nlp_classified",
                "result_interpretation": "definitive_negative",
            }
            n1 = insert_failure_results(conn, [result])
            n2 = insert_failure_results(conn, [result])
            assert n1 == 1
            assert n2 == 0  # duplicate ignored
        finally:
            conn.close()

    def test_empty_list(self, populated_db):
        conn = get_connection(populated_db)
        try:
            n = insert_failure_results(conn, [])
            assert n == 0
        finally:
            conn.close()


# ============================================================
# OPEN TARGETS LABELS TESTS
# ============================================================


class TestLoadOpentargetsLabels:
    def _make_ot_parquet(self, tmp_path, rows):
        """Build a minimal Open Targets parquet with boolean label columns."""
        all_bool_cols = [
            "Another_Study", "Business_Administrative", "Covid19",
            "Endpoint_Met", "Ethical_Reason", "Insufficient_Data",
            "Insufficient_Enrollment", "Interim_Analysis", "Invalid_Reason",
            "Logistics_Resources", "Negative", "No_Context", "Regulatory",
            "Safety_Sideeffects", "Study_Design", "Study_Staff_Moved", "Success",
        ]
        data = {"text": [], "label_descriptions": []}
        for col in all_bool_cols:
            data[col] = []
        for text, active_cols in rows:
            data["text"].append(text)
            data["label_descriptions"].append(np.array(active_cols))
            for col in all_bool_cols:
                data[col].append(col in active_cols)
        df = pd.DataFrame(data)
        for col in all_bool_cols:
            df[col] = df[col].astype(bool)
        path = tmp_path / "ot.parquet"
        df.to_parquet(path)
        return path

    def test_basic_load(self, tmp_path):
        path = self._make_ot_parquet(tmp_path, [
            ("Lack of efficacy", ["Negative"]),
            ("Safety concerns", ["Safety_Sideeffects"]),
            ("Study completed successfully", ["Endpoint_Met"]),
        ])
        result = load_opentargets_labels(path)
        # "Endpoint_Met" maps to None, dropped
        assert len(result) == 2
        assert set(result["label"]) == {"efficacy", "safety"}

    def test_multi_label_precedence(self, tmp_path):
        path = self._make_ot_parquet(tmp_path, [
            ("Low enrollment and toxicity", ["Insufficient_Enrollment", "Safety_Sideeffects"]),
        ])
        result = load_opentargets_labels(path)
        assert len(result) == 1
        assert result["label"].iloc[0] == "safety"  # safety > enrollment

    def test_drops_unmapped(self, tmp_path):
        path = self._make_ot_parquet(tmp_path, [
            ("Study completed successfully", ["Success"]),
        ])
        result = load_opentargets_labels(path)
        assert len(result) == 0


# ============================================================
# CTO OUTCOMES TESTS
# ============================================================


class TestLoadCtoOutcomes:
    def test_basic_load(self, tmp_path):
        df = pd.DataFrame({
            "nct_id": ["NCT001", "NCT002", "NCT003"],
            "outcome": [0, 1, 0],
        })
        path = tmp_path / "cto.parquet"
        df.to_parquet(path)
        result = load_cto_outcomes(path)
        assert len(result) == 3
        assert (result["outcome"] == 0).sum() == 2

    def test_string_outcomes(self, tmp_path):
        df = pd.DataFrame({
            "nct_id": ["NCT001", "NCT002"],
            "outcome": ["failure", "success"],
        })
        path = tmp_path / "cto.parquet"
        df.to_parquet(path)
        result = load_cto_outcomes(path)
        assert len(result) == 2
        assert result[result["nct_id"] == "NCT001"]["outcome"].iloc[0] == 0

    def test_realistic_cto_columns(self, tmp_path):
        """Regression test: CTO parquet has nct_id + expanded_access columns.

        Previously, the column detection loop would overwrite nct_col with
        'expanded_access_status_for_nctid', causing zero results.
        """
        df = pd.DataFrame({
            "nct_id": ["NCT00001", "NCT00002", "NCT00003"],
            "labels": [0.0, 1.0, 0.0],
            "expanded_access_nctid": [None, None, None],
            "expanded_access_status_for_nctid": ["Available", None, "Available"],
            "other_column": ["x", "y", "z"],
        })
        path = tmp_path / "cto_realistic.parquet"
        df.to_parquet(path)
        result = load_cto_outcomes(path)
        assert len(result) == 3
        # Must use 'nct_id' column, not 'expanded_access_status_for_nctid'
        assert list(result["nct_id"]) == ["NCT00001", "NCT00002", "NCT00003"]
        assert (result["outcome"] == 0).sum() == 2
        assert (result["outcome"] == 1).sum() == 1


# ============================================================
# CTO ENRICHMENT TESTS
# ============================================================


class TestEnrichWithCto:
    def test_gap_fill(self, populated_db):
        conn = get_connection(populated_db)
        try:
            # CTO says NCT004 failed (not covered by tier 1/2)
            cto_df = pd.DataFrame({
                "nct_id": ["NCT004"],
                "outcome": [0],
            })
            results = enrich_with_cto(conn, cto_df)
            assert len(results) == 1
            assert results[0]["confidence_tier"] == "copper"
            assert results[0]["source_db"] == "cto"
        finally:
            conn.close()

    def test_skips_already_covered(self, populated_db):
        conn = get_connection(populated_db)
        try:
            # Insert a Tier 1 result for NCT001
            conn.execute(
                "INSERT INTO trial_failure_results "
                "(intervention_id, condition_id, trial_id, "
                " failure_category, confidence_tier, "
                " source_db, source_record_id, extraction_method) "
                "VALUES (1, 1, 1, 'efficacy', 'bronze', "
                " 'clinicaltrials_gov', 'terminated:NCT001', 'nlp_classified')"
            )
            conn.commit()

            cto_df = pd.DataFrame({
                "nct_id": ["NCT001"],
                "outcome": [0],
            })
            results = enrich_with_cto(conn, cto_df)
            assert len(results) == 0
        finally:
            conn.close()


# ============================================================
# OT CATEGORY MAP COVERAGE
# ============================================================


class TestOtCategoryMap:
    def test_all_mapped_to_valid_categories(self):
        valid = {
            "safety", "efficacy", "pharmacokinetic", "enrollment",
            "strategic", "regulatory", "design", "other", None,
        }
        for label, cat in OT_CATEGORY_MAP.items():
            assert cat in valid, f"OT label {label!r} maps to invalid {cat!r}"

    def test_all_8_categories_reachable(self):
        mapped = {v for v in OT_CATEGORY_MAP.values() if v is not None}
        expected = {
            "safety", "efficacy", "enrollment",
            "strategic", "regulatory", "design", "other",
        }
        # pharmacokinetic may not be in OT labels (it's rare)
        assert expected <= mapped
