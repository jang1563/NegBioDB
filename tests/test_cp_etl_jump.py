"""Tests for CP ETL and consensus labeling."""

from pathlib import Path

import pandas as pd
import pytest

from negbiodb_cp.cp_db import get_connection, run_cp_migrations
from negbiodb_cp.etl_jump import (
    assign_confidence_tier,
    assign_outcome_label,
    ingest_jump_tables,
    normalize_observations,
)
from tests.cp_test_utils import (
    CP_MIGRATIONS_DIR,
    build_cp_feature_df,
    build_cp_observations,
    build_cp_orthogonal_evidence_df,
)


def test_normalize_observations_requires_columns():
    with pytest.raises(ValueError):
        normalize_observations(pd.DataFrame({"batch_name": ["B1"]}))


def test_assign_outcome_label_rules():
    assert assign_outcome_label(0.02, 0.9, 1.0, inactive_threshold=0.05) == "inactive"
    assert assign_outcome_label(0.20, 0.8, 0.2, inactive_threshold=0.05) == "toxic_or_artifact"
    assert assign_outcome_label(0.20, 0.9, 0.9, inactive_threshold=0.05) == "strong_phenotype"
    assert assign_outcome_label(0.08, 0.2, 0.9, inactive_threshold=0.05) == "weak_phenotype"


def test_assign_confidence_tier_rules():
    assert assign_confidence_tier(2, 0.1, 0.8, 1.0) == "silver"
    assert assign_confidence_tier(1, 0.1, 0.5, 1.0) == "bronze"
    assert assign_confidence_tier(1, None, 0.5, 1.0) == "copper"


def test_ingest_jump_tables_builds_results_and_features(tmp_path):
    db_path = tmp_path / "cp.db"
    run_cp_migrations(db_path, CP_MIGRATIONS_DIR)
    conn = get_connection(db_path)
    try:
        summary = ingest_jump_tables(
            conn,
            observations=build_cp_observations(),
            profile_features=build_cp_feature_df("profile"),
            image_features=build_cp_feature_df("image"),
            orthogonal_evidence=build_cp_orthogonal_evidence_df(),
        )

        results = pd.read_sql_query(
            """
            SELECT r.outcome_label, r.confidence_tier, r.has_orthogonal_evidence,
                   c.compound_name, b.batch_name
            FROM cp_perturbation_results r
            JOIN compounds c ON r.compound_id = c.compound_id
            JOIN cp_batches b ON r.batch_id = b.batch_id
            ORDER BY c.compound_name, b.batch_name
            """,
            conn,
        )
        observation_count = conn.execute(
            "SELECT COUNT(*) FROM cp_observations WHERE qc_pass = 0"
        ).fetchone()[0]
        profile_count = conn.execute("SELECT COUNT(*) FROM cp_profile_features").fetchone()[0]
        image_count = conn.execute("SELECT COUNT(*) FROM cp_image_features").fetchone()[0]
    finally:
        conn.close()

    assert summary["n_observations"] == len(build_cp_observations())
    assert summary["n_results"] == 6
    assert profile_count == 6
    assert image_count == 6
    assert observation_count == 1

    labels = dict(zip(results["compound_name"] + "::" + results["batch_name"], results["outcome_label"]))
    assert labels["Aspirin::B1"] == "inactive"
    assert labels["Aspirin::B2"] == "inactive"
    assert labels["Ibuprofen::B1"] == "strong_phenotype"
    assert labels["Caffeine::B1"] == "weak_phenotype"
    assert labels["Toxicol::B1"] == "toxic_or_artifact"

    tiers = dict(zip(results["compound_name"] + "::" + results["batch_name"], results["confidence_tier"]))
    assert tiers["Aspirin::B1"] == "gold"
    assert tiers["Aspirin::B2"] == "gold"

    orth = dict(zip(results["compound_name"] + "::" + results["batch_name"], results["has_orthogonal_evidence"]))
    assert orth["Aspirin::B1"] == 1
