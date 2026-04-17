"""Tests for CP ML export pipeline."""

import pandas as pd
import pytest

from negbiodb_cp.cp_db import get_connection
from negbiodb_cp.export import (
    export_cp_feature_tables,
    export_cp_m1_dataset,
    export_cp_m2_dataset,
    generate_all_splits,
    generate_batch_holdout_split,
    generate_cold_compound_split,
    generate_scaffold_split,
)
from tests.cp_test_utils import create_seeded_cp_database


@pytest.fixture
def cp_conn(tmp_path):
    db_path = create_seeded_cp_database(tmp_path)
    conn = get_connection(db_path)
    yield conn
    conn.close()


def test_generate_cold_compound_split_has_no_train_test_leakage(cp_conn):
    generate_cold_compound_split(cp_conn, seed=42)
    df = pd.read_sql_query(
        """
        SELECT r.compound_id, sa.fold
        FROM cp_split_assignments sa
        JOIN cp_split_definitions sd ON sa.split_id = sd.split_id
        JOIN cp_perturbation_results r ON sa.cp_result_id = r.cp_result_id
        WHERE sd.split_name = 'cold_compound_s42'
        """,
        cp_conn,
    )
    train = set(df[df["fold"] == "train"]["compound_id"])
    test = set(df[df["fold"] == "test"]["compound_id"])
    assert not (train & test)


def test_generate_batch_holdout_split_has_no_batch_leakage(cp_conn):
    generate_batch_holdout_split(cp_conn, seed=42)
    df = pd.read_sql_query(
        """
        SELECT r.batch_id, sa.fold
        FROM cp_split_assignments sa
        JOIN cp_split_definitions sd ON sa.split_id = sd.split_id
        JOIN cp_perturbation_results r ON sa.cp_result_id = r.cp_result_id
        WHERE sd.split_name = 'batch_holdout'
        """,
        cp_conn,
    )
    train = set(df[df["fold"] == "train"]["batch_id"])
    test = set(df[df["fold"] == "test"]["batch_id"])
    assert not (train & test)


def test_generate_scaffold_split_preserves_none_scaffold_rows(cp_conn):
    generate_scaffold_split(cp_conn, seed=42)
    df = pd.read_sql_query(
        """
        SELECT c.compound_name, sa.fold
        FROM cp_split_assignments sa
        JOIN cp_split_definitions sd ON sa.split_id = sd.split_id
        JOIN cp_perturbation_results r ON sa.cp_result_id = r.cp_result_id
        JOIN compounds c ON r.compound_id = c.compound_id
        WHERE sd.split_name = 'scaffold_s42'
        """,
        cp_conn,
    )
    assert "Mystery" in set(df["compound_name"])


def test_export_datasets_and_features(cp_conn, tmp_path):
    generate_all_splits(cp_conn, seed=42)
    m1_path, n_m1 = export_cp_m1_dataset(cp_conn, tmp_path)
    m2_path, n_m2 = export_cp_m2_dataset(cp_conn, tmp_path)
    feature_paths = export_cp_feature_tables(cp_conn, tmp_path)

    m1 = pd.read_parquet(m1_path)
    m2 = pd.read_parquet(m2_path)
    profile = pd.read_parquet(feature_paths["profile"])
    image = pd.read_parquet(feature_paths["image"])

    assert n_m1 == len(m1) == 6
    assert n_m2 == len(m2) == 6
    assert "Y" in m1.columns
    assert "outcome_label" in m2.columns
    assert "split_random_s42" in m1.columns
    assert "split_cold_compound_s42" in m1.columns
    assert "split_scaffold_s42" in m1.columns
    assert "split_batch_holdout" in m1.columns
    assert "cp_result_id" in profile.columns
    assert "cp_result_id" in image.columns
