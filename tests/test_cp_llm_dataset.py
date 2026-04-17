"""Tests for CP LLM dataset helpers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from negbiodb_cp.llm_dataset import (
    apply_max_per_compound,
    assign_splits,
    construct_evidence_description,
    construct_l1_context,
    construct_l2_context,
    construct_l3_context,
    construct_l4_context,
    load_cp_annotation_summary,
    load_cp_candidate_pool,
    write_dataset_metadata,
    write_jsonl,
)
from tests.cp_test_utils import create_seeded_cp_database


def sample_row():
    return pd.Series({
        "cp_result_id": 1,
        "compound_id": 1,
        "compound_name": "CmpInactive",
        "inchikey": "TESTINCHIKEY",
        "cell_line_name": "U2OS",
        "batch_name": "B1",
        "dose": 1.0,
        "dose_unit": "uM",
        "timepoint_h": 48.0,
        "num_valid_observations": 2,
        "dmso_distance_mean": 0.023,
        "replicate_reproducibility": 0.91,
        "viability_ratio": 1.00,
        "outcome_label": "inactive",
        "confidence_tier": "gold",
        "has_orthogonal_evidence": 1,
    })


def test_construct_contexts_include_expected_fields():
    row = sample_row()
    evidence = construct_evidence_description(row)
    assert "CmpInactive" in evidence
    assert "U2OS" in evidence
    assert "0.023" in evidence

    assert "A) inactive" in construct_l1_context(row)
    assert "Structured Report" in construct_l2_context(row)
    assert "Orthogonal evidence recorded: yes" in construct_l3_context(row)
    assert "Timepoint: 48" in construct_l4_context(row)


def test_apply_max_per_compound_caps_rows():
    df = pd.DataFrame({
        "compound_id": [1] * 10 + [2] * 3,
        "value": range(13),
    })
    result = apply_max_per_compound(df, max_per_compound=4, rng=np.random.RandomState(42))
    assert len(result[result["compound_id"] == 1]) == 4
    assert len(result[result["compound_id"] == 2]) == 3


def test_assign_splits_and_writers(tmp_path):
    df = pd.DataFrame({"x": range(20)})
    split_df = assign_splits(df, fewshot_size=4, val_size=4, test_size=12, seed=42)
    assert set(split_df["split"]) <= {"fewshot", "val", "test"}

    output_path = tmp_path / "cp.jsonl"
    count = write_jsonl([{"a": 1}, {"a": 2}], output_path)
    assert count == 2
    assert json.loads(output_path.read_text().splitlines()[0])["a"] == 1

    meta_path = write_dataset_metadata(tmp_path, "cp-l1", {"n_records": 2})
    meta = json.loads(meta_path.read_text())
    assert meta["task"] == "cp-l1"
    assert meta["domain"] == "cp"


def test_proxy_db_is_blocked_unless_explicitly_allowed(tmp_path):
    db_path = create_seeded_cp_database(tmp_path, annotation_mode="plate_proxy")

    with pytest.raises(ValueError):
        load_cp_candidate_pool(Path(db_path), min_confidence="bronze")

    allowed = load_cp_candidate_pool(
        Path(db_path),
        min_confidence="bronze",
        allow_proxy_smoke=True,
    )
    assert not allowed.empty

    summary = load_cp_annotation_summary(Path(db_path))
    assert summary["production_ready"] is False
    assert "plate_proxy" in summary["annotation_modes"]
