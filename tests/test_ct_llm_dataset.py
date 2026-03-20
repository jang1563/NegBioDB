"""Tests for CT LLM dataset utilities (src/negbiodb_ct/llm_dataset.py)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from negbiodb_ct.llm_dataset import (
    JSONL_SCHEMA_FIELDS,
    MAX_PER_DRUG,
    THERAPEUTIC_AREA_KEYWORDS,
    apply_max_per_drug,
    assign_splits,
    infer_therapeutic_area,
    is_code_name,
    read_jsonl,
    write_jsonl,
)


# ── infer_therapeutic_area Tests ─────────────────────────────────────────


class TestInferTherapeuticArea:
    def test_oncology(self):
        assert infer_therapeutic_area("Breast Cancer") == "oncology"
        assert infer_therapeutic_area("Non-Small Cell Lung Carcinoma") == "oncology"
        assert infer_therapeutic_area("Acute Myeloid Leukemia") == "oncology"

    def test_cardiology(self):
        assert infer_therapeutic_area("Hypertension") == "cardiology"
        assert infer_therapeutic_area("Coronary Artery Disease") == "cardiology"

    def test_neurology(self):
        assert infer_therapeutic_area("Alzheimer's Disease") == "neurology"
        assert infer_therapeutic_area("Parkinson's Disease") == "neurology"

    def test_other_fallback(self):
        assert infer_therapeutic_area("Acne Vulgaris") == "other"
        assert infer_therapeutic_area("") == "other"
        assert infer_therapeutic_area("Rare Disease XYZ") == "other"

    def test_case_insensitive(self):
        assert infer_therapeutic_area("BREAST CANCER") == "oncology"
        assert infer_therapeutic_area("hypertension") == "cardiology"

    def test_all_areas_have_keywords(self):
        for area, kws in THERAPEUTIC_AREA_KEYWORDS.items():
            assert len(kws) > 0, f"Area {area} has no keywords"


# ── is_code_name Tests ───────────────────────────────────────────────────


class TestIsCodeName:
    def test_typical_code_names(self):
        assert is_code_name("BMS-123456") is True
        assert is_code_name("ABT-737") is True
        assert is_code_name("GSK-12345") is True

    def test_real_drug_names(self):
        assert is_code_name("Imatinib") is False
        assert is_code_name("Aspirin") is False
        assert is_code_name("Trastuzumab") is False

    def test_edge_cases(self):
        assert is_code_name("A-123") is False  # Only 1 letter
        assert is_code_name("ABCDEF-123") is False  # > 5 letters


# ── apply_max_per_drug Tests ─────────────────────────────────────────────


class TestApplyMaxPerDrug:
    def test_caps_at_max(self):
        df = pd.DataFrame({
            "intervention_id": [1] * 20 + [2] * 5,
            "value": range(25),
        })
        result = apply_max_per_drug(df, max_per_drug=10)
        counts = result["intervention_id"].value_counts()
        assert counts[1] == 10
        assert counts[2] == 5

    def test_no_op_under_limit(self):
        df = pd.DataFrame({
            "intervention_id": [1, 1, 2, 2, 3],
            "value": range(5),
        })
        result = apply_max_per_drug(df, max_per_drug=10)
        assert len(result) == 5

    def test_reproducibility(self):
        df = pd.DataFrame({
            "intervention_id": [1] * 50,
            "value": range(50),
        })
        r1 = apply_max_per_drug(df, max_per_drug=10, rng=np.random.RandomState(42))
        r2 = apply_max_per_drug(df, max_per_drug=10, rng=np.random.RandomState(42))
        pd.testing.assert_frame_equal(r1, r2)

    def test_default_max(self):
        assert MAX_PER_DRUG == 10


# ── assign_splits Tests ──────────────────────────────────────────────────


class TestAssignSplits:
    def test_correct_split_sizes(self):
        df = pd.DataFrame({"x": range(100)})
        result = assign_splits(df, fewshot_size=10, val_size=20, test_size=70, seed=42)
        assert (result["split"] == "fewshot").sum() == 10
        assert (result["split"] == "val").sum() == 20
        assert (result["split"] == "test").sum() == 70
        assert len(result) == 100

    def test_reproducibility(self):
        df = pd.DataFrame({"x": range(50)})
        r1 = assign_splits(df, 5, 10, 35, seed=42)
        r2 = assign_splits(df, 5, 10, 35, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_undersized_dataset(self):
        df = pd.DataFrame({"x": range(20)})
        result = assign_splits(df, fewshot_size=10, val_size=5, test_size=100, seed=42)
        # Should adjust test_size to 5 (20 - 10 - 5)
        assert len(result) == 20
        assert (result["split"] == "fewshot").sum() == 10
        assert (result["split"] == "val").sum() == 5
        assert (result["split"] == "test").sum() == 5


# ── JSONL I/O Tests ──────────────────────────────────────────────────────


class TestJSONLIO:
    def test_write_read_roundtrip(self, tmp_path):
        records = [
            {"question_id": "CTL1-001", "task": "CT-L1", "gold_answer": "A"},
            {"question_id": "CTL1-002", "task": "CT-L1", "gold_answer": "B"},
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(records, path)
        loaded = read_jsonl(path)
        assert loaded == records

    def test_empty_write(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        write_jsonl([], path)
        loaded = read_jsonl(path)
        assert loaded == []

    def test_unicode_handling(self, tmp_path):
        records = [{"name": "café", "condition": "Sjögren's syndrome"}]
        path = tmp_path / "unicode.jsonl"
        write_jsonl(records, path)
        loaded = read_jsonl(path)
        assert loaded[0]["name"] == "café"
        assert loaded[0]["condition"] == "Sjögren's syndrome"


# ── Schema Field Tests ───────────────────────────────────────────────────


class TestSchemaFields:
    def test_required_fields_present(self):
        for field in ["question_id", "task", "split", "gold_answer", "context_text"]:
            assert field in JSONL_SCHEMA_FIELDS

    def test_gold_answer_not_correct_answer(self):
        """CT uses gold_answer, NOT DTI's correct_answer."""
        assert "gold_answer" in JSONL_SCHEMA_FIELDS
        assert "correct_answer" not in JSONL_SCHEMA_FIELDS
