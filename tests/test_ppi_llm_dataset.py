"""Tests for PPI LLM dataset utilities (src/negbiodb_ppi/llm_dataset.py)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from negbiodb_ppi.llm_dataset import (
    DETECTION_METHOD_DESCRIPTIONS,
    JSONL_SCHEMA_FIELDS,
    MAX_PER_PROTEIN,
    SOURCE_TO_L1_CATEGORY,
    apply_max_per_protein,
    assign_splits,
    construct_evidence_description,
    construct_l3_context,
    construct_l4_context,
    read_jsonl,
    write_jsonl,
)


# ── SOURCE_TO_L1_CATEGORY Tests ──────────────────────────────────────────


class TestSourceToL1Category:
    def test_intact_maps_to_a(self):
        assert SOURCE_TO_L1_CATEGORY["intact_gold"] == "A"
        assert SOURCE_TO_L1_CATEGORY["intact_silver"] == "A"

    def test_huri_maps_to_b(self):
        assert SOURCE_TO_L1_CATEGORY["huri"] == "B"

    def test_humap_maps_to_c(self):
        assert SOURCE_TO_L1_CATEGORY["humap"] == "C"

    def test_string_maps_to_d(self):
        assert SOURCE_TO_L1_CATEGORY["string"] == "D"

    def test_four_categories(self):
        assert set(SOURCE_TO_L1_CATEGORY.values()) == {"A", "B", "C", "D"}


# ── construct_evidence_description Tests ─────────────────────────────────


class TestConstructEvidenceDescription:
    def test_intact_easy(self):
        rec = {"source_db": "intact", "detection_method": "co-immunoprecipitation",
               "gene_symbol_1": "TP53", "gene_symbol_2": "CDK2"}
        desc = construct_evidence_description(rec, "easy")
        assert "co-immunoprecipitation" in desc
        assert "TP53" in desc
        assert "CDK2" in desc
        assert "No physical interaction" in desc

    def test_huri_easy(self):
        rec = {"source_db": "huri", "gene_symbol_1": "TP53", "gene_symbol_2": "CDK2"}
        desc = construct_evidence_description(rec, "easy")
        assert "yeast two-hybrid" in desc.lower() or "Y2H" in desc

    def test_humap_easy(self):
        rec = {"source_db": "humap", "gene_symbol_1": "X", "gene_symbol_2": "Y"}
        desc = construct_evidence_description(rec, "easy")
        assert "machine learning" in desc.lower() or "co-fractionation" in desc.lower()

    def test_string_easy(self):
        rec = {"source_db": "string", "gene_symbol_1": "X", "gene_symbol_2": "Y"}
        desc = construct_evidence_description(rec, "easy")
        assert "evidence channels" in desc.lower() or "score" in desc.lower()

    def test_difficulty_changes_wording(self):
        rec = {"source_db": "intact", "detection_method": "co-immunoprecipitation",
               "gene_symbol_1": "X", "gene_symbol_2": "Y"}
        easy = construct_evidence_description(rec, "easy")
        hard = construct_evidence_description(rec, "hard")
        assert easy != hard  # Different wording for different difficulties


# ── construct_l3_context Tests ───────────────────────────────────────────


class TestConstructL3Context:
    def test_includes_both_proteins(self):
        rec = {
            "gene_symbol_1": "TP53", "uniprot_1": "P04637", "seq_len_1": 393,
            "function_1": "Tumor suppressor", "location_1": "Nucleus",
            "domains_1": "p53 domain",
            "gene_symbol_2": "INS", "uniprot_2": "P01308", "seq_len_2": 110,
            "function_2": "Insulin", "location_2": "Extracellular",
            "domains_2": "Insulin domain",
            "detection_method": "co-immunoprecipitation",
        }
        ctx = construct_l3_context(rec)
        assert "TP53" in ctx
        assert "INS" in ctx
        assert "P04637" in ctx
        assert "393" in ctx
        assert "Tumor suppressor" in ctx
        assert "Insulin" in ctx
        assert "co-immunoprecipitation" in ctx

    def test_handles_missing_fields(self):
        rec = {"gene_symbol_1": "X", "gene_symbol_2": "Y"}
        ctx = construct_l3_context(rec)
        assert "X" in ctx
        assert "Y" in ctx


# ── construct_l4_context Tests ───────────────────────────────────────────


class TestConstructL4Context:
    def test_minimal_context(self):
        rec = {"gene_symbol_1": "TP53", "gene_symbol_2": "BRCA1"}
        ctx = construct_l4_context(rec)
        assert "TP53" in ctx
        assert "BRCA1" in ctx
        assert "Homo sapiens" in ctx
        assert "tested" in ctx.lower()


# ── apply_max_per_protein Tests ──────────────────────────────────────────


class TestApplyMaxPerProtein:
    def test_caps_at_max(self):
        df = pd.DataFrame({
            "protein_id_1": [1] * 20 + [2] * 5,
            "protein_id_2": [3] * 25,
            "value": range(25),
        })
        result = apply_max_per_protein(df, max_per_protein=10)
        # Protein 1 appears 20 times in column 1 → capped
        counts_p1 = (result["protein_id_1"] == 1).sum()
        assert counts_p1 <= 10

    def test_no_op_under_limit(self):
        df = pd.DataFrame({
            "protein_id_1": [1, 1, 2, 2, 3],
            "protein_id_2": [4, 5, 6, 7, 8],
            "value": range(5),
        })
        result = apply_max_per_protein(df, max_per_protein=10)
        assert len(result) == 5

    def test_reproducibility(self):
        df = pd.DataFrame({
            "protein_id_1": [1] * 50,
            "protein_id_2": [2] * 50,
            "value": range(50),
        })
        r1 = apply_max_per_protein(df, max_per_protein=10, rng=np.random.RandomState(42))
        r2 = apply_max_per_protein(df, max_per_protein=10, rng=np.random.RandomState(42))
        pd.testing.assert_frame_equal(r1, r2)

    def test_default_max(self):
        assert MAX_PER_PROTEIN == 10


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
        assert len(result) == 20
        assert (result["split"] == "fewshot").sum() == 10
        assert (result["split"] == "val").sum() == 5
        assert (result["split"] == "test").sum() == 5


# ── JSONL I/O Tests ──────────────────────────────────────────────────────


class TestJSONLIO:
    def test_write_read_roundtrip(self, tmp_path):
        records = [
            {"question_id": "PPIL1-001", "task": "ppi-l1", "gold_answer": "A"},
            {"question_id": "PPIL1-002", "task": "ppi-l1", "gold_answer": "B"},
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
        records = [{"name": "α-synuclein", "function": "Sjögren's protein"}]
        path = tmp_path / "unicode.jsonl"
        write_jsonl(records, path)
        loaded = read_jsonl(path)
        assert loaded[0]["name"] == "α-synuclein"
        assert loaded[0]["function"] == "Sjögren's protein"


# ── Schema Field Tests ───────────────────────────────────────────────────


class TestSchemaFields:
    def test_required_fields_present(self):
        for field in ["question_id", "task", "split", "gold_answer", "context_text"]:
            assert field in JSONL_SCHEMA_FIELDS

    def test_gold_answer_not_correct_answer(self):
        """PPI uses gold_answer (consistent with CT domain)."""
        assert "gold_answer" in JSONL_SCHEMA_FIELDS
        assert "correct_answer" not in JSONL_SCHEMA_FIELDS


# ── Detection Method Descriptions Tests ──────────────────────────────────


class TestDetectionMethodDescriptions:
    def test_has_common_methods(self):
        assert "co-immunoprecipitation" in DETECTION_METHOD_DESCRIPTIONS
        assert "pull down" in DETECTION_METHOD_DESCRIPTIONS
        assert "two hybrid" in DETECTION_METHOD_DESCRIPTIONS

    def test_descriptions_are_readable(self):
        for method, desc in DETECTION_METHOD_DESCRIPTIONS.items():
            assert len(desc) > 5, f"Description for {method} too short"
