"""Tests for GE LLM dataset builder."""

import json
from pathlib import Path

import pandas as pd
import pytest

from negbiodb_depmap.llm_dataset import (
    apply_max_per_gene,
    assign_splits,
    construct_evidence_description,
    construct_l3_context,
    construct_l4_context,
    write_jsonl,
    write_dataset_metadata,
)


@pytest.fixture
def sample_row():
    return pd.Series({
        "gene_symbol": "TP53",
        "model_id": "ACH-000001",
        "ccle_name": "A549_LUNG",
        "lineage": "Lung",
        "primary_disease": "Non-Small Cell Lung Cancer",
        "gene_effect_score": 0.05,
        "dependency_probability": 0.10,
        "evidence_type": "crispr_nonessential",
        "source_db": "depmap",
        "confidence_tier": "gold",
        "is_common_essential": 0,
        "is_reference_nonessential": 1,
        "gene_degree": 1500,
    })


class TestConstructEvidence:
    def test_basic_fields(self, sample_row):
        text = construct_evidence_description(sample_row)
        assert "TP53" in text
        assert "A549_LUNG" in text
        assert "Lung" in text
        assert "CRISPR" in text

    def test_gene_effect_included(self, sample_row):
        text = construct_evidence_description(sample_row)
        assert "0.050" in text

    def test_dependency_prob_included(self, sample_row):
        text = construct_evidence_description(sample_row)
        assert "0.100" in text

    def test_rnai_evidence(self, sample_row):
        sample_row["evidence_type"] = "rnai_nonessential"
        text = construct_evidence_description(sample_row)
        assert "RNAi" in text


class TestConstructL3:
    def test_includes_gene_function(self, sample_row):
        text = construct_l3_context(sample_row, gene_description="Tumor suppressor")
        assert "Tumor suppressor" in text

    def test_includes_reference_flag(self, sample_row):
        text = construct_l3_context(sample_row)
        assert "non-essential gene set" in text

    def test_includes_degree(self, sample_row):
        text = construct_l3_context(sample_row)
        assert "1500" in text


class TestConstructL4:
    def test_minimal_context(self, sample_row):
        text = construct_l4_context(sample_row)
        assert "TP53" in text
        assert "A549_LUNG" in text
        assert "Lung" in text
        # Should NOT include scores
        assert "0.050" not in text


class TestApplyMaxPerGene:
    def test_caps_per_gene(self):
        df = pd.DataFrame({
            "gene_id": [1] * 10 + [2] * 5,
            "value": range(15),
        })
        result = apply_max_per_gene(df, max_per_gene=3)
        assert len(result[result["gene_id"] == 1]) == 3
        assert len(result[result["gene_id"] == 2]) == 3

    def test_no_change_below_max(self):
        df = pd.DataFrame({
            "gene_id": [1, 2, 3],
            "value": [10, 20, 30],
        })
        result = apply_max_per_gene(df, max_per_gene=5)
        assert len(result) == 3


class TestAssignSplits:
    def test_all_assigned(self):
        df = pd.DataFrame({"x": range(100)})
        result = assign_splits(df)
        assert "split" in result.columns
        assert all(result["split"].isin(["train", "val", "test"]))

    def test_approximate_ratios(self):
        df = pd.DataFrame({"x": range(1000)})
        result = assign_splits(df)
        counts = result["split"].value_counts()
        assert 650 < counts["train"] < 750
        assert 50 < counts["val"] < 150


class TestWriteJsonl:
    def test_writes_file(self, tmp_path):
        records = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        path = tmp_path / "test.jsonl"
        count = write_jsonl(records, path)
        assert count == 2
        assert path.exists()

        # Verify content
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["text"] == "hello"


class TestWriteMetadata:
    def test_writes_metadata(self, tmp_path):
        write_dataset_metadata(
            tmp_path, task="ge-l1",
            stats={"n_records": 100, "split_counts": {"train": 70, "val": 10, "test": 20}},
        )
        meta_path = tmp_path / "ge-l1_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["task"] == "ge-l1"
        assert meta["domain"] == "ge"
        assert meta["n_records"] == 100
