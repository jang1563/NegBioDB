"""Tests for VP LLM dataset builder utilities."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from negbiodb_vp.llm_dataset import (
    ACMG_CRITERIA_DESCRIPTIONS,
    CLASSIFICATION_TO_L1_ANSWER,
    CONSEQUENCE_DESCRIPTIONS,
    FEWSHOT_SEEDS,
    JSONL_SCHEMA_FIELDS,
    MAX_PER_GENE,
    apply_max_per_gene,
    assign_splits,
    construct_l1_context,
    construct_l2_context,
    construct_l3_context,
    construct_l4_context,
    read_jsonl,
    write_dataset_metadata,
    write_jsonl,
)


# ── Classification Mapping Tests ────────────────────────────────────────


class TestClassificationToL1Answer:
    def test_4way_mapping(self):
        assert CLASSIFICATION_TO_L1_ANSWER["pathogenic"] == "A"
        assert CLASSIFICATION_TO_L1_ANSWER["likely_benign"] == "B"
        assert CLASSIFICATION_TO_L1_ANSWER["uncertain_significance"] == "C"
        assert CLASSIFICATION_TO_L1_ANSWER["benign"] == "D"

    def test_all_values_in_abcd(self):
        assert set(CLASSIFICATION_TO_L1_ANSWER.values()) == {"A", "B", "C", "D"}

    def test_likely_pathogenic_grouped(self):
        assert CLASSIFICATION_TO_L1_ANSWER["likely_pathogenic"] == "A"

    def test_benign_likely_benign_grouped(self):
        assert CLASSIFICATION_TO_L1_ANSWER["benign/likely_benign"] == "D"


# ── Context Construction Tests ──────────────────────────────────────────


class TestConstructL1Context:
    def test_contains_gene(self):
        record = {
            "gene_symbol": "BRCA1", "chromosome": "17", "position": 43094464,
            "ref_allele": "G", "alt_allele": "A", "consequence_type": "missense",
            "disease_name": "Hereditary breast cancer",
        }
        ctx = construct_l1_context(record)
        assert "BRCA1" in ctx
        assert "chr17" in ctx

    def test_includes_scores(self):
        record = {
            "gene_symbol": "TP53", "chromosome": "17", "position": 7579472,
            "ref_allele": "G", "alt_allele": "A", "consequence_type": "missense",
            "disease_name": "Li-Fraumeni",
            "cadd_phred": 28.5, "revel_score": 0.92,
            "alphamissense_score": 0.85, "alphamissense_class": "likely_pathogenic",
        }
        ctx = construct_l1_context(record)
        assert "CADD" in ctx
        assert "REVEL" in ctx
        assert "AlphaMissense" in ctx

    def test_includes_constraint(self):
        record = {
            "gene_symbol": "BRCA1", "chromosome": "17", "position": 43094464,
            "ref_allele": "G", "alt_allele": "A", "consequence_type": "missense",
            "disease_name": "test", "pli_score": 0.999, "loeuf_score": 0.23,
        }
        ctx = construct_l1_context(record)
        assert "pLI" in ctx
        assert "LOEUF" in ctx

    def test_handles_missing_scores(self):
        record = {
            "gene_symbol": "MTHFR", "chromosome": "1", "position": 11856378,
            "ref_allele": "C", "alt_allele": "T", "consequence_type": "synonymous",
            "disease_name": "test",
        }
        ctx = construct_l1_context(record)
        assert "MTHFR" in ctx
        # Should not crash with missing scores

    def test_includes_hgvs(self):
        record = {
            "gene_symbol": "BRCA1", "chromosome": "17", "position": 43094464,
            "ref_allele": "G", "alt_allele": "A", "consequence_type": "missense",
            "disease_name": "test",
            "hgvs_coding": "c.5123C>A", "hgvs_protein": "p.Ala1708Asp",
        }
        ctx = construct_l1_context(record)
        assert "c.5123C>A" in ctx
        assert "p.Ala1708Asp" in ctx


class TestConstructL2Context:
    def test_report_format(self):
        record = {
            "gene_symbol": "BRCA1", "hgvs_coding": "c.5123C>A",
            "hgvs_protein": "p.Ala1708Asp", "consequence_type": "missense",
            "disease_name": "Hereditary breast cancer",
            "classification": "benign", "gnomad_af_global": 0.023,
            "cadd_phred": 12.5, "revel_score": 0.1,
            "acmg_criteria": '["BA1", "BS1"]',
        }
        ctx = construct_l2_context(record)
        assert "CLINICAL VARIANT INTERPRETATION REPORT" in ctx
        assert "BRCA1" in ctx
        assert "ACMG/AMP" in ctx
        assert "BA1" in ctx

    def test_mentions_total_variants(self):
        record = {
            "gene_symbol": "TP53", "hgvs_coding": "c.215C>G",
            "consequence_type": "missense", "disease_name": "Li-Fraumeni",
            "classification": "likely_benign",
        }
        ctx = construct_l2_context(record)
        assert "1 variant" in ctx

    def test_acmg_descriptions_included(self):
        record = {
            "gene_symbol": "SCN5A", "hgvs_coding": "c.100A>G",
            "consequence_type": "synonymous", "disease_name": "Brugada",
            "classification": "benign",
            "acmg_criteria": '["BA1", "BP7"]',
        }
        ctx = construct_l2_context(record)
        assert "standalone benign" in ctx  # BA1 description
        assert "Synonymous" in ctx  # BP7 description


class TestConstructL3Context:
    def test_rich_context(self):
        record = {
            "gene_symbol": "MTHFR", "chromosome": "1", "position": 11856378,
            "ref_allele": "C", "alt_allele": "T",
            "hgvs_coding": "c.665C>T", "hgvs_protein": "p.Ala222Val",
            "consequence_type": "missense", "classification": "benign",
            "disease_name": "Homocystinuria", "inheritance_pattern": "AR",
            "gnomad_af_global": 0.34, "gnomad_af_nfe": 0.36,
            "cadd_phred": 23.1, "revel_score": 0.23,
            "pli_score": 0.001, "loeuf_score": 1.2,
            "clingen_validity": "Definitive",
        }
        ctx = construct_l3_context(record)
        assert "MTHFR" in ctx
        assert "Homocystinuria" in ctx
        assert "AR" in ctx
        assert "ClinGen" in ctx
        assert "NFE" in ctx

    def test_handles_missing_fields(self):
        record = {
            "gene_symbol": "UNKNOWN", "chromosome": "1", "position": 100,
            "ref_allele": "A", "alt_allele": "G",
            "consequence_type": "synonymous", "classification": "benign",
            "disease_name": "not specified",
        }
        ctx = construct_l3_context(record)
        assert "UNKNOWN" in ctx
        # Should not crash

    def test_includes_all_scores(self):
        record = {
            "gene_symbol": "TP53", "chromosome": "17", "position": 7579472,
            "ref_allele": "G", "alt_allele": "A",
            "consequence_type": "missense", "classification": "benign",
            "disease_name": "Li-Fraumeni",
            "cadd_phred": 28.5, "revel_score": 0.92,
            "alphamissense_score": 0.85, "alphamissense_class": "likely_pathogenic",
            "phylop_score": 5.2, "gerp_score": 4.8,
            "sift_score": 0.001, "polyphen2_score": 0.995,
        }
        ctx = construct_l3_context(record)
        for score_name in ["CADD", "REVEL", "AlphaMissense", "PhyloP", "GERP", "SIFT", "PolyPhen2"]:
            assert score_name in ctx


class TestConstructL4Context:
    def test_minimal_context(self):
        record = {
            "gene_symbol": "BRCA2", "chromosome": "13", "position": 32340300,
            "ref_allele": "A", "alt_allele": "G",
            "hgvs_coding": "c.7397T>C", "hgvs_protein": "p.Val2466Ala",
            "consequence_type": "missense", "disease_name": "Hereditary breast cancer",
        }
        ctx = construct_l4_context(record)
        assert "BRCA2" in ctx
        assert "assessed" in ctx.lower()
        assert "pathogenicity" in ctx.lower()


# ── Sampling Tests ──────────────────────────────────────────────────────


class TestApplyMaxPerGene:
    def test_no_capping_needed(self):
        df = pd.DataFrame({
            "gene_id": [1, 2, 3, 4, 5],
            "variant_id": [10, 20, 30, 40, 50],
        })
        result = apply_max_per_gene(df, max_per_gene=10)
        assert len(result) == 5

    def test_capping_applied(self):
        # Gene 1 appears 15 times, should be capped to MAX_PER_GENE
        df = pd.DataFrame({
            "gene_id": [1] * 15 + [2] * 3,
            "variant_id": list(range(18)),
        })
        result = apply_max_per_gene(df, max_per_gene=MAX_PER_GENE)
        gene1_count = (result["gene_id"] == 1).sum()
        assert gene1_count <= MAX_PER_GENE

    def test_reproducibility(self):
        df = pd.DataFrame({
            "gene_id": [1] * 20,
            "variant_id": list(range(20)),
        })
        r1 = apply_max_per_gene(df, max_per_gene=5, rng=np.random.RandomState(42))
        r2 = apply_max_per_gene(df, max_per_gene=5, rng=np.random.RandomState(42))
        assert r1["variant_id"].tolist() == r2["variant_id"].tolist()

    def test_no_gene_id_column(self):
        df = pd.DataFrame({"variant_id": [1, 2, 3]})
        result = apply_max_per_gene(df)
        assert len(result) == 3


class TestAssignSplits:
    def test_correct_sizes(self):
        df = pd.DataFrame({"id": range(100)})
        result = assign_splits(df, fewshot_size=10, val_size=10, test_size=80, seed=42)
        assert (result["split"] == "fewshot").sum() == 10
        assert (result["split"] == "val").sum() == 10
        assert (result["split"] == "test").sum() == 80

    def test_small_dataset(self):
        df = pd.DataFrame({"id": range(15)})
        result = assign_splits(df, fewshot_size=5, val_size=5, test_size=100, seed=42)
        assert (result["split"] == "fewshot").sum() == 5
        assert (result["split"] == "val").sum() == 5
        assert (result["split"] == "test").sum() == 5  # Adjusted

    def test_reproducibility(self):
        df = pd.DataFrame({"id": range(50)})
        r1 = assign_splits(df, 5, 5, 40, seed=42)
        r2 = assign_splits(df, 5, 5, 40, seed=42)
        assert r1["id"].tolist() == r2["id"].tolist()


# ── I/O Tests ───────────────────────────────────────────────────────────


class TestJSONLIO:
    def test_write_read_roundtrip(self, tmp_path):
        records = [
            {"question_id": "vp-l1-001", "gold_answer": "A", "context_text": "Gene: BRCA1"},
            {"question_id": "vp-l1-002", "gold_answer": "D", "context_text": "Gene: MTHFR"},
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(records, path)
        loaded = read_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["gold_answer"] == "A"
        assert loaded[1]["question_id"] == "vp-l1-002"

    def test_unicode(self, tmp_path):
        records = [{"text": "variant p.Ala222Val — benign"}]
        path = tmp_path / "unicode.jsonl"
        write_jsonl(records, path)
        loaded = read_jsonl(path)
        assert loaded[0]["text"] == "variant p.Ala222Val — benign"

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        write_jsonl([], path)
        loaded = read_jsonl(path)
        assert loaded == []


class TestWriteMetadata:
    def test_creates_file(self, tmp_path):
        write_dataset_metadata(tmp_path, "vp-l1", {"n_records": 100})
        meta_path = tmp_path / "vp_l1_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["task"] == "vp-l1"
        assert meta["n_records"] == 100
        assert "created" in meta


# ── Schema and Constants Tests ──────────────────────────────────────────


class TestSchemaFields:
    def test_required_fields(self):
        assert "question_id" in JSONL_SCHEMA_FIELDS
        assert "gold_answer" in JSONL_SCHEMA_FIELDS
        assert "context_text" in JSONL_SCHEMA_FIELDS

    def test_uses_gold_answer_not_correct_answer(self):
        assert "gold_answer" in JSONL_SCHEMA_FIELDS
        assert "correct_answer" not in JSONL_SCHEMA_FIELDS


class TestConsequenceDescriptions:
    def test_has_missense(self):
        assert "missense" in CONSEQUENCE_DESCRIPTIONS

    def test_has_synonymous(self):
        assert "synonymous" in CONSEQUENCE_DESCRIPTIONS

    def test_descriptions_are_human_readable(self):
        for key, desc in CONSEQUENCE_DESCRIPTIONS.items():
            assert len(desc) > 10  # Not just a single word


class TestACMGCriteriaDescriptions:
    def test_has_ba1(self):
        assert "BA1" in ACMG_CRITERIA_DESCRIPTIONS
        assert "5%" in ACMG_CRITERIA_DESCRIPTIONS["BA1"]

    def test_has_benign_criteria(self):
        for code in ["BA1", "BS1", "BS2", "BS3", "BS4", "BP1", "BP4", "BP7"]:
            assert code in ACMG_CRITERIA_DESCRIPTIONS

    def test_descriptions_are_informative(self):
        for code, desc in ACMG_CRITERIA_DESCRIPTIONS.items():
            assert len(desc) > 20


class TestFewshotSeeds:
    def test_three_seeds(self):
        assert len(FEWSHOT_SEEDS) == 3

    def test_includes_42(self):
        assert 42 in FEWSHOT_SEEDS
