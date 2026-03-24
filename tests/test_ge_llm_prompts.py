"""Tests for GE LLM prompt formatting (GE-L1 through GE-L4)."""

import pytest

from negbiodb_depmap.llm_prompts import (
    GE_SYSTEM_PROMPT,
    GE_TASK_FORMATTERS,
    format_ge_l1_prompt,
    format_ge_l2_prompt,
    format_ge_l3_prompt,
    format_ge_l4_prompt,
    format_ge_prompt,
)


@pytest.fixture
def sample_record():
    return {
        "context_text": "Gene: TP53\nCell line: A549_LUNG (Lung)\nScreen: CRISPR-Cas9\nGene effect: 0.05",
        "gene_symbol": "TP53",
        "cell_line": "A549_LUNG",
        "lineage": "Lung",
    }


@pytest.fixture
def fewshot_examples():
    return [
        {"context_text": "Gene: BRAF\nCell line: SKMEL5_SKIN", "gold_answer": "B"},
        {"context_text": "Gene: RPS3\nCell line: HeLa_CERVIX", "gold_answer": "A"},
        {"context_text": "Gene: OR1A1\nCell line: MCF7_BREAST", "gold_answer": "C"},
    ]


class TestL1Prompts:
    def test_zero_shot_returns_tuple(self, sample_record):
        sys, user = format_ge_l1_prompt(sample_record)
        assert sys == GE_SYSTEM_PROMPT
        assert "TP53" in user
        assert "A)" in user
        assert "D)" in user

    def test_zero_shot_has_answer_format(self, sample_record):
        _, user = format_ge_l1_prompt(sample_record)
        assert "single letter" in user.lower()

    def test_fewshot_includes_examples(self, sample_record, fewshot_examples):
        _, user = format_ge_l1_prompt(sample_record, config="3-shot", fewshot_examples=fewshot_examples)
        assert "BRAF" in user
        assert "Answer: B" in user

    def test_fewshot_no_examples_falls_back(self, sample_record):
        _, user = format_ge_l1_prompt(sample_record, config="3-shot", fewshot_examples=None)
        assert "Categories:" in user


class TestL2Prompts:
    def test_zero_shot_json_fields(self, sample_record):
        _, user = format_ge_l2_prompt(sample_record)
        assert "gene_name" in user
        assert "essentiality_status" in user
        assert "JSON" in user

    def test_fewshot_includes_extraction(self, sample_record):
        examples = [
            {
                "context_text": "Gene: EGFR",
                "gold_extraction": {"genes": [{"gene_name": "EGFR"}]},
            },
        ]
        _, user = format_ge_l2_prompt(sample_record, config="3-shot", fewshot_examples=examples)
        assert "EGFR" in user


class TestL3Prompts:
    def test_zero_shot_structure(self, sample_record):
        _, user = format_ge_l3_prompt(sample_record)
        assert "TP53" in user
        assert "NON-ESSENTIAL" in user
        assert "biological plausibility" in user.lower()

    def test_fewshot_truncation(self, sample_record):
        long_example = {
            "context_text": "x " * 1000,
            "gold_reasoning": "Because " * 500,
        }
        _, user = format_ge_l3_prompt(
            sample_record, config="3-shot", fewshot_examples=[long_example]
        )
        assert "[...]" in user


class TestL4Prompts:
    def test_zero_shot_tested_format(self, sample_record):
        _, user = format_ge_l4_prompt(sample_record)
        assert "tested" in user.lower()
        assert "untested" in user.lower()

    def test_fewshot_includes_examples(self, sample_record):
        examples = [
            {"context_text": "Gene: TP53\nCell line: A549", "gold_answer": "tested"},
        ]
        _, user = format_ge_l4_prompt(sample_record, config="3-shot", fewshot_examples=examples)
        assert "Answer: tested" in user


class TestDispatch:
    def test_all_tasks_registered(self):
        assert set(GE_TASK_FORMATTERS.keys()) == {"ge-l1", "ge-l2", "ge-l3", "ge-l4"}

    def test_dispatch(self, sample_record):
        sys, user = format_ge_prompt("ge-l1", sample_record)
        assert sys == GE_SYSTEM_PROMPT
        assert "TP53" in user

    def test_invalid_task(self, sample_record):
        with pytest.raises(ValueError):
            format_ge_prompt("ge-l99", sample_record)
