"""Tests for CT LLM prompt templates (src/negbiodb_ct/llm_prompts.py)."""

import pytest

from negbiodb_ct.llm_prompts import (
    CATEGORY_TO_MCQ,
    CT_L1_CATEGORIES,
    CT_SYSTEM_PROMPT,
    CT_TASK_FORMATTERS,
    FEWSHOT_SEEDS,
    format_ct_l1_prompt,
    format_ct_l2_prompt,
    format_ct_l3_prompt,
    format_ct_l4_prompt,
    format_ct_prompt,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_l1_record():
    return {
        "context_text": (
            "Drug: Imatinib\nCondition: Breast Cancer\n"
            "Phase: phase_3\nEndpoint: Overall Survival\n"
            "p-value: 0.34\nEffect size: 0.12"
        ),
        "gold_answer": "B",
    }


@pytest.fixture
def sample_l2_record():
    return {
        "context_text": (
            "Drug: Sorafenib\nCondition: Liver Cancer\n"
            'Phase: phase_2\nTermination text: "The study was '
            'terminated due to a higher-than-expected rate of '
            'hepatotoxicity in the treatment arm."'
        ),
        "gold_answer": "safety",
    }


@pytest.fixture
def sample_l3_record():
    return {
        "context_text": (
            "Drug: Trastuzumab\nType: monoclonal_antibody\n"
            "Condition: Triple-negative breast cancer\n"
            "Therapeutic area: oncology\n"
            "Phase: phase_3\nEndpoint: Progression-Free Survival\n"
            "p-value: 0.08\nEffect size (HR): 0.92\nCI: [0.81, 1.05]"
        ),
        "gold_answer": "efficacy",
    }


@pytest.fixture
def sample_l4_record():
    return {
        "context_text": (
            "Drug: Metformin\nDrug type: small_molecule\n"
            "Condition: Alzheimer's Disease\n"
            "Therapeutic area: neurology"
        ),
        "gold_answer": "tested",
    }


@pytest.fixture
def l1_fewshot_examples():
    return [
        {"context_text": "Drug: X\nCondition: Y\np-value: 0.001", "gold_answer": "A"},
        {"context_text": "Drug: Z\nCondition: W\np-value: 0.45", "gold_answer": "B"},
        {"context_text": "Drug: Q\nCondition: R\nEnrollment: 12", "gold_answer": "C"},
    ]


# ── CT-L1 Tests ──────────────────────────────────────────────────────────


class TestCTL1Prompt:
    def test_zero_shot_format(self, sample_l1_record):
        system, user = format_ct_l1_prompt(sample_l1_record, "zero-shot")
        assert system == CT_SYSTEM_PROMPT
        assert "Imatinib" in user
        assert "Breast Cancer" in user
        assert "A)" in user
        assert "E)" in user
        assert "single letter" in user

    def test_three_shot_format(self, sample_l1_record, l1_fewshot_examples):
        system, user = format_ct_l1_prompt(sample_l1_record, "3-shot", l1_fewshot_examples)
        assert "examples" in user.lower()
        assert "---" in user
        assert "Answer: A" in user
        assert "Answer: B" in user
        assert "Imatinib" in user

    def test_all_five_categories_in_prompt(self, sample_l1_record):
        _, user = format_ct_l1_prompt(sample_l1_record, "zero-shot")
        for letter in "ABCDE":
            assert f"{letter})" in user, f"Category {letter} missing from prompt"

    def test_category_descriptions_match(self, sample_l1_record):
        _, user = format_ct_l1_prompt(sample_l1_record, "zero-shot")
        assert "Safety" in user
        assert "Efficacy" in user
        assert "Enrollment" in user
        assert "Strategic" in user
        assert "Other" in user


# ── CT-L2 Tests ──────────────────────────────────────────────────────────


class TestCTL2Prompt:
    def test_zero_shot_format(self, sample_l2_record):
        system, user = format_ct_l2_prompt(sample_l2_record, "zero-shot")
        assert system == CT_SYSTEM_PROMPT
        assert "hepatotoxicity" in user
        assert "failure_category" in user
        assert "JSON" in user

    def test_all_seven_fields_in_prompt(self, sample_l2_record):
        _, user = format_ct_l2_prompt(sample_l2_record, "zero-shot")
        for field in [
            "failure_category", "failure_subcategory", "affected_system",
            "severity_indicator", "quantitative_evidence", "decision_maker",
            "patient_impact",
        ]:
            assert field in user, f"Field {field} missing from L2 prompt"

    def test_three_shot_format(self, sample_l2_record):
        examples = [
            {
                "context_text": "Drug: A\nCondition: B\nTermination text: \"Futility\"",
                "gold_extraction": {"failure_category": "efficacy"},
            },
            {
                "context_text": "Drug: C\nCondition: D\nTermination text: \"Slow accrual\"",
                "gold_extraction": {"failure_category": "enrollment"},
            },
        ]
        _, user = format_ct_l2_prompt(sample_l2_record, "3-shot", examples)
        assert "---" in user
        assert "Futility" in user
        assert "Slow accrual" in user


# ── CT-L3 Tests ──────────────────────────────────────────────────────────


class TestCTL3Prompt:
    def test_zero_shot_format(self, sample_l3_record):
        system, user = format_ct_l3_prompt(sample_l3_record, "zero-shot")
        assert system == CT_SYSTEM_PROMPT
        assert "Trastuzumab" in user
        assert "FAILURE" in user

    def test_four_dimensions_in_prompt(self, sample_l3_record):
        _, user = format_ct_l3_prompt(sample_l3_record, "zero-shot")
        for dim in ["Mechanism", "Evidence interpretation", "Clinical factors", "Broader context"]:
            assert dim in user, f"Dimension '{dim}' missing from L3 prompt"

    def test_three_shot_format(self, sample_l3_record):
        examples = [
            {
                "context_text": "Drug: X\nCondition: Y",
                "gold_reasoning": "The drug failed because...",
            },
        ]
        _, user = format_ct_l3_prompt(sample_l3_record, "3-shot", examples)
        assert "The drug failed because" in user


# ── CT-L4 Tests ──────────────────────────────────────────────────────────


class TestCTL4Prompt:
    def test_zero_shot_format(self, sample_l4_record):
        system, user = format_ct_l4_prompt(sample_l4_record, "zero-shot")
        assert system == CT_SYSTEM_PROMPT
        assert "Metformin" in user
        assert "tested" in user.lower()
        assert "untested" in user.lower()

    def test_tested_untested_instruction(self, sample_l4_record):
        _, user = format_ct_l4_prompt(sample_l4_record, "zero-shot")
        assert "tested" in user
        assert "untested" in user
        assert "ClinicalTrials.gov" in user

    def test_evidence_guidance(self, sample_l4_record):
        _, user = format_ct_l4_prompt(sample_l4_record, "zero-shot")
        assert "evidence" in user.lower()

    def test_three_shot_format(self, sample_l4_record):
        examples = [
            {"context_text": "Drug: A\nCondition: B", "gold_answer": "tested"},
            {"context_text": "Drug: C\nCondition: D", "gold_answer": "untested"},
        ]
        _, user = format_ct_l4_prompt(sample_l4_record, "3-shot", examples)
        assert "---" in user
        assert "Answer: tested" in user
        assert "Answer: untested" in user


# ── Dispatch Tests ───────────────────────────────────────────────────────


class TestFormatDispatch:
    def test_valid_task_ids(self, sample_l1_record):
        for task_id in ["ct-l1", "ct-l2", "ct-l3", "ct-l4"]:
            system, user = format_ct_prompt(task_id, sample_l1_record)
            assert system == CT_SYSTEM_PROMPT
            assert len(user) > 0

    def test_invalid_task_raises(self, sample_l1_record):
        with pytest.raises(ValueError, match="Unknown task"):
            format_ct_prompt("l1", sample_l1_record)

    def test_all_formatters_registered(self):
        assert set(CT_TASK_FORMATTERS.keys()) == {"ct-l1", "ct-l2", "ct-l3", "ct-l4"}


# ── Constants Tests ──────────────────────────────────────────────────────


class TestConstants:
    def test_category_to_mcq_covers_all_8(self):
        expected = {"safety", "efficacy", "enrollment", "strategic",
                    "design", "regulatory", "pharmacokinetic", "other"}
        assert set(CATEGORY_TO_MCQ.keys()) == expected

    def test_category_to_mcq_maps_to_abcde(self):
        assert set(CATEGORY_TO_MCQ.values()) == {"A", "B", "C", "D", "E"}

    def test_l1_categories_five_way(self):
        assert set(CT_L1_CATEGORIES.keys()) == {"A", "B", "C", "D", "E"}

    def test_fewshot_seeds_count(self):
        assert len(FEWSHOT_SEEDS) == 3
        assert all(isinstance(s, int) for s in FEWSHOT_SEEDS)
