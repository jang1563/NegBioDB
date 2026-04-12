"""Tests for MD LLM prompt formatting (L1-L4)."""

import json

import pytest

from negbiodb_md.llm_prompts import (
    MD_SYSTEM_PROMPT,
    format_l1_prompt,
    format_l2_prompt,
    format_l3_prompt,
    format_l4_prompt,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def l1_record():
    return {
        "task": "md_l1",
        "record_id": "r1",
        "context": "Study: MTBLS1. Platform: lc_ms. Biofluid: blood. n=60 T2D vs 55 controls.",
        "question": "Which metabolite is NOT a significant biomarker for type 2 diabetes mellitus in this study?",
        "choices": {"A": "alanine", "B": "glucose", "C": "leucine", "D": "palmitate"},
        "gold_answer": "A",
        "metadata": {"disease_name": "type 2 diabetes mellitus"},
    }


@pytest.fixture
def l2_record():
    return {
        "task": "md_l2",
        "record_id": "r2",
        "context": (
            "Study MTBLS1 (MetaboLights, lc_ms, blood, n=60 T2D, 55 controls). "
            "Metabolite: alanine. p=0.35. FDR=0.55. fold_change=0.92. Outcome: not_significant."
        ),
        "gold_fields": {
            "metabolite": "alanine",
            "disease": "type 2 diabetes mellitus",
            "fold_change": 0.92,
            "platform": "lc_ms",
            "biofluid": "blood",
            "outcome": "not_significant",
        },
    }


@pytest.fixture
def l3_record():
    return {
        "task": "md_l3",
        "record_id": "r3",
        "context": (
            "Study MTBLS1. Metabolite: alanine. Disease: type 2 diabetes mellitus. "
            "Platform: lc_ms. Biofluid: blood. n=60 disease, 55 control. "
            "p=0.35, FDR=0.55. Not significant."
        ),
        "question": "Explain why alanine was not found to be a significant biomarker for type 2 diabetes mellitus in this study.",
        "gold_reasoning": (
            "Alanine is a gluconeogenic amino acid. In T2D, alanine metabolism can be altered, "
            "but the observed p=0.35 suggests no significant difference in this cohort. "
            "The blood lc_ms platform may not capture tissue-level changes. "
            "With n=60/55, the study may be underpowered to detect small effect sizes."
        ),
        "rubric_axes": ["metabolite_biology", "disease_mechanism", "study_context", "alternative_hypothesis"],
    }


@pytest.fixture
def l4_record_real():
    return {
        "task": "md_l4",
        "record_id": "r4",
        "context": (
            "Metabolite: urea. Disease: colorectal cancer. "
            "Source: MetaboLights (MTBLS2). Platform: nmr. Biofluid: urine. "
            "p=0.42, FDR=0.60. Not significant."
        ),
        "label": 1,
        "label_text": "real",
    }


@pytest.fixture
def l4_record_synthetic():
    return {
        "task": "md_l4",
        "record_id": "r5",
        "context": (
            "Metabolite: phenylalanine. Disease: Alzheimer disease. "
            "This pair has not been experimentally tested in any deposited metabolomics study."
        ),
        "label": 0,
        "label_text": "synthetic",
    }


# ── MD_SYSTEM_PROMPT ──────────────────────────────────────────────────────────

def test_system_prompt_mentions_metabolomics():
    assert "metabolomics" in MD_SYSTEM_PROMPT.lower()


def test_system_prompt_nonempty():
    assert len(MD_SYSTEM_PROMPT) > 20


# ── L1 prompt formatting ──────────────────────────────────────────────────────

def test_format_l1_zero_shot_contains_choices(l1_record):
    prompt = format_l1_prompt(l1_record)
    assert "alanine" in prompt
    assert "glucose" in prompt
    assert "leucine" in prompt
    assert "palmitate" in prompt


def test_format_l1_zero_shot_contains_context(l1_record):
    prompt = format_l1_prompt(l1_record)
    assert "MTBLS1" in prompt


def test_format_l1_zero_shot_answer_format(l1_record):
    prompt = format_l1_prompt(l1_record)
    # Should instruct single letter response
    assert any(letter in prompt for letter in ["A", "B", "C", "D"])


def test_format_l1_few_shot_contains_examples(l1_record):
    examples = [
        {
            "context": "Study: example study. Platform: nmr.",
            "question": "Which metabolite is NOT a significant biomarker for diabetes in this study?",
            "choices": {"A": "ex_met1", "B": "ex_met2", "C": "ex_met3", "D": "ex_met4"},
            "gold_answer": "C",
        }
    ]
    prompt = format_l1_prompt(l1_record, few_shot_examples=examples)
    assert "ex_met1" in prompt
    assert "ex_met3" in prompt  # gold answer for example
    # Also contains the actual question choices
    assert "alanine" in prompt


def test_format_l1_few_shot_example_shows_answer(l1_record):
    examples = [
        {
            "context": "Example context.",
            "question": "Which metabolite is NOT a significant biomarker for cancer in this study?",
            "choices": {"A": "met_a", "B": "met_b", "C": "met_c", "D": "met_d"},
            "gold_answer": "B",
        }
    ]
    prompt = format_l1_prompt(l1_record, few_shot_examples=examples)
    assert "B" in prompt  # gold answer appears in examples


# ── L2 prompt formatting ──────────────────────────────────────────────────────

def test_format_l2_zero_shot_contains_context(l2_record):
    prompt = format_l2_prompt(l2_record)
    assert "alanine" in prompt or "MTBLS1" in prompt


def test_format_l2_zero_shot_contains_field_names(l2_record):
    prompt = format_l2_prompt(l2_record)
    for field in ["metabolite", "disease", "outcome", "platform", "biofluid"]:
        assert field in prompt


def test_format_l2_zero_shot_requests_json(l2_record):
    prompt = format_l2_prompt(l2_record)
    assert "JSON" in prompt or "json" in prompt.lower()


def test_format_l2_few_shot_includes_gold_example(l2_record):
    examples = [
        {
            "context": "Example study context.",
            "gold_fields": {
                "metabolite": "example_metabolite",
                "disease": "example disease",
                "outcome": "significant",
            },
        }
    ]
    prompt = format_l2_prompt(l2_record, few_shot_examples=examples)
    assert "example_metabolite" in prompt
    assert "significant" in prompt


def test_format_l2_few_shot_gold_is_valid_json(l2_record):
    """Gold fields in few-shot should be serialized as valid JSON."""
    examples = [
        {
            "context": "Context.",
            "gold_fields": {"metabolite": "glucose", "outcome": "not_significant"},
        }
    ]
    prompt = format_l2_prompt(l2_record, few_shot_examples=examples)
    # The JSON should appear in prompt
    assert '"glucose"' in prompt
    assert '"not_significant"' in prompt


# ── L3 prompt formatting ──────────────────────────────────────────────────────

def test_format_l3_zero_shot_contains_question(l3_record):
    prompt = format_l3_prompt(l3_record)
    assert "alanine" in prompt
    assert "type 2 diabetes" in prompt.lower()


def test_format_l3_zero_shot_contains_rubric_axes(l3_record):
    prompt = format_l3_prompt(l3_record)
    assert "metabolite" in prompt.lower()
    assert "disease" in prompt.lower()
    assert "study" in prompt.lower()


def test_format_l3_zero_shot_nonempty(l3_record):
    prompt = format_l3_prompt(l3_record)
    assert len(prompt) > 50


def test_format_l3_few_shot_includes_gold_reasoning(l3_record):
    examples = [
        {
            "context": "Example metabolomics context.",
            "question": "Why was glucose not significant?",
            "gold_reasoning": "Glucose regulation is tightly controlled in this population.",
        }
    ]
    prompt = format_l3_prompt(l3_record, few_shot_examples=examples)
    assert "Glucose regulation" in prompt


# ── L4 prompt formatting ──────────────────────────────────────────────────────

def test_format_l4_zero_shot_contains_real_synthetic_options(l4_record_real):
    prompt = format_l4_prompt(l4_record_real)
    assert "A" in prompt
    assert "B" in prompt
    assert "Real" in prompt or "real" in prompt
    assert "Synthetic" in prompt or "synthetic" in prompt


def test_format_l4_zero_shot_contains_context(l4_record_real):
    prompt = format_l4_prompt(l4_record_real)
    assert "urea" in prompt or "colorectal" in prompt


def test_format_l4_zero_shot_answer_format(l4_record_real):
    prompt = format_l4_prompt(l4_record_real)
    # Should instruct single letter response
    assert "A" in prompt and "B" in prompt


def test_format_l4_few_shot_real_example_shows_a(l4_record_real):
    examples = [
        {
            "context": "MetaboLights study context for real finding.",
            "label": 1,
            "label_text": "real",
        }
    ]
    prompt = format_l4_prompt(l4_record_real, few_shot_examples=examples)
    # Real label should be marked as A
    assert "A" in prompt
    assert "Real" in prompt or "real" in prompt.lower()


def test_format_l4_few_shot_synthetic_example_shows_b(l4_record_synthetic):
    examples = [
        {
            "context": "Random metabolite-disease pair never tested.",
            "label": 0,
            "label_text": "synthetic",
        }
    ]
    prompt = format_l4_prompt(l4_record_synthetic, few_shot_examples=examples)
    assert "B" in prompt
    assert "Synthetic" in prompt or "synthetic" in prompt.lower()


# ── Prompt length sanity checks ───────────────────────────────────────────────

def test_l1_zero_shot_reasonable_length(l1_record):
    prompt = format_l1_prompt(l1_record)
    # Should be at least 100 chars but not absurdly long (< 2000 chars for zero-shot)
    assert 100 < len(prompt) < 2000


def test_l2_zero_shot_reasonable_length(l2_record):
    prompt = format_l2_prompt(l2_record)
    assert 100 < len(prompt) < 2000


def test_l3_zero_shot_reasonable_length(l3_record):
    prompt = format_l3_prompt(l3_record)
    assert 100 < len(prompt) < 3000
