"""Tests for VP LLM prompt templates (VP-L1 through VP-L4)."""

import pytest

from negbiodb_vp.llm_prompts import (
    VP_L1_CATEGORIES,
    VP_SYSTEM_PROMPT,
    VP_TASK_FORMATTERS,
    FEWSHOT_SEEDS,
    _truncate_text,
    format_vp_l1_prompt,
    format_vp_l2_prompt,
    format_vp_l3_prompt,
    format_vp_l4_prompt,
    format_vp_prompt,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def l1_record():
    return {
        "context_text": (
            "Gene: BRCA1\n"
            "Variant: chr17:43094464 G>A\n"
            "HGVS coding: c.5123C>A\n"
            "HGVS protein: p.Ala1708Asp\n"
            "Consequence: missense variant (amino acid substitution)\n"
            "Condition: Hereditary breast cancer\n"
            "Inheritance pattern: AD\n"
            "gnomAD global allele frequency: 0.000012\n"
            "Computational scores: CADD Phred=28.5, REVEL=0.920, AlphaMissense=0.850 (likely_pathogenic)\n"
            "Gene constraint: pLI=0.999, LOEUF=0.230"
        ),
        "gold_answer": "A",
    }


@pytest.fixture
def l1_fewshot_examples():
    return [
        {
            "context_text": "Gene: TP53\nVariant: chr17:7579472 G>A\nConsequence: missense",
            "gold_answer": "A",
        },
        {
            "context_text": "Gene: MTHFR\nVariant: chr1:11856378 C>T\nConsequence: missense",
            "gold_answer": "D",
        },
        {
            "context_text": "Gene: SCN5A\nVariant: chr3:38592065 A>G\nConsequence: synonymous",
            "gold_answer": "B",
        },
    ]


@pytest.fixture
def l2_record():
    return {
        "context_text": (
            "CLINICAL VARIANT INTERPRETATION REPORT\n\n"
            "Patient variant: BRCA1 c.5123C>A\n"
            "Protein change: p.Ala1708Asp\n"
            "Variant type: missense\n"
            "Condition evaluated: Hereditary breast cancer\n\n"
            "Classification: Benign\n"
            "Classification method: ACMG/AMP\n\n"
            "EVIDENCE SUMMARY:\n"
            "- Population frequency: gnomAD global AF = 0.023000\n"
            "ACMG criteria applied: BA1, BS1\n"
        ),
        "gold_extraction": {
            "variants": [
                {
                    "gene": "BRCA1",
                    "hgvs": "c.5123C>A",
                    "classification": "benign",
                    "acmg_criteria_met": ["BA1", "BS1"],
                    "population_frequency": 0.023,
                    "condition": "Hereditary breast cancer",
                }
            ],
            "total_variants_discussed": 1,
            "classification_method": "ACMG/AMP",
        },
    }


@pytest.fixture
def l3_record():
    return {
        "context_text": (
            "Variant: MTHFR chr1:11856378 C>T\n"
            "  HGVS coding: c.665C>T\n"
            "  HGVS protein: p.Ala222Val\n"
            "  Consequence: missense variant (amino acid substitution)\n"
            "  Classification: benign\n\n"
            "Condition: Homocystinuria\n"
            "  Inheritance: AR\n\n"
            "Population frequencies (gnomAD): Global: 0.340000, NFE: 0.360000\n"
            "Computational scores: CADD Phred=23.1, REVEL=0.230\n"
            "Gene constraint: pLI=0.001, LOEUF=1.200"
        ),
        "gold_reasoning": (
            "The MTHFR c.665C>T (p.Ala222Val) variant is a well-characterized common "
            "polymorphism. With a gnomAD global allele frequency of 34%, it far exceeds "
            "the BA1 standalone benign threshold."
        ),
    }


@pytest.fixture
def l4_record():
    return {
        "context_text": (
            "Gene: BRCA2\n"
            "Variant: chr13:32340300 A>G\n"
            "HGVS: c.7397T>C\n"
            "Protein: p.Val2466Ala\n"
            "Consequence: missense\n"
            "Condition: Hereditary breast cancer\n\n"
            "Has this variant-disease pair been assessed for pathogenicity in "
            "clinical variant databases?"
        ),
        "gold_answer": "tested",
    }


# ── VP-L1 Tests ─────────────────────────────────────────────────────────


class TestVPL1Prompt:
    def test_zero_shot_returns_tuple(self, l1_record):
        sys, user = format_vp_l1_prompt(l1_record, config="zero-shot")
        assert isinstance(sys, str)
        assert isinstance(user, str)

    def test_zero_shot_contains_context(self, l1_record):
        _, user = format_vp_l1_prompt(l1_record)
        assert "BRCA1" in user
        assert "chr17:43094464" in user

    def test_zero_shot_contains_categories(self, l1_record):
        _, user = format_vp_l1_prompt(l1_record)
        assert "A)" in user
        assert "B)" in user
        assert "C)" in user
        assert "D)" in user

    def test_zero_shot_answer_format(self, l1_record):
        _, user = format_vp_l1_prompt(l1_record)
        assert "ONLY a single letter" in user

    def test_3shot_contains_examples(self, l1_record, l1_fewshot_examples):
        _, user = format_vp_l1_prompt(l1_record, "3-shot", l1_fewshot_examples)
        assert "TP53" in user
        assert "MTHFR" in user
        assert "SCN5A" in user
        assert "Answer: A" in user
        assert "Answer: D" in user

    def test_3shot_contains_separator(self, l1_record, l1_fewshot_examples):
        _, user = format_vp_l1_prompt(l1_record, "3-shot", l1_fewshot_examples)
        assert "---" in user

    def test_system_prompt_is_vp(self, l1_record):
        sys, _ = format_vp_l1_prompt(l1_record)
        assert sys == VP_SYSTEM_PROMPT
        assert "clinical geneticist" in sys

    def test_4way_categories(self):
        assert len(VP_L1_CATEGORIES) == 4
        assert set(VP_L1_CATEGORIES.keys()) == {"A", "B", "C", "D"}

    def test_categories_contain_vp_terms(self):
        all_text = " ".join(VP_L1_CATEGORIES.values()).lower()
        assert "pathogenic" in all_text
        assert "benign" in all_text
        assert "uncertain" in all_text

    def test_no_fewshot_falls_back_to_zero_shot(self, l1_record):
        _, user_zs = format_vp_l1_prompt(l1_record, "zero-shot")
        _, user_3s_empty = format_vp_l1_prompt(l1_record, "3-shot", [])
        # Both should produce zero-shot format (no examples section)
        assert "examples" not in user_zs.lower().split("categories")[0]


# ── VP-L2 Tests ─────────────────────────────────────────────────────────


class TestVPL2Prompt:
    def test_zero_shot_json_fields(self, l2_record):
        _, user = format_vp_l2_prompt(l2_record)
        assert "variants" in user
        assert "acmg_criteria_met" in user
        assert "classification_method" in user

    def test_3shot_contains_examples(self, l2_record):
        examples = [
            {
                "context_text": "Report about CFTR variant",
                "gold_extraction": {
                    "variants": [{"gene": "CFTR", "hgvs": "c.1521_1523delCTT"}],
                    "total_variants_discussed": 1,
                    "classification_method": "ACMG/AMP",
                },
            }
        ]
        _, user = format_vp_l2_prompt(l2_record, "3-shot", examples)
        assert "CFTR" in user
        assert "Extracted:" in user

    def test_asks_for_json(self, l2_record):
        _, user = format_vp_l2_prompt(l2_record)
        assert "valid JSON" in user


# ── VP-L3 Tests ─────────────────────────────────────────────────────────


class TestVPL3Prompt:
    def test_zero_shot_contains_4_dimensions(self, l3_record):
        _, user = format_vp_l3_prompt(l3_record)
        assert "Population reasoning" in user
        assert "Computational evidence" in user
        assert "Functional reasoning" in user
        assert "Gene-disease specificity" in user

    def test_zero_shot_asks_for_paragraphs(self, l3_record):
        _, user = format_vp_l3_prompt(l3_record)
        assert "3-5 paragraphs" in user

    def test_3shot_includes_gold_reasoning(self, l3_record):
        examples = [
            {
                "context_text": "Variant: BRCA2 c.7397T>C\nContext details...",
                "gold_reasoning": "This variant is benign because of high population frequency.",
            }
        ]
        _, user = format_vp_l3_prompt(l3_record, "3-shot", examples)
        assert "Explanation:" in user
        assert "high population frequency" in user

    def test_3shot_truncates_long_context(self, l3_record):
        long_context = "A " * 2000  # 4000 chars
        examples = [
            {
                "context_text": long_context,
                "gold_reasoning": "Explanation text",
            }
        ]
        _, user = format_vp_l3_prompt(l3_record, "3-shot", examples)
        assert "[...]" in user

    def test_3shot_truncates_long_reasoning(self, l3_record):
        long_reasoning = "B " * 1000  # 2000 chars
        examples = [
            {
                "context_text": "Short context",
                "gold_reasoning": long_reasoning,
            }
        ]
        _, user = format_vp_l3_prompt(l3_record, "3-shot", examples)
        assert "[...]" in user

    def test_mentions_benign_classification(self, l3_record):
        _, user = format_vp_l3_prompt(l3_record)
        assert "benign" in user.lower()

    def test_mentions_ba1_bs1(self, l3_record):
        _, user = format_vp_l3_prompt(l3_record)
        assert "BA1" in user
        assert "BS1" in user


class TestTruncateText:
    def test_short_text_unchanged(self):
        assert _truncate_text("hello", 100) == "hello"

    def test_long_text_truncated(self):
        text = "word " * 100
        result = _truncate_text(text, 50)
        assert result.endswith("[...]")
        assert len(result) <= 60  # 50 + word boundary + [...]

    def test_empty_string(self):
        assert _truncate_text("", 100) == ""

    def test_exact_boundary(self):
        text = "abcde"
        assert _truncate_text(text, 5) == "abcde"


# ── VP-L4 Tests ─────────────────────────────────────────────────────────


class TestVPL4Prompt:
    def test_zero_shot_binary(self, l4_record):
        _, user = format_vp_l4_prompt(l4_record)
        assert "tested" in user.lower()
        assert "untested" in user.lower()

    def test_zero_shot_asks_for_evidence(self, l4_record):
        _, user = format_vp_l4_prompt(l4_record)
        assert "evidence" in user.lower()

    def test_mentions_clinvar(self, l4_record):
        _, user = format_vp_l4_prompt(l4_record)
        assert "ClinVar" in user

    def test_3shot_includes_examples(self, l4_record):
        examples = [
            {
                "context_text": "Gene: TP53\nVariant: chr17:7579472 G>A\n...",
                "gold_answer": "tested",
            },
            {
                "context_text": "Gene: OBSCN\nVariant: chr1:228432000 C>T\n...",
                "gold_answer": "untested",
            },
        ]
        _, user = format_vp_l4_prompt(l4_record, "3-shot", examples)
        assert "TP53" in user
        assert "OBSCN" in user
        assert "Answer: tested" in user
        assert "Answer: untested" in user


# ── Dispatch Tests ──────────────────────────────────────────────────────


class TestFormatDispatch:
    def test_all_tasks_registered(self):
        assert set(VP_TASK_FORMATTERS.keys()) == {"vp-l1", "vp-l2", "vp-l3", "vp-l4"}

    def test_dispatch_returns_tuple(self, l1_record):
        result = format_vp_prompt("vp-l1", l1_record)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_invalid_task_raises(self, l1_record):
        with pytest.raises(ValueError, match="Unknown task"):
            format_vp_prompt("vp-l5", l1_record)


# ── Constants Tests ─────────────────────────────────────────────────────


class TestConstants:
    def test_4way_categories(self):
        assert len(VP_L1_CATEGORIES) == 4

    def test_fewshot_seeds_count(self):
        assert len(FEWSHOT_SEEDS) == 3

    def test_system_prompt_clinical_genetics(self):
        assert "clinical geneticist" in VP_SYSTEM_PROMPT
        assert "ACMG" in VP_SYSTEM_PROMPT
