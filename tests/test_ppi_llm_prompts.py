"""Tests for PPI LLM prompt templates (src/negbiodb_ppi/llm_prompts.py)."""

import pytest

from negbiodb_ppi.llm_prompts import (
    FEWSHOT_SEEDS,
    PPI_L1_CATEGORIES,
    PPI_SYSTEM_PROMPT,
    PPI_TASK_FORMATTERS,
    _L3_MAX_EXAMPLE_CHARS,
    _L3_MAX_REASONING_CHARS,
    _truncate_text,
    format_ppi_l1_prompt,
    format_ppi_l2_prompt,
    format_ppi_l3_prompt,
    format_ppi_l4_prompt,
    format_ppi_prompt,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_l1_record():
    return {
        "context_text": (
            "Protein 1: TP53 (P04637)\n"
            "Protein 2: BRCA1 (P38398)\n\n"
            "A co-immunoprecipitation (co-IP) assay was performed to test "
            "for direct binding between TP53 and BRCA1. No physical "
            "interaction was detected under the experimental conditions used."
        ),
        "gold_answer": "A",
    }


@pytest.fixture
def sample_l2_record():
    return {
        "context_text": (
            "A study using co-immunoprecipitation (co-IP) assay tested "
            "multiple protein pairs for physical interaction.\n\n"
            "- TP53 and CDK2: no interaction detected\n"
            "- BRCA1 and ESR1: no interaction detected"
        ),
        "gold_answer": "co-immunoprecipitation (co-IP) assay",
        "gold_extraction": {
            "non_interacting_pairs": [
                {"protein_1": "TP53", "protein_2": "CDK2",
                 "method": "co-immunoprecipitation (co-IP) assay", "evidence_strength": "strong"},
                {"protein_1": "BRCA1", "protein_2": "ESR1",
                 "method": "co-immunoprecipitation (co-IP) assay", "evidence_strength": "strong"},
            ],
            "total_negative_count": 2,
            "positive_interactions_mentioned": False,
        },
    }


@pytest.fixture
def sample_l3_record():
    return {
        "context_text": (
            "Protein 1: TP53 (P04637, 393 AA)\n"
            "  Function: Acts as a tumor suppressor.\n"
            "  Location: Nucleus\n"
            "  Domains: p53 DNA-binding domain\n\n"
            "Protein 2: INS (P01308, 110 AA)\n"
            "  Function: Regulates glucose metabolism.\n"
            "  Location: Extracellular\n"
            "  Domains: Insulin/IGF domain\n\n"
            "Experimental evidence: co-immunoprecipitation assay confirmed "
            "no physical interaction."
        ),
        "gold_answer": "intact",
    }


@pytest.fixture
def sample_l4_record():
    return {
        "context_text": (
            "Protein 1: TP53 (Cellular tumor antigen p53)\n"
            "Protein 2: BRCA1 (Breast cancer type 1 susceptibility protein)\n"
            "Organism: Homo sapiens\n\n"
            "Has this protein pair been experimentally tested for physical interaction?"
        ),
        "gold_answer": "tested",
    }


@pytest.fixture
def l1_fewshot_examples():
    return [
        {"context_text": "co-IP found no binding between X and Y", "gold_answer": "A"},
        {"context_text": "Y2H screen negative for X-Y pair", "gold_answer": "B"},
        {"context_text": "ML predicts X and Y not in same complex", "gold_answer": "C"},
    ]


# ── PPI-L1 Tests ──────────────────────────────────────────────────────────


class TestPPIL1Prompt:
    def test_zero_shot_format(self, sample_l1_record):
        system, user = format_ppi_l1_prompt(sample_l1_record, "zero-shot")
        assert system == PPI_SYSTEM_PROMPT
        assert "TP53" in user
        assert "BRCA1" in user
        assert "A)" in user
        assert "D)" in user
        assert "single letter" in user

    def test_three_shot_format(self, sample_l1_record, l1_fewshot_examples):
        system, user = format_ppi_l1_prompt(sample_l1_record, "3-shot", l1_fewshot_examples)
        assert "examples" in user.lower()
        assert "---" in user
        assert "Answer: A" in user
        assert "Answer: B" in user
        assert "TP53" in user

    def test_all_four_categories_in_prompt(self, sample_l1_record):
        _, user = format_ppi_l1_prompt(sample_l1_record, "zero-shot")
        for letter in "ABCD":
            assert f"{letter})" in user, f"Category {letter} missing from prompt"

    def test_no_e_category(self, sample_l1_record):
        """PPI uses 4-way (A-D) unlike CT's 5-way (A-E)."""
        _, user = format_ppi_l1_prompt(sample_l1_record, "zero-shot")
        assert "E)" not in user

    def test_category_descriptions_match(self, sample_l1_record):
        _, user = format_ppi_l1_prompt(sample_l1_record, "zero-shot")
        assert "Direct experimental" in user
        assert "Systematic screen" in user
        assert "Computational inference" in user
        assert "Database score absence" in user


# ── PPI-L2 Tests ──────────────────────────────────────────────────────────


class TestPPIL2Prompt:
    def test_zero_shot_format(self, sample_l2_record):
        system, user = format_ppi_l2_prompt(sample_l2_record, "zero-shot")
        assert system == PPI_SYSTEM_PROMPT
        assert "TP53" in user
        assert "non_interacting_pairs" in user
        assert "JSON" in user

    def test_required_json_fields_in_prompt(self, sample_l2_record):
        _, user = format_ppi_l2_prompt(sample_l2_record, "zero-shot")
        for field in ["non_interacting_pairs", "total_negative_count",
                       "positive_interactions_mentioned"]:
            assert field in user, f"Field {field} missing from L2 prompt"

    def test_three_shot_format(self, sample_l2_record):
        examples = [
            {
                "context_text": "Y2H screen tested A-B pair. No interaction.",
                "gold_extraction": {"non_interacting_pairs": [{"protein_1": "A", "protein_2": "B"}]},
            },
        ]
        _, user = format_ppi_l2_prompt(sample_l2_record, "3-shot", examples)
        assert "Y2H" in user
        assert "Extracted:" in user


# ── PPI-L3 Tests ──────────────────────────────────────────────────────────


class TestPPIL3Prompt:
    def test_zero_shot_format(self, sample_l3_record):
        system, user = format_ppi_l3_prompt(sample_l3_record, "zero-shot")
        assert system == PPI_SYSTEM_PROMPT
        assert "TP53" in user
        assert "NOT" in user

    def test_four_dimensions_in_prompt(self, sample_l3_record):
        _, user = format_ppi_l3_prompt(sample_l3_record, "zero-shot")
        for dim in ["Biological plausibility", "Structural reasoning",
                     "Mechanistic completeness", "Specificity"]:
            assert dim in user, f"Dimension '{dim}' missing from L3 prompt"

    def test_three_shot_format(self, sample_l3_record):
        examples = [
            {
                "context_text": "Protein 1: X\nProtein 2: Y",
                "gold_reasoning": "The proteins are in different compartments...",
            },
        ]
        _, user = format_ppi_l3_prompt(sample_l3_record, "3-shot", examples)
        assert "different compartments" in user

    def test_gold_reasoning_appears_in_prompt(self, sample_l3_record):
        """gold_reasoning field shows up in Explanation section."""
        reasoning = "These proteins have distinct biological roles."
        examples = [
            {
                "context_text": "Protein 1: A\nProtein 2: B",
                "gold_reasoning": reasoning,
            },
        ]
        _, user = format_ppi_l3_prompt(sample_l3_record, "3-shot", examples)
        assert reasoning in user
        assert "Explanation:" in user

    def test_missing_gold_reasoning_shows_na(self, sample_l3_record):
        """Records without gold_reasoning should show 'N/A'."""
        examples = [
            {"context_text": "Protein 1: A\nProtein 2: B"},
        ]
        _, user = format_ppi_l3_prompt(sample_l3_record, "3-shot", examples)
        assert "Explanation:\nN/A" in user

    def test_truncation_of_long_context(self, sample_l3_record):
        """Long context_text in fewshot examples gets truncated."""
        long_context = "A " * 2000  # ~4000 chars, well above _L3_MAX_EXAMPLE_CHARS
        examples = [
            {
                "context_text": long_context,
                "gold_reasoning": "Short reasoning.",
            },
        ]
        _, user = format_ppi_l3_prompt(sample_l3_record, "3-shot", examples)
        # The full 4000-char context should NOT appear
        assert long_context not in user
        # But truncation marker should
        assert "[...]" in user

    def test_truncation_of_long_reasoning(self, sample_l3_record):
        """Long gold_reasoning gets truncated."""
        long_reasoning = "B " * 1000  # ~2000 chars, above _L3_MAX_REASONING_CHARS
        examples = [
            {
                "context_text": "Short context.",
                "gold_reasoning": long_reasoning,
            },
        ]
        _, user = format_ppi_l3_prompt(sample_l3_record, "3-shot", examples)
        assert long_reasoning not in user
        assert "[...]" in user

    def test_short_text_not_truncated(self, sample_l3_record):
        """Short fewshot text should not be truncated."""
        short_context = "Protein 1: X\nProtein 2: Y"
        examples = [
            {
                "context_text": short_context,
                "gold_reasoning": "Short.",
            },
        ]
        _, user = format_ppi_l3_prompt(sample_l3_record, "3-shot", examples)
        assert short_context in user
        assert "Short." in user
        # No truncation marker for short text
        # (only check within the examples section)


class TestTruncateText:
    def test_short_text_unchanged(self):
        assert _truncate_text("hello world", 100) == "hello world"

    def test_exact_limit_unchanged(self):
        text = "a" * 100
        assert _truncate_text(text, 100) == text

    def test_long_text_truncated(self):
        text = "word " * 300  # 1500 chars
        result = _truncate_text(text, 100)
        assert len(result) <= 110  # allow for "[...]"
        assert result.endswith("[...]")

    def test_truncation_at_word_boundary(self):
        text = "abcdefghij klmnopqrst uvwxyz"
        result = _truncate_text(text, 15)
        # Should truncate before second word if possible
        assert "[...]" in result
        assert len(result.replace(" [...]", "")) <= 15


# ── PPI-L4 Tests ──────────────────────────────────────────────────────────


class TestPPIL4Prompt:
    def test_zero_shot_format(self, sample_l4_record):
        system, user = format_ppi_l4_prompt(sample_l4_record, "zero-shot")
        assert system == PPI_SYSTEM_PROMPT
        assert "TP53" in user
        assert "tested" in user.lower()
        assert "untested" in user.lower()

    def test_tested_untested_instruction(self, sample_l4_record):
        _, user = format_ppi_l4_prompt(sample_l4_record, "zero-shot")
        assert "tested" in user
        assert "untested" in user

    def test_evidence_guidance(self, sample_l4_record):
        _, user = format_ppi_l4_prompt(sample_l4_record, "zero-shot")
        assert "evidence" in user.lower()

    def test_three_shot_format(self, sample_l4_record):
        examples = [
            {"context_text": "Protein 1: X\nProtein 2: Y", "gold_answer": "tested"},
            {"context_text": "Protein 1: A\nProtein 2: B", "gold_answer": "untested"},
        ]
        _, user = format_ppi_l4_prompt(sample_l4_record, "3-shot", examples)
        assert "---" in user
        assert "Answer: tested" in user
        assert "Answer: untested" in user


# ── Dispatch Tests ───────────────────────────────────────────────────────


class TestFormatDispatch:
    def test_valid_task_ids(self, sample_l1_record):
        for task_id in ["ppi-l1", "ppi-l2", "ppi-l3", "ppi-l4"]:
            system, user = format_ppi_prompt(task_id, sample_l1_record)
            assert system == PPI_SYSTEM_PROMPT
            assert len(user) > 0

    def test_invalid_task_raises(self, sample_l1_record):
        with pytest.raises(ValueError, match="Unknown task"):
            format_ppi_prompt("l1", sample_l1_record)

    def test_all_formatters_registered(self):
        assert set(PPI_TASK_FORMATTERS.keys()) == {"ppi-l1", "ppi-l2", "ppi-l3", "ppi-l4"}


# ── Constants Tests ──────────────────────────────────────────────────────


class TestConstants:
    def test_l1_categories_four_way(self):
        """PPI uses 4-way (A-D), NOT 5-way like CT."""
        assert set(PPI_L1_CATEGORIES.keys()) == {"A", "B", "C", "D"}

    def test_fewshot_seeds_count(self):
        assert len(FEWSHOT_SEEDS) == 3
        assert all(isinstance(s, int) for s in FEWSHOT_SEEDS)

    def test_system_prompt_contains_protein(self):
        assert "protein" in PPI_SYSTEM_PROMPT.lower()
