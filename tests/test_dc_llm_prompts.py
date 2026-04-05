"""Tests for DC LLM prompt template module."""

import pytest

from negbiodb_dc.llm_prompts import (
    DC_L1_ANSWER_FORMAT,
    DC_L1_CATEGORIES,
    DC_L4_ANSWER_FORMAT,
    DC_SYSTEM_PROMPT,
    DC_TASK_FORMATTERS,
    FEWSHOT_SEEDS,
    format_dc_l1_prompt,
    format_dc_l2_prompt,
    format_dc_l3_prompt,
    format_dc_l4_prompt,
    format_dc_prompt,
)


# ── Constants tests ───────────────────────────────────────────────────


class TestConstants:
    def test_system_prompt_nonempty(self):
        assert len(DC_SYSTEM_PROMPT) > 20

    def test_l1_categories(self):
        assert set(DC_L1_CATEGORIES.keys()) == {"A", "B", "C", "D"}

    def test_fewshot_seeds(self):
        assert len(FEWSHOT_SEEDS) == 3

    def test_task_formatters_keys(self):
        assert set(DC_TASK_FORMATTERS.keys()) == {"dc-l1", "dc-l2", "dc-l3", "dc-l4"}


# ── L1 prompt tests ──────────────────────────────────────────────────


class TestL1Prompt:
    @pytest.fixture
    def record(self):
        return {
            "context_text": "Drug A: Aspirin\nDrug B: Ibuprofen\nTargets: PTGS1",
            "gold_answer": "C",
        }

    def test_zero_shot(self, record):
        sys, user = format_dc_l1_prompt(record, config="zero-shot")
        assert sys == DC_SYSTEM_PROMPT
        assert "Aspirin" in user
        assert "single letter" in user.lower()

    def test_3_shot(self, record):
        examples = [
            {"context_text": "Drug A: X\nDrug B: Y", "gold_answer": "A"},
            {"context_text": "Drug A: P\nDrug B: Q", "gold_answer": "B"},
            {"context_text": "Drug A: M\nDrug B: N", "gold_answer": "D"},
        ]
        sys, user = format_dc_l1_prompt(record, config="3-shot", fewshot_examples=examples)
        assert "Drug A: X" in user
        assert "Answer: A" in user
        assert "Aspirin" in user

    def test_no_examples_falls_back_to_zero_shot(self, record):
        _, user_zero = format_dc_l1_prompt(record, config="zero-shot")
        _, user_3shot_empty = format_dc_l1_prompt(record, config="3-shot", fewshot_examples=None)
        assert user_zero == user_3shot_empty


# ── L2 prompt tests ──────────────────────────────────────────────────


class TestL2Prompt:
    @pytest.fixture
    def record(self):
        return {
            "context_text": "Drug Combination Report: Aspirin + Ibuprofen\nShared targets: 2",
        }

    def test_zero_shot(self, record):
        sys, user = format_dc_l2_prompt(record, config="zero-shot")
        assert "JSON" in user
        assert "interaction_type" in user
        assert "mechanism_of_interaction" in user

    def test_3_shot(self, record):
        examples = [
            {
                "context_text": "Report: X + Y",
                "gold_extraction": {"interaction_type": "synergistic"},
            }
        ]
        sys, user = format_dc_l2_prompt(record, config="3-shot", fewshot_examples=examples)
        assert "Report: X + Y" in user
        assert "synergistic" in user


# ── L3 prompt tests ──────────────────────────────────────────────────


class TestL3Prompt:
    @pytest.fixture
    def record(self):
        return {
            "context_text": "Drug Combination: Aspirin + Caffeine\nOutcome: Antagonistic",
        }

    def test_zero_shot(self, record):
        sys, user = format_dc_l3_prompt(record, config="zero-shot")
        assert "antagonistic" in user.lower()
        assert "Mechanistic reasoning" in user or "mechanistic" in user.lower()

    def test_3_shot_truncation(self, record):
        long_context = "A" * 2000
        examples = [
            {"context_text": long_context, "gold_reasoning": "B" * 1000},
        ]
        _, user = format_dc_l3_prompt(record, config="3-shot", fewshot_examples=examples)
        assert "[...]" in user


# ── L4 prompt tests ──────────────────────────────────────────────────


class TestL4Prompt:
    @pytest.fixture
    def record(self):
        return {
            "context_text": "Drug A: Aspirin\nDrug B: Caffeine",
            "gold_answer": "tested",
        }

    def test_zero_shot(self, record):
        sys, user = format_dc_l4_prompt(record, config="zero-shot")
        assert "tested" in user.lower() and "untested" in user.lower()
        assert "DrugComb" in user or "ALMANAC" in user

    def test_3_shot(self, record):
        examples = [
            {"context_text": "Drug A: X\nDrug B: Y", "gold_answer": "tested"},
        ]
        _, user = format_dc_l4_prompt(record, config="3-shot", fewshot_examples=examples)
        assert "Answer: tested" in user


# ── Dispatch tests ────────────────────────────────────────────────────


class TestDispatch:
    def test_format_dc_prompt_l1(self):
        rec = {"context_text": "test context"}
        sys, user = format_dc_prompt("dc-l1", rec)
        assert sys == DC_SYSTEM_PROMPT
        assert "test context" in user

    def test_format_dc_prompt_l2(self):
        rec = {"context_text": "test context"}
        sys, user = format_dc_prompt("dc-l2", rec)
        assert "JSON" in user

    def test_format_dc_prompt_l3(self):
        rec = {"context_text": "test context"}
        sys, user = format_dc_prompt("dc-l3", rec)
        assert "antagonistic" in user.lower()

    def test_format_dc_prompt_l4(self):
        rec = {"context_text": "test context"}
        sys, user = format_dc_prompt("dc-l4", rec)
        assert "tested" in user.lower()

    def test_invalid_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            format_dc_prompt("dc-l5", {"context_text": "x"})
