"""Tests for CP prompt formatting."""

from negbiodb_cp.llm_prompts import format_cp_l3_judge_prompt, format_cp_prompt


def test_format_cp_prompt_zero_shot_l1():
    system, user = format_cp_prompt(
        "cp-l1",
        {"context_text": "Example context"},
        "zero-shot",
    )
    assert "assay evidence only" in system
    assert "Answer with one letter only" in user


def test_format_cp_prompt_three_shot_reads_metadata_fields():
    record = {"context_text": "Target context"}
    examples = [{
        "context_text": "Example context",
        "gold_answer": "A",
        "gold_extraction": {"field": "value"},
        "gold_reasoning": "Grounded explanation",
    }]

    _, l2_user = format_cp_prompt("cp-l2", record, "3-shot", fewshot_examples=examples)
    assert '"field": "value"' in l2_user

    _, l3_user = format_cp_prompt("cp-l3", record, "3-shot", fewshot_examples=examples)
    assert "Grounded explanation" in l3_user


def test_format_cp_l3_judge_prompt_uses_reference_reasoning():
    system, user = format_cp_l3_judge_prompt(
        {
            "context_text": "Cell Painting summary",
            "gold_category": "inactive",
            "metadata": {"gold_reasoning": "Stay close to DMSO and do not speculate."},
        },
        "Candidate explanation",
    )
    assert "return json only" in system.lower()
    assert "Stay close to DMSO" in user
    assert "Candidate explanation" in user
