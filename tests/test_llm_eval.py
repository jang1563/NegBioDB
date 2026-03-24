"""Tests for LLM benchmark evaluation functions."""

import json

import pytest

from negbiodb.llm_eval import (
    compute_all_llm_metrics,
    evaluate_l1,
    evaluate_l2,
    evaluate_l3,
    evaluate_l4,
    parse_l1_answer,
    parse_l2_response,
    parse_l3_judge_scores,
    parse_l4_answer,
)


# ── L1 MCQ Parsing ──────────────────────────────────────────────────────────


class TestParseL1Answer:
    def test_single_letter(self):
        assert parse_l1_answer("A") == "A"
        assert parse_l1_answer("b") == "B"
        assert parse_l1_answer("C") == "C"
        assert parse_l1_answer("d") == "D"

    def test_with_text(self):
        assert parse_l1_answer("A) Active") == "A"
        assert parse_l1_answer("The answer is B") == "B"

    def test_lowercase(self):
        assert parse_l1_answer("a") == "A"

    def test_invalid(self):
        assert parse_l1_answer("") is None
        assert parse_l1_answer("xyz") is None

    def test_whitespace(self):
        assert parse_l1_answer("  A  ") == "A"

    # M-5: MCQ-specific patterns
    def test_answer_colon(self):
        assert parse_l1_answer("Answer: B") == "B"

    def test_answer_is(self):
        assert parse_l1_answer("The answer is C") == "C"

    def test_parenthesized(self):
        assert parse_l1_answer("(D)") == "D"

    def test_letter_dot(self):
        assert parse_l1_answer("A. This is active") == "A"

    def test_choice_colon(self):
        assert parse_l1_answer("Choice: B") == "B"


# ── L1 MCQ Evaluation ────────────────────────────────────────────────────────


class TestEvaluateL1:
    def test_perfect(self):
        preds = ["A", "B", "C", "D"]
        gold = ["A", "B", "C", "D"]
        result = evaluate_l1(preds, gold)
        assert result["accuracy"] == 1.0
        assert result["mcc"] == 1.0
        assert result["parse_rate"] == 1.0

    def test_all_wrong(self):
        preds = ["B", "A", "D", "C"]
        gold = ["A", "B", "C", "D"]
        result = evaluate_l1(preds, gold)
        assert result["accuracy"] == 0.0

    def test_partial(self):
        preds = ["A", "B", "A", "D"]
        gold = ["A", "B", "C", "D"]
        result = evaluate_l1(preds, gold)
        assert result["accuracy"] == 0.75

    def test_unparseable(self):
        preds = ["A", "xyz", "C"]
        gold = ["A", "B", "C"]
        result = evaluate_l1(preds, gold)
        assert result["n_valid"] == 2
        assert result["parse_rate"] == pytest.approx(2 / 3)

    def test_empty(self):
        result = evaluate_l1([], [])
        assert result["accuracy"] == 0.0
        assert result["n_total"] == 0

    def test_per_class_accuracy(self):
        preds = ["A", "A", "B", "B"]
        gold = ["A", "A", "B", "C"]
        classes = ["active", "active", "inactive", "inconclusive"]
        result = evaluate_l1(preds, gold, classes)
        assert result["per_class_accuracy"]["active"] == 1.0
        assert result["per_class_accuracy"]["inactive"] == 1.0
        assert result["per_class_accuracy"]["inconclusive"] == 0.0


# ── L2 JSON Parsing ──────────────────────────────────────────────────────────


class TestParseL2Response:
    def test_plain_json(self):
        r = '{"negative_results": [], "total_inactive_count": 0}'
        result = parse_l2_response(r)
        assert result is not None
        assert result["total_inactive_count"] == 0

    def test_code_fenced_json(self):
        r = '```json\n{"total_inactive_count": 5}\n```'
        result = parse_l2_response(r)
        assert result is not None
        assert result["total_inactive_count"] == 5

    def test_json_in_text(self):
        r = 'Here is the extraction:\n{"total_inactive_count": 3}\nDone.'
        result = parse_l2_response(r)
        assert result is not None

    def test_invalid_json(self):
        assert parse_l2_response("not json at all") is None

    def test_empty(self):
        assert parse_l2_response("") is None

    def test_code_fenced_python(self):
        """Code fence with non-json language tag should be stripped."""
        r = '```python\n{"total_inactive_count": 7}\n```'
        result = parse_l2_response(r)
        assert result is not None
        assert result["total_inactive_count"] == 7

    def test_code_fenced_bare(self):
        """Code fence with no language tag should be stripped."""
        r = '```\n{"total_inactive_count": 2}\n```'
        result = parse_l2_response(r)
        assert result is not None
        assert result["total_inactive_count"] == 2


# ── L2 Evaluation ─────────────────────────────────────────────────────────────


class TestEvaluateL2:
    def test_perfect_extraction(self):
        pred = json.dumps(
            {
                "negative_results": [
                    {"compound": "aspirin", "target": "EGFR"}
                ],
                "total_inactive_count": 1,
                "positive_results_mentioned": False,
            }
        )
        gold = {
            "negative_results": [
                {"compound": "aspirin", "target": "EGFR"}
            ],
            "total_inactive_count": 1,
            "positive_results_mentioned": False,
        }
        result = evaluate_l2([pred], [gold])
        assert result["schema_compliance"] == 1.0
        assert result["entity_f1"] == 1.0

    def test_missing_entity(self):
        pred = json.dumps({"negative_results": [], "total_inactive_count": 0})
        gold = {
            "negative_results": [
                {"compound": "aspirin", "target": "EGFR"}
            ],
            "total_inactive_count": 1,
        }
        result = evaluate_l2([pred], [gold])
        assert result["entity_f1"] == 0.0


# ── L3 Judge Parsing ──────────────────────────────────────────────────────────


class TestParseL3JudgeScores:
    def test_valid(self):
        r = '{"accuracy": 4, "reasoning": 3, "completeness": 5, "specificity": 2}'
        scores = parse_l3_judge_scores(r)
        assert scores is not None
        assert scores["accuracy"] == 4.0
        assert scores["specificity"] == 2.0

    def test_out_of_range(self):
        r = '{"accuracy": 6, "reasoning": 3, "completeness": 5, "specificity": 2}'
        scores = parse_l3_judge_scores(r)
        assert scores is None  # accuracy=6 is out of range

    def test_incomplete(self):
        r = '{"accuracy": 4, "reasoning": 3}'
        scores = parse_l3_judge_scores(r)
        assert scores is None  # missing dimensions


# ── L3 Evaluation ─────────────────────────────────────────────────────────────


class TestEvaluateL3:
    def test_basic(self):
        scores = [
            {"accuracy": 4, "reasoning": 3, "completeness": 5, "specificity": 2},
            {"accuracy": 3, "reasoning": 4, "completeness": 4, "specificity": 3},
        ]
        result = evaluate_l3(scores)
        assert result["accuracy"]["mean"] == 3.5
        assert result["overall"]["mean"] == pytest.approx(3.5)
        assert result["n_valid"] == 2

    def test_empty(self):
        result = evaluate_l3([])
        assert result["accuracy"]["mean"] == 0.0
        # L-4: Empty return must have all expected keys
        assert "overall" in result
        assert result["overall"]["mean"] == 0.0
        assert result["n_valid"] == 0
        assert result["n_total"] == 0

    def test_with_none(self):
        scores = [
            {"accuracy": 4, "reasoning": 3, "completeness": 5, "specificity": 2},
            None,
        ]
        result = evaluate_l3(scores)
        assert result["n_valid"] == 1


# ── L4 Parsing ────────────────────────────────────────────────────────────────


class TestParseL4Answer:
    def test_tested(self):
        answer, evidence = parse_l4_answer("tested\nChEMBL CHEMBL25")
        assert answer == "tested"
        assert "ChEMBL" in evidence

    def test_untested(self):
        answer, evidence = parse_l4_answer("untested")
        assert answer == "untested"

    def test_tested_in_sentence(self):
        answer, _ = parse_l4_answer("This pair has been tested.")
        assert answer == "tested"

    def test_empty(self):
        answer, _ = parse_l4_answer("")
        assert answer is None

    # M-4: "not tested" negation patterns
    def test_not_tested(self):
        answer, _ = parse_l4_answer("not tested")
        assert answer == "untested"

    def test_has_not_been_tested(self):
        answer, _ = parse_l4_answer("This pair has not been tested.")
        assert answer == "untested"

    def test_not_tested_sentence(self):
        answer, _ = parse_l4_answer("This compound has not been tested against this target.")
        assert answer == "untested"

    def test_never_tested(self):
        answer, _ = parse_l4_answer("This pair has never tested positive.")
        # "never tested" should match untested
        answer2, _ = parse_l4_answer("never tested")
        assert answer2 == "untested"

    def test_never_been_tested(self):
        answer, _ = parse_l4_answer("This compound has never been tested against EGFR.")
        assert answer == "untested"

    def test_hasnt_been_tested(self):
        answer, _ = parse_l4_answer("This compound hasn't been tested against EGFR.")
        assert answer == "untested"

    def test_no_evidence_of_testing(self):
        answer, _ = parse_l4_answer("No evidence of testing for this pair.")
        assert answer == "untested"


# ── L4 Evaluation ─────────────────────────────────────────────────────────────


class TestEvaluateL4:
    def test_perfect(self):
        preds = ["tested", "untested", "tested", "untested"]
        gold = ["tested", "untested", "tested", "untested"]
        result = evaluate_l4(preds, gold)
        assert result["accuracy"] == 1.0
        assert result["mcc"] == 1.0

    def test_with_evidence(self):
        preds = [
            "tested\nChEMBL CHEMBL25 compound tested with IC50 measurement in biochemical assay",
            "untested",
            "tested\nPubChem AID123456 confirmed inactive in dose-response screening",
            "tested",
        ]
        gold = ["tested", "untested", "tested", "tested"]
        result = evaluate_l4(preds, gold)
        assert result["accuracy"] == 1.0
        assert result["evidence_citation_rate"] == pytest.approx(2 / 3)

    # M-10: Short filler evidence should not count
    def test_evidence_short_filler_rejected(self):
        """Evidence under 50 chars without keywords should not count."""
        preds = [
            "tested\nyes it was",  # short filler, no keywords
        ]
        gold = ["tested"]
        result = evaluate_l4(preds, gold)
        assert result["evidence_citation_rate"] == 0.0

    def test_evidence_keyword_short_rejected(self):
        """Short evidence WITH a keyword should NOT count (AND logic: >50 AND keyword)."""
        preds = [
            "tested\nChEMBL ID found",  # has keyword but too short
        ]
        gold = ["tested"]
        result = evaluate_l4(preds, gold)
        assert result["evidence_citation_rate"] == 0.0

    def test_temporal(self):
        preds = ["tested", "tested", "untested", "untested"]
        gold = ["tested", "untested", "tested", "untested"]
        temporal = ["pre_2023", "pre_2023", "post_2024", "post_2024"]
        result = evaluate_l4(preds, gold, temporal)
        assert result["accuracy_pre_2023"] == 0.5
        assert result["accuracy_post_2024"] == 0.5

    def test_contamination_gap_flagged(self):
        """Gap > 0.15 should set contamination_flag=True."""
        # 4 pre_2023 correct, 0 post_2024 correct → gap = 1.0
        preds = ["tested", "untested", "untested", "tested"]
        gold = ["tested", "untested", "tested", "untested"]
        temporal = ["pre_2023", "pre_2023", "post_2024", "post_2024"]
        result = evaluate_l4(preds, gold, temporal)
        assert result["accuracy_pre_2023"] == 1.0
        assert result["accuracy_post_2024"] == 0.0
        assert result["contamination_gap"] == 1.0
        assert result["contamination_flag"] is True

    def test_contamination_gap_not_flagged(self):
        """Gap <= 0.15 should set contamination_flag=False."""
        # Equal accuracy across temporal groups → gap = 0.0
        preds = ["tested", "untested", "tested", "untested"]
        gold = ["tested", "untested", "tested", "untested"]
        temporal = ["pre_2023", "pre_2023", "post_2024", "post_2024"]
        result = evaluate_l4(preds, gold, temporal)
        assert result["accuracy_pre_2023"] == 1.0
        assert result["accuracy_post_2024"] == 1.0
        assert result["contamination_gap"] == 0.0
        assert result["contamination_flag"] is False


# ── Dispatch ──────────────────────────────────────────────────────────────────


class TestComputeAllLLMMetrics:
    def test_l1_dispatch(self):
        gold = [
            {"correct_answer": "A", "class": "active"},
            {"correct_answer": "B", "class": "inactive"},
        ]
        result = compute_all_llm_metrics("l1", ["A", "B"], gold)
        assert result["accuracy"] == 1.0

    def test_l4_dispatch(self):
        gold = [
            {"correct_answer": "tested"},
            {"correct_answer": "untested"},
        ]
        result = compute_all_llm_metrics("l4", ["tested", "untested"], gold)
        assert result["accuracy"] == 1.0

    def test_invalid_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            compute_all_llm_metrics("l99", [], [])
