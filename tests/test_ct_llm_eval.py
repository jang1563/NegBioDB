"""Tests for CT LLM evaluation functions (src/negbiodb_ct/llm_eval.py)."""

import json

import pytest

from negbiodb_ct.llm_eval import (
    CT_EVIDENCE_KEYWORDS,
    CT_L2_REQUIRED_FIELDS,
    CT_L3_JUDGE_PROMPT,
    compute_all_ct_llm_metrics,
    evaluate_ct_l1,
    evaluate_ct_l2,
    evaluate_ct_l3,
    evaluate_ct_l4,
    parse_ct_l1_answer,
    parse_ct_l2_response,
    parse_ct_l3_judge_scores,
    parse_ct_l4_answer,
)


# ── CT-L1 Parser Tests ───────────────────────────────────────────────────


class TestParseCTL1Answer:
    def test_single_letter_a_through_e(self):
        for letter in "ABCDE":
            assert parse_ct_l1_answer(letter) == letter

    def test_case_insensitive(self):
        assert parse_ct_l1_answer("a") == "A"
        assert parse_ct_l1_answer("e") == "E"

    def test_answer_colon_format(self):
        assert parse_ct_l1_answer("Answer: E") == "E"
        assert parse_ct_l1_answer("Answer: B") == "B"

    def test_answer_is_format(self):
        assert parse_ct_l1_answer("The answer is C") == "C"

    def test_parenthesized(self):
        assert parse_ct_l1_answer("(D) Strategic discontinuation") == "D"

    def test_letter_dot_format(self):
        assert parse_ct_l1_answer("A. Safety issue") == "A"

    def test_letter_with_explanation(self):
        assert parse_ct_l1_answer("B\nThis trial failed to show efficacy") == "B"

    def test_empty_returns_none(self):
        assert parse_ct_l1_answer("") is None

    def test_no_valid_letter_returns_none(self):
        assert parse_ct_l1_answer("I don't know the answer") is None

    def test_e_category_recognized(self):
        """CT uses E (5-way) unlike DTI (4-way A-D)."""
        assert parse_ct_l1_answer("E") == "E"
        assert parse_ct_l1_answer("Answer: E") == "E"
        assert parse_ct_l1_answer("(E)") == "E"


# ── CT-L1 Evaluator Tests ───────────────────────────────────────────────


class TestEvaluateCTL1:
    def test_perfect_accuracy(self):
        preds = ["A", "B", "C", "D", "E"]
        golds = ["A", "B", "C", "D", "E"]
        result = evaluate_ct_l1(preds, golds)
        assert result["accuracy"] == 1.0
        assert result["parse_rate"] == 1.0

    def test_zero_accuracy(self):
        preds = ["B", "A", "D", "C", "A"]
        golds = ["A", "B", "C", "D", "E"]
        result = evaluate_ct_l1(preds, golds)
        assert result["accuracy"] == 0.0

    def test_per_class_accuracy(self):
        preds = ["A", "B", "C"]
        golds = ["A", "B", "D"]
        classes = ["safety", "efficacy", "strategic"]
        result = evaluate_ct_l1(preds, golds, gold_classes=classes)
        assert "per_class_accuracy" in result
        assert result["per_class_accuracy"]["safety"] == 1.0
        assert result["per_class_accuracy"]["strategic"] == 0.0

    def test_per_difficulty_accuracy(self):
        preds = ["A", "A", "B", "B"]
        golds = ["A", "B", "B", "A"]
        diffs = ["easy", "easy", "hard", "hard"]
        result = evaluate_ct_l1(preds, golds, difficulties=diffs)
        assert "per_difficulty_accuracy" in result
        assert result["per_difficulty_accuracy"]["easy"] == 0.5
        assert result["per_difficulty_accuracy"]["hard"] == 0.5

    def test_parse_failures(self):
        preds = ["A", "no valid response here at all", "C"]
        golds = ["A", "B", "C"]
        result = evaluate_ct_l1(preds, golds)
        assert result["n_valid"] == 2
        assert result["parse_rate"] == pytest.approx(2 / 3)
        assert result["accuracy"] == 1.0  # Both parsed correctly

    def test_empty_predictions(self):
        result = evaluate_ct_l1([], [])
        assert result["accuracy"] == 0.0
        assert result["n_total"] == 0

    def test_all_unparseable(self):
        preds = ["xyz", "hello", "???"]
        golds = ["A", "B", "C"]
        result = evaluate_ct_l1(preds, golds)
        assert result["accuracy"] == 0.0
        assert result["n_valid"] == 0


# ── CT-L2 Parser Tests ──────────────────────────────────────────────────


class TestParseCTL2Response:
    def test_valid_json(self):
        obj = {"failure_category": "efficacy", "failure_subcategory": "futility"}
        result = parse_ct_l2_response(json.dumps(obj))
        assert result == obj

    def test_json_with_code_fences(self):
        raw = '```json\n{"failure_category": "safety"}\n```'
        result = parse_ct_l2_response(raw)
        assert result["failure_category"] == "safety"

    def test_json_embedded_in_text(self):
        raw = 'Here is the result: {"failure_category": "enrollment"} as expected.'
        result = parse_ct_l2_response(raw)
        assert result["failure_category"] == "enrollment"

    def test_invalid_json(self):
        assert parse_ct_l2_response("not json at all") is None

    def test_partial_fields(self):
        obj = {"failure_category": "safety"}  # Missing other fields
        result = parse_ct_l2_response(json.dumps(obj))
        assert result is not None
        assert result["failure_category"] == "safety"


# ── CT-L2 Evaluator Tests ───────────────────────────────────────────────


class TestEvaluateCTL2:
    def test_perfect_schema_compliance(self):
        pred_obj = {f: "test" for f in CT_L2_REQUIRED_FIELDS}
        pred_obj["quantitative_evidence"] = True
        preds = [json.dumps(pred_obj)]
        golds = [{"gold_answer": "efficacy", "failure_category": "efficacy"}]
        result = evaluate_ct_l2(preds, golds)
        assert result["schema_compliance"] == 1.0
        assert result["parse_rate"] == 1.0

    def test_category_accuracy(self):
        pred_obj = {"failure_category": "safety", "failure_subcategory": "toxicity",
                     "affected_system": "liver", "severity_indicator": "severe",
                     "quantitative_evidence": False, "decision_maker": "dsmb",
                     "patient_impact": "hepatic injury"}
        preds = [json.dumps(pred_obj)]
        golds = [{"gold_answer": "safety"}]
        result = evaluate_ct_l2(preds, golds)
        assert result["category_accuracy"] == 1.0

    def test_wrong_category(self):
        pred_obj = {"failure_category": "efficacy"}
        preds = [json.dumps(pred_obj)]
        golds = [{"gold_answer": "safety"}]
        result = evaluate_ct_l2(preds, golds)
        assert result["category_accuracy"] == 0.0

    def test_parse_rate(self):
        preds = ['{"failure_category": "safety"}', "not json", '{"failure_category": "efficacy"}']
        golds = [{"gold_answer": "safety"}, {"gold_answer": "efficacy"}, {"gold_answer": "efficacy"}]
        result = evaluate_ct_l2(preds, golds)
        assert result["parse_rate"] == pytest.approx(2 / 3)

    def test_empty_predictions(self):
        result = evaluate_ct_l2([], [])
        assert result["n_total"] == 0


# ── CT-L3 Judge Score Parser Tests ───────────────────────────────────────


class TestParseCTL3JudgeScores:
    def test_valid_scores(self):
        resp = json.dumps({"accuracy": 4, "reasoning": 3, "completeness": 5, "specificity": 2})
        scores = parse_ct_l3_judge_scores(resp)
        assert scores == {"accuracy": 4.0, "reasoning": 3.0, "completeness": 5.0, "specificity": 2.0}

    def test_out_of_range(self):
        resp = json.dumps({"accuracy": 6, "reasoning": 0, "completeness": 3, "specificity": 3})
        scores = parse_ct_l3_judge_scores(resp)
        assert scores is None  # 6 and 0 are out of range

    def test_missing_dimension(self):
        resp = json.dumps({"accuracy": 4, "reasoning": 3, "completeness": 5})
        scores = parse_ct_l3_judge_scores(resp)
        assert scores is None  # specificity missing

    def test_invalid_json(self):
        scores = parse_ct_l3_judge_scores("not json")
        assert scores is None


# ── CT-L3 Evaluator Tests ───────────────────────────────────────────────


class TestEvaluateCTL3:
    def test_aggregation(self):
        scores = [
            {"accuracy": 4.0, "reasoning": 3.0, "completeness": 5.0, "specificity": 2.0},
            {"accuracy": 2.0, "reasoning": 5.0, "completeness": 3.0, "specificity": 4.0},
        ]
        result = evaluate_ct_l3(scores)
        assert result["accuracy"]["mean"] == pytest.approx(3.0)
        assert result["reasoning"]["mean"] == pytest.approx(4.0)
        assert result["overall"]["mean"] == pytest.approx(3.5)
        assert result["n_valid"] == 2

    def test_none_handling(self):
        scores = [
            {"accuracy": 4.0, "reasoning": 3.0, "completeness": 5.0, "specificity": 2.0},
            None,
        ]
        result = evaluate_ct_l3(scores)
        assert result["n_valid"] == 1
        assert result["n_total"] == 2

    def test_all_none(self):
        result = evaluate_ct_l3([None, None])
        assert result["n_valid"] == 0
        assert result["accuracy"]["mean"] == 0.0

    def test_empty(self):
        result = evaluate_ct_l3([])
        assert result["n_valid"] == 0


# ── CT-L4 Parser Tests ──────────────────────────────────────────────────


class TestParseCTL4Answer:
    def test_tested(self):
        answer, evidence = parse_ct_l4_answer("tested\nNCT01234567 completed in 2020")
        assert answer == "tested"
        assert "NCT01234567" in evidence

    def test_untested(self):
        answer, evidence = parse_ct_l4_answer("untested\nNo registered trials found")
        assert answer == "untested"
        assert "No registered" in evidence

    def test_not_tested_variant(self):
        answer, _ = parse_ct_l4_answer("not tested\nReasoning...")
        assert answer == "untested"

    def test_not_been_tested_variant(self):
        answer, _ = parse_ct_l4_answer("This combination has not been tested\nEvidence...")
        assert answer == "untested"

    def test_no_evidence(self):
        answer, evidence = parse_ct_l4_answer("tested")
        assert answer == "tested"
        assert evidence is None

    def test_empty(self):
        answer, evidence = parse_ct_l4_answer("")
        assert answer is None
        assert evidence is None


# ── CT-L4 Evaluator Tests ───────────────────────────────────────────────


class TestEvaluateCTL4:
    def test_perfect_accuracy(self):
        preds = ["tested\nNCT123", "untested\nNo trials"]
        golds = ["tested", "untested"]
        result = evaluate_ct_l4(preds, golds)
        assert result["accuracy"] == 1.0

    def test_temporal_pre_2020_post_2023(self):
        """CT uses pre_2020/post_2023, NOT DTI's pre_2023/post_2024."""
        preds = ["tested\nNCT001", "tested\nNCT002", "untested\nNone", "tested\nNCT003"]
        golds = ["tested", "tested", "untested", "untested"]
        temporal = ["pre_2020", "post_2023", "pre_2020", "post_2023"]
        result = evaluate_ct_l4(preds, golds, temporal_groups=temporal)
        # pre_2020: tested→tested (correct), untested→untested (correct) → 100%
        assert result["accuracy_pre_2020"] == 1.0
        # post_2023: tested→tested (correct), untested→tested (wrong) → 50%
        assert result["accuracy_post_2023"] == 0.5

    def test_contamination_flag(self):
        """Flag when pre_2020 accuracy exceeds post_2023 by >15%."""
        preds = ["tested\nA", "tested\nB", "tested\nC", "untested\nD",
                  "tested\nE", "tested\nF", "tested\nG", "tested\nH"]
        golds = ["tested", "tested", "tested", "untested",
                  "untested", "untested", "untested", "untested"]
        temporal = ["pre_2020", "pre_2020", "pre_2020", "pre_2020",
                     "post_2023", "post_2023", "post_2023", "post_2023"]
        result = evaluate_ct_l4(preds, golds, temporal)
        # pre_2020: 3 correct + 1 correct = 4/4 = 100%
        # post_2023: 0/4 = 0%
        assert result["contamination_flag"] is True
        assert result["contamination_gap"] == pytest.approx(1.0)

    def test_no_contamination(self):
        preds = ["tested\nA", "untested\nB"]
        golds = ["tested", "untested"]
        temporal = ["pre_2020", "post_2023"]
        result = evaluate_ct_l4(preds, golds, temporal)
        assert result["contamination_flag"] is False

    def test_evidence_citation_rate(self):
        preds = [
            "tested\nNCT01234567 showed positive results",  # has NCT → evidence
            "tested\nI think so",  # too short, no keywords
        ]
        golds = ["tested", "tested"]
        result = evaluate_ct_l4(preds, golds)
        assert result["evidence_citation_rate"] == 0.5

    def test_ct_evidence_keywords(self):
        """CT-specific keywords differ from DTI."""
        assert "nct" in CT_EVIDENCE_KEYWORDS
        assert "clinicaltrials" in CT_EVIDENCE_KEYWORDS
        assert "fda" in CT_EVIDENCE_KEYWORDS
        assert "eudract" in CT_EVIDENCE_KEYWORDS
        # DTI keywords should NOT be here
        assert "chembl" not in CT_EVIDENCE_KEYWORDS
        assert "pubchem" not in CT_EVIDENCE_KEYWORDS

    def test_empty(self):
        result = evaluate_ct_l4([], [])
        assert result["accuracy"] == 0.0


# ── Dispatch Tests ───────────────────────────────────────────────────────


class TestDispatch:
    def test_ct_l1_dispatch(self):
        preds = ["A", "B"]
        gold = [{"gold_answer": "A", "gold_category": "safety", "difficulty": "easy"},
                {"gold_answer": "B", "gold_category": "efficacy", "difficulty": "hard"}]
        result = compute_all_ct_llm_metrics("ct-l1", preds, gold)
        assert result["accuracy"] == 1.0

    def test_ct_l4_dispatch(self):
        preds = ["tested\nNCT123", "untested\nNone"]
        gold = [{"gold_answer": "tested", "temporal_group": "pre_2020"},
                {"gold_answer": "untested", "temporal_group": "post_2023"}]
        result = compute_all_ct_llm_metrics("ct-l4", preds, gold)
        assert result["accuracy"] == 1.0

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            compute_all_ct_llm_metrics("l1", ["A"], [{"gold_answer": "A"}])

    def test_ct_l2_dispatch(self):
        pred_obj = {"failure_category": "efficacy"}
        preds = [json.dumps(pred_obj)]
        gold = [{"gold_answer": "efficacy"}]
        result = compute_all_ct_llm_metrics("ct-l2", preds, gold)
        assert result["category_accuracy"] == 1.0

    def test_ct_l3_dispatch(self):
        resp = json.dumps({"accuracy": 4, "reasoning": 3, "completeness": 5, "specificity": 2})
        result = compute_all_ct_llm_metrics("ct-l3", [resp], [{}])
        assert result["n_valid"] == 1
