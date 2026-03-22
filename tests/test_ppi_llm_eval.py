"""Tests for PPI LLM evaluation functions (src/negbiodb_ppi/llm_eval.py)."""

import json

import pytest

from negbiodb_ppi.llm_eval import (
    PPI_EVIDENCE_KEYWORDS,
    PPI_L2_REQUIRED_FIELDS,
    PPI_L3_JUDGE_PROMPT,
    compute_all_ppi_llm_metrics,
    evaluate_ppi_l1,
    evaluate_ppi_l2,
    evaluate_ppi_l3,
    evaluate_ppi_l4,
    parse_ppi_l1_answer,
    parse_ppi_l2_response,
    parse_ppi_l3_judge_scores,
    parse_ppi_l4_answer,
)


# ── PPI-L1 Parser Tests ───────────────────────────────────────────────────


class TestParsePPIL1Answer:
    def test_single_letter_a_through_d(self):
        for letter in "ABCD":
            assert parse_ppi_l1_answer(letter) == letter

    def test_case_insensitive(self):
        assert parse_ppi_l1_answer("a") == "A"
        assert parse_ppi_l1_answer("d") == "D"

    def test_answer_colon_format(self):
        assert parse_ppi_l1_answer("Answer: D") == "D"
        assert parse_ppi_l1_answer("Answer: B") == "B"

    def test_answer_is_format(self):
        assert parse_ppi_l1_answer("The answer is C") == "C"

    def test_parenthesized(self):
        assert parse_ppi_l1_answer("(D) Database score") == "D"

    def test_letter_dot_format(self):
        assert parse_ppi_l1_answer("A. Direct experimental") == "A"

    def test_letter_with_explanation(self):
        assert parse_ppi_l1_answer("B\nThis was a systematic screen") == "B"

    def test_empty_returns_none(self):
        assert parse_ppi_l1_answer("") is None

    def test_no_valid_letter_returns_none(self):
        assert parse_ppi_l1_answer("I don't know the answer") is None

    def test_no_e_category(self):
        """PPI uses 4-way (A-D), E should fallback to None or other."""
        # "E" is not in _PPI_L1_LETTERS, so should return None
        assert parse_ppi_l1_answer("E") is None


# ── PPI-L1 Evaluator Tests ───────────────────────────────────────────────


class TestEvaluatePPIL1:
    def test_perfect_accuracy(self):
        preds = ["A", "B", "C", "D"]
        golds = ["A", "B", "C", "D"]
        result = evaluate_ppi_l1(preds, golds)
        assert result["accuracy"] == 1.0
        assert result["parse_rate"] == 1.0

    def test_zero_accuracy(self):
        preds = ["B", "A", "D", "C"]
        golds = ["A", "B", "C", "D"]
        result = evaluate_ppi_l1(preds, golds)
        assert result["accuracy"] == 0.0

    def test_per_class_accuracy(self):
        preds = ["A", "B", "C"]
        golds = ["A", "B", "D"]
        classes = ["intact", "huri", "string"]
        result = evaluate_ppi_l1(preds, golds, gold_classes=classes)
        assert "per_class_accuracy" in result
        assert result["per_class_accuracy"]["intact"] == 1.0
        assert result["per_class_accuracy"]["string"] == 0.0

    def test_per_difficulty_accuracy(self):
        preds = ["A", "A", "B", "B"]
        golds = ["A", "B", "B", "A"]
        diffs = ["easy", "easy", "hard", "hard"]
        result = evaluate_ppi_l1(preds, golds, difficulties=diffs)
        assert "per_difficulty_accuracy" in result
        assert result["per_difficulty_accuracy"]["easy"] == 0.5
        assert result["per_difficulty_accuracy"]["hard"] == 0.5

    def test_parse_failures(self):
        preds = ["A", "no valid response here at all", "C"]
        golds = ["A", "B", "C"]
        result = evaluate_ppi_l1(preds, golds)
        assert result["n_valid"] == 2
        assert result["parse_rate"] == pytest.approx(2 / 3)
        assert result["accuracy"] == 1.0

    def test_empty_predictions(self):
        result = evaluate_ppi_l1([], [])
        assert result["accuracy"] == 0.0
        assert result["n_total"] == 0

    def test_all_unparseable(self):
        preds = ["xyz", "hello", "???"]
        golds = ["A", "B", "C"]
        result = evaluate_ppi_l1(preds, golds)
        assert result["accuracy"] == 0.0
        assert result["n_valid"] == 0


# ── PPI-L2 Parser Tests ──────────────────────────────────────────────────


class TestParsePPIL2Response:
    def test_valid_json(self):
        obj = {"non_interacting_pairs": [{"protein_1": "TP53", "protein_2": "CDK2"}],
               "total_negative_count": 1}
        result = parse_ppi_l2_response(json.dumps(obj))
        assert result["total_negative_count"] == 1

    def test_json_with_code_fences(self):
        raw = '```json\n{"non_interacting_pairs": [], "total_negative_count": 0}\n```'
        result = parse_ppi_l2_response(raw)
        assert result["total_negative_count"] == 0

    def test_json_embedded_in_text(self):
        raw = 'Here is the result: {"non_interacting_pairs": []} as expected.'
        result = parse_ppi_l2_response(raw)
        assert result is not None
        assert "non_interacting_pairs" in result

    def test_invalid_json(self):
        assert parse_ppi_l2_response("not json at all") is None


# ── PPI-L2 Evaluator Tests ───────────────────────────────────────────────


class TestEvaluatePPIL2:
    def test_perfect_entity_matching(self):
        pred = json.dumps({
            "non_interacting_pairs": [
                {"protein_1": "TP53", "protein_2": "CDK2", "method": "co-IP", "evidence_strength": "strong"},
            ],
            "total_negative_count": 1,
            "positive_interactions_mentioned": False,
        })
        gold = {
            "gold_extraction": {
                "non_interacting_pairs": [
                    {"protein_1": "TP53", "protein_2": "CDK2", "method": "co-IP", "evidence_strength": "strong"},
                ],
                "total_negative_count": 1,
                "positive_interactions_mentioned": False,
            }
        }
        result = evaluate_ppi_l2([pred], [gold])
        assert result["entity_f1"] == 1.0
        assert result["schema_compliance"] == 1.0
        assert result["parse_rate"] == 1.0

    def test_missing_pair(self):
        pred = json.dumps({
            "non_interacting_pairs": [
                {"protein_1": "TP53", "protein_2": "CDK2"},
            ],
            "total_negative_count": 1,
            "positive_interactions_mentioned": False,
        })
        gold = {
            "gold_extraction": {
                "non_interacting_pairs": [
                    {"protein_1": "TP53", "protein_2": "CDK2"},
                    {"protein_1": "BRCA1", "protein_2": "ESR1"},
                ],
                "total_negative_count": 2,
                "positive_interactions_mentioned": False,
            }
        }
        result = evaluate_ppi_l2([pred], [gold])
        assert result["entity_recall"] == 0.5
        assert result["entity_precision"] == 1.0

    def test_count_accuracy(self):
        pred = json.dumps({
            "non_interacting_pairs": [],
            "total_negative_count": 3,
            "positive_interactions_mentioned": False,
        })
        gold = {"gold_extraction": {"non_interacting_pairs": [],
                "total_negative_count": 3, "positive_interactions_mentioned": False}}
        result = evaluate_ppi_l2([pred], [gold])
        assert result["count_accuracy"] == 1.0

    def test_parse_rate(self):
        preds = ['{"non_interacting_pairs": []}', "not json", '{"non_interacting_pairs": []}']
        golds = [
            {"gold_extraction": {"non_interacting_pairs": []}},
            {"gold_extraction": {"non_interacting_pairs": []}},
            {"gold_extraction": {"non_interacting_pairs": []}},
        ]
        result = evaluate_ppi_l2(preds, golds)
        assert result["parse_rate"] == pytest.approx(2 / 3)

    def test_empty_predictions(self):
        result = evaluate_ppi_l2([], [])
        assert result["n_total"] == 0


# ── PPI-L3 Judge Score Parser Tests ───────────────────────────────────────


class TestParsePPIL3JudgeScores:
    def test_valid_scores(self):
        resp = json.dumps({
            "biological_plausibility": 4,
            "structural_reasoning": 3,
            "mechanistic_completeness": 5,
            "specificity": 2,
        })
        scores = parse_ppi_l3_judge_scores(resp)
        assert scores == {
            "biological_plausibility": 4.0,
            "structural_reasoning": 3.0,
            "mechanistic_completeness": 5.0,
            "specificity": 2.0,
        }

    def test_out_of_range(self):
        resp = json.dumps({
            "biological_plausibility": 6,
            "structural_reasoning": 0,
            "mechanistic_completeness": 3,
            "specificity": 3,
        })
        scores = parse_ppi_l3_judge_scores(resp)
        assert scores is None  # 6 and 0 are out of range

    def test_missing_dimension(self):
        resp = json.dumps({
            "biological_plausibility": 4,
            "structural_reasoning": 3,
            "mechanistic_completeness": 5,
        })
        scores = parse_ppi_l3_judge_scores(resp)
        assert scores is None  # specificity missing

    def test_invalid_json(self):
        scores = parse_ppi_l3_judge_scores("not json")
        assert scores is None

    def test_ppi_specific_dimensions(self):
        """PPI L3 uses different dimensions than CT L3."""
        resp = json.dumps({
            "biological_plausibility": 4,
            "structural_reasoning": 3,
            "mechanistic_completeness": 5,
            "specificity": 2,
        })
        scores = parse_ppi_l3_judge_scores(resp)
        assert "biological_plausibility" in scores
        assert "structural_reasoning" in scores


# ── PPI-L3 Evaluator Tests ───────────────────────────────────────────────


class TestEvaluatePPIL3:
    def test_aggregation(self):
        scores = [
            {"biological_plausibility": 4.0, "structural_reasoning": 3.0,
             "mechanistic_completeness": 5.0, "specificity": 2.0},
            {"biological_plausibility": 2.0, "structural_reasoning": 5.0,
             "mechanistic_completeness": 3.0, "specificity": 4.0},
        ]
        result = evaluate_ppi_l3(scores)
        assert result["biological_plausibility"]["mean"] == pytest.approx(3.0)
        assert result["structural_reasoning"]["mean"] == pytest.approx(4.0)
        assert result["overall"]["mean"] == pytest.approx(3.5)
        assert result["n_valid"] == 2

    def test_none_handling(self):
        scores = [
            {"biological_plausibility": 4.0, "structural_reasoning": 3.0,
             "mechanistic_completeness": 5.0, "specificity": 2.0},
            None,
        ]
        result = evaluate_ppi_l3(scores)
        assert result["n_valid"] == 1
        assert result["n_total"] == 2

    def test_all_none(self):
        result = evaluate_ppi_l3([None, None])
        assert result["n_valid"] == 0
        assert result["biological_plausibility"]["mean"] == 0.0

    def test_empty(self):
        result = evaluate_ppi_l3([])
        assert result["n_valid"] == 0


# ── PPI-L4 Parser Tests ──────────────────────────────────────────────────


class TestParsePPIL4Answer:
    def test_tested(self):
        answer, evidence = parse_ppi_l4_answer("tested\nIntAct curated non-interaction")
        assert answer == "tested"
        assert "IntAct" in evidence

    def test_untested(self):
        answer, evidence = parse_ppi_l4_answer("untested\nNo interaction databases found")
        assert answer == "untested"
        assert "No interaction" in evidence

    def test_not_tested_variant(self):
        answer, _ = parse_ppi_l4_answer("not tested\nReasoning...")
        assert answer == "untested"

    def test_not_been_tested_variant(self):
        answer, _ = parse_ppi_l4_answer("This pair has not been tested\nEvidence...")
        assert answer == "untested"

    def test_no_evidence(self):
        answer, evidence = parse_ppi_l4_answer("tested")
        assert answer == "tested"
        assert evidence is None

    def test_empty(self):
        answer, evidence = parse_ppi_l4_answer("")
        assert answer is None
        assert evidence is None


# ── PPI-L4 Evaluator Tests ───────────────────────────────────────────────


class TestEvaluatePPIL4:
    def test_perfect_accuracy(self):
        preds = ["tested\nIntAct", "untested\nNo data"]
        golds = ["tested", "untested"]
        result = evaluate_ppi_l4(preds, golds)
        assert result["accuracy"] == 1.0

    def test_temporal_pre_2015_post_2020(self):
        """PPI uses pre_2015/post_2020 (not CT's pre_2020/post_2023)."""
        preds = ["tested\nIntAct", "tested\nHuRI", "untested\nNone", "tested\nBioGRID"]
        golds = ["tested", "tested", "untested", "untested"]
        temporal = ["pre_2015", "post_2020", "pre_2015", "post_2020"]
        result = evaluate_ppi_l4(preds, golds, temporal_groups=temporal)
        assert result["accuracy_pre_2015"] == 1.0
        assert result["accuracy_post_2020"] == 0.5

    def test_contamination_flag(self):
        """Flag when pre_2015 accuracy exceeds post_2020 by >15%."""
        preds = ["tested\nA", "tested\nB", "tested\nC", "untested\nD",
                  "tested\nE", "tested\nF", "tested\nG", "tested\nH"]
        golds = ["tested", "tested", "tested", "untested",
                  "untested", "untested", "untested", "untested"]
        temporal = ["pre_2015", "pre_2015", "pre_2015", "pre_2015",
                     "post_2020", "post_2020", "post_2020", "post_2020"]
        result = evaluate_ppi_l4(preds, golds, temporal)
        assert result["contamination_flag"] is True
        assert result["contamination_gap"] == pytest.approx(1.0)

    def test_no_contamination(self):
        preds = ["tested\nA", "untested\nB"]
        golds = ["tested", "untested"]
        temporal = ["pre_2015", "post_2020"]
        result = evaluate_ppi_l4(preds, golds, temporal)
        assert result["contamination_flag"] is False

    def test_evidence_citation_rate(self):
        preds = [
            "tested\nThis pair was tested in IntAct using co-IP assay",
            "tested\nI think so",  # too short, no keywords
        ]
        golds = ["tested", "tested"]
        result = evaluate_ppi_l4(preds, golds)
        assert result["evidence_citation_rate"] == 0.5

    def test_ppi_evidence_keywords(self):
        """PPI-specific keywords differ from CT and DTI."""
        assert "intact" in PPI_EVIDENCE_KEYWORDS
        assert "huri" in PPI_EVIDENCE_KEYWORDS
        assert "biogrid" in PPI_EVIDENCE_KEYWORDS
        assert "co-ip" in PPI_EVIDENCE_KEYWORDS
        assert "two-hybrid" in PPI_EVIDENCE_KEYWORDS
        assert "pulldown" in PPI_EVIDENCE_KEYWORDS
        # CT/DTI keywords should NOT be here
        assert "nct" not in PPI_EVIDENCE_KEYWORDS
        assert "chembl" not in PPI_EVIDENCE_KEYWORDS

    def test_empty(self):
        result = evaluate_ppi_l4([], [])
        assert result["accuracy"] == 0.0


# ── Dispatch Tests ───────────────────────────────────────────────────────


class TestDispatch:
    def test_ppi_l1_dispatch(self):
        preds = ["A", "B"]
        gold = [{"gold_answer": "A", "gold_category": "intact", "difficulty": "easy"},
                {"gold_answer": "B", "gold_category": "huri", "difficulty": "hard"}]
        result = compute_all_ppi_llm_metrics("ppi-l1", preds, gold)
        assert result["accuracy"] == 1.0

    def test_ppi_l4_dispatch(self):
        preds = ["tested\nIntAct", "untested\nNone"]
        gold = [{"gold_answer": "tested", "temporal_group": "pre_2015"},
                {"gold_answer": "untested", "temporal_group": "post_2020"}]
        result = compute_all_ppi_llm_metrics("ppi-l4", preds, gold)
        assert result["accuracy"] == 1.0

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            compute_all_ppi_llm_metrics("l1", ["A"], [{"gold_answer": "A"}])

    def test_ppi_l2_dispatch(self):
        pred_obj = {"non_interacting_pairs": [{"protein_1": "X", "protein_2": "Y"}],
                     "total_negative_count": 1, "positive_interactions_mentioned": False}
        preds = [json.dumps(pred_obj)]
        gold = [{"gold_extraction": {"non_interacting_pairs": [{"protein_1": "X", "protein_2": "Y"}],
                 "total_negative_count": 1, "positive_interactions_mentioned": False}}]
        result = compute_all_ppi_llm_metrics("ppi-l2", preds, gold)
        assert result["entity_f1"] == 1.0

    def test_ppi_l3_dispatch(self):
        resp = json.dumps({
            "biological_plausibility": 4,
            "structural_reasoning": 3,
            "mechanistic_completeness": 5,
            "specificity": 2,
        })
        result = compute_all_ppi_llm_metrics("ppi-l3", [resp], [{}])
        assert result["n_valid"] == 1
