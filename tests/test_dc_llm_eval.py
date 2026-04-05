"""Tests for DC LLM evaluation module."""

import json

import numpy as np
import pytest

from negbiodb_dc.llm_eval import (
    DC_L3_JUDGE_PROMPT,
    compute_all_dc_llm_metrics,
    evaluate_dc_l1,
    evaluate_dc_l2,
    evaluate_dc_l3,
    evaluate_dc_l4,
    parse_dc_l1_answer,
    parse_dc_l2_response,
    parse_dc_l3_judge_scores,
    parse_dc_l4_answer,
)


# ── L1 Parser tests ──────────────────────────────────────────────────


class TestL1Parser:
    def test_single_letter(self):
        assert parse_dc_l1_answer("A") == "A"
        assert parse_dc_l1_answer("C") == "C"

    def test_lowercase(self):
        assert parse_dc_l1_answer("b") == "B"

    def test_answer_colon_pattern(self):
        assert parse_dc_l1_answer("Answer: B") == "B"
        assert parse_dc_l1_answer("Answer:C") == "C"

    def test_parenthesized(self):
        assert parse_dc_l1_answer("(D)") == "D"

    def test_letter_with_explanation(self):
        assert parse_dc_l1_answer("A\nBecause the drugs amplify each other") == "A"

    def test_word_boundary(self):
        assert parse_dc_l1_answer("The answer is definitely C based on the evidence") == "C"

    def test_empty(self):
        assert parse_dc_l1_answer("") is None
        assert parse_dc_l1_answer("ERROR: timeout") is None

    def test_invalid(self):
        assert parse_dc_l1_answer("The result seems somewhere in the middle of the spectrum") is None


# ── L1 Evaluation tests ──────────────────────────────────────────────


class TestL1Eval:
    def test_perfect(self):
        preds = ["A", "B", "C", "D"]
        golds = ["A", "B", "C", "D"]
        result = evaluate_dc_l1(preds, golds)
        assert result["accuracy"] == 1.0
        assert result["mcc"] == 1.0
        assert result["parse_rate"] == 1.0

    def test_all_wrong(self):
        preds = ["B", "A", "D", "C"]
        golds = ["A", "B", "C", "D"]
        result = evaluate_dc_l1(preds, golds)
        assert result["accuracy"] == 0.0

    def test_partial_valid(self):
        preds = ["A", "gibberish", "C"]
        golds = ["A", "B", "C"]
        result = evaluate_dc_l1(preds, golds)
        assert result["n_valid"] == 2
        assert result["n_total"] == 3
        assert result["accuracy"] == 1.0  # Both valid are correct

    def test_empty_predictions(self):
        result = evaluate_dc_l1([], [])
        assert result["n_total"] == 0

    def test_all_invalid(self):
        preds = ["xyz", "abc"]
        golds = ["A", "B"]
        result = evaluate_dc_l1(preds, golds)
        assert result["accuracy"] == 0.0
        assert result["n_valid"] == 0


# ── L2 Parser tests ──────────────────────────────────────────────────


class TestL2Parser:
    def test_direct_json(self):
        data = {"interaction_type": "antagonistic", "shared_targets": ["PTGS1"]}
        result = parse_dc_l2_response(json.dumps(data))
        assert result == data

    def test_markdown_code_block(self):
        raw = '```json\n{"interaction_type": "synergistic"}\n```'
        result = parse_dc_l2_response(raw)
        assert result["interaction_type"] == "synergistic"

    def test_json_in_text(self):
        raw = 'Here is the result: {"interaction_type": "additive"} as described.'
        result = parse_dc_l2_response(raw)
        assert result["interaction_type"] == "additive"

    def test_empty(self):
        assert parse_dc_l2_response("") is None

    def test_invalid_json(self):
        assert parse_dc_l2_response("not json at all") is None


# ── L2 Evaluation tests ──────────────────────────────────────────────


class TestL2Eval:
    def _gold(self, interaction="antagonistic", mechanism="competitive_binding",
              pathways=None, targets=None):
        return {
            "gold_extraction": {
                "drug_a": {"name": "Aspirin"},
                "drug_b": {"name": "Ibuprofen"},
                "shared_targets": targets or ["PTGS1", "PTGS2"],
                "interaction_type": interaction,
                "mechanism_of_interaction": mechanism,
                "affected_pathways": pathways or ["cyclooxygenase pathway"],
            }
        }

    def test_perfect_extraction(self):
        gold = self._gold()
        pred_json = json.dumps(gold["gold_extraction"])
        result = evaluate_dc_l2([pred_json], [gold])
        assert result["parse_rate"] == 1.0
        assert result["schema_compliance"] == 1.0
        assert result["interaction_accuracy"] == 1.0
        assert result["mechanism_accuracy"] == 1.0

    def test_wrong_interaction(self):
        gold = self._gold(interaction="antagonistic")
        pred = dict(gold["gold_extraction"])
        pred["interaction_type"] = "synergistic"
        result = evaluate_dc_l2([json.dumps(pred)], [gold])
        assert result["interaction_accuracy"] == 0.0

    def test_wrong_mechanism(self):
        gold = self._gold(mechanism="competitive_binding")
        pred = dict(gold["gold_extraction"])
        pred["mechanism_of_interaction"] = "pathway_crosstalk"
        result = evaluate_dc_l2([json.dumps(pred)], [gold])
        assert result["mechanism_accuracy"] == 0.0

    def test_pathway_f1(self):
        gold = self._gold(pathways=["pathway_a", "pathway_b"])
        pred = dict(gold["gold_extraction"])
        pred["affected_pathways"] = ["pathway_a", "pathway_c"]  # 1 TP, 1 FP, 1 FN
        result = evaluate_dc_l2([json.dumps(pred)], [gold])
        assert 0.4 < result["pathway_f1"] < 0.6  # F1 = 2/(2+1+1) ≈ 0.5

    def test_target_f1(self):
        gold = self._gold(targets=["PTGS1", "PTGS2"])
        pred = dict(gold["gold_extraction"])
        pred["shared_targets"] = ["PTGS1"]  # 1 TP, 0 FP, 1 FN → F1 = 2/3
        result = evaluate_dc_l2([json.dumps(pred)], [gold])
        assert 0.6 < result["target_f1"] < 0.7

    def test_unparseable_predictions(self):
        gold = self._gold()
        result = evaluate_dc_l2(["not json"], [gold])
        assert result["parse_rate"] == 0.0
        assert result["n_parsed"] == 0

    def test_empty(self):
        result = evaluate_dc_l2([], [])
        assert result["n_total"] == 0

    def test_mechanism_f1_composite(self):
        gold = self._gold(mechanism="competitive_binding", pathways=["path_a"])
        pred = dict(gold["gold_extraction"])
        # Correct mechanism, correct pathway → both 1.0 → mechanism_f1=1.0
        result = evaluate_dc_l2([json.dumps(pred)], [gold])
        assert result["mechanism_f1"] == 1.0


# ── L3 Parser tests ──────────────────────────────────────────────────


class TestL3Parser:
    def test_json_format(self):
        raw = json.dumps({
            "mechanistic_reasoning": 4,
            "pathway_analysis": 3,
            "pharmacological_context": 5,
            "therapeutic_relevance": 4,
        })
        scores = parse_dc_l3_judge_scores(raw)
        assert scores is not None
        assert scores["mechanistic_reasoning"] == 4.0
        assert scores["pathway_analysis"] == 3.0

    def test_json_in_code_block(self):
        raw = '```json\n{"mechanistic_reasoning": 3, "pathway_analysis": 4, "pharmacological_context": 2, "therapeutic_relevance": 5}\n```'
        scores = parse_dc_l3_judge_scores(raw)
        assert scores is not None
        assert len(scores) == 4

    def test_key_value_format(self):
        raw = (
            "mechanistic_reasoning: 4\n"
            "pathway_analysis: 3\n"
            "pharmacological_context: 5\n"
            "therapeutic_relevance: 4"
        )
        scores = parse_dc_l3_judge_scores(raw)
        assert scores is not None
        assert scores["pharmacological_context"] == 5.0

    def test_empty(self):
        assert parse_dc_l3_judge_scores("") is None
        assert parse_dc_l3_judge_scores("no scores here") is None

    def test_out_of_range_ignored(self):
        raw = json.dumps({
            "mechanistic_reasoning": 6,  # Out of 1-5 range
            "pathway_analysis": 3,
            "pharmacological_context": 0,  # Out of range
            "therapeutic_relevance": 4,
        })
        scores = parse_dc_l3_judge_scores(raw)
        assert scores is not None
        assert "mechanistic_reasoning" not in scores
        assert scores["pathway_analysis"] == 3.0


# ── L3 Evaluation tests ──────────────────────────────────────────────


class TestL3Eval:
    def test_perfect_scores(self):
        outputs = [
            json.dumps({
                "mechanistic_reasoning": 5,
                "pathway_analysis": 5,
                "pharmacological_context": 5,
                "therapeutic_relevance": 5,
            })
            for _ in range(5)
        ]
        result = evaluate_dc_l3(outputs)
        assert result["overall_mean"] == 5.0
        assert result["overall_std"] == 0.0
        assert result["n_parsed"] == 5

    def test_mixed_scores(self):
        outputs = [
            json.dumps({
                "mechanistic_reasoning": 3,
                "pathway_analysis": 4,
                "pharmacological_context": 2,
                "therapeutic_relevance": 5,
            }),
            json.dumps({
                "mechanistic_reasoning": 5,
                "pathway_analysis": 4,
                "pharmacological_context": 4,
                "therapeutic_relevance": 3,
            }),
        ]
        result = evaluate_dc_l3(outputs)
        assert result["mechanistic_reasoning_mean"] == 4.0
        assert result["pathway_analysis_mean"] == 4.0

    def test_empty(self):
        result = evaluate_dc_l3([])
        assert result["n_total"] == 0
        assert result["overall_mean"] == 0.0

    def test_unparseable(self):
        result = evaluate_dc_l3(["bad output", "also bad"])
        assert result["n_parsed"] == 0
        assert result["overall_mean"] == 0.0


# ── L3 Judge prompt tests ────────────────────────────────────────────


class TestL3JudgePrompt:
    def test_prompt_has_placeholders(self):
        assert "{context_text}" in DC_L3_JUDGE_PROMPT
        assert "{response_text}" in DC_L3_JUDGE_PROMPT

    def test_prompt_has_dimensions(self):
        assert "mechanistic_reasoning" in DC_L3_JUDGE_PROMPT
        assert "pathway_analysis" in DC_L3_JUDGE_PROMPT
        assert "pharmacological_context" in DC_L3_JUDGE_PROMPT
        assert "therapeutic_relevance" in DC_L3_JUDGE_PROMPT


# ── L4 Parser tests ──────────────────────────────────────────────────


class TestL4Parser:
    def test_tested(self):
        ans, ev = parse_dc_l4_answer("tested\nFound in DrugComb")
        assert ans == "tested"
        assert "DrugComb" in ev

    def test_untested(self):
        ans, ev = parse_dc_l4_answer("untested\nNo evidence")
        assert ans == "untested"

    def test_not_tested_phrase(self):
        ans, _ = parse_dc_l4_answer("not tested\nReason...")
        assert ans == "untested"

    def test_has_not_been_tested(self):
        ans, _ = parse_dc_l4_answer("This combination has not been tested\nBecause...")
        assert ans == "untested"

    def test_empty(self):
        ans, ev = parse_dc_l4_answer("")
        assert ans is None
        assert ev is None

    def test_error_prefix(self):
        ans, ev = parse_dc_l4_answer("ERROR: timeout")
        assert ans is None


# ── L4 Evaluation tests ──────────────────────────────────────────────


class TestL4Eval:
    def test_perfect(self):
        preds = ["tested\nevidence", "untested\nno info"]
        golds = ["tested", "untested"]
        result = evaluate_dc_l4(preds, golds)
        assert result["accuracy"] == 1.0
        assert result["mcc"] == 1.0

    def test_all_wrong(self):
        preds = ["untested\n", "tested\n"]
        golds = ["tested", "untested"]
        result = evaluate_dc_l4(preds, golds)
        assert result["accuracy"] == 0.0

    def test_evidence_citation_rate(self):
        preds = [
            "tested\nThis combination was tested in DrugComb synergy screen with ZIP scores",
            "tested\nyes",
        ]
        golds = ["tested", "tested"]
        result = evaluate_dc_l4(preds, golds)
        assert result["evidence_citation_rate"] == 0.5

    def test_temporal_groups(self):
        preds = ["tested\nDrugComb", "tested\nALMANAC", "untested\n", "untested\n"]
        golds = ["tested", "tested", "untested", "untested"]
        groups = ["classic_combos", "recent_combos", "untested_plausible", "untested_rare"]
        result = evaluate_dc_l4(preds, golds, temporal_groups=groups)
        assert result["accuracy_classic_combos"] == 1.0
        assert result["accuracy_recent_combos"] == 1.0
        assert result["accuracy_untested_plausible"] == 1.0

    def test_contamination_gap(self):
        # Classic combos all correct, recent all wrong → gap > 0.20
        preds = ["tested\nyes", "tested\nyes", "untested\nno", "untested\nno"]
        golds = ["tested", "untested", "untested", "untested"]
        groups = ["classic_combos", "recent_combos", "untested_plausible", "untested_rare"]
        result = evaluate_dc_l4(preds, golds, temporal_groups=groups)
        assert result["accuracy_classic_combos"] == 1.0
        assert result["accuracy_recent_combos"] == 0.0
        assert result["contamination_gap"] == 1.0
        assert result["contamination_flag"] is True

    def test_empty(self):
        result = evaluate_dc_l4([], [])
        assert result["n_total"] == 0

    def test_all_invalid(self):
        result = evaluate_dc_l4(["xyz", "abc"], ["tested", "untested"])
        assert result["n_valid"] == 0
        assert result["accuracy"] == 0.0


# ── Dispatch tests ────────────────────────────────────────────────────


class TestDispatch:
    def test_l1_dispatch(self):
        result = compute_all_dc_llm_metrics("dc-l1", ["A", "B"], [{"gold_answer": "A"}, {"gold_answer": "B"}])
        assert result["accuracy"] == 1.0

    def test_l2_dispatch(self):
        gold = {"gold_extraction": {"interaction_type": "antagonistic", "shared_targets": []}}
        pred = json.dumps({"interaction_type": "antagonistic", "shared_targets": []})
        result = compute_all_dc_llm_metrics("dc-l2", [pred], [gold])
        assert result["interaction_accuracy"] == 1.0

    def test_l3_dispatch(self):
        output = json.dumps({
            "mechanistic_reasoning": 4, "pathway_analysis": 3,
            "pharmacological_context": 5, "therapeutic_relevance": 4,
        })
        result = compute_all_dc_llm_metrics("dc-l3", [output], [{}])
        assert result["n_parsed"] == 1

    def test_l4_dispatch(self):
        result = compute_all_dc_llm_metrics(
            "dc-l4", ["tested\nyes"], [{"gold_answer": "tested"}]
        )
        assert result["accuracy"] == 1.0

    def test_l4_with_temporal(self):
        result = compute_all_dc_llm_metrics(
            "dc-l4",
            ["tested\nyes", "untested\nno"],
            [
                {"gold_answer": "tested", "temporal_group": "classic_combos"},
                {"gold_answer": "untested", "temporal_group": "untested_rare"},
            ],
        )
        assert "accuracy_classic_combos" in result

    def test_invalid_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            compute_all_dc_llm_metrics("dc-l5", [], [])
