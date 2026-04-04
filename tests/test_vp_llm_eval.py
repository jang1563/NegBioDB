"""Tests for VP LLM evaluation functions (VP-L1 through VP-L4)."""

import json

import pytest

from negbiodb_vp.llm_eval import (
    VP_EVIDENCE_KEYWORDS,
    VP_L2_REQUIRED_FIELDS,
    VP_L3_JUDGE_PROMPT,
    _normalize_classification,
    _normalize_criteria,
    compute_all_vp_llm_metrics,
    evaluate_vp_l1,
    evaluate_vp_l2,
    evaluate_vp_l3,
    evaluate_vp_l4,
    parse_vp_l1_answer,
    parse_vp_l2_response,
    parse_vp_l3_judge_scores,
    parse_vp_l4_answer,
)


# ── VP-L1 Tests ──────────────────────────────────────────────────────────


class TestParseVPL1Answer:
    def test_single_letter(self):
        assert parse_vp_l1_answer("A") == "A"
        assert parse_vp_l1_answer("D") == "D"

    def test_lowercase(self):
        assert parse_vp_l1_answer("b") == "B"

    def test_answer_colon(self):
        assert parse_vp_l1_answer("Answer: C") == "C"

    def test_answer_is(self):
        assert parse_vp_l1_answer("The answer is B") == "B"

    def test_parenthesized(self):
        assert parse_vp_l1_answer("(D)") == "D"

    def test_letter_period(self):
        assert parse_vp_l1_answer("A. Pathogenic") == "A"

    def test_empty(self):
        assert parse_vp_l1_answer("") is None

    def test_no_valid_letter(self):
        assert parse_vp_l1_answer("Not sure about this variant") is None

    def test_rejects_E(self):
        assert parse_vp_l1_answer("E") is None

    def test_classification_colon(self):
        assert parse_vp_l1_answer("Classification: D") == "D"


class TestEvaluateVPL1:
    def test_perfect_accuracy(self):
        preds = ["A", "B", "C", "D"]
        gold = ["A", "B", "C", "D"]
        result = evaluate_vp_l1(preds, gold)
        assert result["accuracy"] == 1.0
        assert result["mcc"] == 1.0

    def test_zero_accuracy(self):
        preds = ["D", "C", "B", "A"]
        gold = ["A", "B", "C", "D"]
        result = evaluate_vp_l1(preds, gold)
        assert result["accuracy"] == 0.0

    def test_parse_rate(self):
        preds = ["A", "invalid", "C", "no answer"]
        gold = ["A", "B", "C", "D"]
        result = evaluate_vp_l1(preds, gold)
        assert result["parse_rate"] == 0.5
        assert result["n_valid"] == 2

    def test_per_class_accuracy(self):
        preds = ["A", "D", "C", "D"]
        gold = ["A", "B", "C", "D"]
        classes = ["pathogenic", "likely_benign", "vus", "benign"]
        result = evaluate_vp_l1(preds, gold, gold_classes=classes)
        assert "per_class_accuracy" in result
        assert result["per_class_accuracy"]["pathogenic"] == 1.0
        assert result["per_class_accuracy"]["likely_benign"] == 0.0

    def test_per_difficulty(self):
        preds = ["A", "B", "C", "D"]
        gold = ["A", "B", "C", "D"]
        diffs = ["easy", "easy", "hard", "hard"]
        result = evaluate_vp_l1(preds, gold, difficulties=diffs)
        assert result["per_difficulty_accuracy"]["easy"] == 1.0
        assert result["per_difficulty_accuracy"]["hard"] == 1.0

    def test_empty_predictions(self):
        result = evaluate_vp_l1([], [])
        assert result["accuracy"] == 0.0
        assert result["parse_rate"] == 0.0

    def test_all_unparseable(self):
        preds = ["garbage", "nonsense"]
        gold = ["A", "B"]
        result = evaluate_vp_l1(preds, gold)
        assert result["n_valid"] == 0
        assert result["parse_rate"] == 0.0


# ── VP-L2 Tests ──────────────────────────────────────────────────────────


class TestParseVPL2Response:
    def test_valid_json(self):
        data = {"variants": [], "total_variants_discussed": 0}
        result = parse_vp_l2_response(json.dumps(data))
        assert result is not None
        assert result["total_variants_discussed"] == 0

    def test_markdown_fence(self):
        data = {"variants": [], "total_variants_discussed": 1}
        raw = f"```json\n{json.dumps(data)}\n```"
        result = parse_vp_l2_response(raw)
        assert result is not None

    def test_embedded_json(self):
        raw = "Here is the extraction:\n{\"variants\": [], \"total_variants_discussed\": 0}\nDone."
        result = parse_vp_l2_response(raw)
        assert result is not None

    def test_invalid_json(self):
        assert parse_vp_l2_response("not json at all") is None


class TestNormalizeClassification:
    def test_exact_match(self):
        assert _normalize_classification("benign") == "benign"
        assert _normalize_classification("pathogenic") == "pathogenic"

    def test_with_spaces(self):
        assert _normalize_classification("likely benign") == "likely_benign"
        assert _normalize_classification("likely pathogenic") == "likely_pathogenic"

    def test_vus_alias(self):
        assert _normalize_classification("VUS") == "uncertain_significance"

    def test_case_insensitive(self):
        assert _normalize_classification("Benign") == "benign"
        assert _normalize_classification("PATHOGENIC") == "pathogenic"


class TestNormalizeCriteria:
    def test_basic(self):
        assert _normalize_criteria(["BA1", "BS1"]) == {"BA1", "BS1"}

    def test_lowercase(self):
        assert _normalize_criteria(["ba1", "bs1"]) == {"BA1", "BS1"}

    def test_invalid_filtered(self):
        assert _normalize_criteria(["BA1", "XX9", "BP4"]) == {"BA1", "BP4"}

    def test_all_acmg_benign_codes(self):
        codes = ["BA1", "BS1", "BS2", "BS3", "BS4", "BP1", "BP2", "BP3", "BP4", "BP5", "BP6", "BP7"]
        result = _normalize_criteria(codes)
        assert len(result) == 12


class TestEvaluateVPL2:
    def test_perfect_extraction(self):
        pred = json.dumps({
            "variants": [{"gene": "BRCA1", "hgvs": "c.5123C>A", "classification": "benign",
                          "acmg_criteria_met": ["BA1", "BS1"]}],
            "total_variants_discussed": 1,
            "classification_method": "ACMG/AMP",
        })
        gold = [{
            "gold_extraction": {
                "variants": [{"gene": "BRCA1", "hgvs": "c.5123C>A", "classification": "benign",
                              "acmg_criteria_met": ["BA1", "BS1"]}],
                "total_variants_discussed": 1,
                "classification_method": "ACMG/AMP",
            }
        }]
        result = evaluate_vp_l2([pred], gold)
        assert result["field_f1"] == 1.0
        assert result["classification_accuracy"] == 1.0
        assert result["criteria_f1"] == 1.0
        assert result["schema_compliance"] == 1.0

    def test_wrong_classification(self):
        pred = json.dumps({
            "variants": [{"gene": "BRCA1", "hgvs": "c.5123C>A", "classification": "pathogenic",
                          "acmg_criteria_met": ["BA1"]}],
            "total_variants_discussed": 1,
            "classification_method": "ACMG/AMP",
        })
        gold = [{
            "gold_extraction": {
                "variants": [{"gene": "BRCA1", "hgvs": "c.5123C>A", "classification": "benign",
                              "acmg_criteria_met": ["BA1"]}],
                "total_variants_discussed": 1,
                "classification_method": "ACMG/AMP",
            }
        }]
        result = evaluate_vp_l2([pred], gold)
        assert result["field_f1"] == 1.0  # variant matched
        assert result["classification_accuracy"] == 0.0  # but classification wrong

    def test_missing_criteria(self):
        pred = json.dumps({
            "variants": [{"gene": "BRCA1", "hgvs": "c.5123C>A", "classification": "benign",
                          "acmg_criteria_met": ["BA1"]}],
            "total_variants_discussed": 1,
            "classification_method": "ACMG/AMP",
        })
        gold = [{
            "gold_extraction": {
                "variants": [{"gene": "BRCA1", "hgvs": "c.5123C>A", "classification": "benign",
                              "acmg_criteria_met": ["BA1", "BS1", "BP4"]}],
                "total_variants_discussed": 1,
                "classification_method": "ACMG/AMP",
            }
        }]
        result = evaluate_vp_l2([pred], gold)
        assert result["criteria_precision"] == 1.0  # 1/1 correct
        assert result["criteria_recall"] == pytest.approx(1 / 3)  # 1/3 found
        assert result["criteria_f1"] < 1.0

    def test_unparseable(self):
        result = evaluate_vp_l2(["not json"], [{"gold_extraction": {"variants": [{"gene": "X", "hgvs": "c.1A>G", "acmg_criteria_met": ["BA1"]}]}}])
        assert result["parse_rate"] == 0.0
        assert result["field_f1"] == 0.0

    def test_schema_compliance(self):
        # Missing required field
        pred = json.dumps({"variants": []})
        gold = [{"gold_extraction": {"variants": [], "total_variants_discussed": 0, "classification_method": "ACMG/AMP"}}]
        result = evaluate_vp_l2([pred], gold)
        assert result["schema_compliance"] == 0.0

    def test_case_insensitive_gene_matching(self):
        pred = json.dumps({
            "variants": [{"gene": "brca1", "hgvs": "c.5123C>A", "classification": "benign",
                          "acmg_criteria_met": []}],
            "total_variants_discussed": 1,
            "classification_method": "ACMG/AMP",
        })
        gold = [{
            "gold_extraction": {
                "variants": [{"gene": "BRCA1", "hgvs": "c.5123C>A", "classification": "benign",
                              "acmg_criteria_met": []}],
                "total_variants_discussed": 1,
                "classification_method": "ACMG/AMP",
            }
        }]
        result = evaluate_vp_l2([pred], gold)
        assert result["field_f1"] == 1.0


# ── VP-L3 Tests ──────────────────────────────────────────────────────────


class TestParseVPL3JudgeScores:
    def test_valid_scores(self):
        raw = json.dumps({
            "population_reasoning": 4,
            "computational_evidence": 3,
            "functional_reasoning": 5,
            "gene_disease_specificity": 4,
        })
        result = parse_vp_l3_judge_scores(raw)
        assert result is not None
        assert result["population_reasoning"] == 4.0
        assert result["gene_disease_specificity"] == 4.0

    def test_out_of_range(self):
        raw = json.dumps({
            "population_reasoning": 6,
            "computational_evidence": 3,
            "functional_reasoning": 5,
            "gene_disease_specificity": 4,
        })
        result = parse_vp_l3_judge_scores(raw)
        assert result is None  # 6 > 5

    def test_missing_dimension(self):
        raw = json.dumps({
            "population_reasoning": 4,
            "computational_evidence": 3,
            "functional_reasoning": 5,
        })
        result = parse_vp_l3_judge_scores(raw)
        assert result is None  # missing gene_disease_specificity

    def test_invalid_json(self):
        assert parse_vp_l3_judge_scores("not json") is None


class TestEvaluateVPL3:
    def test_aggregation(self):
        scores = [
            {"population_reasoning": 4, "computational_evidence": 3,
             "functional_reasoning": 5, "gene_disease_specificity": 4},
            {"population_reasoning": 2, "computational_evidence": 3,
             "functional_reasoning": 3, "gene_disease_specificity": 2},
        ]
        result = evaluate_vp_l3(scores)
        assert result["population_reasoning"]["mean"] == 3.0
        assert result["overall"]["mean"] == pytest.approx(3.25)

    def test_all_none(self):
        result = evaluate_vp_l3([None, None])
        assert result["n_valid"] == 0
        assert result["overall"]["mean"] == 0.0

    def test_mixed_none(self):
        scores = [
            {"population_reasoning": 4, "computational_evidence": 4,
             "functional_reasoning": 4, "gene_disease_specificity": 4},
            None,
        ]
        result = evaluate_vp_l3(scores)
        assert result["n_valid"] == 1
        assert result["overall"]["mean"] == 4.0

    def test_vp_specific_dimensions(self):
        scores = [
            {"population_reasoning": 5, "computational_evidence": 5,
             "functional_reasoning": 5, "gene_disease_specificity": 5},
        ]
        result = evaluate_vp_l3(scores)
        assert "population_reasoning" in result
        assert "computational_evidence" in result
        assert "functional_reasoning" in result
        assert "gene_disease_specificity" in result


class TestVPL3JudgePrompt:
    def test_prompt_contains_vp_dimensions(self):
        assert "population_reasoning" in VP_L3_JUDGE_PROMPT
        assert "computational_evidence" in VP_L3_JUDGE_PROMPT
        assert "functional_reasoning" in VP_L3_JUDGE_PROMPT
        assert "gene_disease_specificity" in VP_L3_JUDGE_PROMPT

    def test_prompt_asks_for_json(self):
        assert "JSON" in VP_L3_JUDGE_PROMPT


# ── VP-L4 Tests ──────────────────────────────────────────────────────────


class TestParseVPL4Answer:
    def test_tested(self):
        answer, evidence = parse_vp_l4_answer("tested\nFound in ClinVar")
        assert answer == "tested"
        assert evidence == "Found in ClinVar"

    def test_untested(self):
        answer, _ = parse_vp_l4_answer("untested\nNo ClinVar entry")
        assert answer == "untested"

    def test_not_assessed(self):
        answer, _ = parse_vp_l4_answer("not assessed\nRare variant")
        assert answer == "untested"

    def test_has_not_been_tested(self):
        answer, _ = parse_vp_l4_answer("This has not been tested\nDetails")
        assert answer == "untested"

    def test_empty(self):
        answer, evidence = parse_vp_l4_answer("")
        assert answer is None
        assert evidence is None

    def test_evidence_multiline(self):
        answer, evidence = parse_vp_l4_answer("tested\nLine 1\nLine 2")
        assert answer == "tested"
        assert "Line 1" in evidence
        assert "Line 2" in evidence

    def test_no_evidence_line(self):
        answer, evidence = parse_vp_l4_answer("tested")
        assert answer == "tested"
        assert evidence is None

    def test_ambiguous_no_keywords(self):
        answer, _ = parse_vp_l4_answer("I am not sure about this variant")
        assert answer is None


class TestEvaluateVPL4:
    def test_perfect(self):
        preds = ["tested\nClinVar entry", "untested\nNo data"]
        gold = ["tested", "untested"]
        result = evaluate_vp_l4(preds, gold)
        assert result["accuracy"] == 1.0
        assert result["mcc"] == 1.0

    def test_all_wrong(self):
        preds = ["untested\nNo data", "tested\nClinVar"]
        gold = ["tested", "untested"]
        result = evaluate_vp_l4(preds, gold)
        assert result["accuracy"] == 0.0

    def test_temporal_groups_vp(self):
        preds = ["tested\nx", "tested\nx", "untested\nx", "untested\nx"]
        gold = ["tested", "tested", "untested", "untested"]
        temporal = ["pre_2020", "post_2023", "untested_trick", "untested_rare"]
        result = evaluate_vp_l4(preds, gold, temporal)
        assert result["accuracy_pre_2020"] == 1.0
        assert result["accuracy_post_2023"] == 1.0
        assert result["accuracy_untested_trick"] == 1.0

    def test_contamination_flag(self):
        # Pre-2020 much better than post-2023 → contamination
        preds = ["tested\nx", "tested\nx", "untested\nx", "tested\nx"]
        gold = ["tested", "tested", "tested", "tested"]
        temporal = ["pre_2020", "pre_2020", "post_2023", "post_2023"]
        result = evaluate_vp_l4(preds, gold, temporal)
        assert result["accuracy_pre_2020"] == 1.0
        assert result["accuracy_post_2023"] == 0.5
        assert result["contamination_gap"] == 0.5
        assert result["contamination_flag"] is True

    def test_no_contamination(self):
        preds = ["tested\nx", "tested\nx"]
        gold = ["tested", "tested"]
        temporal = ["pre_2020", "post_2023"]
        result = evaluate_vp_l4(preds, gold, temporal)
        assert result["contamination_flag"] is False

    def test_evidence_citation_rate(self):
        preds = [
            "tested\nThis variant has been submitted to ClinVar by multiple clinical laboratories with concordant benign classifications.",
            "tested\nyes",
        ]
        gold = ["tested", "tested"]
        result = evaluate_vp_l4(preds, gold)
        assert result["evidence_citation_rate"] == 0.5  # Only first has >50 chars AND keyword

    def test_evidence_keywords_are_vp_specific(self):
        assert "clinvar" in VP_EVIDENCE_KEYWORDS
        assert "acmg" in VP_EVIDENCE_KEYWORDS
        assert "gnomad" in VP_EVIDENCE_KEYWORDS
        assert "pathogenic" in VP_EVIDENCE_KEYWORDS

    def test_parse_rate(self):
        preds = ["tested\nx", "garbage output"]
        gold = ["tested", "untested"]
        result = evaluate_vp_l4(preds, gold)
        assert result["parse_rate"] == 0.5

    def test_contamination_threshold_020(self):
        """VP uses 0.20 threshold (not PPI's 0.15)."""
        preds = ["tested\nx", "untested\nx"]
        gold = ["tested", "tested"]
        temporal = ["pre_2020", "post_2023"]
        result = evaluate_vp_l4(preds, gold, temporal)
        gap = result["contamination_gap"]
        # Gap = 1.0 - 0.0 = 1.0
        assert result["contamination_flag"] is True

    def test_empty_predictions(self):
        result = evaluate_vp_l4([], [])
        assert result["accuracy"] == 0.0


# ── Dispatch Tests ──────────────────────────────────────────────────────


class TestDispatch:
    def test_l1(self):
        gold = [{"gold_answer": "A", "gold_category": "pathogenic", "difficulty": "easy"}]
        result = compute_all_vp_llm_metrics("vp-l1", ["A"], gold)
        assert result["accuracy"] == 1.0

    def test_l2(self):
        pred = json.dumps({"variants": [], "total_variants_discussed": 0, "classification_method": "ACMG/AMP"})
        gold = [{"gold_extraction": {"variants": [], "total_variants_discussed": 0, "classification_method": "ACMG/AMP"}}]
        result = compute_all_vp_llm_metrics("vp-l2", [pred], gold)
        assert result["schema_compliance"] == 1.0

    def test_l3(self):
        scores_json = json.dumps({
            "population_reasoning": 4, "computational_evidence": 3,
            "functional_reasoning": 5, "gene_disease_specificity": 4,
        })
        result = compute_all_vp_llm_metrics("vp-l3", [scores_json], [{}])
        assert result["overall"]["mean"] == 4.0

    def test_l4(self):
        gold = [{"gold_answer": "tested", "temporal_group": "pre_2020"}]
        result = compute_all_vp_llm_metrics("vp-l4", ["tested\nClinVar"], gold)
        assert result["accuracy"] == 1.0

    def test_invalid_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            compute_all_vp_llm_metrics("vp-l5", [], [])
