"""Tests for GE LLM evaluation module (parsing + metrics)."""

import pytest

from negbiodb_depmap.llm_eval import (
    compute_all_ge_llm_metrics,
    evaluate_ge_l1,
    evaluate_ge_l2,
    evaluate_ge_l3,
    evaluate_ge_l4,
    parse_ge_l1_answer,
    parse_ge_l2_response,
    parse_ge_l3_judge_scores,
    parse_ge_l4_answer,
)


# ── L1 parsing ────────────────────────────────────────────────────────────


class TestParseL1:
    def test_single_letter(self):
        assert parse_ge_l1_answer("C") == "C"

    def test_lowercase(self):
        assert parse_ge_l1_answer("b") == "B"

    def test_answer_prefix(self):
        assert parse_ge_l1_answer("Answer: A") == "A"

    def test_parenthesized(self):
        assert parse_ge_l1_answer("(D)") == "D"

    def test_with_explanation(self):
        assert parse_ge_l1_answer("C\nBecause the gene is non-essential...") == "C"

    def test_empty(self):
        assert parse_ge_l1_answer("") is None

    def test_no_valid_letter(self):
        # Text without any A-D characters
        assert parse_ge_l1_answer("no help here with this question") is None

    def test_embedded_letter(self):
        assert parse_ge_l1_answer("I think B is correct") == "B"


class TestEvaluateL1:
    def test_perfect_accuracy(self):
        preds = ["A", "B", "C", "D"]
        gold = ["A", "B", "C", "D"]
        result = evaluate_ge_l1(preds, gold)
        assert result["accuracy"] == 1.0
        assert result["valid_rate"] == 1.0

    def test_zero_accuracy(self):
        preds = ["B", "C", "D", "A"]
        gold = ["A", "B", "C", "D"]
        result = evaluate_ge_l1(preds, gold)
        assert result["accuracy"] == 0.0

    def test_invalid_responses(self):
        # "nonsense" has no A-D letters, so it's truly unparseable
        preds = ["nonsense", "A"]
        gold = ["A", "A"]
        result = evaluate_ge_l1(preds, gold)
        assert result["n_valid"] == 1
        assert result["valid_rate"] == 0.5

    def test_empty_predictions(self):
        result = evaluate_ge_l1([], [])
        assert result["accuracy"] == 0.0


# ── L2 parsing ────────────────────────────────────────────────────────────


class TestParseL2:
    def test_valid_json(self):
        raw = '{"genes": [{"gene_name": "TP53"}], "total_genes_mentioned": 1, "screen_type": "CRISPR"}'
        result = parse_ge_l2_response(raw)
        assert result is not None
        assert result["total_genes_mentioned"] == 1

    def test_markdown_json(self):
        raw = '```json\n{"genes": [], "total_genes_mentioned": 0, "screen_type": "RNAi"}\n```'
        result = parse_ge_l2_response(raw)
        assert result is not None

    def test_json_with_text(self):
        raw = 'Here is the extraction:\n{"genes": [], "total_genes_mentioned": 0, "screen_type": "CRISPR"}'
        result = parse_ge_l2_response(raw)
        assert result is not None

    def test_invalid_json(self):
        assert parse_ge_l2_response("not json at all") is None

    def test_empty(self):
        assert parse_ge_l2_response("") is None


class TestEvaluateL2:
    def test_perfect_parse(self):
        preds = ['{"genes": [], "total_genes_mentioned": 0, "screen_type": "CRISPR"}']
        gold = [{"genes": [], "total_genes_mentioned": 0, "screen_type": "CRISPR"}]
        result = evaluate_ge_l2(preds, gold)
        assert result["parse_rate"] == 1.0
        assert result["schema_compliance"] == 1.0

    def test_missing_fields(self):
        preds = ['{"genes": []}']
        gold = [{"genes": [], "total_genes_mentioned": 0, "screen_type": "CRISPR"}]
        result = evaluate_ge_l2(preds, gold)
        assert result["schema_compliance"] == 0.0

    def test_unparseable(self):
        preds = ["invalid"]
        gold = [{"genes": []}]
        result = evaluate_ge_l2(preds, gold)
        assert result["parse_rate"] == 0.0

    def test_essentiality_accuracy_correct(self):
        preds = ['{"genes": [{"gene_name": "TP53", "essentiality_status": "non-essential"}], "total_genes_mentioned": 1, "screen_type": "CRISPR"}']
        gold = [{"genes": [{"gene_name": "TP53", "essentiality_status": "non-essential"}], "total_genes_mentioned": 1, "screen_type": "CRISPR"}]
        result = evaluate_ge_l2(preds, gold)
        assert result["essentiality_accuracy"] == 1.0
        assert result["essentiality_n"] == 1

    def test_essentiality_accuracy_wrong(self):
        preds = ['{"genes": [{"gene_name": "TP53", "essentiality_status": "essential"}], "total_genes_mentioned": 1, "screen_type": "CRISPR"}']
        gold = [{"genes": [{"gene_name": "TP53", "essentiality_status": "non-essential"}], "total_genes_mentioned": 1, "screen_type": "CRISPR"}]
        result = evaluate_ge_l2(preds, gold)
        assert result["essentiality_accuracy"] == 0.0

    def test_legacy_essentiality_findings_key(self):
        # gold uses old schema key 'essentiality_findings' — should still work
        preds = ['{"genes": [{"gene_name": "BRCA1", "essentiality_status": "non-essential"}], "total_genes_mentioned": 1, "screen_type": "CRISPR"}']
        gold = [{"essentiality_findings": [{"gene_name": "BRCA1", "essentiality_status": "non-essential"}], "total_gene_count": 1, "screen_type": "CRISPR"}]
        result = evaluate_ge_l2(preds, gold)
        assert result["essentiality_accuracy"] == 1.0


# ── L3 parsing ────────────────────────────────────────────────────────────


class TestParseL3:
    def test_standard_format(self):
        raw = """biological_plausibility: 4
pathway_reasoning: 3
context_specificity: 5
mechanistic_depth: 4"""
        result = parse_ge_l3_judge_scores(raw)
        assert result is not None
        assert result["biological_plausibility"] == 4.0
        assert result["context_specificity"] == 5.0

    def test_json_format(self):
        raw = '{"biological_plausibility": 3, "pathway_reasoning": 4, "context_specificity": 3, "mechanistic_depth": 2}'
        result = parse_ge_l3_judge_scores(raw)
        assert result is not None
        assert result["pathway_reasoning"] == 4.0

    def test_empty(self):
        assert parse_ge_l3_judge_scores("") is None


class TestEvaluateL3:
    def test_basic_evaluation(self):
        judge_outputs = [
            "biological_plausibility: 4\npathway_reasoning: 3\ncontext_specificity: 5\nmechanistic_depth: 4",
            "biological_plausibility: 3\npathway_reasoning: 4\ncontext_specificity: 3\nmechanistic_depth: 3",
        ]
        result = evaluate_ge_l3(judge_outputs)
        assert result["n_parsed"] == 2
        assert result["biological_plausibility_mean"] == 3.5
        assert result["overall_mean"] > 0

    def test_no_parseable(self):
        result = evaluate_ge_l3(["invalid", "garbage"])
        assert result["n_parsed"] == 0
        assert result["overall_mean"] == 0.0


# ── L4 parsing ────────────────────────────────────────────────────────────


class TestParseL4:
    def test_tested(self):
        assert parse_ge_l4_answer("tested") == "tested"

    def test_untested(self):
        assert parse_ge_l4_answer("untested") == "untested"

    def test_with_evidence(self):
        assert parse_ge_l4_answer("tested\nThis gene is in DepMap 22Q2") == "tested"

    def test_case_insensitive(self):
        assert parse_ge_l4_answer("UNTESTED") == "untested"

    def test_embedded(self):
        assert parse_ge_l4_answer("I believe this is untested") == "untested"

    def test_tested_priority_over_untested(self):
        # "untested" contains "tested" — check correct parsing
        assert parse_ge_l4_answer("untested because...") == "untested"

    def test_empty(self):
        assert parse_ge_l4_answer("") is None


class TestEvaluateL4:
    def test_perfect(self):
        preds = ["tested", "untested", "tested"]
        gold = ["tested", "untested", "tested"]
        result = evaluate_ge_l4(preds, gold)
        assert result["accuracy"] == 1.0
        assert result["mcc"] == 1.0

    def test_all_wrong(self):
        preds = ["untested", "tested"]
        gold = ["tested", "untested"]
        result = evaluate_ge_l4(preds, gold)
        assert result["accuracy"] == 0.0

    def test_distribution(self):
        preds = ["tested", "tested", "untested"]
        gold = ["tested", "tested", "untested"]
        result = evaluate_ge_l4(preds, gold)
        assert result["prediction_distribution"]["tested"] == 2
        assert result["prediction_distribution"]["untested"] == 1


# ── Dispatch ──────────────────────────────────────────────────────────────


class TestDispatch:
    def test_l1_dispatch(self):
        result = compute_all_ge_llm_metrics("ge-l1", ["A", "B"], ["A", "B"])
        assert "accuracy" in result

    def test_l4_dispatch(self):
        result = compute_all_ge_llm_metrics("ge-l4", ["tested"], ["tested"])
        assert "accuracy" in result

    def test_l2_dispatch(self):
        pred = '{"genes": ["TP53"], "total_genes_mentioned": 1, "screen_type": "CRISPR"}'
        gold = [{"genes": ["TP53"], "total_genes_mentioned": 1, "screen_type": "CRISPR"}]
        result = compute_all_ge_llm_metrics("ge-l2", [pred], gold)
        assert "parse_rate" in result
        assert "schema_compliance" in result
        assert "field_f1" in result
        assert result["parse_rate"] == 1.0

    def test_l2_dispatch_with_full_record(self):
        pred = '{"genes": ["TP53"], "total_genes_mentioned": 1, "screen_type": "CRISPR"}'
        gold_records = [
            {
                "question_id": "GEL2-001",
                "task": "ge-l2",
                "gold_extraction": {
                    "genes": ["TP53"],
                    "total_genes_mentioned": 1,
                    "screen_type": "CRISPR",
                },
                "gold_answer": "non-essential",
            }
        ]
        result = compute_all_ge_llm_metrics("ge-l2", [pred], gold_records)
        assert result["parse_rate"] == 1.0
        assert "field_f1" in result

    def test_l3_dispatch(self):
        result = compute_all_ge_llm_metrics(
            "ge-l3",
            ["This gene is non-essential because it has redundant paralogs."],
            [{}],
        )
        assert "n_parsed" in result
        assert result["n_parsed"] == 0
        assert result["overall_mean"] == 0.0

    def test_invalid_task(self):
        with pytest.raises(ValueError):
            compute_all_ge_llm_metrics("ge-l99", [], [])
