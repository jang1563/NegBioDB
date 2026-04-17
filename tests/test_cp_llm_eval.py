"""Tests for CP LLM evaluation helpers."""

from negbiodb_cp.llm_eval import (
    compute_all_cp_llm_metrics,
    evaluate_cp_l3,
    parse_cp_l1_answer,
    parse_cp_l2_response,
    parse_cp_l3_judge_scores,
    parse_cp_l4_answer,
)


def test_parse_cp_l1_answer_variants():
    assert parse_cp_l1_answer("A") == "A"
    assert parse_cp_l1_answer("Answer: c") == "C"
    assert parse_cp_l1_answer("(D)") == "D"


def test_parse_cp_l2_response_json_block():
    parsed = parse_cp_l2_response("```json\n{\"compound_identifier\":\"Cmp\"}\n```")
    assert parsed["compound_identifier"] == "Cmp"


def test_parse_cp_l3_and_evaluate():
    score = parse_cp_l3_judge_scores(
        "{\"evidence_grounding\": 4, \"assay_reasoning\": 5, \"specificity\": 3, \"non_speculation\": 4}"
    )
    metrics = evaluate_cp_l3([score, score])
    assert metrics["n_valid"] == 2
    assert metrics["overall_mean"] >= 4.0


def test_parse_cp_l4_answer():
    answer, evidence = parse_cp_l4_answer("tested\nCell Painting evidence cites DMSO distance")
    assert answer == "tested"
    assert "DMSO" in evidence


def test_compute_all_cp_llm_metrics_l2_uses_metadata_gold_extraction():
    predictions = [
        (
            "{\"compound_identifier\":\"CmpInactive\",\"dose\":\"1\",\"dose_unit\":\"uM\","
            "\"cell_line\":\"U2OS\",\"batch_id\":\"B1\",\"dmso_distance_summary\":\"0.020\","
            "\"reproducibility_summary\":\"0.900\",\"qc_summary\":\"1.000\","
            "\"outcome_label\":\"inactive\"}"
        )
    ]
    gold = [{
        "gold_answer": "inactive",
        "gold_category": "inactive",
        "metadata": {
            "gold_extraction": {
                "compound_identifier": "CmpInactive",
                "dose": "1",
                "dose_unit": "uM",
                "cell_line": "U2OS",
                "batch_id": "B1",
                "dmso_distance_summary": "0.020",
                "reproducibility_summary": "0.900",
                "qc_summary": "1.000",
                "outcome_label": "inactive",
            }
        },
    }]
    metrics = compute_all_cp_llm_metrics("cp-l2", predictions, gold)
    assert metrics["parse_rate"] == 1.0
    assert metrics["schema_compliance"] == 1.0
    assert metrics["field_accuracy"] == 1.0
