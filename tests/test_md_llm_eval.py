"""Tests for MD LLM evaluation metrics (L1-L4)."""

import pytest
from negbiodb_md.llm_eval import (
    eval_l1,
    eval_l2,
    eval_l3_with_judge,
    eval_l4,
    parse_l1_response,
    parse_l2_response,
    parse_l4_response,
)


# ── L1 parsing ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("response,expected", [
    ("A", "A"),
    ("b", "B"),
    ("C.", "C"),
    ("The answer is D.", "D"),
    ("  A  ", "A"),
    ("", None),
    ("none", None),
])
def test_parse_l1_response(response, expected):
    assert parse_l1_response(response) == expected


# ── L1 evaluation ─────────────────────────────────────────────────────────────

def test_eval_l1_perfect():
    records = [
        {"gold_answer": "A"},
        {"gold_answer": "B"},
        {"gold_answer": "C"},
        {"gold_answer": "D"},
    ]
    responses = ["A", "B", "C", "D"]
    result = eval_l1(records, responses)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["n"] == 4


def test_eval_l1_random():
    records = [{"gold_answer": "A"}, {"gold_answer": "A"}, {"gold_answer": "A"}, {"gold_answer": "A"}]
    responses = ["B", "C", "D", "A"]
    result = eval_l1(records, responses)
    assert result["accuracy"] == pytest.approx(0.25)


def test_eval_l1_empty():
    result = eval_l1([], [])
    assert result["n"] == 0
    assert result["mcc"] is None


# ── L2 parsing ────────────────────────────────────────────────────────────────

def test_parse_l2_response_valid_json():
    resp = '{"metabolite": "glucose", "disease": "T2D", "outcome": "not_significant"}'
    parsed = parse_l2_response(resp)
    assert parsed is not None
    assert parsed["metabolite"] == "glucose"


def test_parse_l2_response_with_code_fence():
    resp = "```json\n{\"metabolite\": \"alanine\", \"outcome\": \"significant\"}\n```"
    parsed = parse_l2_response(resp)
    assert parsed is not None
    assert parsed["metabolite"] == "alanine"


def test_parse_l2_response_invalid():
    parsed = parse_l2_response("Not JSON at all")
    assert parsed is None


def test_parse_l2_response_empty():
    assert parse_l2_response("") is None


# ── L2 evaluation ─────────────────────────────────────────────────────────────

def test_eval_l2_perfect():
    records = [
        {"gold_fields": {
            "metabolite": "glucose",
            "disease": "type 2 diabetes mellitus",
            "fold_change": None,
            "platform": "lc_ms",
            "biofluid": "blood",
            "outcome": "not_significant",
        }}
    ]
    responses = ['{"metabolite": "glucose", "disease": "type 2 diabetes mellitus", '
                 '"fold_change": null, "platform": "lc_ms", "biofluid": "blood", '
                 '"outcome": "not_significant"}']
    result = eval_l2(records, responses)
    assert result["field_f1"] == pytest.approx(1.0)
    assert result["schema_compliance"] == pytest.approx(1.0)


def test_eval_l2_platform_alias():
    records = [{"gold_fields": {"platform": "lc_ms", "outcome": "not_significant"}}]
    responses = ['{"platform": "lc-ms", "outcome": "not_significant"}']
    result = eval_l2(records, responses)
    # platform lc-ms should match lc_ms
    assert result["per_field_accuracy"]["platform"] == pytest.approx(1.0)


def test_eval_l2_schema_noncompliant():
    records = [{"gold_fields": {"outcome": "significant"}}]
    responses = ["This metabolite is significant."]
    result = eval_l2(records, responses)
    assert result["schema_compliance"] == pytest.approx(0.0)


# ── L3 evaluation ─────────────────────────────────────────────────────────────

def test_eval_l3_with_judge_computes_mean():
    records = [{"record_id": "r1"}, {"record_id": "r2"}]
    responses = ["response 1", "response 2"]
    judge_scores = [
        {"metabolite_biology": 4, "disease_mechanism": 3, "study_context": 5, "alternative_hypothesis": 4},
        {"metabolite_biology": 5, "disease_mechanism": 5, "study_context": 4, "alternative_hypothesis": 3},
    ]
    result = eval_l3_with_judge(records, responses, judge_scores)
    assert result["overall_mean"] == pytest.approx(4.125)
    assert result["per_axis"]["metabolite_biology"] == pytest.approx(4.5)


def test_eval_l3_with_empty_judge_scores():
    result = eval_l3_with_judge([], [], [])
    assert result["overall_mean"] is None


# ── L4 parsing ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("response,expected", [
    ("A", 1),
    ("B", 0),
    ("The answer is A", 1),
    ("B) Synthetic", 0),
    ("This is a real finding from a study", 1),
    ("This is synthetic data", 0),
    ("", None),
])
def test_parse_l4_response(response, expected):
    assert parse_l4_response(response) == expected


# ── L4 evaluation ─────────────────────────────────────────────────────────────

def test_eval_l4_perfect():
    records = [{"label": 1}, {"label": 0}, {"label": 1}, {"label": 0}]
    responses = ["A", "B", "A", "B"]
    result = eval_l4(records, responses)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["n"] == 4


def test_eval_l4_random():
    records = [{"label": 1}] * 4 + [{"label": 0}] * 4
    responses = ["A", "B", "A", "B", "A", "B", "A", "B"]
    result = eval_l4(records, responses)
    assert result["n"] == 8


def test_eval_l4_empty():
    result = eval_l4([], [])
    assert result["n"] == 0
    assert result["mcc"] is None
