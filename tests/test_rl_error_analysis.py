"""Tests for negbiorl.error_analysis."""

from unittest.mock import patch

import pytest

from negbiorl.error_analysis import classify_l1_errors, classify_l4_errors, summarize_errors


class TestSummarizeErrors:
    def test_none_tier_does_not_crash(self):
        """summarize_errors must handle None tier without TypeError on sorted()."""
        errors = [
            {"error_type": "wrong_class", "tier": None},
            {"error_type": "wrong_class", "tier": "gold"},
            {"error_type": "parse_failure", "tier": None},
        ]
        result = summarize_errors(errors)
        assert result["total_errors"] == 3
        # None tiers should map to "unknown"
        assert "unknown" in result["per_tier"]
        assert "gold" in result["per_tier"]

    def test_empty_errors(self):
        result = summarize_errors([])
        assert result["total_errors"] == 0
        assert result["error_distribution"] == {}

    def test_distribution_fractions(self):
        errors = [
            {"error_type": "wrong_class", "tier": "gold"},
            {"error_type": "wrong_class", "tier": "gold"},
            {"error_type": "parse_failure", "tier": "gold"},
        ]
        result = summarize_errors(errors)
        assert result["error_distribution"]["wrong_class"]["fraction"] == pytest.approx(2/3)
        assert result["error_distribution"]["parse_failure"]["fraction"] == pytest.approx(1/3)


class TestClassifyL1Errors:
    def test_wrong_class_detected(self):
        """Parsed answer != gold → wrong_class."""
        export = [{"question_id": "q1", "correct_answer": "B", "class": "negative"}]
        predictions = [{"question_id": "q1", "prediction": "A"}]

        with patch("negbiorl.error_analysis.get_l1_parser", return_value=lambda t: t[0] if t else None):
            errors = classify_l1_errors(predictions, export, "dti")
        assert len(errors) == 1
        assert errors[0]["error_type"] == "wrong_class"

    def test_parse_failure_detected(self):
        """Unparseable prediction → parse_failure."""
        export = [{"question_id": "q1", "correct_answer": "B", "class": "negative"}]
        predictions = [{"question_id": "q1", "prediction": "garbage"}]

        with patch("negbiorl.error_analysis.get_l1_parser", return_value=lambda t: None):
            errors = classify_l1_errors(predictions, export, "dti")
        assert len(errors) == 1
        assert errors[0]["error_type"] == "parse_failure"

    def test_correct_skipped(self):
        """Correct parsed answer should not appear in errors."""
        export = [{"question_id": "q1", "correct_answer": "B", "class": "negative"}]
        predictions = [{"question_id": "q1", "prediction": "B. because..."}]

        with patch("negbiorl.error_analysis.get_l1_parser", return_value=lambda t: t[0] if t else None):
            errors = classify_l1_errors(predictions, export, "dti")
        assert len(errors) == 0

    def test_tier_defaults_to_unknown(self):
        """Missing tier → 'unknown'."""
        export = [{"question_id": "q1", "correct_answer": "B", "class": "negative"}]
        predictions = [{"question_id": "q1", "prediction": "A"}]

        with patch("negbiorl.error_analysis.get_l1_parser", return_value=lambda t: t[0] if t else None):
            errors = classify_l1_errors(predictions, export, "dti")
        assert errors[0]["tier"] == "unknown"


class TestClassifyL4Errors:
    def test_positivity_bias(self):
        """Gold=untested, predicted=tested → positivity_bias."""
        export = [{"question_id": "q1", "gold_answer": "untested"}]
        predictions = [{"question_id": "q1", "prediction": "tested"}]

        with patch("negbiorl.error_analysis.parse_l4_unified", return_value=("tested", None)):
            errors = classify_l4_errors(predictions, export, "ct")
        assert len(errors) == 1
        assert errors[0]["error_type"] == "positivity_bias"

    def test_false_alarm(self):
        """Gold=tested, predicted=untested → false_alarm."""
        export = [{"question_id": "q1", "gold_answer": "tested"}]
        predictions = [{"question_id": "q1", "prediction": "untested"}]

        with patch("negbiorl.error_analysis.parse_l4_unified", return_value=("untested", None)):
            errors = classify_l4_errors(predictions, export, "ct")
        assert len(errors) == 1
        assert errors[0]["error_type"] == "false_alarm"

    def test_correct_skipped(self):
        """Correct predictions should not be in errors."""
        export = [{"question_id": "q1", "gold_answer": "tested"}]
        predictions = [{"question_id": "q1", "prediction": "tested"}]

        with patch("negbiorl.error_analysis.parse_l4_unified", return_value=("tested", None)):
            errors = classify_l4_errors(predictions, export, "ct")
        assert len(errors) == 0

    def test_tier_defaults_to_unknown(self):
        """Missing tier should become 'unknown', not None."""
        export = [{"question_id": "q1", "gold_answer": "untested"}]
        predictions = [{"question_id": "q1", "prediction": "tested"}]

        with patch("negbiorl.error_analysis.parse_l4_unified", return_value=("tested", None)):
            errors = classify_l4_errors(predictions, export, "ct")
        assert errors[0]["tier"] == "unknown"
