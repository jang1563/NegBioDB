"""Tests for negbiorl.eval_pipeline."""

from unittest.mock import patch

import pytest

from negbiorl.eval_pipeline import evaluate_l1, evaluate_l4, evaluate_before_after


class TestEvaluateL1:
    @patch("negbiorl.eval_pipeline.load_export")
    @patch("negbiorl.eval_pipeline.get_l1_parser")
    def test_uses_parser(self, mock_get_parser, mock_load_export):
        """L1 eval must use domain-specific parser, not raw text."""
        mock_load_export.return_value = [
            {"question_id": "q1", "correct_answer": "B"},
            {"question_id": "q2", "correct_answer": "A"},
        ]
        # Simulate parser that extracts letter from verbose text
        mock_get_parser.return_value = lambda text: text[0] if text else None

        predictions = [
            {"question_id": "q1", "prediction": "B. The answer is B because..."},
            {"question_id": "q2", "prediction": "A. This is correct."},
        ]
        result = evaluate_l1(predictions, "dti")
        assert result["accuracy"] == 1.0
        assert result["parse_rate"] == 1.0
        mock_get_parser.assert_called_once_with("dti")

    @patch("negbiorl.eval_pipeline.load_export")
    @patch("negbiorl.eval_pipeline.get_l1_parser")
    def test_parse_failure_counted(self, mock_get_parser, mock_load_export):
        mock_load_export.return_value = [
            {"question_id": "q1", "correct_answer": "B"},
        ]
        mock_get_parser.return_value = lambda text: None  # always fails

        predictions = [{"question_id": "q1", "prediction": "garbage"}]
        result = evaluate_l1(predictions, "dti")
        assert result["parse_rate"] == 0.0


class TestEvaluateL4:
    @patch("negbiorl.eval_pipeline.load_export")
    @patch("negbiorl.eval_pipeline.parse_l4_unified")
    @patch("negbiorl.eval_pipeline.get_domain")
    def test_uses_parser(self, mock_get_domain, mock_parse, mock_load_export):
        """L4 eval must use domain-specific parser, not raw text."""
        mock_get_domain.return_value = {"temporal_groups": []}
        mock_load_export.return_value = [
            {"question_id": "q1", "gold_answer": "tested"},
            {"question_id": "q2", "gold_answer": "untested"},
        ]
        mock_parse.side_effect = [("tested", None), ("untested", None)]

        predictions = [
            {"question_id": "q1", "prediction": "tested\nEvidence: lots of data"},
            {"question_id": "q2", "prediction": "untested\nNo evidence found"},
        ]
        result = evaluate_l4(predictions, "ct")
        assert result["accuracy"] == 1.0
        assert mock_parse.call_count == 2

    @patch("negbiorl.eval_pipeline.load_export")
    @patch("negbiorl.eval_pipeline.parse_l4_unified")
    @patch("negbiorl.eval_pipeline.get_domain")
    def test_parse_failure_counted(self, mock_get_domain, mock_parse, mock_load_export):
        mock_get_domain.return_value = {"temporal_groups": []}
        mock_load_export.return_value = [
            {"question_id": "q1", "gold_answer": "tested"},
        ]
        mock_parse.return_value = (None, None)

        predictions = [{"question_id": "q1", "prediction": "garbage"}]
        result = evaluate_l4(predictions, "ct")
        assert result["parse_rate"] == 0.0


class TestEvaluateBeforeAfter:
    @patch("negbiorl.eval_pipeline.evaluate_l1")
    def test_delta_excludes_n(self, mock_eval_l1):
        mock_eval_l1.side_effect = [
            {"accuracy": 0.5, "mcc": 0.1, "parse_rate": 0.9, "n": 100},
            {"accuracy": 0.7, "mcc": 0.3, "parse_rate": 0.95, "n": 100},
        ]
        result = evaluate_before_after([], [], "dti", "l1")
        assert "delta_n" not in result["deltas"]
        assert abs(result["deltas"]["delta_accuracy"] - 0.2) < 1e-10
        assert abs(result["deltas"]["delta_mcc"] - 0.2) < 1e-10
