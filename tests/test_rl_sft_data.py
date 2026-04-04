"""Tests for negbiorl.sft_data — SFT and GRPO dataset construction."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from negbiorl.sft_data import (
    build_grpo_record,
    build_sft_record,
    save_dataset,
)


# ---------------------------------------------------------------------------
# Mock prompt formatter
# ---------------------------------------------------------------------------

def _mock_format_prompt(domain, task, record):
    return ("You are a scientist.", f"Question about {task}: {record.get('question_id', 'Q')}")


# ---------------------------------------------------------------------------
# SFT records
# ---------------------------------------------------------------------------

class TestBuildSftRecord:
    @patch("negbiorl.sft_data._format_prompt", side_effect=_mock_format_prompt)
    def test_sft_record_structure(self, mock_fmt):
        rec = build_sft_record(
            domain="dti",
            task="l1",
            record={"question_id": "L1-0001", "correct_answer": "B"},
            gold_response="B",
        )
        assert "messages" in rec
        assert len(rec["messages"]) == 3
        assert rec["messages"][0]["role"] == "system"
        assert rec["messages"][1]["role"] == "user"
        assert rec["messages"][2]["role"] == "assistant"
        assert rec["messages"][2]["content"] == "B"
        assert rec["domain"] == "dti"
        assert rec["task"] == "l1"

    @patch("negbiorl.sft_data._format_prompt", side_effect=_mock_format_prompt)
    def test_sft_record_gold_in_assistant(self, mock_fmt):
        rec = build_sft_record(
            domain="ct",
            task="l4",
            record={"question_id": "L4-0001", "gold_answer": "untested"},
            gold_response="untested",
        )
        assert rec["messages"][2]["content"] == "untested"


# ---------------------------------------------------------------------------
# GRPO records
# ---------------------------------------------------------------------------

class TestBuildGrpoRecord:
    @patch("negbiorl.sft_data._format_prompt", side_effect=_mock_format_prompt)
    def test_grpo_record_structure(self, mock_fmt):
        rec = build_grpo_record(
            domain="dti",
            task="l1",
            record={"question_id": "L1-0001", "correct_answer": "B", "difficulty": "hard"},
        )
        assert "prompt" in rec
        assert len(rec["prompt"]) == 2  # system + user, no assistant
        assert rec["prompt"][0]["role"] == "system"
        assert rec["prompt"][1]["role"] == "user"
        assert rec["gold_answer"] == "B"
        assert rec["task"] == "l1"
        assert rec["domain"] == "dti"
        assert rec["difficulty"] == "hard"

    @patch("negbiorl.sft_data._format_prompt", side_effect=_mock_format_prompt)
    def test_grpo_uses_domain_gold_field(self, mock_fmt):
        # DTI uses 'correct_answer'
        rec = build_grpo_record(
            domain="dti",
            task="l1",
            record={"correct_answer": "C", "question_id": "Q1"},
        )
        assert rec["gold_answer"] == "C"

    @patch("negbiorl.sft_data._format_prompt", side_effect=_mock_format_prompt)
    def test_grpo_ct_uses_gold_answer(self, mock_fmt):
        # CT uses 'gold_answer'
        rec = build_grpo_record(
            domain="ct",
            task="l4",
            record={"gold_answer": "tested", "question_id": "Q1"},
        )
        assert rec["gold_answer"] == "tested"

    @patch("negbiorl.sft_data._format_prompt", side_effect=_mock_format_prompt)
    def test_tier_from_metadata(self, mock_fmt):
        rec = build_grpo_record(
            domain="ppi",
            task="l1",
            record={
                "gold_answer": "A",
                "question_id": "Q1",
                "metadata": {"confidence_tier": "gold"},
            },
        )
        assert rec["tier"] == "gold"


# ---------------------------------------------------------------------------
# save_dataset
# ---------------------------------------------------------------------------

class TestSaveDataset:
    def test_save_and_count(self, tmp_path):
        records = [
            {"prompt": "hello", "task": "l1"},
            {"prompt": "world", "task": "l4"},
        ]
        path = tmp_path / "subdir" / "dataset.jsonl"
        count = save_dataset(records, path)
        assert count == 2
        assert path.exists()

        # Verify content
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["prompt"] == "hello"

    def test_save_empty(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        count = save_dataset([], path)
        assert count == 0
        assert path.exists()
