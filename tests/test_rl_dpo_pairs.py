"""Tests for negbiorl.dpo_pairs — DPO pair construction."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from negbiorl.dpo_pairs import (
    build_l3_dpo_pairs,
    build_l4_dpo_pairs,
    build_all_dpo_pairs,
    save_dpo_pairs,
)


class TestSaveDpoPairs:
    def test_save_and_count(self, tmp_path):
        pairs = [
            {"prompt": [{"role": "user", "content": "q"}], "chosen": "A", "rejected": "B"},
            {"prompt": [{"role": "user", "content": "q2"}], "chosen": "C", "rejected": "D"},
        ]
        path = tmp_path / "dpo.jsonl"
        count = save_dpo_pairs(pairs, path)
        assert count == 2

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        loaded = json.loads(lines[0])
        assert loaded["chosen"] == "A"
        assert loaded["rejected"] == "B"

    def test_save_creates_parent(self, tmp_path):
        path = tmp_path / "a" / "b" / "dpo.jsonl"
        save_dpo_pairs([], path)
        assert path.exists()


class TestBuildAllDpoPairs:
    @patch("negbiorl.dpo_pairs.build_l3_dpo_pairs", return_value=[
        {"task": "l3", "chosen": "good response", "rejected": "bad response"},
    ])
    @patch("negbiorl.dpo_pairs.build_l4_dpo_pairs", return_value=[
        {"task": "l4", "chosen": "tested", "rejected": "untested"},
    ])
    def test_combines_l3_and_l4(self, mock_l4, mock_l3):
        pairs = build_all_dpo_pairs(domains=["dti"])
        assert len(pairs) == 2
        assert pairs[0]["task"] == "l3"
        assert pairs[1]["task"] == "l4"

    @patch("negbiorl.dpo_pairs.build_l3_dpo_pairs", return_value=[
        {"task": "l3", "chosen": "good response", "rejected": "ERROR: HTTP Error 429"},
    ])
    @patch("negbiorl.dpo_pairs.build_l4_dpo_pairs", return_value=[
        {"task": "l4", "chosen": "tested", "rejected": "untested"},
    ])
    def test_filters_error_responses(self, mock_l4, mock_l3):
        pairs = build_all_dpo_pairs(domains=["dti"])
        assert len(pairs) == 1
        assert pairs[0]["task"] == "l4"  # L3 pair with error filtered out
