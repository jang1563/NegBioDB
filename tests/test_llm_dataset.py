"""Tests for LLM benchmark dataset integrity."""

import json
from collections import Counter
from pathlib import Path

import pytest

EXPORTS_DIR = Path(__file__).resolve().parent.parent / "exports" / "llm_benchmarks"


def _load_jsonl(filename: str) -> list[dict]:
    """Load JSONL file from exports/llm_benchmarks/."""
    path = EXPORTS_DIR / filename
    if not path.exists():
        pytest.skip(f"{path} not found (run build scripts first)")
    with open(path) as f:
        return [json.loads(line) for line in f]


# ── L1 MCQ Dataset ────────────────────────────────────────────────────────────


class TestL1Dataset:
    @pytest.fixture
    def records(self):
        return _load_jsonl("l1_mcq.jsonl")

    def test_total_count(self, records):
        # After cross-class dedup + per-class compound cap, count may be < 2000
        assert len(records) >= 1800
        assert len(records) <= 2000

    def test_class_distribution(self, records):
        counts = Counter(r["class"] for r in records)
        # After dedup, classes may be slightly below target
        assert counts["active"] >= 350
        assert counts["inactive"] >= 750
        assert counts["inconclusive"] >= 200  # DAVIS 68-compound panel limits this
        assert counts["conditional"] >= 350

    def test_correct_answers(self, records):
        answer_map = {
            "active": "A",
            "inactive": "B",
            "inconclusive": "C",
            "conditional": "D",
        }
        for r in records:
            assert r["correct_answer"] == answer_map[r["class"]]

    def test_split_distribution(self, records):
        counts = Counter(r["split"] for r in records)
        assert counts["fewshot"] == 200
        assert counts["val"] == 200
        assert counts["test"] >= 1400  # rest goes to test

    def test_fewshot_balanced(self, records):
        """Each class should have 50 few-shot examples."""
        fewshot = [r for r in records if r["split"] == "fewshot"]
        counts = Counter(r["class"] for r in fewshot)
        for cls in ["active", "inactive", "inconclusive", "conditional"]:
            assert counts[cls] == 50

    def test_required_fields(self, records):
        required = [
            "question_id", "class", "correct_answer", "difficulty",
            "compound_name", "compound_smiles", "target_uniprot",
            "context_text", "split",
        ]
        for r in records:
            for field in required:
                assert field in r, f"Missing {field} in {r.get('question_id')}"

    def test_unique_question_ids(self, records):
        ids = [r["question_id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_difficulty_levels(self, records):
        difficulties = set(r["difficulty"] for r in records)
        assert difficulties.issubset({"easy", "medium", "hard"})

    def test_context_text_not_empty(self, records):
        for r in records:
            assert len(r["context_text"]) > 50

    def test_no_cross_class_pair_conflicts(self, records):
        """C-2: Same compound-target pair must not appear in multiple classes."""
        pair_classes = {}
        for r in records:
            ik = r.get("compound_inchikey", "")[:14]
            uni = r.get("target_uniprot", "")
            pair = (ik, uni)
            pair_classes.setdefault(pair, set()).add(r["class"])
        conflicts = {p: cls for p, cls in pair_classes.items() if len(cls) > 1}
        assert len(conflicts) == 0, f"Cross-class conflicts: {len(conflicts)}"


# ── L4 Tested/Untested Dataset ───────────────────────────────────────────────


class TestL4Dataset:
    @pytest.fixture
    def records(self):
        return _load_jsonl("l4_tested_untested.jsonl")

    def test_total_count(self, records):
        assert len(records) == 500

    def test_class_balance(self, records):
        counts = Counter(r["class"] for r in records)
        assert counts["tested"] == 250
        assert counts["untested"] == 250

    def test_temporal_split(self, records):
        """Tested pairs should have temporal groups."""
        tested = [r for r in records if r["class"] == "tested"]
        temporal = Counter(r.get("temporal_group") for r in tested)
        assert temporal["pre_2023"] == 125
        assert temporal["post_2024"] == 125

    def test_untested_types(self, records):
        untested = [r for r in records if r["class"] == "untested"]
        types = Counter(r.get("untested_type") for r in untested)
        assert types["trick"] == 125
        assert types["tdark"] == 125

    def test_split_distribution(self, records):
        counts = Counter(r["split"] for r in records)
        assert counts["fewshot"] == 50
        assert counts["val"] == 50
        assert counts["test"] == 400

    def test_correct_answers(self, records):
        for r in records:
            assert r["correct_answer"] == r["class"]

    def test_unique_question_ids(self, records):
        ids = [r["question_id"] for r in records]
        assert len(ids) == len(set(ids))


# ── L3 Reasoning Pilot ───────────────────────────────────────────────────────


class TestL3Dataset:
    @pytest.fixture
    def records(self):
        return _load_jsonl("l3_reasoning_pilot.jsonl")

    def test_total_count(self, records):
        assert len(records) == 50

    def test_split_distribution(self, records):
        counts = Counter(r["split"] for r in records)
        assert counts["fewshot"] == 5
        assert counts["val"] == 5
        assert counts["test"] == 40

    def test_required_fields(self, records):
        required = [
            "question_id", "compound_name", "compound_smiles",
            "target_uniprot", "context_text", "split",
        ]
        for r in records:
            for field in required:
                assert field in r

    def test_evidence_quality(self, records):
        """All L3 pairs should be silver quality."""
        for r in records:
            assert r.get("evidence_quality") == "silver"
