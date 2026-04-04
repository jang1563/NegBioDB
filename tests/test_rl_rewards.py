"""Tests for negbiorl.rewards — GRPO reward functions."""

import pytest

from negbiorl.rewards import (
    DEFAULT_REWARD_FUNCS,
    DEFAULT_REWARD_WEIGHTS,
    evidence_reward_func,
    format_reward_func,
    l1_reward_func,
    l4_reward_func,
)


# ---------------------------------------------------------------------------
# Helper: wrap text as trl completion format
# ---------------------------------------------------------------------------

def _comp(text: str) -> list[dict[str, str]]:
    """Wrap text as trl chat completion format."""
    return [{"role": "assistant", "content": text}]


# ---------------------------------------------------------------------------
# L1 reward
# ---------------------------------------------------------------------------

class TestL1Reward:
    def test_correct_answer_gets_1(self):
        rewards = l1_reward_func(
            completions=[_comp("The answer is B.")],
            gold_answer=["B"],
            task=["l1"],
            domain=["dti"],
        )
        assert rewards == [1.0]

    def test_wrong_answer_gets_0(self):
        rewards = l1_reward_func(
            completions=[_comp("The answer is A.")],
            gold_answer=["B"],
            task=["l1"],
            domain=["dti"],
        )
        assert rewards == [0.0]

    def test_non_l1_returns_none(self):
        rewards = l1_reward_func(
            completions=[_comp("tested")],
            gold_answer=["tested"],
            task=["l4"],
            domain=["dti"],
        )
        assert rewards == [None]

    def test_unparseable_gets_0(self):
        rewards = l1_reward_func(
            completions=[_comp("I don't know the answer")],
            gold_answer=["A"],
            task=["l1"],
            domain=["dti"],
        )
        assert rewards == [0.0]

    def test_case_insensitive(self):
        rewards = l1_reward_func(
            completions=[_comp("b")],
            gold_answer=["B"],
            task=["l1"],
            domain=["dti"],
        )
        assert rewards == [1.0]

    def test_batch_mixed_tasks(self):
        rewards = l1_reward_func(
            completions=[_comp("A"), _comp("tested"), _comp("C")],
            gold_answer=["A", "tested", "B"],
            task=["l1", "l4", "l1"],
            domain=["dti", "ct", "ppi"],
        )
        assert rewards[0] == 1.0   # correct L1
        assert rewards[1] is None  # L4 → None
        assert rewards[2] == 0.0   # wrong L1

    def test_ct_5way_e_option(self):
        """CT domain has 5-way MCQ (A-E)."""
        rewards = l1_reward_func(
            completions=[_comp("E")],
            gold_answer=["E"],
            task=["l1"],
            domain=["ct"],
        )
        assert rewards == [1.0]


# ---------------------------------------------------------------------------
# L4 reward
# ---------------------------------------------------------------------------

class TestL4Reward:
    def test_correct_tested(self):
        rewards = l4_reward_func(
            completions=[_comp("tested\nEvidence: ChEMBL CHEMBL25")],
            gold_answer=["tested"],
            task=["l4"],
            domain=["dti"],
        )
        assert rewards == [1.0]

    def test_wrong_prediction(self):
        rewards = l4_reward_func(
            completions=[_comp("untested")],
            gold_answer=["tested"],
            task=["l4"],
            domain=["dti"],
        )
        assert rewards == [0.0]

    def test_non_l4_returns_none(self):
        rewards = l4_reward_func(
            completions=[_comp("B")],
            gold_answer=["B"],
            task=["l1"],
            domain=["dti"],
        )
        assert rewards == [None]

    def test_unparseable_gets_0(self):
        rewards = l4_reward_func(
            completions=[_comp("I'm not sure about this")],
            gold_answer=["tested"],
            task=["l4"],
            domain=["dti"],
        )
        assert rewards == [0.0]


# ---------------------------------------------------------------------------
# Evidence reward
# ---------------------------------------------------------------------------

class TestEvidenceReward:
    def test_high_evidence_score(self):
        text = "This was tested. ChEMBL data shows IC50 of 500nM. PubChem bioassay AID123 confirms. DOI: 10.1234"
        rewards = evidence_reward_func(
            completions=[_comp(text)],
            task=["l4"],
            domain=["dti"],
        )
        assert rewards[0] is not None
        assert rewards[0] >= 0.5  # multiple keyword matches

    def test_no_evidence(self):
        rewards = evidence_reward_func(
            completions=[_comp("tested")],
            task=["l4"],
            domain=["dti"],
        )
        assert rewards[0] is not None
        assert rewards[0] == 0.0  # no keywords

    def test_non_l4_returns_none(self):
        rewards = evidence_reward_func(
            completions=[_comp("A")],
            task=["l1"],
            domain=["dti"],
        )
        assert rewards == [None]

    def test_domain_specific_keywords(self):
        """PPI domain should match PPI-specific keywords."""
        text = "UniProt P12345 and BioGRID data from co-IP experiments"
        rewards = evidence_reward_func(
            completions=[_comp(text)],
            task=["l4"],
            domain=["ppi"],
        )
        assert rewards[0] is not None
        assert rewards[0] >= 0.5

    def test_max_score_capped_at_1(self):
        text = "ChEMBL PubChem BindingDB IC50 Ki Kd pChEMBL bioassay PMID DOI"
        rewards = evidence_reward_func(
            completions=[_comp(text)],
            task=["l4"],
            domain=["dti"],
        )
        assert rewards[0] == 1.0


# ---------------------------------------------------------------------------
# Format reward
# ---------------------------------------------------------------------------

class TestFormatReward:
    def test_l1_with_letter(self):
        rewards = format_reward_func(
            completions=[_comp("B. The compound is inactive.")],
            task=["l1"],
        )
        assert rewards[0] == 0.5

    def test_l1_without_letter(self):
        rewards = format_reward_func(
            completions=[_comp("The compound shows no activity against the target.")],
            task=["l1"],
        )
        assert rewards[0] == 0.0

    def test_l4_with_tested(self):
        rewards = format_reward_func(
            completions=[_comp("tested\nThis was tested in ChEMBL")],
            task=["l4"],
        )
        assert rewards[0] == 0.5

    def test_non_l1_l4_returns_none(self):
        rewards = format_reward_func(
            completions=[_comp("some text")],
            task=["l3"],
        )
        assert rewards == [None]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_funcs_length(self):
        assert len(DEFAULT_REWARD_FUNCS) == 4

    def test_default_weights_length(self):
        assert len(DEFAULT_REWARD_WEIGHTS) == 4

    def test_weights_match_funcs(self):
        assert len(DEFAULT_REWARD_FUNCS) == len(DEFAULT_REWARD_WEIGHTS)

    def test_primary_rewards_weighted_1(self):
        assert DEFAULT_REWARD_WEIGHTS[0] == 1.0  # l1
        assert DEFAULT_REWARD_WEIGHTS[1] == 1.0  # l4
