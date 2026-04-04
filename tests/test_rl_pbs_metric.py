"""Tests for negbiorl.pbs_metric — Publication Bias Score."""

import math
import pytest

from negbiorl.pbs_metric import (
    compute_pbs,
    compute_pbs_by_tier,
    compute_pbs_delta,
    compute_multi_domain_pbs,
)


class TestComputePBS:
    def test_perfect_classifier(self):
        """Perfect classifier: PBS = P(pos|neg) - P(pos|pos) = 0 - 1 = -1."""
        result = compute_pbs(
            predictions=["tested", "tested", "untested", "untested"],
            gold_answers=["tested", "tested", "untested", "untested"],
        )
        assert result["pbs"] == -1.0
        assert result["p_pos_given_pos"] == 1.0
        assert result["p_pos_given_neg"] == 0.0

    def test_pure_positivity_bias(self):
        """Always predicts 'tested': PBS = 1 - 1 = 0."""
        result = compute_pbs(
            predictions=["tested", "tested", "tested", "tested"],
            gold_answers=["tested", "tested", "untested", "untested"],
        )
        # P(pos|neg) = 1.0, P(pos|pos) = 1.0 → PBS = 0
        assert result["pbs"] == 0.0

    def test_biased_toward_positive_on_negatives(self):
        """Model says 'tested' for negatives more than for positives."""
        result = compute_pbs(
            predictions=["untested", "untested", "tested", "tested"],
            gold_answers=["tested", "tested", "untested", "untested"],
        )
        # P(pos|pos)=0.0, P(pos|neg)=1.0 → PBS=1.0
        assert result["pbs"] == 1.0

    def test_negative_bias(self):
        """Model predicts 'untested' everywhere."""
        result = compute_pbs(
            predictions=["untested", "untested", "untested", "untested"],
            gold_answers=["tested", "tested", "untested", "untested"],
        )
        # P(pos|pos)=0.0, P(pos|neg)=0.0 → PBS=0.0
        assert result["pbs"] == 0.0

    def test_realistic_case(self):
        """Typical failure-blind model: predicts tested for 80% of negatives."""
        preds = ["tested"] * 8 + ["untested"] * 2 + ["tested"] * 10
        golds = ["untested"] * 10 + ["tested"] * 10
        result = compute_pbs(preds, golds)
        # P(pos|neg)=0.8, P(pos|pos)=1.0 → PBS=-0.2
        assert abs(result["pbs"] - (-0.2)) < 1e-10

    def test_empty_true_pos(self):
        """No true positives → NaN."""
        result = compute_pbs(
            predictions=["tested", "untested"],
            gold_answers=["untested", "untested"],
        )
        assert math.isnan(result["pbs"])

    def test_empty_true_neg(self):
        """No true negatives → NaN."""
        result = compute_pbs(
            predictions=["tested", "untested"],
            gold_answers=["tested", "tested"],
        )
        assert math.isnan(result["pbs"])

    def test_none_predictions_skipped(self):
        result = compute_pbs(
            predictions=[None, "tested", "untested"],
            gold_answers=["tested", "tested", "untested"],
        )
        assert result["n_true_pos"] == 1
        assert result["n_true_neg"] == 1


class TestPBSByTier:
    def test_tier_stratification(self):
        result = compute_pbs_by_tier(
            predictions=["tested", "untested", "tested", "untested"],
            gold_answers=["tested", "untested", "untested", "tested"],
            tiers=["gold", "gold", "silver", "silver"],
        )
        assert "gold" in result
        assert "silver" in result
        # Gold: perfect → PBS = 0 - 1 = -1
        assert result["gold"]["pbs"] == -1.0
        # Silver: reversed → PBS=1.0
        assert result["silver"]["pbs"] == 1.0


class TestPBSDelta:
    def test_bias_reduction(self):
        before = {"pbs": 0.5}
        after = {"pbs": 0.1}
        delta = compute_pbs_delta(before, after)
        assert delta["delta_pbs"] == -0.4
        assert delta["bias_reduced"] is True

    def test_bias_increase(self):
        before = {"pbs": 0.1}
        after = {"pbs": 0.5}
        delta = compute_pbs_delta(before, after)
        assert delta["delta_pbs"] == 0.4
        assert delta["bias_reduced"] is False

    def test_nan_handling(self):
        delta = compute_pbs_delta({"pbs": float("nan")}, {"pbs": 0.5})
        assert math.isnan(delta["delta_pbs"])


class TestMultiDomainPBS:
    def test_aggregation(self):
        result = compute_multi_domain_pbs({
            "dti": {
                "predictions": ["tested", "tested"],
                "gold_answers": ["tested", "untested"],
            },
            "ct": {
                "predictions": ["tested", "untested"],
                "gold_answers": ["tested", "untested"],
            },
        })
        assert result["n_domains"] == 2
        assert "dti" in result["per_domain"]
        assert "ct" in result["per_domain"]
        assert not math.isnan(result["mean_pbs"])
