"""Tests for negbiodb.metrics — ML evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from negbiodb.metrics import (
    auprc,
    auroc,
    bedroc,
    compute_all_metrics,
    enrichment_factor,
    evaluate_splits,
    log_auc,
    mcc,
    save_results,
    summarize_runs,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def perfect_data():
    """Perfect classifier: actives score=1.0, inactives score=0.0."""
    y_true = np.array([1] * 50 + [0] * 950)
    y_score = np.array([1.0] * 50 + [0.0] * 950)
    return y_true, y_score


@pytest.fixture
def random_data():
    """Random classifier: uniform random scores, ~5% prevalence."""
    rng = np.random.RandomState(42)
    y_true = np.zeros(1000, dtype=int)
    y_true[:50] = 1
    rng.shuffle(y_true)
    y_score = rng.uniform(0, 1, 1000)
    return y_true, y_score


@pytest.fixture
def good_data():
    """Good classifier: actives biased high, inactives biased low."""
    rng = np.random.RandomState(42)
    y_true = np.array([1] * 100 + [0] * 900)
    y_score = np.concatenate([
        rng.beta(5, 2, 100),   # actives: skewed high
        rng.beta(2, 5, 900),   # inactives: skewed low
    ])
    return y_true, y_score


# ------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------

class TestInputValidation:
    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            auroc(np.array([0, 1]), np.array([0.5]))

    def test_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 2"):
            auroc(np.array([1]), np.array([0.5]))

    def test_non_binary_labels(self):
        with pytest.raises(ValueError, match="only 0 and 1"):
            auroc(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))

    def test_nan_in_scores(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            auroc(np.array([0, 1]), np.array([0.5, float("nan")]))

    def test_inf_in_scores(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            auroc(np.array([0, 1]), np.array([0.5, float("inf")]))

    def test_accepts_lists(self):
        """Python lists should be accepted (not just numpy arrays)."""
        result = auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert result == 1.0

    def test_accepts_pandas_series(self):
        """Pandas Series should be accepted."""
        import pandas as pd
        y_true = pd.Series([0, 0, 1, 1])
        y_score = pd.Series([0.1, 0.2, 0.8, 0.9])
        result = auroc(y_true, y_score)
        assert result == 1.0


# ------------------------------------------------------------------
# AUROC
# ------------------------------------------------------------------

class TestAUROC:
    def test_perfect(self, perfect_data):
        y_true, y_score = perfect_data
        assert auroc(y_true, y_score) == 1.0

    def test_random_near_half(self, random_data):
        y_true, y_score = random_data
        result = auroc(y_true, y_score)
        assert 0.3 < result < 0.7  # random should be around 0.5

    def test_single_class_returns_nan(self):
        y_true = np.zeros(100)
        y_score = np.random.RandomState(0).uniform(0, 1, 100)
        with pytest.warns(match="only one class"):
            result = auroc(y_true, y_score)
        assert np.isnan(result)


# ------------------------------------------------------------------
# AUPRC
# ------------------------------------------------------------------

class TestAUPRC:
    def test_perfect(self, perfect_data):
        y_true, y_score = perfect_data
        assert auprc(y_true, y_score) == 1.0

    def test_good_better_than_random(self, good_data):
        y_true, y_score = good_data
        result = auprc(y_true, y_score)
        random_baseline = y_true.sum() / len(y_true)  # = 0.1
        assert result > random_baseline

    def test_single_class_returns_nan(self):
        y_true = np.ones(100)
        y_score = np.random.RandomState(0).uniform(0, 1, 100)
        with pytest.warns(match="only one class"):
            result = auprc(y_true, y_score)
        assert np.isnan(result)


# ------------------------------------------------------------------
# MCC
# ------------------------------------------------------------------

class TestMCC:
    def test_perfect(self, perfect_data):
        y_true, y_score = perfect_data
        assert mcc(y_true, y_score) == 1.0

    def test_random_near_zero(self, random_data):
        y_true, y_score = random_data
        result = mcc(y_true, y_score)
        assert -0.2 < result < 0.2  # random should be near 0

    def test_custom_threshold(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.3, 0.6, 0.8])
        # Threshold 0.5: pred=[0,0,1,1] → MCC=1.0
        assert mcc(y_true, y_score, threshold=0.5) == 1.0
        # Threshold 0.05: pred=[1,1,1,1] → all same class → NaN
        with pytest.warns(match="all predictions are the same"):
            result = mcc(y_true, y_score, threshold=0.05)
        assert np.isnan(result)

    def test_single_class_returns_nan(self):
        y_true = np.zeros(100)
        y_score = np.random.RandomState(0).uniform(0, 1, 100)
        with pytest.warns(match="only one class"):
            result = mcc(y_true, y_score)
        assert np.isnan(result)


# ------------------------------------------------------------------
# LogAUC
# ------------------------------------------------------------------

class TestLogAUC:
    def test_perfect_near_one(self, perfect_data):
        """Perfect classifier should have LogAUC close to 1.0."""
        y_true, y_score = perfect_data
        result = log_auc(y_true, y_score)
        assert result > 0.99

    def test_random_baseline(self, random_data):
        """Random classifier should be near the random baseline."""
        y_true, y_score = random_data
        result = log_auc(y_true, y_score)
        # Random baseline for [0.001, 0.1] is ~0.0215
        assert 0.0 < result < 0.15

    def test_good_between(self, good_data):
        """Good classifier between random baseline and perfect."""
        y_true, y_score = good_data
        result = log_auc(y_true, y_score)
        assert 0.3 < result < 1.0

    def test_custom_fpr_range(self, good_data):
        """Different FPR range should give different value."""
        y_true, y_score = good_data
        r1 = log_auc(y_true, y_score, fpr_range=(0.001, 0.1))
        r2 = log_auc(y_true, y_score, fpr_range=(0.01, 0.5))
        assert r1 != r2

    def test_single_class_returns_nan(self):
        y_true = np.zeros(100)
        y_score = np.random.RandomState(0).uniform(0, 1, 100)
        with pytest.warns(match="only one class"):
            result = log_auc(y_true, y_score)
        assert np.isnan(result)

    def test_monotonicity_with_quality(self):
        """Better classifier should have higher LogAUC."""
        rng = np.random.RandomState(123)
        y_true = np.array([1] * 50 + [0] * 950)

        # Good: actives scored high
        y_good = np.concatenate([rng.beta(8, 2, 50), rng.beta(2, 8, 950)])
        # Weak: slight separation
        y_weak = np.concatenate([rng.beta(3, 2, 50), rng.beta(2, 3, 950)])

        assert log_auc(y_true, y_good) > log_auc(y_true, y_weak)

    def test_normalized_range(self, good_data):
        """LogAUC should be in [0, 1] for valid inputs."""
        y_true, y_score = good_data
        result = log_auc(y_true, y_score)
        assert 0.0 <= result <= 1.0

    def test_worst_classifier_below_random(self):
        """Worst classifier (actives at bottom) should score below random baseline."""
        rng = np.random.RandomState(99)
        y_true = np.array([1] * 50 + [0] * 950)
        # Reverse ranking: inactives get high scores, actives get low scores
        y_score = np.concatenate([rng.uniform(0.0, 0.3, 50), rng.uniform(0.7, 1.0, 950)])
        result = log_auc(y_true, y_score)
        # Random baseline ~0.0215; worst classifier should be well below
        assert result < 0.01


# ------------------------------------------------------------------
# BEDROC
# ------------------------------------------------------------------

class TestBEDROC:
    def test_perfect_near_one(self, perfect_data):
        """Perfect classifier should have BEDROC close to 1.0."""
        y_true, y_score = perfect_data
        result = bedroc(y_true, y_score, alpha=20.0)
        assert result > 0.99

    def test_random_near_ra(self, random_data):
        """Random classifier BEDROC should be near active fraction (ra)."""
        y_true, y_score = random_data
        ra = y_true.sum() / len(y_true)  # 0.05
        result = bedroc(y_true, y_score, alpha=20.0)
        # Random should be roughly near ra, allow wide margin
        assert 0.0 < result < 0.3

    def test_good_between(self, good_data):
        """Good classifier between random and perfect."""
        y_true, y_score = good_data
        result = bedroc(y_true, y_score, alpha=20.0)
        ra = y_true.sum() / len(y_true)
        assert ra < result < 1.0

    def test_different_alpha(self, good_data):
        """Higher alpha emphasizes top ranks more."""
        y_true, y_score = good_data
        b20 = bedroc(y_true, y_score, alpha=20.0)
        b80 = bedroc(y_true, y_score, alpha=80.0)
        # Both should be valid BEDROC scores
        assert 0.0 < b20 <= 1.0
        assert 0.0 < b80 <= 1.0

    def test_no_actives_returns_nan(self):
        y_true = np.zeros(100)
        y_score = np.random.RandomState(0).uniform(0, 1, 100)
        with pytest.warns(match="no actives"):
            result = bedroc(y_true, y_score)
        assert np.isnan(result)

    def test_all_actives_returns_one(self):
        """When all samples are active, BEDROC should be 1.0."""
        y_true = np.ones(100)
        y_score = np.random.RandomState(0).uniform(0, 1, 100)
        result = bedroc(y_true, y_score)
        assert result == 1.0

    def test_accepts_lists(self):
        result = bedroc([1, 0, 0, 0, 0], [0.9, 0.1, 0.2, 0.3, 0.4])
        assert 0.0 < result <= 1.0


# ------------------------------------------------------------------
# Cross-validation with hand-computed values
# ------------------------------------------------------------------

class TestCrossValidation:
    def test_bedroc_actives_at_top(self):
        """Actives ranked at top should give high BEDROC."""
        # 2 actives at ranks 1 and 2 out of 10
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        result = bedroc(y_true, y_score, alpha=20.0)
        assert result > 0.8  # actives at top → high BEDROC

    def test_bedroc_actives_at_bottom(self):
        """Actives ranked at bottom should give low BEDROC."""
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        result = bedroc(y_true, y_score, alpha=20.0)
        assert result < 0.2  # actives at bottom → low BEDROC

    def test_log_auc_perfect_small(self):
        """Small perfect classifier: LogAUC should be 1.0."""
        # 5 actives scored [1.0, 0.9, 0.8, 0.7, 0.6]
        # 95 inactives scored [0.0-0.5] range
        y_true = np.array([1] * 5 + [0] * 95)
        y_score = np.concatenate([
            np.linspace(1.0, 0.6, 5),
            np.linspace(0.5, 0.0, 95),
        ])
        result = log_auc(y_true, y_score)
        assert result > 0.95  # near-perfect separation


# ------------------------------------------------------------------
# Enrichment Factor
# ------------------------------------------------------------------

class TestEnrichmentFactor:
    def test_perfect_ef1(self, perfect_data):
        """Perfect classifier: all 50 actives in top 1% (=10 slots).
        EF@1% = (10/10) / (50/1000) = 1.0/0.05 = 20.0"""
        y_true, y_score = perfect_data
        result = enrichment_factor(y_true, y_score, percentage=1.0)
        assert result == 20.0

    def test_perfect_ef5(self, perfect_data):
        """Perfect classifier: all 50 actives in top 5% (=50 slots).
        EF@5% = (50/50) / (50/1000) = 1.0/0.05 = 20.0"""
        y_true, y_score = perfect_data
        result = enrichment_factor(y_true, y_score, percentage=5.0)
        assert result == 20.0

    def test_random_near_one(self, random_data):
        """Random classifier EF should be near 1.0."""
        y_true, y_score = random_data
        result = enrichment_factor(y_true, y_score, percentage=5.0)
        assert 0.0 < result < 3.0  # allow variance

    def test_ef100_always_one(self, good_data):
        """EF@100% is always exactly 1.0."""
        y_true, y_score = good_data
        result = enrichment_factor(y_true, y_score, percentage=100.0)
        assert result == pytest.approx(1.0)

    def test_no_actives_returns_nan(self):
        y_true = np.zeros(100)
        y_score = np.random.RandomState(0).uniform(0, 1, 100)
        with pytest.warns(match="no actives"):
            result = enrichment_factor(y_true, y_score)
        assert np.isnan(result)

    def test_ef_monotonicity(self, good_data):
        """For a good classifier, EF@1% >= EF@5% >= EF@10%."""
        y_true, y_score = good_data
        ef1 = enrichment_factor(y_true, y_score, percentage=1.0)
        ef5 = enrichment_factor(y_true, y_score, percentage=5.0)
        ef10 = enrichment_factor(y_true, y_score, percentage=10.0)
        # Good classifiers should concentrate actives at the top
        assert ef1 >= ef5 >= ef10

    def test_all_same_score(self):
        """When all scores are identical, EF should be near 1.0."""
        y_true = np.array([1] * 10 + [0] * 90)
        y_score = np.full(100, 0.5)
        result = enrichment_factor(y_true, y_score, percentage=10.0)
        assert result == pytest.approx(1.0)

    def test_invalid_percentage(self):
        with pytest.raises(ValueError, match="percentage must be in"):
            enrichment_factor(np.array([0, 1]), np.array([0.1, 0.9]), percentage=0.0)
        with pytest.raises(ValueError, match="percentage must be in"):
            enrichment_factor(np.array([0, 1]), np.array([0.1, 0.9]), percentage=-5.0)


# ------------------------------------------------------------------
# Additional edge cases (from 3-agent review)
# ------------------------------------------------------------------

class TestLogAUCEdgeCases:
    def test_all_same_score(self):
        """All same scores → near random baseline (~0.0215)."""
        y_true = np.array([1] * 50 + [0] * 950)
        y_score = np.full(1000, 0.5)
        result = log_auc(y_true, y_score)
        assert 0.0 < result < 0.1

    def test_invalid_fpr_range_zero(self):
        with pytest.raises(ValueError, match="lower bound must be > 0"):
            log_auc(np.array([0, 1]), np.array([0.1, 0.9]), fpr_range=(0.0, 1.0))

    def test_invalid_fpr_range_inverted(self):
        with pytest.raises(ValueError, match="lower must be < upper"):
            log_auc(np.array([0, 1]), np.array([0.1, 0.9]), fpr_range=(0.5, 0.1))

    def test_invalid_fpr_range_upper_gt_one(self):
        with pytest.raises(ValueError, match="upper bound must be <= 1.0"):
            log_auc(np.array([0, 1]), np.array([0.1, 0.9]), fpr_range=(0.001, 1.5))


class TestBEDROCEdgeCases:
    def test_all_same_score(self):
        """All same scores with randomized tie order → BEDROC near ra."""
        y_true = np.array([1] * 50 + [0] * 950)
        rng = np.random.RandomState(0)
        y_true = y_true[rng.permutation(len(y_true))]
        y_score = np.full(1000, 0.5)
        result = bedroc(y_true, y_score)
        ra = 50 / 1000  # 0.05
        assert result == pytest.approx(ra, abs=0.25)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be > 0"):
            bedroc(np.array([0, 1]), np.array([0.1, 0.9]), alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be > 0"):
            bedroc(np.array([0, 1]), np.array([0.1, 0.9]), alpha=-5.0)

    def test_extreme_alpha_large(self):
        """Very large alpha should not crash (numerical stability)."""
        y_true = np.array([1] * 10 + [0] * 90)
        rng = np.random.RandomState(42)
        y_score = np.concatenate([rng.beta(5, 2, 10), rng.beta(2, 5, 90)])
        result = bedroc(y_true, y_score, alpha=500.0)
        assert 0.0 <= result <= 1.0

    def test_extreme_alpha_1500(self):
        """alpha=1500 previously caused sinh/cosh overflow → NaN."""
        y_true = np.array([1] * 10 + [0] * 90)
        rng = np.random.RandomState(42)
        y_score = np.concatenate([rng.beta(5, 2, 10), rng.beta(2, 5, 90)])
        result = bedroc(y_true, y_score, alpha=1500.0)
        assert np.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_small_alpha_no_crash(self):
        """Very small alpha (1e-20) is numerically degenerate (formula is 0/0
        as alpha→0). Should return NaN with UserWarning, not crash or produce -1e15."""
        y_true = np.array([1] * 10 + [0] * 90)
        y_score = np.concatenate([np.ones(10), np.zeros(90)])
        with pytest.warns(UserWarning, match="non-finite"):
            result = bedroc(y_true, y_score, alpha=1e-20)
        assert np.isnan(result)

    def test_exactly_one_active(self):
        """BEDROC with exactly 1 active should compute without error."""
        rng = np.random.RandomState(7)
        y_true = np.zeros(100, dtype=int)
        y_true[0] = 1  # 1 active
        y_score = rng.uniform(0, 1, 100)
        y_score[0] = 1.0  # active ranked first
        result = bedroc(y_true, y_score, alpha=20.0)
        assert np.isfinite(result)
        assert 0.0 <= result <= 1.0
        # Active is ranked first → high BEDROC
        assert result > 0.5


# ------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_returns_all_keys(self, good_data):
        y_true, y_score = good_data
        result = compute_all_metrics(y_true, y_score)
        expected_keys = {"auroc", "auprc", "mcc", "log_auc", "bedroc", "ef_1pct", "ef_5pct"}
        assert set(result.keys()) == expected_keys

    def test_values_in_range(self, good_data):
        y_true, y_score = good_data
        result = compute_all_metrics(y_true, y_score)
        assert 0.0 <= result["auroc"] <= 1.0
        assert 0.0 <= result["auprc"] <= 1.0
        assert 0.0 <= result["log_auc"] <= 1.0
        assert 0.0 <= result["bedroc"] <= 1.0
        assert result["ef_1pct"] >= 0.0
        assert result["ef_5pct"] >= 0.0

    def test_matches_individual(self, good_data):
        y_true, y_score = good_data
        result = compute_all_metrics(y_true, y_score)
        assert result["auroc"] == auroc(y_true, y_score)
        assert result["auprc"] == auprc(y_true, y_score)
        assert result["log_auc"] == log_auc(y_true, y_score)
        assert result["bedroc"] == bedroc(y_true, y_score)


class TestSummarizeRuns:
    def test_mean_std(self):
        runs = [
            {"auroc": 0.8, "auprc": 0.6},
            {"auroc": 0.9, "auprc": 0.7},
            {"auroc": 0.85, "auprc": 0.65},
        ]
        result = summarize_runs(runs)
        assert result["auroc"]["mean"] == pytest.approx(0.85, abs=1e-10)
        assert result["auprc"]["mean"] == pytest.approx(0.65, abs=1e-10)
        assert result["auroc"]["std"] > 0

    def test_single_run_zero_std(self):
        runs = [{"auroc": 0.8, "auprc": 0.6}]
        result = summarize_runs(runs)
        assert result["auroc"]["mean"] == 0.8
        assert result["auroc"]["std"] == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            summarize_runs([])


class TestSaveResults:
    def test_json_roundtrip(self, tmp_path):
        import json
        metrics = {"auroc": 0.85, "auprc": 0.7, "mcc": float("nan")}
        path = tmp_path / "results.json"
        save_results(metrics, path, format="json")
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["auroc"] == 0.85
        assert loaded["mcc"] is None  # NaN → null in JSON

    def test_csv_output(self, tmp_path):
        import pandas as pd
        metrics = [
            {"auroc": 0.85, "auprc": 0.7},
            {"auroc": 0.90, "auprc": 0.75},
        ]
        path = tmp_path / "results.csv"
        save_results(metrics, path, format="csv")
        df = pd.read_csv(path)
        assert len(df) == 2
        assert list(df.columns) == ["auroc", "auprc"]

    def test_invalid_format(self, tmp_path):
        with pytest.raises(ValueError, match="format must be"):
            save_results({}, tmp_path / "x.txt", format="xml")

    def test_csv_nested_dict_flattens(self, tmp_path):
        """summarize_runs output (nested dict) should flatten to CSV columns."""
        import pandas as pd
        nested = {"auroc": {"mean": 0.85, "std": 0.01}, "auprc": {"mean": 0.7, "std": 0.02}}
        path = tmp_path / "summary.csv"
        save_results(nested, path, format="csv")
        df = pd.read_csv(path)
        assert "auroc_mean" in df.columns
        assert "auprc_std" in df.columns
        assert df["auroc_mean"].iloc[0] == pytest.approx(0.85)

    def test_csv_list_of_nested_dicts_flattens(self, tmp_path):
        """List of nested dicts should also flatten for CSV."""
        import pandas as pd
        nested_list = [
            {"auroc": {"mean": 0.85, "std": 0.01}},
            {"auroc": {"mean": 0.90, "std": 0.02}},
        ]
        path = tmp_path / "list_nested.csv"
        save_results(nested_list, path, format="csv")
        df = pd.read_csv(path)
        assert "auroc_mean" in df.columns
        assert len(df) == 2
        assert df["auroc_mean"].iloc[1] == pytest.approx(0.90)


# ------------------------------------------------------------------
# evaluate_splits
# ------------------------------------------------------------------

class TestEvaluateSplits:
    def test_per_fold_keys(self):
        """Each fold should have all 7 metric keys with finite values."""
        rng = np.random.RandomState(42)
        y_true = np.array([1] * 30 + [0] * 70)
        y_score = np.concatenate([rng.beta(5, 2, 30), rng.beta(2, 5, 70)])
        # Shuffle so both folds get actives
        idx = rng.permutation(100)
        y_true, y_score = y_true[idx], y_score[idx]
        splits = np.array(["train"] * 50 + ["test"] * 50)
        result = evaluate_splits(y_true, y_score, splits)
        assert set(result.keys()) == {"train", "test"}
        for fold_metrics in result.values():
            expected_keys = {"auroc", "auprc", "mcc", "log_auc", "bedroc", "ef_1pct", "ef_5pct"}
            assert set(fold_metrics.keys()) == expected_keys
            assert np.isfinite(fold_metrics["auroc"])

    def test_fold_independence(self):
        """Each fold result should match independent compute_all_metrics."""
        rng = np.random.RandomState(42)
        y_true = np.array([1] * 30 + [0] * 70)
        y_score = np.concatenate([rng.beta(5, 2, 30), rng.beta(2, 5, 70)])
        idx = rng.permutation(100)
        y_true, y_score = y_true[idx], y_score[idx]
        splits = np.array(["a"] * 50 + ["b"] * 50)
        result = evaluate_splits(y_true, y_score, splits)
        # Verify each fold matches direct computation
        for fold, sl in [("a", slice(0, 50)), ("b", slice(50, 100))]:
            expected = compute_all_metrics(y_true[sl], y_score[sl])
            for key in expected:
                assert result[fold][key] == pytest.approx(expected[key], nan_ok=True)

    def test_mismatched_split_labels_length(self):
        with pytest.raises(ValueError, match="same length"):
            evaluate_splits(np.array([0, 1, 0]), np.array([0.1, 0.9, 0.5]),
                            np.array(["a", "b"]))

    def test_degenerate_fold_returns_nan(self):
        """Fold with <2 samples should return all NaN, not crash."""
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, 0.9, 0.2, 0.8])
        splits = np.array(["a", "a", "a", "b"])  # fold "b" has 1 sample
        with pytest.warns(match="fold 'b' has 1 sample"):
            result = evaluate_splits(y_true, y_score, splits)
        assert all(np.isnan(v) for v in result["b"].values())

    def test_fold_with_zero_actives_returns_nan(self):
        """Fold with ≥2 samples but 0 actives should return all NaN."""
        rng = np.random.RandomState(42)
        y_true = np.array([1] * 10 + [0] * 90)
        y_score = rng.uniform(0, 1, 100)
        # fold "a": 10 actives + 40 inactives; fold "b": 0 actives + 50 inactives
        splits = np.array(["a"] * 50 + ["b"] * 50)
        with pytest.warns(match="only one class"):
            result = evaluate_splits(y_true, y_score, splits)
        assert all(np.isnan(v) for v in result["b"].values())
        # fold "a" should still compute normally
        assert np.isfinite(result["a"]["auroc"])

    def test_integer_fold_labels(self):
        """evaluate_splits should accept integer fold labels (converted to str keys)."""
        rng = np.random.RandomState(42)
        y_true = np.array([1] * 30 + [0] * 70)
        y_score = np.concatenate([rng.beta(5, 2, 30), rng.beta(2, 5, 70)])
        idx = rng.permutation(100)
        y_true, y_score = y_true[idx], y_score[idx]
        splits = np.array([0] * 50 + [1] * 50)
        result = evaluate_splits(y_true, y_score, splits)
        assert "0" in result and "1" in result


# ------------------------------------------------------------------
# Additional review-driven tests
# ------------------------------------------------------------------

class TestReviewFindings:
    def test_compute_all_single_class(self):
        """All-zeros y_true → all NaN, no crash."""
        y_true = np.zeros(100)
        y_score = np.random.RandomState(0).uniform(0, 1, 100)
        with pytest.warns(UserWarning):
            result = compute_all_metrics(y_true, y_score)
        assert len(result) == 7
        assert all(np.isnan(v) for v in result.values())

    def test_summarize_runs_with_nan(self):
        """NaN values in some runs should be skipped by nanmean."""
        runs = [
            {"auroc": 0.8, "auprc": 0.6},
            {"auroc": float("nan"), "auprc": 0.7},
            {"auroc": 0.9, "auprc": float("nan")},
        ]
        result = summarize_runs(runs)
        assert result["auroc"]["mean"] == pytest.approx(0.85)
        assert result["auprc"]["mean"] == pytest.approx(0.65)

    def test_summarize_runs_inconsistent_keys(self):
        """Inconsistent keys should raise ValueError."""
        runs = [{"auroc": 0.8, "auprc": 0.6}, {"auroc": 0.9}]
        with pytest.raises(ValueError, match="inconsistent keys"):
            summarize_runs(runs)


class TestEndToEnd:
    def test_pipeline_json(self, tmp_path, good_data):
        """compute_all_metrics -> summarize_runs -> save_results (JSON)."""
        import json
        y_true, y_score = good_data
        runs = [compute_all_metrics(y_true, y_score + offset)
                for offset in [0.0, 0.01, -0.01]]
        summary = summarize_runs(runs)
        path = tmp_path / "pipeline.json"
        save_results(summary, path, format="json")
        with open(path) as f:
            loaded = json.load(f)
        assert "auroc" in loaded
        assert "mean" in loaded["auroc"]
        assert isinstance(loaded["auroc"]["mean"], float)

    def test_pipeline_csv(self, tmp_path, good_data):
        """compute_all_metrics -> summarize_runs -> save_results (CSV)."""
        import pandas as pd
        y_true, y_score = good_data
        runs = [compute_all_metrics(y_true, y_score)]
        summary = summarize_runs(runs)
        path = tmp_path / "pipeline.csv"
        save_results(summary, path, format="csv")
        df = pd.read_csv(path)
        assert "auroc_mean" in df.columns
        assert "auroc_std" in df.columns

    def test_save_np_integer(self, tmp_path):
        """save_results should handle np.integer values in JSON."""
        import json
        metrics = {"n_samples": np.int64(1000), "auroc": np.float64(0.85)}
        path = tmp_path / "np_types.json"
        save_results(metrics, path, format="json")
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["n_samples"] == 1000
        assert isinstance(loaded["n_samples"], int)
