"""Tests for PPI results collection (scripts_ppi/collect_results.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts_ppi"))

from collect_results import (
    MODEL_ORDER,
    NEG_ORDER,
    SPLIT_ORDER,
    TABLE_METRICS,
    aggregate_over_seeds,
    build_table1,
    filter_results,
    format_markdown,
    load_results,
    summarize_exp1,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result_json(
    run_dir: Path,
    model: str = "siamese_cnn",
    split: str = "random",
    negative: str = "negbiodb",
    dataset: str = "balanced",
    seed: int = 42,
    log_auc: float = 0.300,
) -> Path:
    """Create a fake results.json in a run directory."""
    run_name = f"{model}_{dataset}_{split}_{negative}_seed{seed}"
    d = run_dir / run_name
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "model": model,
        "split": split,
        "negative": negative,
        "dataset": dataset,
        "seed": seed,
        "best_val_log_auc": log_auc,
        "n_train": 1000,
        "n_val": 100,
        "n_test": 200,
        "test_metrics": {
            "log_auc": log_auc,
            "auprc": log_auc * 0.8,
            "bedroc": log_auc * 0.9,
            "ef_1pct": 5.0,
            "ef_5pct": 3.0,
            "mcc": log_auc * 0.5,
            "auroc": log_auc + 0.5,
        },
    }
    path = d / "results.json"
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Create a results directory with 18 runs (3 models x 6 configs)."""
    rd = tmp_path / "ppi_baselines"
    for model in MODEL_ORDER:
        # B1-B3: 3 splits x negbiodb
        for split in ["random", "cold_protein", "cold_both"]:
            _make_result_json(rd, model=model, split=split, negative="negbiodb", log_auc=0.300)
        # E1: 2 controls x random
        _make_result_json(rd, model=model, split="random", negative="uniform_random", log_auc=0.450)
        _make_result_json(rd, model=model, split="random", negative="degree_matched", log_auc=0.400)
        # E4: DDB
        _make_result_json(rd, model=model, split="ddb", negative="negbiodb", log_auc=0.280)
    return rd


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadResults:
    def test_loads_all(self, results_dir: Path):
        df = load_results(results_dir)
        assert len(df) == 18  # 3 models x 6 configs

    def test_empty_dir(self, tmp_path: Path):
        df = load_results(tmp_path / "nonexistent")
        assert df.empty

    def test_has_expected_columns(self, results_dir: Path):
        df = load_results(results_dir)
        for col in ["model", "split", "negative", "dataset", "seed"] + TABLE_METRICS:
            assert col in df.columns

    def test_deduplication(self, results_dir: Path):
        # Create a duplicate run
        _make_result_json(results_dir, model="siamese_cnn", split="random",
                         negative="negbiodb", log_auc=0.350)
        df = load_results(results_dir)
        mask = (df["model"] == "siamese_cnn") & (df["split"] == "random") & (df["negative"] == "negbiodb")
        assert mask.sum() == 1


class TestFilterResults:
    def test_filter_by_model(self, results_dir: Path):
        df = load_results(results_dir)
        filtered = filter_results(df, models=["siamese_cnn"])
        assert len(filtered) == 6
        assert (filtered["model"] == "siamese_cnn").all()

    def test_filter_by_split(self, results_dir: Path):
        df = load_results(results_dir)
        filtered = filter_results(df, splits=["random"])
        assert len(filtered) == 9  # 3 models x 3 negatives

    def test_filter_by_negative(self, results_dir: Path):
        df = load_results(results_dir)
        filtered = filter_results(df, negatives=["negbiodb"])
        assert len(filtered) == 12  # 3 models x 4 splits


class TestBuildTable1:
    def test_ordered_output(self, results_dir: Path):
        df = load_results(results_dir)
        table = build_table1(df)
        assert len(table) == 18
        # First row should be siamese_cnn (first in MODEL_ORDER)
        assert table.iloc[0]["model"] == "siamese_cnn"

    def test_has_metrics(self, results_dir: Path):
        df = load_results(results_dir)
        table = build_table1(df)
        for m in TABLE_METRICS:
            assert m in table.columns


class TestAggregateOverSeeds:
    def test_single_seed(self, results_dir: Path):
        df = load_results(results_dir)
        agg = aggregate_over_seeds(df)
        assert len(agg) == 18  # same as unique configs (1 seed each)
        assert (agg["n_seeds"] == 1).all()

    def test_multi_seed(self, results_dir: Path):
        # Add seed=43 runs
        for model in MODEL_ORDER:
            _make_result_json(results_dir, model=model, split="random",
                             negative="negbiodb", seed=43, log_auc=0.320)
        df = load_results(results_dir)
        agg = aggregate_over_seeds(df)
        mask = (agg["split"] == "random") & (agg["negative"] == "negbiodb")
        assert (agg.loc[mask, "n_seeds"] == 2).all()

    def test_empty(self):
        agg = aggregate_over_seeds(pd.DataFrame())
        assert agg.empty


class TestFormatMarkdown:
    def test_produces_table(self, results_dir: Path):
        df = load_results(results_dir)
        table = build_table1(df)
        md = format_markdown(table)
        assert "| **Model**" in md
        assert "siamese_cnn" in md
        assert len(md.split("\n")) == 20  # header + separator + 18 rows


class TestSummarizeExp1:
    def test_inflation_computed(self, results_dir: Path):
        df = load_results(results_dir)
        exp1_df = df[df["split"] == "random"]
        summary = summarize_exp1(exp1_df)
        assert "NegBioDB=" in summary
        assert "uniform_random=" in summary
        assert "degree_matched=" in summary

    def test_empty_df(self):
        summary = summarize_exp1(pd.DataFrame())
        assert "No Exp 1" in summary


class TestMainCLI:
    def test_main_success(self, results_dir: Path, tmp_path: Path):
        from collect_results import main
        out_dir = tmp_path / "out"
        ret = main([
            "--results-dir", str(results_dir),
            "--out", str(out_dir),
        ])
        assert ret == 0
        assert (out_dir / "table1.csv").exists()
        assert (out_dir / "table1.md").exists()

    def test_main_aggregate(self, results_dir: Path, tmp_path: Path):
        from collect_results import main
        out_dir = tmp_path / "out"
        ret = main([
            "--results-dir", str(results_dir),
            "--out", str(out_dir),
            "--aggregate-seeds",
        ])
        assert ret == 0
        assert (out_dir / "table1_aggregated.csv").exists()

    def test_main_no_results(self, tmp_path: Path):
        from collect_results import main
        ret = main([
            "--results-dir", str(tmp_path / "empty"),
            "--out", str(tmp_path / "out"),
        ])
        assert ret == 1
