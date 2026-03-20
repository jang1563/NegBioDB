"""Tests for the PPI training harness (scripts_ppi/train_baseline.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_ppi"))

from train_baseline import (
    PPIDataset,
    _build_run_name,
    _collate_features,
    _collate_sequence_pair,
    _compute_val_metric,
    _json_safe,
    _resolve_dataset_file,
    _DATASET_MAP,
    _SPLIT_COL_MAP,
    build_model,
    make_dataloader,
    set_seed,
    write_results_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_parquet(tmp_path: Path) -> Path:
    """Create a minimal PPI parquet with split columns."""
    rng = np.random.RandomState(42)
    n = 60
    seqs = ["ACDEFGHIKLMNPQRSTVWY" * 3] * n  # 60-char sequence
    df = pd.DataFrame({
        "pair_id": range(n),
        "uniprot_id_1": [f"P{i:05d}" for i in range(n)],
        "sequence_1": seqs,
        "gene_symbol_1": [f"GEN{i}" for i in range(n)],
        "subcellular_location_1": ["Nucleus"] * (n // 2) + ["Cytoplasm"] * (n // 2),
        "uniprot_id_2": [f"Q{i:05d}" for i in range(n)],
        "sequence_2": seqs,
        "gene_symbol_2": [f"GEN{i + 100}" for i in range(n)],
        "subcellular_location_2": ["Membrane"] * n,
        "Y": ([1, 0] * (n // 2)),  # interleave for balanced splits
        "confidence_tier": [None, "gold"] * (n // 2),
        "num_sources": [1] * n,
        "protein1_degree": rng.randint(1, 50, n).tolist(),
        "protein2_degree": rng.randint(1, 50, n).tolist(),
        "split_random": (["train"] * 42 + ["val"] * 6 + ["test"] * 12),
        "split_cold_protein": (["train"] * 42 + ["val"] * 6 + ["test"] * 12),
        "split_cold_both": (["train"] * 36 + ["val"] * 6 + ["test"] * 12 + [None] * 6),
        "split_degree_balanced": (["train"] * 42 + ["val"] * 6 + ["test"] * 12),
    })
    path = tmp_path / "ppi_m1_balanced.parquet"
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestResolveDatasetFile:
    def test_balanced_negbiodb(self):
        assert _resolve_dataset_file("balanced", "random", "negbiodb") == "ppi_m1_balanced.parquet"

    def test_realistic_negbiodb(self):
        assert _resolve_dataset_file("realistic", "random", "negbiodb") == "ppi_m1_realistic.parquet"

    def test_balanced_uniform_random(self):
        assert _resolve_dataset_file("balanced", "random", "uniform_random") == "ppi_m1_uniform_random.parquet"

    def test_balanced_degree_matched(self):
        assert _resolve_dataset_file("balanced", "random", "degree_matched") == "ppi_m1_degree_matched.parquet"

    def test_ddb_split(self):
        assert _resolve_dataset_file("balanced", "ddb", "negbiodb") == "ppi_m1_balanced_ddb.parquet"

    def test_ddb_non_negbiodb_returns_none(self):
        assert _resolve_dataset_file("balanced", "ddb", "uniform_random") is None

    def test_realistic_uniform_random_returns_none(self):
        assert _resolve_dataset_file("realistic", "random", "uniform_random") is None


class TestBuildRunName:
    def test_format(self):
        assert _build_run_name("siamese_cnn", "balanced", "random", "negbiodb", 42) == \
            "siamese_cnn_balanced_random_negbiodb_seed42"


class TestJsonSafe:
    def test_nan(self):
        assert _json_safe(float("nan")) is None

    def test_inf(self):
        assert _json_safe(float("inf")) is None

    def test_np_float(self):
        assert _json_safe(np.float64(0.5)) == 0.5

    def test_np_int(self):
        assert _json_safe(np.int64(42)) == 42

    def test_nested(self):
        result = _json_safe({"a": np.float64(1.0), "b": [np.int64(2)]})
        assert result == {"a": 1.0, "b": [2]}


class TestWriteResultsJson:
    def test_writes_valid_json(self, tmp_path: Path):
        path = tmp_path / "results.json"
        payload = {"metric": np.float64(0.95), "n": np.int64(100)}
        write_results_json(path, payload)
        with open(path) as f:
            data = json.load(f)
        assert data["metric"] == 0.95
        assert data["n"] == 100


class TestSetSeed:
    def test_deterministic(self):
        set_seed(123)
        a = np.random.random()
        set_seed(123)
        b = np.random.random()
        assert a == b


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestPPIDataset:
    def test_sequence_model(self, dummy_parquet: Path):
        ds = PPIDataset(dummy_parquet, "split_random", "train", "siamese_cnn")
        assert len(ds) == 42
        item = ds[0]
        assert len(item) == 3  # seq1, seq2, label
        assert isinstance(item[0], str)
        assert isinstance(item[2], float)

    def test_mlp_features_model(self, dummy_parquet: Path):
        ds = PPIDataset(dummy_parquet, "split_random", "train", "mlp_features")
        item = ds[0]
        assert len(item) == 7  # seq1, seq2, deg1, deg2, loc1, loc2, label

    def test_cold_both_excludes_nan(self, dummy_parquet: Path):
        ds = PPIDataset(dummy_parquet, "split_cold_both", "train", "siamese_cnn")
        assert len(ds) == 36  # NaN entries excluded

    def test_test_fold(self, dummy_parquet: Path):
        ds = PPIDataset(dummy_parquet, "split_random", "test", "siamese_cnn")
        assert len(ds) == 12


# ---------------------------------------------------------------------------
# Collate tests
# ---------------------------------------------------------------------------


class TestCollate:
    def test_sequence_pair_collate(self, dummy_parquet: Path):
        ds = PPIDataset(dummy_parquet, "split_random", "train", "siamese_cnn")
        batch = [ds[i] for i in range(4)]
        device = torch.device("cpu")
        seq1, seq2, labels = _collate_sequence_pair(batch, device)
        assert seq1.shape == (4, 1000)  # MAX_SEQ_LEN
        assert seq2.shape == (4, 1000)
        assert labels.shape == (4,)
        assert labels.dtype == torch.float32

    def test_features_collate(self, dummy_parquet: Path):
        ds = PPIDataset(dummy_parquet, "split_random", "train", "mlp_features")
        batch = [ds[i] for i in range(4)]
        device = torch.device("cpu")
        features, placeholder, labels = _collate_features(batch, device)
        assert features.shape == (4, 67)  # FEATURE_DIM
        assert placeholder is None
        assert labels.shape == (4,)


# ---------------------------------------------------------------------------
# Dataloader tests
# ---------------------------------------------------------------------------


class TestDataloader:
    def test_make_dataloader_sequence(self, dummy_parquet: Path):
        ds = PPIDataset(dummy_parquet, "split_random", "train", "siamese_cnn")
        loader = make_dataloader(ds, batch_size=8, shuffle=False, device=torch.device("cpu"))
        batch = next(iter(loader))
        assert len(batch) == 3

    def test_make_dataloader_features(self, dummy_parquet: Path):
        ds = PPIDataset(dummy_parquet, "split_random", "train", "mlp_features")
        loader = make_dataloader(ds, batch_size=8, shuffle=False, device=torch.device("cpu"))
        batch = next(iter(loader))
        assert len(batch) == 3
        assert batch[0].shape[1] == 67


# ---------------------------------------------------------------------------
# Model factory tests
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_siamese_cnn(self):
        model = build_model("siamese_cnn")
        from negbiodb_ppi.models.siamese_cnn import SiameseCNN
        assert isinstance(model, SiameseCNN)

    def test_pipr(self):
        model = build_model("pipr")
        from negbiodb_ppi.models.pipr import PIPR
        assert isinstance(model, PIPR)

    def test_mlp_features(self):
        model = build_model("mlp_features")
        from negbiodb_ppi.models.mlp_features import MLPFeatures
        assert isinstance(model, MLPFeatures)

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("unknown")


# ---------------------------------------------------------------------------
# Val metric
# ---------------------------------------------------------------------------


class TestComputeValMetric:
    def test_both_classes(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        val = _compute_val_metric(y_true, y_score)
        assert 0 < val <= 1

    def test_single_class_nan(self):
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3])
        val = _compute_val_metric(y_true, y_score)
        assert np.isnan(val)


# ---------------------------------------------------------------------------
# Split/dataset map completeness
# ---------------------------------------------------------------------------


class TestMaps:
    def test_all_split_cols_exist(self):
        expected_splits = {"random", "cold_protein", "cold_both", "ddb"}
        assert set(_SPLIT_COL_MAP.keys()) == expected_splits

    def test_all_dataset_configs(self):
        expected_keys = {
            ("balanced", "negbiodb"),
            ("realistic", "negbiodb"),
            ("balanced", "uniform_random"),
            ("balanced", "degree_matched"),
            ("balanced", "ddb"),
        }
        assert set(_DATASET_MAP.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Integration: mini training run
# ---------------------------------------------------------------------------


class TestMiniTrainingRun:
    """End-to-end test with a tiny model on dummy data."""

    def test_main_siamese_cnn(self, dummy_parquet: Path, tmp_path: Path):
        from train_baseline import main

        # Place parquet where main expects it
        data_dir = tmp_path / "exports" / "ppi"
        data_dir.mkdir(parents=True)
        import shutil
        shutil.copy(dummy_parquet, data_dir / "ppi_m1_balanced.parquet")

        out_dir = tmp_path / "results" / "ppi_baselines"
        ret = main([
            "--model", "siamese_cnn",
            "--split", "random",
            "--negative", "negbiodb",
            "--dataset", "balanced",
            "--epochs", "2",
            "--patience", "5",
            "--batch_size", "16",
            "--lr", "0.01",
            "--seed", "42",
            "--data_dir", str(data_dir),
            "--output_dir", str(out_dir),
        ])
        assert ret == 0

        run_dir = out_dir / "siamese_cnn_balanced_random_negbiodb_seed42"
        assert run_dir.exists()
        assert (run_dir / "results.json").exists()
        assert (run_dir / "training_log.csv").exists()
        assert (run_dir / "best.pt").exists()

        with open(run_dir / "results.json") as f:
            results = json.load(f)
        assert results["model"] == "siamese_cnn"
        assert results["split"] == "random"
        assert "test_metrics" in results
        assert "log_auc" in results["test_metrics"]

    def test_main_mlp_features(self, dummy_parquet: Path, tmp_path: Path):
        from train_baseline import main

        data_dir = tmp_path / "exports" / "ppi"
        data_dir.mkdir(parents=True)
        import shutil
        shutil.copy(dummy_parquet, data_dir / "ppi_m1_balanced.parquet")

        out_dir = tmp_path / "results" / "ppi_baselines"
        ret = main([
            "--model", "mlp_features",
            "--split", "random",
            "--negative", "negbiodb",
            "--dataset", "balanced",
            "--epochs", "2",
            "--patience", "5",
            "--batch_size", "16",
            "--lr", "0.01",
            "--seed", "42",
            "--data_dir", str(data_dir),
            "--output_dir", str(out_dir),
        ])
        assert ret == 0

        run_dir = out_dir / "mlp_features_balanced_random_negbiodb_seed42"
        with open(run_dir / "results.json") as f:
            results = json.load(f)
        assert results["model"] == "mlp_features"

    def test_main_missing_dataset(self, tmp_path: Path):
        from train_baseline import main

        ret = main([
            "--model", "siamese_cnn",
            "--split", "random",
            "--negative", "negbiodb",
            "--data_dir", str(tmp_path / "nonexistent"),
            "--output_dir", str(tmp_path / "out"),
        ])
        assert ret == 1

    def test_main_invalid_combo(self, tmp_path: Path):
        from train_baseline import main

        ret = main([
            "--model", "siamese_cnn",
            "--split", "ddb",
            "--negative", "negbiodb",
            "--dataset", "realistic",
            "--data_dir", str(tmp_path),
            "--output_dir", str(tmp_path / "out"),
        ])
        assert ret == 1
