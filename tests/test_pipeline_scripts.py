"""Regression tests for experiment orchestration scripts."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import pandas as pd
import pytest

from negbiodb.db import connect, create_database, refresh_all_pairs
from negbiodb.export import _resolve_split_id, export_negative_dataset

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = ROOT / "migrations"


def _load_script_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def migrated_db(tmp_path):
    db_path = tmp_path / "test.db"
    create_database(db_path, MIGRATIONS_DIR)
    return db_path


def _populate_small_db(conn, n_compounds=3, n_targets=2):
    for i in range(1, n_compounds + 1):
        conn.execute(
            """INSERT INTO compounds
            (canonical_smiles, inchikey, inchikey_connectivity)
            VALUES (?, ?, ?)""",
            (f"C{i}", f"KEY{i:011d}SA-N", f"KEY{i:011d}"),
        )
    for j in range(1, n_targets + 1):
        conn.execute(
            "INSERT INTO targets (uniprot_accession, amino_acid_sequence) VALUES (?, ?)",
            (f"P{j:05d}", f"SEQ{j}"),
        )
    for i in range(1, n_compounds + 1):
        for j in range(1, n_targets + 1):
            conn.execute(
                """INSERT INTO negative_results
                (compound_id, target_id, result_type, confidence_tier,
                 activity_type, activity_value, activity_unit,
                 inactivity_threshold, source_db, source_record_id,
                 extraction_method, publication_year)
                VALUES (?, ?, 'hard_negative', 'silver',
                        'IC50', 20000.0, 'nM',
                        10000.0, 'chembl', ?, 'database_direct', ?)""",
                (i, j, f"C:{i}:{j}", 2015 + i),
            )
    refresh_all_pairs(conn)
    conn.commit()


class TestExportVersioning:
    def test_export_uses_latest_split_without_row_duplication(self, migrated_db, tmp_path):
        with connect(migrated_db) as conn:
            _populate_small_db(conn)
            pair_ids = [row[0] for row in conn.execute(
                "SELECT pair_id FROM compound_target_pairs ORDER BY pair_id"
            )]
            conn.execute(
                """INSERT INTO split_definitions
                (split_name, split_strategy, random_seed, train_ratio, val_ratio, test_ratio)
                VALUES ('random_v1', 'random', 1, 0.7, 0.1, 0.2)"""
            )
            old_id = conn.execute(
                "SELECT split_id FROM split_definitions WHERE split_name = 'random_v1'"
            ).fetchone()[0]
            conn.executemany(
                "INSERT INTO split_assignments (pair_id, split_id, fold) VALUES (?, ?, 'train')",
                [(pair_id, old_id) for pair_id in pair_ids],
            )

            conn.execute(
                """INSERT INTO split_definitions
                (split_name, split_strategy, random_seed, train_ratio, val_ratio, test_ratio)
                VALUES ('random_v2', 'random', 2, 0.7, 0.1, 0.2)"""
            )
            new_id = conn.execute(
                "SELECT split_id FROM split_definitions WHERE split_name = 'random_v2'"
            ).fetchone()[0]
            conn.executemany(
                "INSERT INTO split_assignments (pair_id, split_id, fold) VALUES (?, ?, 'val')",
                [(pair_id, new_id) for pair_id in pair_ids],
            )
            conn.commit()

        result = export_negative_dataset(migrated_db, tmp_path / "exports", split_strategies=["random"])
        df = pd.read_parquet(result["parquet_path"])

        assert len(df) == len(pair_ids)
        assert set(df["split_random"]) == {"val"}

    def test_split_resolution_prefers_version_suffix_over_insert_order(self, migrated_db):
        with connect(migrated_db) as conn:
            conn.execute(
                """INSERT INTO split_definitions
                (split_name, split_strategy, version, random_seed, train_ratio, val_ratio, test_ratio)
                VALUES ('random_v2', 'random', '2.0', 2, 0.7, 0.1, 0.2)"""
            )
            newer_semantic_id = conn.execute(
                "SELECT split_id FROM split_definitions WHERE split_name = 'random_v2'"
            ).fetchone()[0]
            conn.execute(
                """INSERT INTO split_definitions
                (split_name, split_strategy, version, random_seed, train_ratio, val_ratio, test_ratio)
                VALUES ('random_backfill', 'random', '1.0', 1, 0.7, 0.1, 0.2)"""
            )
            conn.commit()

            assert _resolve_split_id(conn, "random") == newer_semantic_id


class TestPrepareExpData:
    def test_skip_exp4_does_not_require_pairs_parquet(self, tmp_path):
        module = _load_script_module("prepare_exp_data_test", "scripts/prepare_exp_data.py")

        data_dir = tmp_path / "exports"
        data_dir.mkdir()
        db_path = tmp_path / "negbiodb.db"
        db_path.write_text("")
        pd.DataFrame({"smiles": ["CC"], "inchikey": ["A" * 27], "uniprot_id": ["P00001"], "target_sequence": ["SEQ"], "Y": [1], "split_random": ["train"]}).to_parquet(data_dir / "negbiodb_m1_balanced.parquet")
        pd.DataFrame({"smiles": ["CC"], "inchikey": ["A" * 27], "uniprot_id": ["P00001"], "target_sequence": ["SEQ"]}).to_parquet(data_dir / "chembl_positives_pchembl6.parquet")

        rc = module.main([
            "--data-dir", str(data_dir),
            "--db", str(db_path),
            "--skip-exp1",
            "--skip-exp4",
        ])

        assert rc == 0

    def test_exp4_only_does_not_require_db_or_positives_and_builds_full_task_ddb(self, tmp_path):
        module = _load_script_module("prepare_exp_data_test_exp4_only", "scripts/prepare_exp_data.py")

        data_dir = tmp_path / "exports"
        data_dir.mkdir()
        rows = []
        for i in range(10):
            rows.append({
                "smiles": f"P{i}",
                "inchikey": f"POS{i:011d}ABCDEFGHIJKLM",
                "uniprot_id": "P00001",
                "target_sequence": "SEQ",
                "Y": 1,
                "split_random": "train",
                "split_cold_compound": "train",
                "split_cold_target": "train",
            })
        for i in range(10):
            rows.append({
                "smiles": f"N{i}",
                "inchikey": f"NEG{i:011d}ABCDEFGHIJKLM",
                "uniprot_id": "P00002" if i < 5 else "P00003",
                "target_sequence": "SEQ",
                "Y": 0,
                "split_random": "test",
                "split_cold_compound": "test",
                "split_cold_target": "test",
            })
        pd.DataFrame(rows).to_parquet(data_dir / "negbiodb_m1_balanced.parquet")

        rc = module.main([
            "--data-dir", str(data_dir),
            "--db", str(tmp_path / "missing.db"),
            "--skip-exp1",
        ])

        assert rc == 0
        ddb = pd.read_parquet(data_dir / "negbiodb_m1_balanced_ddb.parquet")
        pos = ddb[ddb["Y"] == 1]
        assert "split_degree_balanced" in ddb.columns
        assert set(pos["split_degree_balanced"]) != {"train"}


class TestTrainBaselineHelpers:
    def test_build_run_name_includes_dataset_and_seed(self):
        module = _load_script_module("train_baseline_test_names", "scripts/train_baseline.py")
        run_name = module._build_run_name("deepdta", "balanced", "random", "negbiodb", 7)
        assert run_name == "deepdta_balanced_random_negbiodb_seed7"

    def test_resolve_dataset_file_rejects_invalid_ddb_combo(self):
        module = _load_script_module("train_baseline_test_resolve", "scripts/train_baseline.py")
        assert module._resolve_dataset_file("balanced", "ddb", "uniform_random") is None
        assert module._resolve_dataset_file("balanced", "ddb", "negbiodb") == "negbiodb_m1_balanced_ddb.parquet"
        assert module._resolve_dataset_file("realistic", "random", "uniform_random") is None
        assert module._resolve_dataset_file("realistic", "random", "degree_matched") is None

    def test_prepare_graph_cache_backfills_missing_smiles(self, tmp_path):
        torch = pytest.importorskip("torch")
        module = _load_script_module("train_baseline_test_cache", "scripts/train_baseline.py")

        parquet_path = tmp_path / "m1.parquet"
        pd.DataFrame({"smiles": ["CC", "CCC"]}).to_parquet(parquet_path)
        cache_path = tmp_path / "graph_cache.pt"
        torch.save({"CC": {"graph": "cached"}}, cache_path)

        fake_graph_module = types.SimpleNamespace(
            smiles_to_graph=lambda smiles: {"graph": smiles}
        )
        original = sys.modules.get("negbiodb.models.graphdta")
        sys.modules["negbiodb.models.graphdta"] = fake_graph_module
        try:
            cache = module._prepare_graph_cache(parquet_path, cache_path)
        finally:
            if original is None:
                del sys.modules["negbiodb.models.graphdta"]
            else:
                sys.modules["negbiodb.models.graphdta"] = original

        assert set(cache) == {"CC", "CCC"}
        saved = torch.load(cache_path, weights_only=False)
        assert set(saved) == {"CC", "CCC"}

    def test_main_rejects_realistic_ddb(self, tmp_path):
        module = _load_script_module("train_baseline_test_ddb", "scripts/train_baseline.py")
        rc = module.main([
            "--model", "deepdta",
            "--split", "ddb",
            "--negative", "negbiodb",
            "--dataset", "realistic",
            "--data_dir", str(tmp_path),
            "--output_dir", str(tmp_path / "results"),
        ])
        assert rc == 1

    def test_main_rejects_realistic_uniform_random(self, tmp_path):
        module = _load_script_module("train_baseline_test_realistic_control", "scripts/train_baseline.py")
        rc = module.main([
            "--model", "deepdta",
            "--split", "random",
            "--negative", "uniform_random",
            "--dataset", "realistic",
            "--data_dir", str(tmp_path),
            "--output_dir", str(tmp_path / "results"),
        ])
        assert rc == 1

    def test_main_writes_results_with_dataset_and_seed_in_run_name(self, tmp_path, monkeypatch):
        module = _load_script_module("train_baseline_test_main", "scripts/train_baseline.py")

        data_dir = tmp_path / "exports"
        output_dir = tmp_path / "results"
        data_dir.mkdir()
        pd.DataFrame({
            "smiles": ["CC", "CCC", "CCCC"],
            "target_sequence": ["SEQ", "SEQ", "SEQ"],
            "Y": [1, 0, 0],
            "split_random": ["train", "val", "test"],
        }).to_parquet(data_dir / "negbiodb_m1_balanced.parquet")

        fake_torch = types.SimpleNamespace(
            device=lambda name: name,
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch

        class DummyDataset:
            def __init__(self, parquet_path, split_col, fold, model, graph_cache):
                self.fold = fold
                self.items = {"train": [1], "val": [1], "test": [1]}[fold]

            def __len__(self):
                return len(self.items)

        class DummyModel:
            def to(self, device):
                return self

            def parameters(self):
                return []

        monkeypatch.setattr(module, "set_seed", lambda seed: None)
        monkeypatch.setattr(module, "DTIDataset", DummyDataset)
        monkeypatch.setattr(module, "make_dataloader", lambda dataset, batch_size, shuffle, device: [dataset.fold])
        monkeypatch.setattr(module, "build_model", lambda model_type: DummyModel())

        def fake_train(model, train_loader, val_loader, epochs, patience, lr, output_dir, device):
            (output_dir / "best.pt").write_text("checkpoint")
            (output_dir / "training_log.csv").write_text("epoch\n1\n")
            return 0.123

        monkeypatch.setattr(module, "train", fake_train)
        monkeypatch.setattr(module, "evaluate", lambda model, test_loader, checkpoint_path, device: {"log_auc": 0.5})

        try:
            rc = module.main([
                "--model", "deepdta",
                "--split", "random",
                "--negative", "negbiodb",
                "--dataset", "balanced",
                "--seed", "9",
                "--data_dir", str(data_dir),
                "--output_dir", str(output_dir),
            ])
        finally:
            if original_torch is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = original_torch

        assert rc == 0
        run_dir = output_dir / "deepdta_balanced_random_negbiodb_seed9"
        results = json.loads((run_dir / "results.json").read_text())
        assert results["run_name"] == "deepdta_balanced_random_negbiodb_seed9"
        assert results["dataset"] == "balanced"
        assert results["seed"] == 9
        assert (run_dir / "best.pt").exists()

    def test_main_writes_strict_json_with_null_for_nan(self, tmp_path, monkeypatch):
        module = _load_script_module("train_baseline_test_nan_json", "scripts/train_baseline.py")

        data_dir = tmp_path / "exports"
        output_dir = tmp_path / "results"
        data_dir.mkdir()
        pd.DataFrame({
            "smiles": ["CC", "CCC", "CCCC"],
            "target_sequence": ["SEQ", "SEQ", "SEQ"],
            "Y": [1, 0, 0],
            "split_random": ["train", "val", "test"],
        }).to_parquet(data_dir / "negbiodb_m1_balanced.parquet")

        fake_torch = types.SimpleNamespace(
            device=lambda name: name,
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch

        class DummyDataset:
            def __init__(self, parquet_path, split_col, fold, model, graph_cache):
                self.fold = fold
                self.items = {"train": [1], "val": [1], "test": [1]}[fold]

            def __len__(self):
                return len(self.items)

        class DummyModel:
            def to(self, device):
                return self

            def parameters(self):
                return []

        monkeypatch.setattr(module, "set_seed", lambda seed: None)
        monkeypatch.setattr(module, "DTIDataset", DummyDataset)
        monkeypatch.setattr(module, "make_dataloader", lambda dataset, batch_size, shuffle, device: [dataset.fold])
        monkeypatch.setattr(module, "build_model", lambda model_type: DummyModel())

        def fake_train(model, train_loader, val_loader, epochs, patience, lr, output_dir, device):
            (output_dir / "best.pt").write_text("checkpoint")
            (output_dir / "training_log.csv").write_text("epoch\n1\n")
            return float("nan")

        monkeypatch.setattr(module, "train", fake_train)
        monkeypatch.setattr(module, "evaluate", lambda model, test_loader, checkpoint_path, device: {"log_auc": float("nan")})

        try:
            rc = module.main([
                "--model", "deepdta",
                "--split", "random",
                "--negative", "negbiodb",
                "--dataset", "balanced",
                "--data_dir", str(data_dir),
                "--output_dir", str(output_dir),
            ])
        finally:
            if original_torch is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = original_torch

        assert rc == 0
        result_text = (output_dir / "deepdta_balanced_random_negbiodb_seed42" / "results.json").read_text()
        assert "\"best_val_log_auc\": null" in result_text
        assert "\"log_auc\": null" in result_text
        assert "NaN" not in result_text


class TestEvalCheckpoint:
    def test_eval_checkpoint_uses_ddb_dataset_and_current_run_name(self, tmp_path, monkeypatch):
        module = _load_script_module("eval_checkpoint_test_main", "scripts/eval_checkpoint.py")

        data_dir = tmp_path / "exports"
        output_dir = tmp_path / "results"
        data_dir.mkdir()
        pd.DataFrame({
            "smiles": ["CC", "CCC", "CCCC"],
            "target_sequence": ["SEQ", "SEQ", "SEQ"],
            "Y": [1, 0, 0],
            "split_degree_balanced": ["train", "val", "test"],
        }).to_parquet(data_dir / "negbiodb_m1_balanced_ddb.parquet")

        run_dir = output_dir / "deepdta_balanced_ddb_negbiodb_seed9"
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_text("checkpoint")
        (run_dir / "training_log.csv").write_text("epoch,val_log_auc\n1,0.7\n")

        fake_torch = types.SimpleNamespace(
            device=lambda name: name,
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch

        seen: dict[str, object] = {}

        class DummyDataset:
            def __init__(self, parquet_path, split_col, fold, model, graph_cache):
                seen["parquet_name"] = parquet_path.name
                seen["split_col"] = split_col
                self.fold = fold
                self.items = {"train": [1], "val": [1], "test": [1]}[fold]

            def __len__(self):
                return len(self.items)

        class DummyModel:
            def to(self, device):
                seen["device"] = device
                return self

        monkeypatch.setattr(module.baseline, "DTIDataset", DummyDataset)
        monkeypatch.setattr(module.baseline, "make_dataloader", lambda dataset, batch_size, shuffle, device: [dataset.fold])
        monkeypatch.setattr(module.baseline, "build_model", lambda model_type: DummyModel())

        def fake_evaluate(model, test_loader, checkpoint_path, device):
            seen["checkpoint"] = checkpoint_path
            return {"log_auc": 0.5}

        monkeypatch.setattr(module.baseline, "evaluate", fake_evaluate)

        try:
            rc = module.main([
                "--model", "deepdta",
                "--split", "ddb",
                "--negative", "negbiodb",
                "--dataset", "balanced",
                "--seed", "9",
                "--data_dir", str(data_dir),
                "--output_dir", str(output_dir),
            ])
        finally:
            if original_torch is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = original_torch

        assert rc == 0
        assert seen["parquet_name"] == "negbiodb_m1_balanced_ddb.parquet"
        assert seen["split_col"] == "split_degree_balanced"
        assert seen["checkpoint"] == run_dir / "best.pt"
        results = json.loads((run_dir / "results.json").read_text())
        assert results["run_name"] == "deepdta_balanced_ddb_negbiodb_seed9"
        assert results["best_val_log_auc"] == pytest.approx(0.7)

    def test_eval_checkpoint_rejects_realistic_ddb(self, tmp_path):
        module = _load_script_module("eval_checkpoint_test_ddb", "scripts/eval_checkpoint.py")
        rc = module.main([
            "--model", "deepdta",
            "--split", "ddb",
            "--negative", "negbiodb",
            "--dataset", "realistic",
            "--data_dir", str(tmp_path),
            "--output_dir", str(tmp_path / "results"),
        ])
        assert rc == 1

    def test_eval_checkpoint_falls_back_to_legacy_run_directory(self, tmp_path, monkeypatch):
        module = _load_script_module("eval_checkpoint_test_legacy", "scripts/eval_checkpoint.py")

        data_dir = tmp_path / "exports"
        output_dir = tmp_path / "results"
        data_dir.mkdir()
        pd.DataFrame({
            "smiles": ["CC", "CCC", "CCCC"],
            "target_sequence": ["SEQ", "SEQ", "SEQ"],
            "Y": [1, 0, 0],
            "split_random": ["train", "val", "test"],
        }).to_parquet(data_dir / "negbiodb_m1_balanced.parquet")

        run_dir = output_dir / "deepdta_random_negbiodb"
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_text("checkpoint")
        (run_dir / "training_log.csv").write_text("epoch,val_log_auc\n1,0.8\n")

        fake_torch = types.SimpleNamespace(
            device=lambda name: name,
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch

        class DummyDataset:
            def __init__(self, parquet_path, split_col, fold, model, graph_cache):
                self.fold = fold
                self.items = {"train": [1], "val": [1], "test": [1]}[fold]

            def __len__(self):
                return len(self.items)

        class DummyModel:
            def to(self, device):
                return self

        monkeypatch.setattr(module.baseline, "DTIDataset", DummyDataset)
        monkeypatch.setattr(module.baseline, "make_dataloader", lambda dataset, batch_size, shuffle, device: [dataset.fold])
        monkeypatch.setattr(module.baseline, "build_model", lambda model_type: DummyModel())
        monkeypatch.setattr(module.baseline, "evaluate", lambda model, test_loader, checkpoint_path, device: {"log_auc": 0.5})

        try:
            rc = module.main([
                "--model", "deepdta",
                "--split", "random",
                "--negative", "negbiodb",
                "--dataset", "balanced",
                "--seed", "42",
                "--data_dir", str(data_dir),
                "--output_dir", str(output_dir),
            ])
        finally:
            if original_torch is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = original_torch

        assert rc == 0
        results = json.loads((run_dir / "results.json").read_text())
        assert results["run_name"] == "deepdta_random_negbiodb"
        assert results["best_val_log_auc"] == pytest.approx(0.8)


class TestCollectResults:
    def test_load_results_prefers_canonical_run_name_for_duplicate_settings(self, tmp_path):
        module = _load_script_module("collect_results_test_dedup", "scripts/collect_results.py")
        results_base = tmp_path / "results" / "baselines"

        legacy_dir = results_base / "deepdta_random_negbiodb"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "results.json").write_text(json.dumps({
            "run_name": "deepdta_random_negbiodb",
            "model": "deepdta",
            "split": "random",
            "negative": "negbiodb",
            "dataset": "balanced",
            "seed": 42,
            "n_train": 1,
            "n_val": 1,
            "n_test": 1,
            "best_val_log_auc": 0.1,
            "test_metrics": {"log_auc": 0.1},
        }))

        canonical_dir = results_base / "deepdta_balanced_random_negbiodb_seed42"
        canonical_dir.mkdir(parents=True)
        (canonical_dir / "results.json").write_text(json.dumps({
            "run_name": "deepdta_balanced_random_negbiodb_seed42",
            "model": "deepdta",
            "split": "random",
            "negative": "negbiodb",
            "dataset": "balanced",
            "seed": 42,
            "n_train": 1,
            "n_val": 1,
            "n_test": 1,
            "best_val_log_auc": 0.9,
            "test_metrics": {"log_auc": 0.9},
        }))

        loaded = module.load_results(results_base)
        assert len(loaded) == 1
        assert loaded.iloc[0]["log_auc"] == pytest.approx(0.9)

    def test_load_results_drops_stale_ddb_runs_by_checkpoint_mtime(self, tmp_path):
        module = _load_script_module("collect_results_test_stale_ddb", "scripts/collect_results.py")
        results_base = tmp_path / "results" / "baselines"
        ddb_reference = tmp_path / "exports" / "negbiodb_m1_balanced_ddb.parquet"
        ddb_reference.parent.mkdir(parents=True)
        ddb_reference.write_text("ddb")

        base_time = time.time() - 1000
        os.utime(ddb_reference, (base_time + 100, base_time + 100))

        stale_dir = results_base / "deepdta_balanced_ddb_negbiodb_seed42"
        stale_dir.mkdir(parents=True)
        stale_json = stale_dir / "results.json"
        stale_json.write_text(json.dumps({
            "run_name": "deepdta_balanced_ddb_negbiodb_seed42",
            "model": "deepdta",
            "split": "ddb",
            "negative": "negbiodb",
            "dataset": "balanced",
            "seed": 42,
            "n_train": 1,
            "n_val": 1,
            "n_test": 1,
            "best_val_log_auc": 0.1,
            "test_metrics": {"log_auc": 0.1},
        }))
        stale_best = stale_dir / "best.pt"
        stale_best.write_text("checkpoint")
        os.utime(stale_best, (base_time, base_time))
        os.utime(stale_json, (base_time + 200, base_time + 200))

        fresh_dir = results_base / "deepdta_balanced_ddb_negbiodb_seed43"
        fresh_dir.mkdir(parents=True)
        fresh_json = fresh_dir / "results.json"
        fresh_json.write_text(json.dumps({
            "run_name": "deepdta_balanced_ddb_negbiodb_seed43",
            "model": "deepdta",
            "split": "ddb",
            "negative": "negbiodb",
            "dataset": "balanced",
            "seed": 43,
            "n_train": 1,
            "n_val": 1,
            "n_test": 1,
            "best_val_log_auc": 0.9,
            "test_metrics": {"log_auc": 0.9},
        }))
        fresh_best = fresh_dir / "best.pt"
        fresh_best.write_text("checkpoint")
        os.utime(fresh_best, (base_time + 200, base_time + 200))

        loaded = module.load_results(results_base, ddb_reference=ddb_reference)
        assert len(loaded) == 1
        assert int(loaded.iloc[0]["seed"]) == 43
        assert loaded.iloc[0]["log_auc"] == pytest.approx(0.9)

    def test_filter_results_by_dataset_and_seed(self):
        module = _load_script_module("collect_results_test_filter", "scripts/collect_results.py")
        df = pd.DataFrame([
            {"model": "deepdta", "dataset": "balanced", "seed": 1, "split": "random", "negative": "negbiodb"},
            {"model": "deepdta", "dataset": "balanced", "seed": 2, "split": "random", "negative": "negbiodb"},
            {"model": "deepdta", "dataset": "realistic", "seed": 1, "split": "random", "negative": "negbiodb"},
        ])

        filtered = module.filter_results(df, dataset="balanced", seeds=[2])
        assert len(filtered) == 1
        assert filtered.iloc[0]["dataset"] == "balanced"
        assert int(filtered.iloc[0]["seed"]) == 2

    def test_filter_results_by_all_axes(self):
        module = _load_script_module("collect_results_test_filter_axes", "scripts/collect_results.py")
        df = pd.DataFrame([
            {"model": "deepdta", "dataset": "balanced", "seed": 1, "split": "random", "negative": "negbiodb"},
            {"model": "graphdta", "dataset": "balanced", "seed": 1, "split": "random", "negative": "negbiodb"},
            {"model": "deepdta", "dataset": "balanced", "seed": 1, "split": "cold_target", "negative": "negbiodb"},
            {"model": "deepdta", "dataset": "balanced", "seed": 1, "split": "random", "negative": "uniform_random"},
        ])

        filtered = module.filter_results(
            df,
            dataset="balanced",
            seeds=[1],
            models=["deepdta"],
            splits=["random"],
            negatives=["negbiodb"],
        )
        assert len(filtered) == 1
        row = filtered.iloc[0]
        assert row["model"] == "deepdta"
        assert row["split"] == "random"
        assert row["negative"] == "negbiodb"

    def test_build_table1_preserves_dataset_and_seed(self):
        module = _load_script_module("collect_results_test_table", "scripts/collect_results.py")
        df = pd.DataFrame([
            {
                "model": "deepdta",
                "split": "random",
                "negative": "negbiodb",
                "dataset": "balanced",
                "seed": 42,
                "n_test": 10,
                "log_auc": 0.5,
                "auprc": 0.4,
                "bedroc": 0.3,
                "ef_1pct": 1.0,
                "ef_5pct": 2.0,
                "mcc": 0.1,
                "auroc": 0.6,
            }
        ])

        table = module.build_table1(df)
        assert list(table.columns[:5]) == ["model", "dataset", "seed", "split", "negative"]

    def test_aggregate_over_seeds_computes_mean_std_and_counts(self):
        module = _load_script_module("collect_results_test_aggregate", "scripts/collect_results.py")
        df = pd.DataFrame([
            {
                "model": "deepdta", "dataset": "balanced", "seed": 1, "split": "random", "negative": "negbiodb",
                "n_test": 10, "log_auc": 0.4, "auprc": 0.3, "bedroc": 0.2, "ef_1pct": 1.0, "ef_5pct": 2.0, "mcc": 0.1, "auroc": 0.6,
            },
            {
                "model": "deepdta", "dataset": "balanced", "seed": 2, "split": "random", "negative": "negbiodb",
                "n_test": 12, "log_auc": 0.6, "auprc": 0.5, "bedroc": 0.4, "ef_1pct": 3.0, "ef_5pct": 4.0, "mcc": 0.3, "auroc": 0.8,
            },
        ])

        agg = module.aggregate_over_seeds(df)
        assert len(agg) == 1
        row = agg.iloc[0]
        assert int(row["n_seeds"]) == 2
        assert row["log_auc_mean"] == pytest.approx(0.5)
        assert row["n_test_mean"] == pytest.approx(11.0)
        assert "log_auc_std" in agg.columns

    def test_format_aggregated_markdown_includes_mean_std(self):
        module = _load_script_module("collect_results_test_agg_md", "scripts/collect_results.py")
        agg = pd.DataFrame([
            {
                "model": "deepdta",
                "dataset": "balanced",
                "split": "random",
                "negative": "negbiodb",
                "n_seeds": 2,
                "log_auc_mean": 0.5,
                "log_auc_std": 0.1,
                "auprc_mean": 0.4,
                "auprc_std": 0.05,
                "bedroc_mean": 0.3,
                "bedroc_std": 0.02,
                "ef_1pct_mean": 1.0,
                "ef_1pct_std": 0.1,
                "ef_5pct_mean": 2.0,
                "ef_5pct_std": 0.2,
                "mcc_mean": 0.1,
                "mcc_std": 0.01,
                "auroc_mean": 0.6,
                "auroc_std": 0.03,
            }
        ])

        md = module.format_aggregated_markdown(agg)
        assert "0.500 +/- 0.100" in md
        assert "| **Seeds** |" in md

    def test_summarize_exp1_groups_by_dataset(self):
        module = _load_script_module("collect_results_test_summary", "scripts/collect_results.py")
        df = pd.DataFrame([
            {"model": "deepdta", "dataset": "balanced", "negative": "negbiodb", "seed": 1, "log_auc": 0.50},
            {"model": "deepdta", "dataset": "balanced", "negative": "uniform_random", "seed": 1, "log_auc": 0.60},
            {"model": "deepdta", "dataset": "balanced", "negative": "degree_matched", "seed": 1, "log_auc": 0.55},
            {"model": "deepdta", "dataset": "realistic", "negative": "negbiodb", "seed": 2, "log_auc": 0.40},
            {"model": "deepdta", "dataset": "realistic", "negative": "uniform_random", "seed": 2, "log_auc": 0.44},
            {"model": "deepdta", "dataset": "realistic", "negative": "degree_matched", "seed": 2, "log_auc": 0.42},
        ])

        summary = module.summarize_exp1(df)
        assert "Dataset=balanced" in summary
        assert "Dataset=realistic" in summary

    def test_summarize_exp1_uses_only_matched_seeds(self):
        module = _load_script_module("collect_results_test_matched_seeds", "scripts/collect_results.py")
        df = pd.DataFrame([
            {"model": "deepdta", "dataset": "balanced", "negative": "negbiodb", "seed": 1, "log_auc": 0.50},
            {"model": "deepdta", "dataset": "balanced", "negative": "uniform_random", "seed": 1, "log_auc": 0.60},
            {"model": "deepdta", "dataset": "balanced", "negative": "degree_matched", "seed": 1, "log_auc": 0.55},
            {"model": "deepdta", "dataset": "balanced", "negative": "negbiodb", "seed": 2, "log_auc": 0.10},
            {"model": "deepdta", "dataset": "balanced", "negative": "uniform_random", "seed": 3, "log_auc": 0.90},
            {"model": "deepdta", "dataset": "balanced", "negative": "degree_matched", "seed": 4, "log_auc": 0.95},
        ])

        summary = module.summarize_exp1(df)
        assert "[n=1]" in summary
        assert "NegBioDB=0.500" in summary

    def test_summarize_exp1_aggregated_uses_only_matched_seeds(self):
        module = _load_script_module("collect_results_test_agg_matched_seeds", "scripts/collect_results.py")
        df = pd.DataFrame([
            {"model": "deepdta", "dataset": "balanced", "negative": "negbiodb", "seed": 1, "log_auc": 0.50},
            {"model": "deepdta", "dataset": "balanced", "negative": "uniform_random", "seed": 1, "log_auc": 0.60},
            {"model": "deepdta", "dataset": "balanced", "negative": "degree_matched", "seed": 1, "log_auc": 0.55},
            {"model": "deepdta", "dataset": "balanced", "negative": "negbiodb", "seed": 2, "log_auc": 0.10},
            {"model": "deepdta", "dataset": "balanced", "negative": "uniform_random", "seed": 3, "log_auc": 0.90},
            {"model": "deepdta", "dataset": "balanced", "negative": "degree_matched", "seed": 4, "log_auc": 0.95},
        ])

        summary = module.summarize_exp1_aggregated(df)
        assert "[n=1]" in summary
        assert "NegBioDB=0.500" in summary
        assert "uniform_random=0.600" in summary
        assert "0.900" not in summary

    def test_main_returns_error_when_filters_remove_all_rows(self, tmp_path):
        module = _load_script_module("collect_results_test_main_filter", "scripts/collect_results.py")
        results_dir = tmp_path / "results" / "baselines" / "run1"
        results_dir.mkdir(parents=True)
        (results_dir / "results.json").write_text(json.dumps({
            "model": "deepdta",
            "split": "random",
            "negative": "negbiodb",
            "dataset": "balanced",
            "seed": 1,
            "n_train": 1,
            "n_val": 1,
            "n_test": 1,
            "best_val_log_auc": 0.1,
            "test_metrics": {"log_auc": 0.2},
        }))

        rc = module.main([
            "--results-dir", str(tmp_path / "results" / "baselines"),
            "--out", str(tmp_path / "out"),
            "--dataset", "realistic",
        ])
        assert rc == 1

    def test_main_writes_aggregated_csv(self, tmp_path):
        module = _load_script_module("collect_results_test_main_agg", "scripts/collect_results.py")
        results_base = tmp_path / "results" / "baselines"
        for seed, log_auc in [(1, 0.2), (2, 0.4)]:
            run_dir = results_base / f"run_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "results.json").write_text(json.dumps({
                "model": "deepdta",
                "split": "random",
                "negative": "negbiodb",
                "dataset": "balanced",
                "seed": seed,
                "n_train": 1,
                "n_val": 1,
                "n_test": 1,
                "best_val_log_auc": 0.1,
                "test_metrics": {
                    "log_auc": log_auc,
                    "auprc": 0.1,
                    "bedroc": 0.1,
                    "ef_1pct": 1.0,
                    "ef_5pct": 1.0,
                    "mcc": 0.1,
                    "auroc": 0.1,
                },
            }))

        out_dir = tmp_path / "out"
        rc = module.main([
            "--results-dir", str(results_base),
            "--out", str(out_dir),
            "--aggregate-seeds",
        ])
        assert rc == 0
        assert (out_dir / "table1_aggregated.csv").exists()
        assert (out_dir / "table1_aggregated.md").exists()
