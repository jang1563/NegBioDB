"""Smoke tests for CP domain scripts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts_cp.build_cp_l3_dataset as build_cp_l3_dataset
import scripts_cp.build_cp_l1_dataset as build_cp_l1_dataset
import scripts_cp.build_cp_l4_dataset as build_cp_l4_dataset
import scripts_cp.load_jump_cp as load_jump_cp
import scripts_cp.export_cp_ml_dataset as export_cp_ml_dataset
import scripts_cp.run_cp_llm_benchmark as run_cp_llm_benchmark
import scripts_cp.train_cp_baseline as train_cp_baseline
from tests.cp_test_utils import create_seeded_cp_db


class _FakeLLMClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, user: str, system: str) -> str:
        if "evidence_grounding" in user and "non_speculation" in user:
            return json.dumps(
                {
                    "evidence_grounding": 5,
                    "assay_reasoning": 4,
                    "specificity": 4,
                    "non_speculation": 5,
                }
            )
        if "one letter only" in user.lower():
            return "A"
        if "evidence-grounded explanation" in user.lower():
            return "The perturbation stays close to DMSO and the explanation does not speculate."
        return "tested\nCell Painting evidence cites DMSO distance and batch context."


def test_cp_export_and_training_smoke(tmp_path):
    db_path = create_seeded_cp_db(tmp_path)
    export_dir = tmp_path / "cp_ml"
    results_dir = tmp_path / "cp_results"

    assert export_cp_ml_dataset.main([
        "--db-path", str(db_path),
        "--output-dir", str(export_dir),
        "--seed", "42",
    ]) == 0

    assert train_cp_baseline.main([
        "--model", "mlp",
        "--task", "m1",
        "--feature-set", "profile",
        "--split", "batch_holdout",
        "--data-dir", str(export_dir),
        "--output-dir", str(results_dir),
        "--smoke-test",
    ]) == 0

    assert train_cp_baseline.main([
        "--model", "mlp",
        "--task", "m1",
        "--feature-set", "image",
        "--split", "batch_holdout",
        "--data-dir", str(export_dir),
        "--output-dir", str(results_dir),
        "--smoke-test",
    ]) == 0

    results_files = list(results_dir.glob("*/results.json"))
    assert results_files


def test_load_and_export_accept_db_alias(tmp_path):
    db_path = tmp_path / "cp.db"
    obs_path = tmp_path / "obs.parquet"
    feature_path = tmp_path / "profile.parquet"
    export_dir = tmp_path / "exports"

    obs = pd.DataFrame(
        [
            {
                "batch_name": "B1",
                "plate_name": "P1",
                "cell_line_name": "U2OS",
                "well_id": "A01",
                "dose": 0.0,
                "dose_unit": "uM",
                "timepoint_h": 48.0,
                "control_type": "dmso",
                "dmso_distance": 0.01,
                "replicate_reproducibility": 1.0,
                "viability_ratio": 1.0,
                "qc_pass": 1,
                "compound_name": "DMSO",
                "canonical_smiles": "CS(C)=O",
            },
            {
                "batch_name": "B1",
                "plate_name": "P1",
                "cell_line_name": "U2OS",
                "well_id": "B01",
                "dose": 1.0,
                "dose_unit": "uM",
                "timepoint_h": 48.0,
                "control_type": "perturbation",
                "dmso_distance": 0.02,
                "replicate_reproducibility": 0.8,
                "viability_ratio": 1.0,
                "qc_pass": 1,
                "compound_name": "Aspirin",
                "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            },
        ]
    )
    features = pd.DataFrame(
        [
            {
                "compound_name": "Aspirin",
                "batch_name": "B1",
                "cell_line_name": "U2OS",
                "dose": 1.0,
                "dose_unit": "uM",
                "timepoint_h": 48.0,
                "feature_source": "synthetic",
                "storage_uri": "s3://synthetic/profile",
                "profile_f1": 0.1,
                "profile_f2": 0.2,
            }
        ]
    )
    obs.to_parquet(obs_path, index=False)
    features.to_parquet(feature_path, index=False)

    assert load_jump_cp.main(
        [
            "--db", str(db_path),
            "--observations", str(obs_path),
            "--profile-features", str(feature_path),
        ]
    ) == 0
    assert export_cp_ml_dataset.main(
        [
            "--db", str(db_path),
            "--output-dir", str(export_dir),
        ]
    ) == 0
    assert (export_dir / "cp_ml_metadata.json").exists()


def test_cp_l1_and_l4_builders_and_benchmark_smoke(tmp_path, monkeypatch):
    db_path = create_seeded_cp_db(tmp_path)
    llm_dir = tmp_path / "cp_llm"
    results_dir = tmp_path / "cp_llm_results"

    assert build_cp_l1_dataset.main([
        "--db-path", str(db_path),
        "--output-dir", str(llm_dir),
        "--n-per-class", "2",
        "--fewshot-per-class", "1",
        "--val-per-class", "0",
    ]) == 0
    assert build_cp_l4_dataset.main([
        "--db-path", str(db_path),
        "--output-dir", str(llm_dir),
        "--n-per-class", "2",
        "--fewshot-per-class", "1",
        "--val-per-class", "0",
    ]) == 0

    monkeypatch.setattr(run_cp_llm_benchmark, "LLMClient", _FakeLLMClient)
    assert run_cp_llm_benchmark.main([
        "--task", "cp-l1",
        "--model", "fake-model",
        "--provider", "openai",
        "--data-dir", str(llm_dir),
        "--output-dir", str(results_dir),
    ]) == 0

    result_path = next(results_dir.glob("*/results.json"))
    metrics = json.loads(result_path.read_text())
    assert "accuracy" in metrics


def test_cp_l3_benchmark_writes_judge_scores(tmp_path, monkeypatch):
    db_path = create_seeded_cp_db(tmp_path)
    llm_dir = tmp_path / "cp_llm"
    results_dir = tmp_path / "cp_llm_results"

    assert build_cp_l3_dataset.main(
        [
            "--db-path", str(db_path),
            "--output-dir", str(llm_dir),
            "--n-records", "4",
            "--fewshot-size", "1",
            "--val-size", "0",
        ]
    ) == 0

    monkeypatch.setattr(run_cp_llm_benchmark, "LLMClient", _FakeLLMClient)
    assert run_cp_llm_benchmark.main(
        [
            "--task", "cp-l3",
            "--model", "fake-model",
            "--provider", "openai",
            "--data-dir", str(llm_dir),
            "--output-dir", str(results_dir),
        ]
    ) == 0

    run_dir = next(results_dir.glob("cp-l3_*"))
    judge_scores = [json.loads(line) for line in (run_dir / "judge_scores.jsonl").read_text().splitlines()]
    assert judge_scores
    assert judge_scores[0]["scores"]["evidence_grounding"] == 5
    metrics = json.loads((run_dir / "results.json").read_text())
    assert metrics["parse_rate"] > 0


def test_cp_l4_can_synthesize_untested_rows_for_single_dose_plate():
    tested = pd.DataFrame(
        [
            {
                "cp_result_id": 1,
                "compound_id": 11,
                "compound_name": "CmpdA",
                "inchikey": "AAAA",
                "cell_line_id": 1,
                "cell_line_name": "U2OS",
                "dose": 10.0,
                "dose_unit": "uM",
                "timepoint_h": 48.0,
                "confidence_tier": "silver",
                "batch_name": "B1",
            },
            {
                "cp_result_id": 2,
                "compound_id": 12,
                "compound_name": "CmpdB",
                "inchikey": "BBBB",
                "cell_line_id": 1,
                "cell_line_name": "U2OS",
                "dose": 10.0,
                "dose_unit": "uM",
                "timepoint_h": 48.0,
                "confidence_tier": "silver",
                "batch_name": "B1",
            },
        ]
    )

    untested = build_cp_l4_dataset._generate_untested_rows(
        tested,
        target_n=4,
        rng=np.random.RandomState(42),
    )

    assert not untested.empty
    assert set(untested["gold_answer"]) == {"untested"}
    assert any(float(v) != 10.0 for v in untested["dose"].dropna())


def test_proxy_smoke_shell_uses_supported_flags():
    shell_path = Path("scripts_cp/run_jump_cp_hpc_smoke.sh")
    text = shell_path.read_text()
    assert "--db-path" in text
    assert "--annotation-mode plate_proxy" in text
    assert "--allow-proxy-smoke" in text


def test_production_smoke_shell_uses_annotated_contract():
    shell_path = Path("scripts_cp/run_jump_cp_production_smoke.sh")
    text = shell_path.read_text()
    assert "--annotation-mode annotated" in text
    assert "--default-compound-dose" in text
    assert "build_cp_l1_dataset.py" in text
    assert "build_cp_l4_dataset.py" in text
