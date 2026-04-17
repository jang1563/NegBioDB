"""Additional smoke tests for CP baseline training script."""

import json
import importlib

from negbiodb_cp.cp_db import get_connection
from negbiodb_cp.export import generate_all_splits, export_cp_feature_tables, export_cp_m1_dataset
import scripts_cp.export_cp_ml_dataset as export_cp_ml_dataset

from tests.cp_test_utils import create_seeded_cp_database


def test_train_cp_baseline_profile_smoke(tmp_path):
    db_path = create_seeded_cp_database(tmp_path)
    export_dir = tmp_path / "cp_ml"
    result_dir = tmp_path / "results"

    conn = get_connection(db_path)
    try:
        generate_all_splits(conn, seed=42)
        export_cp_m1_dataset(conn, export_dir)
        export_cp_feature_tables(conn, export_dir)
    finally:
        conn.close()

    module = importlib.import_module("scripts_cp.train_cp_baseline")
    rc = module.main(
        [
            "--model", "mlp",
            "--task", "m1",
            "--feature-set", "profile",
            "--split", "batch_holdout",
            "--data-dir", str(export_dir),
            "--output-dir", str(result_dir),
            "--smoke-test",
        ]
    )
    assert rc == 0

    run_dir = result_dir / "mlp_m1_profile_batch_holdout_seed42"
    payload = json.loads((run_dir / "results.json").read_text())
    assert "macro_f1" in payload
    assert payload["n_features"] >= 2


def test_train_cp_baseline_blocks_proxy_exports_without_override(tmp_path):
    db_path = create_seeded_cp_database(tmp_path, annotation_mode="plate_proxy")
    export_dir = tmp_path / "cp_ml_proxy"
    result_dir = tmp_path / "results"

    assert export_cp_ml_dataset.main(
        [
            "--db-path", str(db_path),
            "--output-dir", str(export_dir),
            "--allow-proxy-smoke",
        ]
    ) == 0

    module = importlib.import_module("scripts_cp.train_cp_baseline")
    rc = module.main(
        [
            "--model", "mlp",
            "--task", "m1",
            "--feature-set", "profile",
            "--split", "batch_holdout",
            "--data-dir", str(export_dir),
            "--output-dir", str(result_dir),
            "--smoke-test",
        ]
    )
    assert rc == 1

    rc = module.main(
        [
            "--model", "mlp",
            "--task", "m1",
            "--feature-set", "profile",
            "--split", "batch_holdout",
            "--data-dir", str(export_dir),
            "--output-dir", str(result_dir),
            "--smoke-test",
            "--allow-proxy-smoke",
        ]
    )
    assert rc == 0
