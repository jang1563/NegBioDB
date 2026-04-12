#!/bin/bash
#SBATCH --job-name=md_train
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_train_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_train_%j.err

# MD ML baseline training (XGBoost + MLP on CPU)
# Usage: sbatch --export=ALL,TASK=m1,SPLIT=random,SEED=42 slurm/run_md_train_cpu.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-ml"
HOME_ENV="/home/fs01/jak4013/miniconda3/miniconda3/envs/negbiodb-ml"
PROJECT_DIR="${SCRATCH}/negbiodb"

TASK="${TASK:-m1}"
SPLIT="${SPLIT:-random}"
SEED="${SEED:-42}"
MODEL_TYPE="${MODEL_TYPE:-xgboost}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x "${SCRATCH_ENV}/bin/python" ]]; then
        PYTHON_BIN="${SCRATCH_ENV}/bin/python"
    elif [[ -x "${HOME_ENV}/bin/python" ]]; then
        PYTHON_BIN="${HOME_ENV}/bin/python"
    else
        echo "ERROR: Could not find negbiodb-ml python interpreter."
        echo "Checked ${SCRATCH_ENV}/bin/python and ${HOME_ENV}/bin/python"
        exit 1
    fi
fi

ENV_ROOT="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"

export PATH="${ENV_ROOT}/bin:${PATH}"
export CONDA_PREFIX="${ENV_ROOT}"
export PYTHONPATH="${PROJECT_DIR}/src"

echo "=== MD ML Baseline Training ==="
echo "Task: ${TASK}, Split: ${SPLIT}, Seed: ${SEED}, Model: ${MODEL_TYPE}"
echo "Start: $(date)"
echo "Python: ${PYTHON_BIN}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" - <<'PYEOF'
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

task = "${TASK}"
split = "${SPLIT}"
seed = int("${SEED}")
model_type = "${MODEL_TYPE}"
project_dir = Path("${PROJECT_DIR}")

sys.path.insert(0, str(project_dir / "src"))

# Load exported parquet
parquet_path = project_dir / "exports" / "md_ml" / "md_ml.parquet"
if not parquet_path.exists():
    print(f"Parquet not found: {parquet_path}")
    sys.exit(1)

df = pd.read_parquet(parquet_path)
split_col = f"split_{split}"

if split_col not in df.columns:
    print(f"Split column {split_col} not found in parquet")
    sys.exit(1)

feat_cols = [c for c in df.columns if c.startswith("feat_")]
label_col = "is_significant" if task == "m1" else "label_m2"

X_train = df[df[split_col] == "train"][feat_cols].values.astype(np.float32)
y_train = df[df[split_col] == "train"][label_col].values
X_val   = df[df[split_col] == "val"][feat_cols].values.astype(np.float32)
y_val   = df[df[split_col] == "val"][label_col].values
X_test  = df[df[split_col] == "test"][feat_cols].values.astype(np.float32)
y_test  = df[df[split_col] == "test"][label_col].values

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

if model_type == "xgboost":
    from negbiodb_md.models.xgboost_md import train_xgboost_md, evaluate_md
    task_type = "binary" if task == "m1" else "multiclass"
    model = train_xgboost_md(X_train, y_train, X_val, y_val, task=task_type, seed=seed)
    metrics = evaluate_md(model, X_test, y_test, task=task_type)
else:
    from negbiodb_md.models.mlp_md import train_mlp_md, evaluate_mlp_md
    task_type = "binary" if task == "m1" else "multiclass"
    model, scaler = train_mlp_md(X_train, y_train, X_val if len(X_val) > 0 else None,
                                  y_val if len(y_val) > 0 else None,
                                  task=task_type, seed=seed)
    metrics = evaluate_mlp_md(model, scaler, X_test, y_test, task=task_type)

print(f"\nMD-{task.upper()} {split} {model_type} (seed={seed}): {json.dumps(metrics, indent=2)}")

# Save results
out_dir = project_dir / "results" / "md_baselines"
out_dir.mkdir(parents=True, exist_ok=True)
result = {"task": task, "split": split, "model": model_type, "seed": seed, "metrics": metrics}
out_path = out_dir / f"md_{task}_{split}_{model_type}_seed{seed}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Results saved: {out_path}")
PYEOF

echo "End: $(date)"
