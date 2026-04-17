#!/bin/bash
#SBATCH --job-name=rl_eval
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rl_eval_%j.log
#SBATCH --error=logs/rl_eval_%j.err

# === Resolve project root ===
cd "${SLURM_SUBMIT_DIR:-.}"

# === Configuration (override via sbatch --export=ALL,...) ===
ADAPTER="${ADAPTER:?Must set ADAPTER path}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
EVAL_DOMAINS="${EVAL_DOMAINS:-dti ct ppi ge vp}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
BASELINE_DIR="${BASELINE_DIR:-results/negbiorl/phase4_eval/baseline}"
BACKEND="${BACKEND:-vllm}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-}"

SCRATCH="/athena/masonlab/scratch/users/jak4013"
CONDA_ENV="${SCRATCH}/conda_env/negbiorl-train"
OUTPUT_DIR="${OUTPUT_DIR:-results/negbiorl/phase4_eval/${SLURM_JOB_ID}}"

# === Setup ===
set -euo pipefail
mkdir -p logs "${OUTPUT_DIR}"

echo "=== NegBioRL Evaluation ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Adapter: ${ADAPTER}"
echo "Domains: ${EVAL_DOMAINS}"

source /home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PYTHONPATH=src:${PYTHONPATH:-}

# === Step 1: Generate predictions ===
echo "Generating predictions..."
PRED_CMD="python scripts_rl/10_generate_predictions.py \
    --adapter ${ADAPTER} \
    --base-model ${BASE_MODEL} \
    --domains ${EVAL_DOMAINS} \
    --backend ${BACKEND} \
    --output-dir ${OUTPUT_DIR}"

if [ "${BACKEND}" = "vllm" ]; then
    PRED_CMD="${PRED_CMD} --gpu-memory-utilization ${GPU_MEM_UTIL}"
fi
if [ -n "${LOAD_IN_4BIT}" ]; then
    PRED_CMD="${PRED_CMD} --load-in-4bit"
fi
eval ${PRED_CMD}

# === Step 2: Evaluate ===
echo "Evaluating..."
python scripts_rl/07_eval_before_after.py \
    --before-dir "${BASELINE_DIR}" \
    --after-dir "${OUTPUT_DIR}" \
    --domains ${EVAL_DOMAINS} \
    --output "${OUTPUT_DIR}/before_after.json"

# === Step 3: PBS ===
echo "Computing PBS..."
python scripts_rl/09_compute_pbs.py \
    --before-dir "${BASELINE_DIR}" \
    --after-dir "${OUTPUT_DIR}" \
    --domains ${EVAL_DOMAINS} \
    --output "${OUTPUT_DIR}/pbs_analysis.json"

echo "=== Done: ${OUTPUT_DIR} ==="
