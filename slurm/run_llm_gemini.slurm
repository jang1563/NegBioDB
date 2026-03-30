#!/bin/bash
#SBATCH --job-name=llm_gemini
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/llm_gemini_%j.log
#SBATCH --error=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/llm_gemini_%j.err

# Run LLM benchmark with Gemini API (CPU only, long runtime for rate limits).
# Usage:
#   sbatch --export=ALL,TASK=l1,MODEL=gemini-2.5-flash,CONFIG=zero-shot,FS=0 slurm/run_llm_gemini.slurm

set -euo pipefail

# Use scratch conda env copy to bypass NFS /home degradation
SCRATCH="${SCRATCH_DIR:-/path/to/scratch}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

# Defaults
TASK="${TASK:-l1}"
MODEL="${MODEL:-gemini-2.5-flash}"
CONFIG="${CONFIG:-zero-shot}"
FS="${FS:-0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

# Init — use scratch env via PATH (no conda activate, no NFS /home)
export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONUNBUFFERED=1

echo "=== LLM Gemini Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FS: ${FS}, MaxTok: ${MAX_TOKENS}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts/run_llm_benchmark.py" \
    --task "${TASK}" \
    --model "${MODEL}" \
    --provider gemini \
    --config "${CONFIG}" \
    --fewshot-set "${FS}" \
    --max-tokens "${MAX_TOKENS}" \
    --data-dir "${PROJECT_DIR}/exports/llm_benchmarks" \
    --output-dir "${PROJECT_DIR}/results/llm"

echo "Done: $(date)"
