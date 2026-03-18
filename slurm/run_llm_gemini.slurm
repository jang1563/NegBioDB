#!/bin/bash
#SBATCH --job-name=llm_gemini
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/llm_gemini_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/llm_gemini_%j.err

# Run LLM benchmark with Gemini API (CPU only, long runtime for rate limits).
# Usage:
#   sbatch --export=ALL,TASK=l1,MODEL=gemini-2.5-flash,CONFIG=zero-shot,FS=0 slurm/run_llm_gemini.slurm

set -euo pipefail

CONDA_BASE="/home/fs01/jak4013/miniconda3/miniconda3"
SCRATCH="/athena/masonlab/scratch/users/jak4013"
PROJECT_DIR="${SCRATCH}/negbiodb"

# Defaults
TASK="${TASK:-l1}"
MODEL="${MODEL:-gemini-2.5-flash}"
CONFIG="${CONFIG:-zero-shot}"
FS="${FS:-0}"

# Init
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate negbiodb-llm

echo "=== LLM Gemini Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FS: ${FS}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts/run_llm_benchmark.py" \
    --task "${TASK}" \
    --model "${MODEL}" \
    --provider gemini \
    --config "${CONFIG}" \
    --fewshot-set "${FS}" \
    --data-dir "${PROJECT_DIR}/exports/llm_benchmarks" \
    --output-dir "${PROJECT_DIR}/results/llm"

echo "Done: $(date)"
