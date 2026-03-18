#!/bin/bash
#SBATCH --job-name=llm_openai
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/llm_openai_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/llm_openai_%j.err

# Run LLM benchmark with OpenAI API (CPU only, fast with high rate limits).
# Usage:
#   sbatch --export=ALL,TASK=l1,MODEL=gpt-4o-mini,CONFIG=zero-shot,FS=0 slurm/run_llm_openai.slurm

set -euo pipefail

CONDA_BASE="/home/fs01/jak4013/miniconda3/miniconda3"
SCRATCH="/athena/masonlab/scratch/users/jak4013"
PROJECT_DIR="${SCRATCH}/negbiodb"

# Defaults
TASK="${TASK:-l1}"
MODEL="${MODEL:-gpt-4o-mini}"
CONFIG="${CONFIG:-zero-shot}"
FS="${FS:-0}"

# Init
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate negbiodb-llm

echo "=== LLM OpenAI Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FS: ${FS}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts/run_llm_benchmark.py" \
    --task "${TASK}" \
    --model "${MODEL}" \
    --provider openai \
    --config "${CONFIG}" \
    --fewshot-set "${FS}" \
    --data-dir "${PROJECT_DIR}/exports/llm_benchmarks" \
    --output-dir "${PROJECT_DIR}/results/llm"

echo "Done: $(date)"
