#!/bin/bash
#SBATCH --job-name=ppi_llm_openai
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ppi_llm_openai_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ppi_llm_openai_%j.err

# Run PPI LLM benchmark with OpenAI API (CPU only).
# Usage:
#   sbatch --export=ALL,TASK=ppi-l1,MODEL=gpt-4o-mini,CONFIG=zero-shot,FS=0 slurm/run_ppi_llm_openai.slurm

set -euo pipefail

SCRATCH="/athena/masonlab/scratch/users/jak4013"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

# Defaults
TASK="${TASK:-ppi-l1}"
MODEL="${MODEL:-gpt-4o-mini}"
CONFIG="${CONFIG:-zero-shot}"
FS="${FS:-0}"
EXTRA="${EXTRA:-}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"

echo "=== PPI LLM OpenAI Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FS: ${FS}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_ppi/run_ppi_llm_benchmark.py" \
    --task "${TASK}" \
    --model "${MODEL}" \
    --provider openai \
    --config "${CONFIG}" \
    --fewshot-set "${FS}" \
    --data-dir "${PROJECT_DIR}/exports/ppi_llm" \
    --output-dir "${PROJECT_DIR}/results/ppi_llm" \
    ${EXTRA}

echo "Done: $(date)"
