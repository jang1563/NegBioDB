#!/bin/bash
#SBATCH --job-name=md_llm_vllm
#SBATCH --partition=scu-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_llm_vllm_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_llm_vllm_%j.err

# MD LLM benchmark via vLLM (Llama, Qwen open models)
# Usage: sbatch --export=ALL,LEVEL=l1,MODEL=Qwen/Qwen2.5-7B-Instruct slurm/run_md_llm_vllm.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

LEVEL="${LEVEL:-l1}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SHOT="${SHOT:-0shot}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

echo "=== MD LLM vLLM Benchmark ==="
echo "Level: ${LEVEL}, Model: ${MODEL}, Shot: ${SHOT}"
echo "Start: $(date)"

# Start vLLM server in background
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 4096 \
    --dtype auto &
VLLM_PID=$!

# Wait for server to be ready
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM server ready"
        break
    fi
    sleep 5
done

python "${PROJECT_DIR}/scripts_md/08_run_llm_eval.py" \
    --model "${MODEL}" \
    --level "${LEVEL}" \
    --input-dir "${PROJECT_DIR}/exports/md_llm" \
    --output-dir "${PROJECT_DIR}/results/md_llm"

kill ${VLLM_PID} 2>/dev/null || true
echo "End: $(date)"
