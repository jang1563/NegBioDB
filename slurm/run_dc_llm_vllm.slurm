#!/bin/bash
#SBATCH --job-name=dc_llm_vllm
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/dc_llm_vllm_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/dc_llm_vllm_%j.err

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

TASK="${TASK:-dc-l1}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CONFIG="${CONFIG:-zero-shot}"
FEWSHOT_SET="${FEWSHOT_SET:-0}"
VLLM_PORT="${VLLM_PORT:-8000}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

echo "=== DC LLM vLLM Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FewshotSet: ${FEWSHOT_SET}"
echo "Start: $(date)"

# Launch vLLM server in background
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${VLLM_PORT}" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype auto &

VLLM_PID=$!
echo "vLLM server PID: ${VLLM_PID}"

# Ensure vLLM process is killed on exit (including errors)
cleanup() {
    kill "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to be ready
for i in $(seq 1 120); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    sleep 1
done

python "${PROJECT_DIR}/scripts_dc/run_dc_llm_benchmark.py" \
    --task "${TASK}" \
    --config "${CONFIG}" \
    --fewshot-set "${FEWSHOT_SET}" \
    --model "${MODEL}" \
    --provider vllm \
    --api-base "http://localhost:${VLLM_PORT}/v1" \
    --data-dir "${PROJECT_DIR}/exports/dc_llm" \
    --output-dir "${PROJECT_DIR}/results/dc_llm"

echo "End: $(date)"
