#!/bin/bash
#SBATCH --job-name=cp_llm_vllm
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/cp_llm_vllm_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/cp_llm_vllm_%j.err

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

TASK="${TASK:-cp-l1}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CONFIG="${CONFIG:-zero-shot}"
FEWSHOT_SET="${FEWSHOT_SET:-0}"
VLLM_PORT="${VLLM_PORT:-8000}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

echo "=== CP LLM vLLM Benchmark ==="
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

cleanup() {
    kill "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for model readiness
for i in $(seq 1 240); do
    MODELS=$(curl -s "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null || true)
    if echo "${MODELS}" | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if d.get('data') else 1)" 2>/dev/null; then
        echo "vLLM model ready after $((i * 5))s"
        break
    fi
    sleep 5
done

python "${PROJECT_DIR}/scripts_cp/run_cp_llm_benchmark.py" \
    --task "${TASK}" \
    --config "${CONFIG}" \
    --fewshot-set "${FEWSHOT_SET}" \
    --model "${MODEL}" \
    --provider vllm \
    --api-base "http://localhost:${VLLM_PORT}/v1" \
    --data-dir "${PROJECT_DIR}/exports/cp_llm" \
    --output-dir "${PROJECT_DIR}/results/cp_llm"

echo "End: $(date)"
