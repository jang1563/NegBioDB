#!/bin/bash
#SBATCH --job-name=llm_local
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
# Note: g0003 CUDA sometimes hangs; prefer g0001/g0002 but allow all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/llm_local_%j.log
#SBATCH --error=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/llm_local_%j.err

# Run LLM benchmark with local model (vLLM server).
# Usage:
#   sbatch --export=ALL,TASK=l1,MODEL=llama70b,CONFIG=zero-shot,FS=0 slurm/run_llm_local.slurm

set -euo pipefail

# Use scratch conda env copy to bypass NFS /home degradation
SCRATCH="${SCRATCH_DIR:-/path/to/scratch}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"
MODEL_DIR="${SCRATCH}/models"
# Unique port per job to avoid conflicts when multiple jobs share a node
PORT=$((8000 + (SLURM_JOB_ID % 1000)))

# Defaults
TASK="${TASK:-l1}"
MODEL="${MODEL:-llama70b}"
CONFIG="${CONFIG:-zero-shot}"
FS="${FS:-0}"

# Resolve model path
case "${MODEL}" in
    llama70b)
        MODEL_PATH="${MODEL_DIR}/llama-3.3-70b-instruct-awq"
        QUANT_ARG="--quantization awq_marlin"
        MAX_MODEL_LEN=4096
        ;;
    mistral7b)
        MODEL_PATH="${MODEL_DIR}/Mistral-7B-Instruct-v0.3"
        QUANT_ARG=""
        MAX_MODEL_LEN=8192
        ;;
    qwen32b)
        MODEL_PATH="${MODEL_DIR}/Qwen2.5-32B-Instruct-AWQ"
        QUANT_ARG="--quantization awq_marlin"
        MAX_MODEL_LEN=4096
        ;;
    *)
        echo "Unknown model: ${MODEL}"
        exit 1
        ;;
esac

# Init — use scratch env via PATH (no conda activate, no NFS /home)
export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"

echo "=== LLM Local Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FS: ${FS}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi 2>&1 || echo "nvidia-smi failed"
echo "Start: $(date)"

# Start vLLM server in background (redirect stderr to stdout for logging)
VLLM_LOG="${PROJECT_DIR}/logs/vllm_${SLURM_JOB_ID}.log"
echo "Starting vLLM server (log: ${VLLM_LOG})..."
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port ${PORT} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    ${QUANT_ARG} \
    --dtype auto \
    --enforce-eager \
    --trust-remote-code > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

# Wait for server to be ready (up to 40 minutes; NFS latency can slow imports)
echo "Waiting for vLLM server on port ${PORT}..."
SERVER_READY=0
for i in $(seq 1 2400); do
    if curl -s http://127.0.0.1:${PORT}/v1/models > /dev/null 2>&1; then
        echo "vLLM server ready (${i}s)"
        SERVER_READY=1
        break
    fi
    # Check if vLLM process died
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM server process died. Last 50 lines of vLLM log:"
        tail -50 "${VLLM_LOG}" 2>/dev/null || true
        exit 1
    fi
    # Progress indicator every 60s
    if [ $((i % 60)) -eq 0 ]; then
        echo "  Still waiting... (${i}s)"
    fi
    sleep 1
done

if [ "${SERVER_READY}" -eq 0 ]; then
    echo "ERROR: vLLM server not ready after 2400s. Last 50 lines of vLLM log:"
    tail -50 "${VLLM_LOG}" 2>/dev/null || true
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

# Run benchmark
python "${PROJECT_DIR}/scripts/run_llm_benchmark.py" \
    --task "${TASK}" \
    --model "${MODEL_PATH}" \
    --provider vllm \
    --config "${CONFIG}" \
    --fewshot-set "${FS}" \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --data-dir "${PROJECT_DIR}/exports/llm_benchmarks" \
    --output-dir "${PROJECT_DIR}/results/llm"

# Cleanup
kill ${VLLM_PID} 2>/dev/null || true
echo "Done: $(date)"
