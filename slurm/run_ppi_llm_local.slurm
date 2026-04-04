#!/bin/bash
#SBATCH --job-name=ppi_llm_local
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ppi_llm_local_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ppi_llm_local_%j.err

# Run PPI LLM benchmark with local model (vLLM server).
# Usage:
#   sbatch --export=ALL,TASK=ppi-l1,MODEL=llama70b,CONFIG=zero-shot,FS=0 slurm/run_ppi_llm_local.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"
MODEL_DIR="${SCRATCH}/models"
PORT=$((8000 + (SLURM_JOB_ID % 1000)))

# Defaults
TASK="${TASK:-ppi-l1}"
MODEL="${MODEL:-llama70b}"
CONFIG="${CONFIG:-zero-shot}"
FS="${FS:-0}"

# Resolve model path
case "${MODEL}" in
    llama70b)
        MODEL_PATH="${MODEL_DIR}/llama-3.3-70b-instruct-awq"
        QUANT_ARG=""
        MAX_MODEL_LEN=4096
        ;;
    qwen32b)
        MODEL_PATH="${MODEL_DIR}/Qwen2.5-32B-Instruct-AWQ"
        QUANT_ARG=""
        MAX_MODEL_LEN=8192
        ;;
    *)
        echo "Unknown model: ${MODEL}"
        exit 1
        ;;
esac

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"

echo "=== PPI LLM Local Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FS: ${FS}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi 2>&1 || echo "nvidia-smi failed"
echo "Start: $(date)"

# Start vLLM server
VLLM_LOG="${PROJECT_DIR}/logs/vllm_ppi_${SLURM_JOB_ID}.log"
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

# Wait for server ready
echo "Waiting for vLLM server on port ${PORT}..."
SERVER_READY=0
for i in $(seq 1 2400); do
    if curl -s http://127.0.0.1:${PORT}/v1/models > /dev/null 2>&1; then
        echo "vLLM server ready (${i}s)"
        SERVER_READY=1
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM server process died."
        tail -50 "${VLLM_LOG}" 2>/dev/null || true
        exit 1
    fi
    if [ $((i % 60)) -eq 0 ]; then
        echo "  Still waiting... (${i}s)"
    fi
    sleep 1
done

if [ "${SERVER_READY}" -eq 0 ]; then
    echo "ERROR: vLLM server not ready after 2400s."
    tail -50 "${VLLM_LOG}" 2>/dev/null || true
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

# Run benchmark
python "${PROJECT_DIR}/scripts_ppi/run_ppi_llm_benchmark.py" \
    --task "${TASK}" \
    --model "${MODEL_PATH}" \
    --provider vllm \
    --config "${CONFIG}" \
    --fewshot-set "${FS}" \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --data-dir "${PROJECT_DIR}/exports/ppi_llm" \
    --output-dir "${PROJECT_DIR}/results/ppi_llm"

# Cleanup
kill ${VLLM_PID} 2>/dev/null || true
echo "Done: $(date)"
