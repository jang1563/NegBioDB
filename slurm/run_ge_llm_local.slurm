#!/bin/bash
#SBATCH --job-name=ge_llm_local
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ge_llm_local_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ge_llm_local_%j.err

# Run GE LLM benchmark with local model (vLLM server).
# Usage:
#   sbatch --export=ALL,TASK=ge-l1,MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG=zero-shot,FEWSHOT_SET=0 slurm/run_ge_llm_local.slurm

set -euo pipefail

SCRATCH="/athena/masonlab/scratch/users/jak4013"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"
MODEL_DIR="${SCRATCH}/models"
PORT=$((8000 + (SLURM_JOB_ID % 1000)))

# Defaults
TASK="${TASK:-ge-l1}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CONFIG="${CONFIG:-zero-shot}"
FEWSHOT_SET="${FEWSHOT_SET:-0}"

# Resolve model path — use local copy if available, otherwise HuggingFace ID
case "${MODEL}" in
    Qwen/Qwen2.5-7B-Instruct)
        MODEL_PATH="${MODEL}"
        MAX_MODEL_LEN=16384
        ;;
    meta-llama/Llama-3.1-8B-Instruct)
        MODEL_PATH="${MODEL}"
        MAX_MODEL_LEN=4096
        ;;
    Qwen/Qwen2.5-32B-Instruct-AWQ|qwen32b)
        MODEL_PATH="${MODEL_DIR}/Qwen2.5-32B-Instruct-AWQ"
        MAX_MODEL_LEN=8192
        ;;
    *)
        # Pass through as HuggingFace ID
        MODEL_PATH="${MODEL}"
        MAX_MODEL_LEN=4096
        ;;
esac

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"
export HF_HOME="${SCRATCH}/hf_cache"
# Load HuggingFace token for gated models (e.g., Llama)
HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"
if [[ -f "${HF_TOKEN_FILE}" ]]; then
    export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "=== GE LLM Local (vLLM) Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FewshotSet: ${FEWSHOT_SET}"
echo "Model path: ${MODEL_PATH}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi 2>&1 || echo "nvidia-smi failed"
echo "Start: $(date)"

# Start vLLM server
VLLM_LOG="${PROJECT_DIR}/logs/vllm_ge_${SLURM_JOB_ID}.log"
echo "Starting vLLM server on port ${PORT} (log: ${VLLM_LOG})..."
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port ${PORT} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --dtype auto \
    --enforce-eager \
    --trust-remote-code > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

# Wait for server ready (up to 40 min for model download + load)
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
python "${PROJECT_DIR}/scripts_depmap/run_ge_llm_benchmark.py" \
    --task "${TASK}" \
    --config "${CONFIG}" \
    --fewshot-set "${FEWSHOT_SET}" \
    --model "${MODEL}" \
    --provider vllm \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --data-dir "${PROJECT_DIR}/exports/ge_llm" \
    --output-dir "${PROJECT_DIR}/results/ge_llm"

# Cleanup
kill ${VLLM_PID} 2>/dev/null || true
echo "End: $(date)"
