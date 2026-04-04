#!/bin/bash
#SBATCH --job-name=vp_llm_local
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_llm_local_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_llm_local_%j.err

# Run VP LLM benchmark with local model (vLLM server).
# Usage:
#   sbatch --export=ALL,TASK=vp-l1,MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG=zero-shot,FEWSHOT_SET=0 slurm/run_vp_llm_vllm.slurm
#   sbatch --export=ALL,TASK=vp-l1,MODEL=meta-llama/Llama-3.1-8B-Instruct,CONFIG=3-shot,FEWSHOT_SET=1 slurm/run_vp_llm_vllm.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"
MODEL_DIR="${SCRATCH}/models"
PORT=$((8000 + (SLURM_JOB_ID % 1000)))

# Defaults
TASK="${TASK:-vp-l1}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CONFIG="${CONFIG:-zero-shot}"
FEWSHOT_SET="${FEWSHOT_SET:-0}"

# Resolve model path
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
        MODEL_PATH="${MODEL}"
        MAX_MODEL_LEN=4096
        ;;
esac

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"
export HF_HOME="${SCRATCH}/hf_cache"
_NVIDIA_LIBS="$(python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))' 2>/dev/null || echo "")"
if [[ -n "${_NVIDIA_LIBS}" ]]; then
    for _d in "${_NVIDIA_LIBS}"/*/lib; do
        [ -d "${_d}" ] && export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
    done
    echo "Added nvidia libs from ${_NVIDIA_LIBS}"
fi
# Load HuggingFace token for gated models (e.g., Llama).
HF_TOKEN=""
for _tok in "${HOME}/.cache/huggingface/token" \
            "${HF_HOME}/token" \
            "${SCRATCH}/.hf_token"; do
    if [[ -f "${_tok}" ]]; then
        HF_TOKEN="$(cat "${_tok}")"
        echo "Loaded HF token from ${_tok}"
        break
    fi
done
if [[ -n "${HF_TOKEN}" ]]; then
    export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
else
    echo "WARNING: No HF token found — gated models (Llama) will fail 401."
fi

echo "=== VP LLM Local (vLLM) Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FewshotSet: ${FEWSHOT_SET}"
echo "Model path: ${MODEL_PATH}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi 2>&1 || echo "nvidia-smi failed"
echo "Start: $(date)"

# Start vLLM server
VLLM_LOG="${PROJECT_DIR}/logs/vllm_vp_${SLURM_JOB_ID}.log"
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
python "${PROJECT_DIR}/scripts_vp/run_vp_llm_benchmark.py" \
    --task "${TASK}" \
    --config "${CONFIG}" \
    --fewshot-set "${FEWSHOT_SET}" \
    --model "${MODEL}" \
    --provider vllm \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --data-dir "${PROJECT_DIR}/exports/vp_llm" \
    --output-dir "${PROJECT_DIR}/results/vp_llm"

# Cleanup
kill ${VLLM_PID} 2>/dev/null || true
echo "End: $(date)"
