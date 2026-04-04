#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vllm_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vllm_%j.err

# Start vLLM OpenAI-compatible API server on Cayuga.
# Usage:
#   sbatch slurm/start_vllm_server.sh                          # default: Llama 3.3 70B AWQ
#   sbatch --export=ALL,MODEL=mistral7b slurm/start_vllm_server.sh  # Mistral 7B
#
# The server listens on port 8000.  Use SSH tunnel from client scripts:
#   ssh -L 8000:$(hostname):8000 cayuga-login1

set -euo pipefail

# ---- Config ----
CONDA_BASE="${CONDA_PREFIX:-/athena/masonlab/scratch/users/jak4013}/miniconda3"
MODEL_DIR="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}/models"
PORT=8000

MODEL="${MODEL:-llama70b}"  # default model

case "${MODEL}" in
    llama70b)
        MODEL_PATH="${MODEL_DIR}/llama-3.3-70b-instruct-awq"
        QUANTIZATION="awq"
        MAX_MODEL_LEN=4096
        GPU_MEM_UTIL=0.90
        ;;
    mistral7b)
        MODEL_PATH="${MODEL_DIR}/Mistral-7B-Instruct-v0.3"
        QUANTIZATION=""
        MAX_MODEL_LEN=8192
        GPU_MEM_UTIL=0.85
        ;;
    *)
        echo "Unknown model: ${MODEL}. Use llama70b or mistral7b."
        exit 1
        ;;
esac

# ---- Init ----
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate negbiodb-llm

echo "=== vLLM Server ==="
echo "Model: ${MODEL} (${MODEL_PATH})"
echo "Port: ${PORT}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

# ---- Launch server ----
QUANT_ARG=""
if [ -n "${QUANTIZATION}" ]; then
    QUANT_ARG="--quantization ${QUANTIZATION}"
fi

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port ${PORT} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    ${QUANT_ARG} \
    --dtype auto \
    --trust-remote-code
