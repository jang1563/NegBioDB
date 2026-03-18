#!/bin/bash
# Setup negbiodb-llm conda environment on Cayuga HPC
# Run on login node: bash slurm/setup_llm_env.sh
#
# This creates a separate env from negbiodb-ml (torch 2.2.2) to avoid conflicts.
# Installs vLLM for local model serving + google-genai for Gemini API.

set -euo pipefail

# ---- Config ----
ENV_NAME="negbiodb-llm"
CONDA_BASE="/home/fs01/jak4013/miniconda3/miniconda3"
SCRATCH="/athena/masonlab/scratch/users/jak4013"
MODEL_DIR="${SCRATCH}/models"

# ---- Init conda ----
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- Create env ----
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment ${ENV_NAME} already exists. To recreate:"
    echo "  conda env remove -n ${ENV_NAME}"
    echo "  bash slurm/setup_llm_env.sh"
    exit 0
fi

echo "=== Creating conda environment: ${ENV_NAME} ==="
conda create -n "${ENV_NAME}" python=3.11 -y

echo "=== Activating ${ENV_NAME} ==="
conda activate "${ENV_NAME}"

echo "=== Installing packages ==="
# Core: vLLM (includes torch + CUDA)
pip install vllm

# Gemini SDK
pip install google-genai

# Data + ML utilities
pip install pandas pyarrow scikit-learn tqdm pyyaml

# Install negbiodb package (for metrics, db access)
pip install -e "${SCRATCH}/negbiodb"

echo "=== Verifying installation ==="
python -c "import vllm; print(f'vLLM {vllm.__version__} OK')" || echo "vLLM import failed (may need GPU node)"
python -c "from google import genai; print('Gemini SDK OK')"
python -c "import pandas; print('pandas OK')"
python -c "from negbiodb import llm_eval; print('negbiodb OK')"

echo "=== Creating model directory ==="
mkdir -p "${MODEL_DIR}"

echo ""
echo "=== Setup complete ==="
echo "Environment: ${ENV_NAME}"
echo "Model dir:   ${MODEL_DIR}"
echo ""
echo "Next steps:"
echo "  1. Download models:"
echo "     conda activate ${ENV_NAME}"
echo "     huggingface-cli download TheBloke/Llama-3.3-70B-Instruct-AWQ --local-dir ${MODEL_DIR}/Llama-3.3-70B-Instruct-AWQ"
echo "     huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ${MODEL_DIR}/Mistral-7B-Instruct-v0.3"
echo "  2. Test vLLM server:"
echo "     sbatch slurm/start_vllm_server.sh"
