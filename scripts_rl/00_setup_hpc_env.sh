#!/bin/bash
# Setup NegBioRL training environment on Cayuga HPC
# Run once: bash scripts_rl/00_setup_hpc_env.sh
set -euo pipefail

SCRATCH="/athena/masonlab/scratch/users/jak4013"
ENV_DIR="${SCRATCH}/conda_env/negbiorl-train"

echo "=== Creating NegBioRL training environment ==="
echo "Target: ${ENV_DIR}"

# Create env if it doesn't exist
if [ ! -d "${ENV_DIR}" ]; then
    conda create -p "${ENV_DIR}" python=3.11 -y
fi

# Activate
source activate "${ENV_DIR}"

# Core ML/RL deps
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "trl>=1.0.0" "peft>=0.14.0" "transformers>=4.48.0" "accelerate>=1.2.0"
pip install "datasets>=3.2.0" "wandb>=0.19.0"
pip install vllm  # for GRPO generation
pip install pyyaml numpy scipy scikit-learn

# Install negbiorl package in editable mode
cd "${SCRATCH}/negbiodb" || cd /home/jak4013/negbiodb
pip install -e ".[rltrain]" 2>/dev/null || echo "Using PYTHONPATH instead"

echo ""
echo "=== Verification ==="
python -c "import trl; print(f'trl={trl.__version__}')"
python -c "import peft; print(f'peft={peft.__version__}')"
python -c "import transformers; print(f'transformers={transformers.__version__}')"
python -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}')"
python -c "from trl import GRPOTrainer, GRPOConfig; print('GRPOTrainer: OK')"

echo ""
echo "=== Done ==="
echo "Activate with: source activate ${ENV_DIR}"
