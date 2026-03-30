#!/bin/bash
# Set up negbiodb-ml conda environment on Cayuga HPC.
# Run once from a Cayuga login or interactive node.
#
# Usage:
#   bash slurm/setup_env.sh
#
# What it installs:
#   - Python 3.11 + PyTorch 2.2.2 (CUDA 12.1)
#   - torch-geometric (with scatter/sparse wheels for cu121)
#   - rdkit, pandas, pyarrow, scikit-learn, tqdm
#   - negbiodb package (editable install)

set -euo pipefail

NEGBIODB=${SCRATCH_DIR:-/path/to/scratch}/negbiodb
CONDA_SH=${CONDA_PREFIX:-/path/to/conda}/miniconda3/etc/profile.d/conda.sh
ENV_NAME=negbiodb-ml

source "$CONDA_SH"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '$ENV_NAME' already exists. Activating..."
else
    echo "Creating conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"

echo "Installing PyTorch 2.2.2 (CUDA 12.1)..."
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

echo "Installing torch-geometric..."
pip install torch-geometric

echo "Installing torch-scatter and torch-sparse (CUDA 12.1 wheels)..."
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

echo "Installing core dependencies..."
pip install rdkit pandas pyarrow scikit-learn tqdm pyyaml requests mlcroissant

echo "Installing negbiodb package (editable)..."
pip install -e "$NEGBIODB/"

echo "Verifying installation..."
python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
import torch_geometric
print(f'torch_geometric: {torch_geometric.__version__}')
from negbiodb.models.graphdta import GraphDTA, smiles_to_graph
g = smiles_to_graph('CCO')
print(f'smiles_to_graph test: {g}')
print('Setup complete!')
"

echo ""
echo "Environment '$ENV_NAME' ready."
echo "Activate with: conda activate $ENV_NAME"
