#!/bin/bash
#SBATCH --job-name=nes_embed
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/nes_embed_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/nes_embed_%j.err

# NES Step 1: Extract Gemma-3-27B embeddings for NES analysis.
# Runtime ~45 min on A100 40GB (20 records × 4 domains, 4-bit quant).
# After it finishes: rsync results/spinout/idea_nes/domain_embeddings.npz back locally
# and run scripts_rl/11_nes_embedding_analysis.py.

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
# negbiorl-train has bitsandbytes + transformers for 4-bit Gemma-3-27B loading
SCRATCH_ENV="${SCRATCH}/conda_env/negbiorl-train"
PROJECT_DIR="${SCRATCH}/negbiodb"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"

cd "$PROJECT_DIR"

MODEL="${MODEL:-unsloth/gemma-3-27b-it-bnb-4bit}"
N_RECORDS="${N_RECORDS:-20}"
MID_LAYER="${MID_LAYER:-31}"
OUT="${OUT:-results/spinout/idea_nes/domain_embeddings.npz}"

echo "=== NES Embedding Extraction ==="
echo "Model:      $MODEL"
echo "N_RECORDS:  $N_RECORDS"
echo "MID_LAYER:  $MID_LAYER"
echo "OUT:        $OUT"
date
which python
python --version
nvidia-smi || true

mkdir -p logs results/spinout/idea_nes

PYTHONPATH=src python scripts_rl/11a_nes_extract_embeddings.py \
    --model "$MODEL" \
    --n-records "$N_RECORDS" \
    --mid-layer "$MID_LAYER" \
    --out "$OUT"

echo "Exit: $?"
date
ls -lh "$OUT" || true
echo "=== DONE ==="
