#!/bin/bash
#SBATCH --job-name=cp-ingest
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/cp_ingest_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/cp_ingest_%j.err

# Production ingest of JUMP Cell Painting data
# Usage: sbatch --export=ALL,SOURCE=source_4,BATCH=2021_04_26_Batch1,MAX_PLATES=5 slurm/run_cp_ingest.slurm

set -euo pipefail

SCRATCH="/athena/masonlab/scratch/users/jak4013"
PROJECT_DIR="${SCRATCH}/negbiodb"

export SOURCE="${SOURCE:-source_4}"
export BATCH="${BATCH:-2021_04_26_Batch1}"
export MAX_PLATES="${MAX_PLATES:-50}"
export PYTHON="/home/fs01/jak4013/miniconda3/miniconda3/envs/negbiodb-ml/bin/python"
export SCRATCH="${SCRATCH}"

echo "=== CP Production Ingest ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Source: ${SOURCE}, Batch: ${BATCH}, Max plates: ${MAX_PLATES}"
echo "Start: $(date)"

cd "${PROJECT_DIR}"
bash scripts_cp/ingest_jump_production.sh

echo "=== Done: $(date) ==="
