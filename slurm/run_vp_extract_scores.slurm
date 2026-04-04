#!/bin/bash
#SBATCH --job-name=vp_scores
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_scores_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_scores_%j.err

# Export VP score targets and extract CADD/REVEL/AlphaMissense matches on HPC.
# Usage:
#   sbatch --export=ALL,SCORES_DIR=/scratch/vp_scores,OUT_DIR=/scratch/negbiodb/vp_scores \
#       slurm/run_vp_extract_scores.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"
SCORES_DIR="${SCORES_DIR:-${SCRATCH}/vp_scores}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/data/vp/scores/hpc_extract}"

mkdir -p "${OUT_DIR}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

TARGETS_TSV="${OUT_DIR}/vp_score_targets.tsv"
MERGED_TSV="${OUT_DIR}/merged_scores.tsv"

echo "=== VP Score Extraction ==="
echo "Scores dir: ${SCORES_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_vp/export_score_targets.py" \
    --db-path "${PROJECT_DIR}/data/negbiodb_vp.db" \
    --output "${TARGETS_TSV}"

python "${PROJECT_DIR}/scripts_vp/extract_scores_hpc.py" \
    --variant-list "${TARGETS_TSV}" \
    --output "${MERGED_TSV}" \
    --cadd "${SCORES_DIR}/whole_genome_SNVs.tsv.gz" \
    --revel "${SCORES_DIR}/revel-v1.3_all_chromosomes.zip" \
    --alphamissense "${SCORES_DIR}/AlphaMissense_hg38.tsv.gz"

echo "Done: ${MERGED_TSV}"
echo "End: $(date)"
