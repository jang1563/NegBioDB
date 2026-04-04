#!/bin/bash
#SBATCH --job-name=vp_gnomad_sites
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=1-22
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_gnomad_sites_%A_%a.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_gnomad_sites_%A_%a.err

# Extract per-chromosome gnomAD AF and copper TSV shards from the sites VCF.
# Usage:
#   sbatch --export=ALL,VCF_DIR=/scratch/gnomad,OUT_DIR=/scratch/negbiodb/vp_gnomad \
#       slurm/run_vp_extract_gnomad_sites.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"
VCF_DIR="${VCF_DIR:-${SCRATCH}/gnomad}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/data/vp/gnomad/hpc_extract}"
CHR="${CHR:-${SLURM_ARRAY_TASK_ID}}"

mkdir -p "${OUT_DIR}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

VCF_PATH="${VCF_DIR}/gnomad.exomes.v4.1.sites.chr${CHR}.vcf.bgz"
FREQ_OUT="${OUT_DIR}/variant_frequencies.chr${CHR}.tsv"
COPPER_OUT="${OUT_DIR}/copper_variants.chr${CHR}.tsv"

echo "=== VP gnomAD Sites Extraction ==="
echo "Chromosome: ${CHR}"
echo "VCF: ${VCF_PATH}"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_vp/extract_gnomad_sites.py" \
    --db-path "${PROJECT_DIR}/data/negbiodb_vp.db" \
    --vcf "${VCF_PATH}" \
    --frequencies-out "${FREQ_OUT}" \
    --copper-out "${COPPER_OUT}"

echo "Done: ${FREQ_OUT}"
echo "Done: ${COPPER_OUT}"
echo "End: $(date)"
