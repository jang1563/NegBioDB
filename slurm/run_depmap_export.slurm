#!/bin/bash
#SBATCH --job-name=depmap_export
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/depmap_export_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/depmap_export_%j.err

# Export per-gene DepMap Chronos aggregates for 2B analysis.
# After it finishes: rsync exports/ge_gene_aggregates.parquet back locally
# and run scripts/spinout/idea_2b_genetic_buffering.py

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"

cd "$PROJECT_DIR"

DB="${DB:-${PROJECT_DIR}/data/negbiodb_depmap.db}"
OUT="${OUT:-exports/ge_gene_aggregates.parquet}"

echo "=== DepMap Chronos Aggregate Export ==="
echo "DB:  $DB"
echo "OUT: $OUT"
date
which python
python --version

mkdir -p logs exports

PYTHONPATH=src python -u scripts/export_depmap_gene_aggregates.py \
    --db "$DB" \
    --out "$OUT"

echo "Exit: $?"
date
ls -lh "$OUT"
echo "=== DONE ==="
