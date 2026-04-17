#!/bin/bash
# Submit full MD domain pipeline on Cayuga HPC with job dependencies.
# Run from the project root on Cayuga login node:
#   cd /athena/masonlab/scratch/users/jak4013/negbiodb
#   bash scripts_md/submit_md_pipeline.sh
#
# Jobs submitted:
#   1. md-ingest       (~12h) — HMDB + MetaboLights + NMDR + aggregate
#   2. md-build-llm    (~30m, after job 1) — build L1-L4 datasets
#   3. md-llm-gemini   (~1h, after job 2) — L4 eval: gemini-2.5-flash
#   4. md-llm-gpt      (~1h, after job 2) — L4 eval: gpt-4o-mini
#   5. md-llm-haiku    (~1h, after job 2) — L4 eval: claude-haiku-4-5
#
# After all jobs complete, rsync results back to local:
#   rsync -av cayuga-login1:/athena/.../negbiodb/results/md_llm/ results/md_llm/
#   rsync -av cayuga-login1:/athena/.../negbiodb/exports/md_llm/ exports/md_llm/

set -euo pipefail

SBATCH=/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch
PROJECT_DIR=/athena/masonlab/scratch/users/jak4013/negbiodb

echo "=== Submitting MD domain pipeline ==="
echo "Project dir: ${PROJECT_DIR}"
echo ""

# Check project dir exists
if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "ERROR: project dir not found: ${PROJECT_DIR}"
    exit 1
fi

cd "${PROJECT_DIR}"

# Create log dir if needed
mkdir -p /athena/masonlab/scratch/users/jak4013/negbiodb/logs

# Job 1: Full ingest (~12h)
JOB1=$($SBATCH --parsable slurm/run_md_ingest.slurm)
echo "Submitted job 1 (md-ingest):     ${JOB1}"

# Job 2: Build LLM datasets (~30m, after job 1)
JOB2=$($SBATCH --parsable \
    --dependency=afterok:${JOB1} \
    --job-name=md-build-llm \
    --partition=scu-cpu \
    --cpus-per-task=2 \
    --mem=8G \
    --time=1:00:00 \
    --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_build_llm_%j.log \
    --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_build_llm_%j.err \
    --wrap="cd ${PROJECT_DIR} && source ~/.api_keys && PYTHONPATH=${PROJECT_DIR}/src \
        \${SCRATCH:-/athena/masonlab/scratch/users/jak4013}/conda_env/negbiodb-llm/bin/python \
        scripts_md/07_build_llm_datasets.py && echo 'Build complete'")
echo "Submitted job 2 (md-build-llm):  ${JOB2} (depends on ${JOB1})"

# Jobs 3-5: L4 eval for each API model (all after job 2)
for MODEL_KEY in "gemini-2.5-flash" "gpt-4o-mini" "claude-haiku-4-5"; do
    JOB_N=$($SBATCH --parsable \
        --dependency=afterok:${JOB2} \
        --export=ALL,LEVEL=l4,MODEL=${MODEL_KEY},SHOT=0shot \
        slurm/run_md_llm_api.slurm)
    echo "Submitted L4 eval (${MODEL_KEY}): ${JOB_N} (depends on ${JOB2})"
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor with: squeue -u jak4013"
echo ""
echo "After completion, sync results:"
echo "  rsync -av cayuga-login1:${PROJECT_DIR}/results/md_llm/ results/md_llm/"
echo "  rsync -av cayuga-login1:${PROJECT_DIR}/exports/md_llm/ exports/md_llm/"
