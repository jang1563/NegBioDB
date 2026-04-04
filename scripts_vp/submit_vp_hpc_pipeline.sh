#!/bin/bash
# VP HPC Pipeline Submission Script
# Runs on Cayuga login node. Orchestrates the full VP data pipeline.
#
# Usage:
#   bash scripts_vp/submit_vp_hpc_pipeline.sh [stage]
#
# Stages:
#   download   — Download gnomAD sites VCF + CADD/REVEL/AlphaMissense
#   extract    — Submit gnomAD extraction + score extraction jobs
#   merge      — Merge per-chromosome shards and load into DB
#   export     — Export ML dataset + build LLM datasets
#   train      — Submit ML training jobs (XGBoost/MLP/ESM2/GNN)
#   llm        — Submit LLM benchmark jobs (5 models × 4 levels)
#   all        — Run all stages in order (interactive)

set -euo pipefail

SCRATCH="/athena/masonlab/scratch/users/jak4013"
PROJECT_DIR="${SCRATCH}/negbiodb"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
VCF_DIR="${SCRATCH}/gnomad_vp"
SCORES_DIR="${SCRATCH}/vp_scores"
GNOMAD_OUT="${PROJECT_DIR}/data/vp/gnomad/hpc_extract"
SCORES_OUT="${PROJECT_DIR}/data/vp/scores/hpc_extract"
SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

STAGE="${1:-help}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

case "${STAGE}" in
  download)
    log "=== Stage: Download Data ==="

    # gnomAD sites VCF (exomes, ~15-20 GB)
    mkdir -p "${VCF_DIR}"
    log "Downloading gnomAD v4.1 exome sites VCF..."
    gsutil -m cp \
      "gs://gcp-public-data--gnomad/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr{1..22}.vcf.bgz" \
      "${VCF_DIR}/"

    # CADD (~80 GB)
    mkdir -p "${SCORES_DIR}"
    log "Downloading CADD v1.7 prescored SNVs..."
    wget -c -P "${SCORES_DIR}" \
      "https://kircherlab.bihealth.org/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz"

    # REVEL (~900 MB)
    log "Downloading REVEL v1.3..."
    wget -c -P "${SCORES_DIR}" \
      "https://zenodo.org/records/7072866/files/revel-v1.3_all_chromosomes.zip"

    # AlphaMissense (~550 MB)
    log "Downloading AlphaMissense..."
    wget -c -P "${SCORES_DIR}" \
      "https://zenodo.org/records/10813168/files/AlphaMissense_hg38.tsv.gz"

    log "Downloads complete."
    ;;

  extract)
    log "=== Stage: Submit Extraction Jobs ==="
    mkdir -p "${PROJECT_DIR}/logs" "${GNOMAD_OUT}" "${SCORES_OUT}"

    # gnomAD sites extraction (22-job array)
    GNOMAD_JOB=$($SBATCH \
      --export=ALL,VCF_DIR="${VCF_DIR}",OUT_DIR="${GNOMAD_OUT}",SCRATCH_DIR="${SCRATCH}" \
      "${PROJECT_DIR}/slurm/run_vp_extract_gnomad_sites.slurm" \
      | awk '{print $NF}')
    log "gnomAD extraction job: ${GNOMAD_JOB}"

    # Score extraction
    SCORE_JOB=$($SBATCH \
      --export=ALL,SCORES_DIR="${SCORES_DIR}",OUT_DIR="${SCORES_OUT}",SCRATCH_DIR="${SCRATCH}" \
      "${PROJECT_DIR}/slurm/run_vp_extract_scores.slurm" \
      | awk '{print $NF}')
    log "Score extraction job: ${SCORE_JOB}"

    log "Jobs submitted. Monitor with: squeue -u jak4013"
    ;;

  merge)
    log "=== Stage: Merge Shards + Load ==="

    # Merge gnomAD shards
    log "Merging gnomAD frequency shards..."
    python "${PROJECT_DIR}/scripts_vp/merge_gnomad_extracts.py" \
      --input-glob "${GNOMAD_OUT}/variant_frequencies.chr*.tsv" \
      --output "${PROJECT_DIR}/data/vp/gnomad/variant_frequencies.tsv"

    log "Merging gnomAD copper shards..."
    python "${PROJECT_DIR}/scripts_vp/merge_gnomad_extracts.py" \
      --input-glob "${GNOMAD_OUT}/copper_variants.chr*.tsv" \
      --output "${PROJECT_DIR}/data/vp/gnomad/copper_variants.tsv"

    # Load gnomAD frequencies + copper tier
    log "Loading gnomAD frequencies into DB..."
    python "${PROJECT_DIR}/scripts_vp/load_gnomad.py" \
      --db-path "${PROJECT_DIR}/data/negbiodb_vp.db" \
      --frequencies "${PROJECT_DIR}/data/vp/gnomad/variant_frequencies.tsv"

    log "Loading copper tier into DB..."
    python "${PROJECT_DIR}/scripts_vp/load_gnomad.py" \
      --db-path "${PROJECT_DIR}/data/negbiodb_vp.db" \
      --copper "${PROJECT_DIR}/data/vp/gnomad/copper_variants.tsv"

    # Load computational scores
    log "Loading scores into DB..."
    python "${PROJECT_DIR}/scripts_vp/load_scores.py" \
      --db-path "${PROJECT_DIR}/data/negbiodb_vp.db" \
      --scores "${SCORES_OUT}/merged_scores.tsv"

    # Refresh pairs
    log "Refreshing variant-disease pairs..."
    python -c "
from negbiodb_vp.vp_db import get_connection, refresh_all_vp_pairs
conn = get_connection('${PROJECT_DIR}/data/negbiodb_vp.db')
n = refresh_all_vp_pairs(conn); conn.commit(); conn.close()
print(f'Refreshed {n:,} pairs')
"

    log "Merge + load complete."
    ;;

  export)
    log "=== Stage: Export Datasets ==="

    # ML export
    log "Exporting ML dataset..."
    python "${PROJECT_DIR}/scripts_vp/export_vp_ml_dataset.py" \
      --db "${PROJECT_DIR}/data/negbiodb_vp.db" \
      --output-dir "${PROJECT_DIR}/exports/vp_ml" \
      --seed 42

    # LLM datasets
    log "Building LLM L1 dataset..."
    python "${PROJECT_DIR}/scripts_vp/build_vp_l1_dataset.py" \
      --db "${PROJECT_DIR}/data/negbiodb_vp.db"

    log "Building LLM L2 dataset..."
    python "${PROJECT_DIR}/scripts_vp/build_vp_l2_dataset.py" \
      --db "${PROJECT_DIR}/data/negbiodb_vp.db"

    log "Building LLM L3 dataset..."
    python "${PROJECT_DIR}/scripts_vp/build_vp_l3_dataset.py" \
      --db "${PROJECT_DIR}/data/negbiodb_vp.db"

    log "Building LLM L4 dataset..."
    python "${PROJECT_DIR}/scripts_vp/build_vp_l4_dataset.py" \
      --db "${PROJECT_DIR}/data/negbiodb_vp.db"

    log "Datasets exported."
    ;;

  train)
    log "=== Stage: Submit ML Training ==="
    mkdir -p "${PROJECT_DIR}/logs"

    for SEED in 42 43 44; do
      for DATASET in m1_balanced m1_realistic; do
        for SPLIT in random cold_gene cold_disease cold_both degree_balanced temporal; do
          # XGBoost (CPU)
          $SBATCH --export=ALL,MODEL=xgboost,DATASET="${DATASET}",SPLIT="${SPLIT}",SEED="${SEED}",SCRATCH_DIR="${SCRATCH}" \
            "${PROJECT_DIR}/slurm/run_vp_train_cpu.slurm"

          # MLP (CPU)
          $SBATCH --export=ALL,MODEL=mlp,DATASET="${DATASET}",SPLIT="${SPLIT}",SEED="${SEED}",EPOCHS=25,SCRATCH_DIR="${SCRATCH}" \
            "${PROJECT_DIR}/slurm/run_vp_train_cpu.slurm"
        done
      done
    done

    log "CPU training jobs submitted (XGBoost + MLP × 2 datasets × 6 splits × 3 seeds = 72 jobs)"

    # GPU models — only if ESM2 embeddings and gene graph are available
    if [ -f "${PROJECT_DIR}/data/vp/esm2_embeddings.parquet" ]; then
      for SEED in 42 43 44; do
        for DATASET in m1_balanced m1_realistic; do
          for SPLIT in random cold_gene cold_disease cold_both degree_balanced temporal; do
            $SBATCH --export=ALL,MODEL=esm2,DATASET="${DATASET}",SPLIT="${SPLIT}",SEED="${SEED}",\
ESM2_EMBEDDINGS="${PROJECT_DIR}/data/vp/esm2_embeddings.parquet",SCRATCH_DIR="${SCRATCH}" \
              "${PROJECT_DIR}/slurm/run_vp_train_gpu.slurm"
          done
        done
      done
      log "ESM2-VP GPU jobs submitted."
    else
      log "SKIP: ESM2 embeddings not found. Run precompute_esm2 first."
    fi

    if [ -f "${PROJECT_DIR}/data/vp/string_gene_graph.pkl" ]; then
      for SEED in 42 43 44; do
        for DATASET in m1_balanced m1_realistic; do
          for SPLIT in random cold_gene cold_disease cold_both degree_balanced temporal; do
            $SBATCH --export=ALL,MODEL=gnn,DATASET="${DATASET}",SPLIT="${SPLIT}",SEED="${SEED}",\
GENE_GRAPH="${PROJECT_DIR}/data/vp/string_gene_graph.pkl",SCRATCH_DIR="${SCRATCH}" \
              "${PROJECT_DIR}/slurm/run_vp_train_gpu.slurm"
          done
        done
      done
      log "VariantGNN GPU jobs submitted."
    else
      log "SKIP: Gene graph not found. Run build_gene_graph first."
    fi
    ;;

  llm)
    log "=== Stage: Submit LLM Benchmark ==="
    mkdir -p "${PROJECT_DIR}/logs"

    for TASK in vp-l1 vp-l2 vp-l3 vp-l4; do
      for FS in 0 3; do
        CONFIG="zero-shot"
        if [ "${FS}" -eq 3 ]; then CONFIG="3-shot"; fi

        # Gemini
        $SBATCH --export=ALL,TASK="${TASK}",MODEL=gemini-2.5-flash,CONFIG="${CONFIG}",FS="${FS}",SCRATCH_DIR="${SCRATCH}" \
          "${PROJECT_DIR}/slurm/run_vp_llm_gemini.slurm"

        # OpenAI
        $SBATCH --export=ALL,TASK="${TASK}",MODEL=gpt-4o-mini,CONFIG="${CONFIG}",FS="${FS}",SCRATCH_DIR="${SCRATCH}" \
          "${PROJECT_DIR}/slurm/run_vp_llm_openai.slurm"

        # Haiku
        $SBATCH --export=ALL,TASK="${TASK}",MODEL=claude-haiku-4-5-20251001,CONFIG="${CONFIG}",FS="${FS}",SCRATCH_DIR="${SCRATCH}" \
          "${PROJECT_DIR}/slurm/run_vp_llm_openai.slurm"

        # Llama (vLLM)
        $SBATCH --export=ALL,TASK="${TASK}",MODEL=meta-llama/Llama-3.1-8B-Instruct,CONFIG="${CONFIG}",FS="${FS}",SCRATCH_DIR="${SCRATCH}" \
          "${PROJECT_DIR}/slurm/run_vp_llm_vllm.slurm"

        # Qwen (vLLM)
        $SBATCH --export=ALL,TASK="${TASK}",MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG="${CONFIG}",FS="${FS}",SCRATCH_DIR="${SCRATCH}" \
          "${PROJECT_DIR}/slurm/run_vp_llm_vllm.slurm"
      done
    done

    log "LLM jobs submitted (5 models × 4 tasks × 2 configs = 40 jobs)"
    ;;

  verify)
    log "=== Stage: Verify DB State ==="
    python -c "
from negbiodb_vp.vp_db import get_connection
conn = get_connection('${PROJECT_DIR}/data/negbiodb_vp.db')
tables = ['genes', 'variants', 'diseases', 'vp_submissions', 'vp_negative_results', 'variant_disease_pairs']
for t in tables:
    count = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'{t}: {count:,}')
print()
af = conn.execute('SELECT COUNT(*) FROM variants WHERE gnomad_af_global IS NOT NULL').fetchone()[0]
cadd = conn.execute('SELECT COUNT(*) FROM variants WHERE cadd_phred IS NOT NULL').fetchone()[0]
revel = conn.execute('SELECT COUNT(*) FROM variants WHERE revel_score IS NOT NULL').fetchone()[0]
am = conn.execute('SELECT COUNT(*) FROM variants WHERE alphamissense_score IS NOT NULL').fetchone()[0]
copper = conn.execute(\"SELECT COUNT(*) FROM vp_negative_results WHERE source_db = 'gnomad'\").fetchone()[0]
print(f'gnomAD AF annotated: {af:,}')
print(f'CADD annotated: {cadd:,}')
print(f'REVEL annotated: {revel:,}')
print(f'AlphaMissense annotated: {am:,}')
print(f'Copper tier results: {copper:,}')
conn.close()
"
    ;;

  help|*)
    echo "VP HPC Pipeline Submission Script"
    echo ""
    echo "Usage: bash scripts_vp/submit_vp_hpc_pipeline.sh <stage>"
    echo ""
    echo "Stages (in order):"
    echo "  download  — Download gnomAD VCF + CADD/REVEL/AlphaMissense"
    echo "  extract   — Submit gnomAD + score extraction SLURM jobs"
    echo "  merge     — Merge shards + load into VP DB"
    echo "  export    — Export ML dataset + build LLM datasets"
    echo "  train     — Submit ML training jobs"
    echo "  llm       — Submit LLM benchmark jobs"
    echo "  verify    — Verify DB state (counts + annotations)"
    echo ""
    echo "Dependency order:"
    echo "  download → extract → merge → export → {train, llm}"
    ;;
esac
