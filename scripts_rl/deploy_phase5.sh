#!/bin/bash
# Deploy NegBioRL Phase 5 experiments to Cayuga HPC.
#
# Usage:
#   bash scripts_rl/deploy_phase5.sh [data|sync|tier1|tier2|aggregate|all|help]
#
# Steps:
#   data      — Build single-domain datasets locally (8 files)
#   sync      — rsync code + data to HPC
#   tier1     — Submit all Tier 1 jobs (baseline, shared SFT, transfer matrix, ablations)
#   tier2     — Submit Tier 2 jobs (G=2, G=16, Gemma4) — requires S_MULTI from tier1
#   aggregate — Submit aggregation jobs (transfer matrix, before/after, PBS)
#   all       — data + sync + tier1

set -euo pipefail

HPC_HOST="cayuga-login1"
LOCAL_ROOT="/Users/jak4013/Dropbox/Bioinformatics/Claude/Negative_result_DB"
REMOTE_ROOT="/athena/masonlab/scratch/users/jak4013/negbiodb"
SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"

# Phase 5 paths
P5_CHECKPOINTS="results/negbiorl/phase5_checkpoints"
P5_EVAL="results/negbiorl/phase5_eval"
P5_TRANSFER="results/negbiorl/phase5_transfer"
P5_DATA="results/negbiorl/phase2_training_data"

ACTION="${1:-help}"

# ---------------------------------------------------------------------------
build_single_domain_data() {
    echo "=== Building single-domain datasets ==="
    cd "${LOCAL_ROOT}"
    export PYTHONPATH=src

    for DOMAIN in dti ct ppi ge; do
        echo "  Building SFT for ${DOMAIN}..."
        python scripts_rl/02_build_sft_dataset.py \
            --domains "${DOMAIN}" --output-name "sft_${DOMAIN}.jsonl"

        echo "  Building GRPO for ${DOMAIN}..."
        python scripts_rl/03_build_grpo_dataset.py \
            --domains "${DOMAIN}" --output-name "grpo_${DOMAIN}.jsonl"
    done

    echo "Done. Files:"
    ls -la "${P5_DATA}"/sft_{dti,ct,ppi,ge}.jsonl "${P5_DATA}"/grpo_{dti,ct,ppi,ge}.jsonl 2>/dev/null || true
}

# ---------------------------------------------------------------------------
sync_to_hpc() {
    echo "=== Syncing to HPC ==="
    rsync -avz --progress \
        --include='src/***' \
        --include='scripts_rl/***' \
        --include='configs/***' \
        --include='slurm/***' \
        --include='results/negbiorl/***' \
        --include='exports/***' \
        --include='pyproject.toml' \
        --include='config.yaml' \
        --exclude='data/' \
        --exclude='*.db' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='*.pyc' \
        --exclude='paper/' \
        --exclude='research/' \
        --exclude='tests/' \
        "${LOCAL_ROOT}/" "${HPC_HOST}:${REMOTE_ROOT}/"
    echo "Done."
}

# ---------------------------------------------------------------------------
submit_baseline() {
    echo "=== Submitting Qwen2.5-7B baseline ==="
    local BL_JOB
    BL_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && mkdir -p logs && ${SBATCH} --parsable \
        --export=ALL,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,OUTPUT_DIR=${P5_EVAL}/baseline_qwen25 \
        slurm/run_rl_baseline.slurm")
    echo "  Baseline submitted: ${BL_JOB}"
    echo "${BL_JOB}"
}

# ---------------------------------------------------------------------------
submit_shared_sft() {
    echo "=== Submitting shared SFT (S_MULTI) ==="
    local SFT_JOB
    SFT_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && mkdir -p logs && ${SBATCH} --parsable \
        --export=ALL,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG=configs/negbiorl/sft_qwen25_7b.yaml,DATASET=${P5_DATA}/sft_dataset.jsonl,OUTPUT_DIR=${P5_CHECKPOINTS}/sft_multi \
        --job-name=rl_sft_multi \
        slurm/run_rl_sft.slurm")
    echo "  S_MULTI submitted: ${SFT_JOB}"
    echo "${SFT_JOB}"
}

# ---------------------------------------------------------------------------
submit_single_domain_chain() {
    # Args: $1=domain
    local DOMAIN="$1"
    echo "  --- Chain P${DOMAIN} ---"

    # SFT (single-domain)
    local S_JOB
    S_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --export=ALL,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG=configs/negbiorl/sft_qwen25_7b.yaml,DATASET=${P5_DATA}/sft_${DOMAIN}.jsonl,OUTPUT_DIR=${P5_CHECKPOINTS}/sft_${DOMAIN} \
        --job-name=rl_sft_${DOMAIN} \
        slurm/run_rl_sft.slurm")
    echo "    SFT(${DOMAIN}): ${S_JOB}"

    # GRPO (single-domain, 4 epochs)
    local G_JOB
    G_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --dependency=afterok:${S_JOB} \
        --export=ALL,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG=configs/negbiorl/grpo_qwen25_7b_4ep.yaml,SFT_ADAPTER=${P5_CHECKPOINTS}/sft_${DOMAIN},DATASET=${P5_DATA}/grpo_${DOMAIN}.jsonl,OUTPUT_DIR=${P5_CHECKPOINTS}/grpo_${DOMAIN},EPOCHS=4 \
        --job-name=rl_grpo_${DOMAIN} \
        slurm/run_rl_grpo.slurm")
    echo "    GRPO(${DOMAIN}): ${G_JOB} (depends on ${S_JOB})"

    # Eval on all 5 domains → phase5_transfer/{domain}/
    local E_JOB
    E_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --dependency=afterok:${G_JOB} \
        --export=ALL,ADAPTER=${P5_CHECKPOINTS}/grpo_${DOMAIN}/final,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,OUTPUT_DIR=${P5_TRANSFER}/${DOMAIN},BASELINE_DIR=${P5_EVAL}/baseline_qwen25 \
        slurm/run_rl_eval.slurm")
    echo "    Eval(${DOMAIN}→all): ${E_JOB} (depends on ${G_JOB})"
}

# ---------------------------------------------------------------------------
submit_transfer_matrix() {
    echo "=== Submitting transfer matrix chains (P8-P11) ==="
    for DOMAIN in dti ct ppi ge; do
        submit_single_domain_chain "${DOMAIN}"
    done
}

# ---------------------------------------------------------------------------
submit_method_ablation() {
    # Requires: S_MULTI job ID as $1
    local S_MULTI_JOB="$1"
    echo "=== Submitting method ablations (P2c, P6, E7) ==="

    # P2c: GRPO G=8 from shared SFT (runs FIRST to create merge cache)
    local P2C_JOB
    P2C_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --dependency=afterok:${S_MULTI_JOB} \
        --export=ALL,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG=configs/negbiorl/grpo_qwen25_7b.yaml,SFT_ADAPTER=${P5_CHECKPOINTS}/sft_multi,OUTPUT_DIR=${P5_CHECKPOINTS}/grpo_multi_G8 \
        --job-name=rl_grpo_p2c \
        slurm/run_rl_grpo.slurm")
    echo "  P2c (GRPO G=8): ${P2C_JOB} (depends on S_MULTI:${S_MULTI_JOB})"

    # E2c: Eval P2c on all 5 domains → phase5_transfer/multi/
    local E2C_JOB
    E2C_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --dependency=afterok:${P2C_JOB} \
        --export=ALL,ADAPTER=${P5_CHECKPOINTS}/grpo_multi_G8/final,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,OUTPUT_DIR=${P5_TRANSFER}/multi,BASELINE_DIR=${P5_EVAL}/baseline_qwen25 \
        slurm/run_rl_eval.slurm")
    echo "  E2c: ${E2C_JOB} (depends on ${P2C_JOB})"

    # P6: DPO from shared SFT (depends on P2c to ensure merge cache exists)
    local P6_JOB
    P6_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --dependency=afterok:${P2C_JOB} \
        --export=ALL,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG=configs/negbiorl/dpo_qwen25_7b.yaml,SFT_ADAPTER=${P5_CHECKPOINTS}/sft_multi,OUTPUT_DIR=${P5_CHECKPOINTS}/dpo_multi \
        --job-name=rl_dpo_p6 \
        slurm/run_rl_dpo.slurm")
    echo "  P6 (DPO): ${P6_JOB} (depends on P2c:${P2C_JOB})"

    # E6: Eval DPO on all 5 domains
    local E6_JOB
    E6_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --dependency=afterok:${P6_JOB} \
        --export=ALL,ADAPTER=${P5_CHECKPOINTS}/dpo_multi/final,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,OUTPUT_DIR=${P5_EVAL}/dpo_multi,BASELINE_DIR=${P5_EVAL}/baseline_qwen25 \
        slurm/run_rl_eval.slurm")
    echo "  E6: ${E6_JOB} (depends on ${P6_JOB})"

    # E7: Eval SFT-only (use SFT adapter directly, no GRPO)
    local E7_JOB
    E7_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --dependency=afterok:${S_MULTI_JOB} \
        --export=ALL,ADAPTER=${P5_CHECKPOINTS}/sft_multi/final,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,OUTPUT_DIR=${P5_EVAL}/sft_only,BASELINE_DIR=${P5_EVAL}/baseline_qwen25 \
        slurm/run_rl_eval.slurm")
    echo "  E7 (SFT-only): ${E7_JOB} (depends on S_MULTI:${S_MULTI_JOB})"
}

# ---------------------------------------------------------------------------
submit_tier1() {
    echo "=== Submitting Tier 1 ==="

    # Baseline (independent)
    submit_baseline

    # Shared SFT (independent)
    local S_MULTI_JOB
    S_MULTI_JOB=$(submit_shared_sft | tail -1)

    # Transfer matrix chains P8-P11 (independent of S_MULTI)
    submit_transfer_matrix

    # Method ablations P2c/P6/E7 (depend on S_MULTI)
    submit_method_ablation "${S_MULTI_JOB}"

    echo ""
    echo "=== Tier 1 submitted. S_MULTI job: ${S_MULTI_JOB} ==="
    echo "Save this for Tier 2: S_MULTI_JOB=${S_MULTI_JOB}"
}

# ---------------------------------------------------------------------------
submit_tier2() {
    # Requires: S_MULTI job ID as $1
    local S_MULTI_JOB="${1:?Must provide S_MULTI job ID as argument}"
    echo "=== Submitting Tier 2 ==="

    # P4: G=2 ablation from shared SFT
    local P4_JOB
    P4_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --dependency=afterok:${S_MULTI_JOB} \
        --export=ALL,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG=configs/negbiorl/grpo_qwen25_7b_G2.yaml,SFT_ADAPTER=${P5_CHECKPOINTS}/sft_multi,OUTPUT_DIR=${P5_CHECKPOINTS}/grpo_multi_G2,NUM_GEN=2 \
        --job-name=rl_grpo_G2 \
        slurm/run_rl_grpo.slurm")
    echo "  P4 (G=2): ${P4_JOB}"

    ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} \
        --dependency=afterok:${P4_JOB} \
        --export=ALL,ADAPTER=${P5_CHECKPOINTS}/grpo_multi_G2/final,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,OUTPUT_DIR=${P5_EVAL}/grpo_G2,BASELINE_DIR=${P5_EVAL}/baseline_qwen25 \
        slurm/run_rl_eval.slurm"
    echo "  E4: submitted (depends on ${P4_JOB})"

    # P5: G=16 ablation from shared SFT (needs longer time limit)
    local P5_JOB
    P5_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --dependency=afterok:${S_MULTI_JOB} \
        --time=32:00:00 \
        --export=ALL,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG=configs/negbiorl/grpo_qwen25_7b_G16.yaml,SFT_ADAPTER=${P5_CHECKPOINTS}/sft_multi,OUTPUT_DIR=${P5_CHECKPOINTS}/grpo_multi_G16,NUM_GEN=16 \
        --job-name=rl_grpo_G16 \
        slurm/run_rl_grpo.slurm")
    echo "  P5 (G=16): ${P5_JOB}"

    ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} \
        --dependency=afterok:${P5_JOB} \
        --export=ALL,ADAPTER=${P5_CHECKPOINTS}/grpo_multi_G16/final,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,OUTPUT_DIR=${P5_EVAL}/grpo_G16,BASELINE_DIR=${P5_EVAL}/baseline_qwen25 \
        slurm/run_rl_eval.slurm"
    echo "  E5: submitted (depends on ${P5_JOB})"

    # P12: Gemma4-31B (independent of S_MULTI — no SFT, GRPO from base)
    local P12_JOB
    P12_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --export=ALL,BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit,CONFIG=configs/negbiorl/grpo_gemma4_31b.yaml,OUTPUT_DIR=${P5_CHECKPOINTS}/grpo_gemma4,NUM_GEN=2,EPOCHS=2 \
        --job-name=rl_grpo_gemma4 \
        --time=16:00:00 \
        slurm/run_rl_grpo.slurm")
    echo "  P12 (Gemma4 GRPO): ${P12_JOB}"

    ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} \
        --dependency=afterok:${P12_JOB} \
        --export=ALL,ADAPTER=${P5_CHECKPOINTS}/grpo_gemma4/final,BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit,OUTPUT_DIR=${P5_EVAL}/grpo_gemma4,BASELINE_DIR=${P5_EVAL}/baseline_qwen25,BACKEND=hf,LOAD_IN_4BIT=1 \
        slurm/run_rl_eval.slurm"
    echo "  E12 (Gemma4 eval): submitted (depends on ${P12_JOB})"

    echo "=== Tier 2 submitted ==="
}

# ---------------------------------------------------------------------------
submit_aggregation() {
    echo "=== Submitting aggregation (Wave 2) ==="
    ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && python scripts_rl/08_cross_domain_transfer.py \
        --results-dir ${P5_TRANSFER}/ --task l4 \
        --output-dir ${P5_TRANSFER}/"
    echo "=== Aggregation done ==="
}

# ---------------------------------------------------------------------------
case "${ACTION}" in
    data)      build_single_domain_data ;;
    sync)      sync_to_hpc ;;
    tier1)     submit_tier1 ;;
    tier2)     submit_tier2 "${2:-}" ;;
    aggregate) submit_aggregation ;;
    all)
        build_single_domain_data
        sync_to_hpc
        submit_tier1
        ;;
    help|*)
        echo "Usage: bash scripts_rl/deploy_phase5.sh [data|sync|tier1|tier2|aggregate|all]"
        echo ""
        echo "  data      — Build 8 single-domain datasets locally"
        echo "  sync      — rsync code + data to HPC"
        echo "  tier1     — Submit Tier 1: baseline + S_MULTI + P8-P11 + P2c/P6/E7"
        echo "  tier2 ID  — Submit Tier 2: G=2 + G=16 + Gemma4 (needs S_MULTI job ID)"
        echo "  aggregate — Run transfer matrix aggregation on HPC"
        echo "  all       — data + sync + tier1"
        ;;
esac
