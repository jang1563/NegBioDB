#!/bin/bash
# Deploy NegBioRL code + data to Cayuga HPC and submit jobs.
#
# Usage:
#   bash scripts_rl/deploy_to_hpc.sh [sync|setup|baseline|p1|p2|p3|all]
#
# Steps:
#   sync     — rsync code + data to HPC
#   setup    — create conda env on HPC
#   baseline — submit Qwen3-8B baseline evaluation
#   p1       — submit P1: Qwen3-8B SFT → GRPO
#   p2       — submit P2: Qwen2.5-7B SFT → GRPO
#   p3       — submit P3: Llama-3.1-8B SFT → GRPO
#   all      — sync + setup + baseline + p1

set -euo pipefail

HPC_HOST="cayuga-login1"
LOCAL_ROOT="/Users/jak4013/Dropbox/Bioinformatics/Claude/Negative_result_DB"
REMOTE_ROOT="/athena/masonlab/scratch/users/jak4013/negbiodb"
SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"

ACTION="${1:-help}"

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
setup_env() {
    echo "=== Setting up HPC environment ==="
    ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && bash scripts_rl/00_setup_hpc_env.sh"
}

# ---------------------------------------------------------------------------
submit_baseline() {
    echo "=== Submitting baseline (Qwen3-8B, no fine-tuning) ==="
    ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && mkdir -p logs && ${SBATCH} slurm/run_rl_baseline.slurm"
}

# ---------------------------------------------------------------------------
submit_p1() {
    echo "=== P1: Qwen3-8B SFT → GRPO ==="
    # SFT first
    local SFT_JOB
    SFT_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && mkdir -p logs && ${SBATCH} --parsable slurm/run_rl_sft.slurm")
    echo "  SFT submitted: ${SFT_JOB}"

    # GRPO depends on SFT output — use afterok dependency
    # SFT adapter will be at results/negbiorl/phase3_checkpoints/sft_${SFT_JOB}/
    local GRPO_JOB
    GRPO_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --dependency=afterok:${SFT_JOB} \
        --export=ALL,SFT_ADAPTER=results/negbiorl/phase3_checkpoints/sft_${SFT_JOB} \
        slurm/run_rl_grpo.slurm")
    echo "  GRPO submitted: ${GRPO_JOB} (depends on ${SFT_JOB})"

    # Eval depends on GRPO + baseline
    ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} \
        --dependency=afterok:${GRPO_JOB} \
        --export=ALL,ADAPTER=results/negbiorl/phase3_checkpoints/grpo_G8_${GRPO_JOB}/final \
        slurm/run_rl_eval.slurm"
    echo "  Eval submitted (depends on ${GRPO_JOB})"
}

# ---------------------------------------------------------------------------
submit_p2() {
    echo "=== P2: Qwen2.5-7B SFT → GRPO ==="
    local SFT_JOB
    SFT_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && mkdir -p logs && ${SBATCH} --parsable \
        --export=ALL,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG=configs/negbiorl/sft_qwen25_7b.yaml \
        slurm/run_rl_sft.slurm")
    echo "  SFT submitted: ${SFT_JOB}"

    local GRPO_JOB
    GRPO_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --dependency=afterok:${SFT_JOB} \
        --export=ALL,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG=configs/negbiorl/grpo_qwen25_7b.yaml,SFT_ADAPTER=results/negbiorl/phase3_checkpoints/sft_${SFT_JOB} \
        slurm/run_rl_grpo.slurm")
    echo "  GRPO submitted: ${GRPO_JOB} (depends on ${SFT_JOB})"

    ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} \
        --dependency=afterok:${GRPO_JOB} \
        --export=ALL,ADAPTER=results/negbiorl/phase3_checkpoints/grpo_G8_${GRPO_JOB}/final,BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
        slurm/run_rl_eval.slurm"
    echo "  Eval submitted (depends on ${GRPO_JOB})"
}

# ---------------------------------------------------------------------------
submit_p3() {
    echo "=== P3: Llama-3.1-8B SFT → GRPO ==="
    local SFT_JOB
    SFT_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && mkdir -p logs && ${SBATCH} --parsable \
        --export=ALL,BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct,CONFIG=configs/negbiorl/sft_llama31_8b.yaml \
        slurm/run_rl_sft.slurm")
    echo "  SFT submitted: ${SFT_JOB}"

    local GRPO_JOB
    GRPO_JOB=$(ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} --parsable \
        --dependency=afterok:${SFT_JOB} \
        --export=ALL,BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct,CONFIG=configs/negbiorl/grpo_llama31_8b.yaml,SFT_ADAPTER=results/negbiorl/phase3_checkpoints/sft_${SFT_JOB} \
        slurm/run_rl_grpo.slurm")
    echo "  GRPO submitted: ${GRPO_JOB} (depends on ${SFT_JOB})"

    ssh "${HPC_HOST}" "cd ${REMOTE_ROOT} && ${SBATCH} \
        --dependency=afterok:${GRPO_JOB} \
        --export=ALL,ADAPTER=results/negbiorl/phase3_checkpoints/grpo_G8_${GRPO_JOB}/final,BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
        slurm/run_rl_eval.slurm"
    echo "  Eval submitted (depends on ${GRPO_JOB})"
}

# ---------------------------------------------------------------------------
case "${ACTION}" in
    sync)     sync_to_hpc ;;
    setup)    setup_env ;;
    baseline) submit_baseline ;;
    p1)       submit_p1 ;;
    p2)       submit_p2 ;;
    p3)       submit_p3 ;;
    all)
        sync_to_hpc
        setup_env
        submit_baseline
        submit_p1
        ;;
    help|*)
        echo "Usage: bash scripts_rl/deploy_to_hpc.sh [sync|setup|baseline|p1|p2|p3|all]"
        echo ""
        echo "  sync     — rsync code + data to HPC"
        echo "  setup    — create conda env on HPC (run once)"
        echo "  baseline — submit Qwen3-8B baseline evaluation"
        echo "  p1       — submit P1: Qwen3-8B SFT → GRPO → eval pipeline"
        echo "  p2       — submit P2: Qwen2.5-7B SFT → GRPO → eval pipeline"
        echo "  p3       — submit P3: Llama-3.1-8B SFT → GRPO → eval pipeline"
        echo "  all      — sync + setup + baseline + p1"
        ;;
esac
