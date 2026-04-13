#!/bin/bash
# Submit MD downstream jobs on Cayuga using the working Slurm v24.05.2 binaries.
#
# Usage examples:
#   bash scripts_md/submit_md_jobs.sh train
#   bash scripts_md/submit_md_jobs.sh train --task m2 --split cold_both --model-type mlp
#   bash scripts_md/submit_md_jobs.sh llm-api --level l2 --model gemini-2.5-flash
#   bash scripts_md/submit_md_jobs.sh llm-vllm --level l1 --model Qwen/Qwen2.5-7B-Instruct
#   bash scripts_md/submit_md_jobs.sh l3-judge --responses-file md_l3_responses_gemini-2.5-flash_0shot.jsonl
#   bash scripts_md/submit_md_jobs.sh after-ingest
#   bash scripts_md/submit_md_jobs.sh status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH="${SBATCH_BIN:-/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch}"
SQUEUE="${SQUEUE_BIN:-/opt/ohpc/pub/software/slurm/24.05.2/bin/squeue}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts_md/submit_md_jobs.sh train [--task m1|m2] [--split name] [--seed N] [--model-type xgboost|mlp] [--dry-run]
  bash scripts_md/submit_md_jobs.sh llm-api [--level l1|l2|l3|l4] [--model MODEL] [--shot 0shot|3shot] [--fewshot-set N] [--dry-run]
  bash scripts_md/submit_md_jobs.sh llm-vllm [--level l1|l2|l3|l4] [--model MODEL] [--shot 0shot|3shot] [--dry-run]
  bash scripts_md/submit_md_jobs.sh l3-judge [--responses-file FILE] [--judge MODEL] [--dry-run]
  bash scripts_md/submit_md_jobs.sh after-ingest [--dry-run]
  bash scripts_md/submit_md_jobs.sh status [--job-id ID]
EOF
}

require_slurm_bin() {
    local bin="$1"
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: missing executable Slurm binary: $bin" >&2
        exit 1
    fi
}

submit_job() {
    local export_str="$1"
    local script_path="$2"
    local dry_run="$3"

    echo "SBATCH: $SBATCH"
    echo "Script: $script_path"
    echo "Export: $export_str"
    if [[ "$dry_run" == "true" ]]; then
        echo "[dry-run] submission skipped"
    else
        "$SBATCH" --export="$export_str" "$script_path"
    fi
}

submit_train() {
    local task="m1"
    local split="random"
    local seed="42"
    local model_type="xgboost"
    local dry_run="false"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --task) task="$2"; shift 2 ;;
            --split) split="$2"; shift 2 ;;
            --seed) seed="$2"; shift 2 ;;
            --model-type) model_type="$2"; shift 2 ;;
            --dry-run) dry_run="true"; shift ;;
            *) echo "Unknown train option: $1" >&2; usage; exit 1 ;;
        esac
    done

    submit_job \
        "ALL,TASK=${task},SPLIT=${split},SEED=${seed},MODEL_TYPE=${model_type}" \
        "$PROJECT_ROOT/slurm/run_md_train_cpu.slurm" \
        "$dry_run"
}

submit_llm_api() {
    local level="l1"
    local model="gemini-2.5-flash"
    local shot="0shot"
    local fewshot_set=""
    local dry_run="false"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --level) level="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --shot) shot="$2"; shift 2 ;;
            --fewshot-set) fewshot_set="$2"; shift 2 ;;
            --dry-run) dry_run="true"; shift ;;
            *) echo "Unknown llm-api option: $1" >&2; usage; exit 1 ;;
        esac
    done

    local export_str="ALL,LEVEL=${level},MODEL=${model},SHOT=${shot}"
    if [[ -n "$fewshot_set" ]]; then
        export_str+=",FEWSHOT_SET=${fewshot_set}"
    fi

    submit_job "$export_str" "$PROJECT_ROOT/slurm/run_md_llm_api.slurm" "$dry_run"
}

submit_llm_vllm() {
    local level="l1"
    local model="Qwen/Qwen2.5-7B-Instruct"
    local shot="0shot"
    local dry_run="false"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --level) level="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --shot) shot="$2"; shift 2 ;;
            --dry-run) dry_run="true"; shift ;;
            *) echo "Unknown llm-vllm option: $1" >&2; usage; exit 1 ;;
        esac
    done

    submit_job \
        "ALL,LEVEL=${level},MODEL=${model},SHOT=${shot}" \
        "$PROJECT_ROOT/slurm/run_md_llm_vllm.slurm" \
        "$dry_run"
}

submit_l3_judge() {
    local responses_file="md_l3_responses_claude-sonnet-4-6_0shot.jsonl"
    local judge="gemini-2.5-flash"
    local dry_run="false"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --responses-file) responses_file="$2"; shift 2 ;;
            --judge) judge="$2"; shift 2 ;;
            --dry-run) dry_run="true"; shift ;;
            *) echo "Unknown l3-judge option: $1" >&2; usage; exit 1 ;;
        esac
    done

    submit_job \
        "ALL,RESPONSES_FILE=${responses_file},JUDGE=${judge}" \
        "$PROJECT_ROOT/slurm/run_md_l3_judge.slurm" \
        "$dry_run"
}

submit_after_ingest() {
    local dry_run="false"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) dry_run="true"; shift ;;
            *) echo "Unknown after-ingest option: $1" >&2; usage; exit 1 ;;
        esac
    done

    echo "Submitting default MD post-export jobs..."
    submit_job \
        "ALL,TASK=m1,SPLIT=random,SEED=42,MODEL_TYPE=xgboost" \
        "$PROJECT_ROOT/slurm/run_md_train_cpu.slurm" \
        "$dry_run"
    echo ""
    submit_job \
        "ALL,LEVEL=l1,MODEL=gemini-2.5-flash,SHOT=0shot" \
        "$PROJECT_ROOT/slurm/run_md_llm_api.slurm" \
        "$dry_run"
    echo ""
    submit_job \
        "ALL,LEVEL=l1,MODEL=Qwen/Qwen2.5-7B-Instruct,SHOT=0shot" \
        "$PROJECT_ROOT/slurm/run_md_llm_vllm.slurm" \
        "$dry_run"
    echo ""
    echo "When L3 responses finish, submit the judge step with:"
    echo "  bash scripts_md/submit_md_jobs.sh l3-judge --responses-file md_l3_responses_<model>_0shot.jsonl"
}

show_status() {
    local job_id=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --job-id) job_id="$2"; shift 2 ;;
            *) echo "Unknown status option: $1" >&2; usage; exit 1 ;;
        esac
    done

    if [[ -n "$job_id" ]]; then
        "$SQUEUE" -j "$job_id" -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
    else
        "$SQUEUE" -u "$USER" -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" \
            | awk 'NR == 1 || $3 ~ /^md/'
    fi
}

COMMAND="${1:-}"
if [[ -z "$COMMAND" ]]; then
    usage
    exit 1
fi
shift

if [[ "$COMMAND" == "help" || "$COMMAND" == "--help" || "$COMMAND" == "-h" ]]; then
    usage
    exit 0
fi

require_slurm_bin "$SBATCH"
require_slurm_bin "$SQUEUE"

case "$COMMAND" in
    train) submit_train "$@" ;;
    llm-api) submit_llm_api "$@" ;;
    llm-vllm) submit_llm_vllm "$@" ;;
    l3-judge) submit_l3_judge "$@" ;;
    after-ingest) submit_after_ingest "$@" ;;
    status) show_status "$@" ;;
    *) echo "Unknown command: $COMMAND" >&2; usage; exit 1 ;;
esac
