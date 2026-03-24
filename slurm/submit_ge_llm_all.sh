#!/bin/bash
# Submit all GE LLM benchmark jobs on Cayuga.
#
# Grid: 5 models × 4 tasks × (1 zero-shot + 3 three-shot seeds) = 80 jobs
#
# Provider routing (GE-specific SLURM templates):
#   Qwen2.5-7B, Llama-3.1-8B → run_ge_llm_local.slurm  (GPU, vLLM)
#   gpt-4o-mini               → run_ge_llm_openai.slurm  (CPU, API)
#   gemini-2.5-flash           → run_ge_llm_gemini.slurm  (CPU, API)
#   claude-haiku-4-5           → run_ge_llm_anthropic.slurm (CPU, API)
#
# Usage:
#   bash slurm/submit_ge_llm_all.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

GE_TASKS=("ge-l1" "ge-l2" "ge-l3" "ge-l4")
CONFIGS=("zero-shot" "3-shot")
FEWSHOT_SETS=(0 1 2)

# Model definitions
declare -A MODEL_ARG
declare -A MODEL_SLURM

MODEL_ARG["claude-haiku-4-5"]="claude-haiku-4-5"
MODEL_SLURM["claude-haiku-4-5"]="run_ge_llm_anthropic.slurm"

MODEL_ARG["gpt-4o-mini"]="gpt-4o-mini"
MODEL_SLURM["gpt-4o-mini"]="run_ge_llm_openai.slurm"

MODEL_ARG["gemini-2.5-flash"]="gemini-2.5-flash"
MODEL_SLURM["gemini-2.5-flash"]="run_ge_llm_gemini.slurm"

MODEL_ARG["qwen7b"]="Qwen/Qwen2.5-7B-Instruct"
MODEL_SLURM["qwen7b"]="run_ge_llm_local.slurm"

MODEL_ARG["llama8b"]="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SLURM["llama8b"]="run_ge_llm_local.slurm"

TOTAL=0
SUBMITTED=0

for MODEL in "${!MODEL_ARG[@]}"; do
    MODEL_NAME="${MODEL_ARG[$MODEL]}"
    SLURM_TMPL="${MODEL_SLURM[$MODEL]}"

    for TASK in "${GE_TASKS[@]}"; do
        for CONFIG in "${CONFIGS[@]}"; do
            if [[ "$CONFIG" == "zero-shot" ]]; then
                FS_SETS=(0)
            else
                FS_SETS=("${FEWSHOT_SETS[@]}")
            fi

            for FS in "${FS_SETS[@]}"; do
                TOTAL=$((TOTAL + 1))
                JOB_NAME="ge_${TASK}_${MODEL}_${CONFIG}_fs${FS}"

                if $DRY_RUN; then
                    echo "  [DRY] $JOB_NAME → $SLURM_TMPL"
                else
                    echo "  Submitting: $JOB_NAME"
                    $SBATCH \
                        --job-name="$JOB_NAME" \
                        --export="ALL,TASK=$TASK,MODEL=$MODEL_NAME,CONFIG=$CONFIG,FEWSHOT_SET=$FS" \
                        "$SCRIPT_DIR/$SLURM_TMPL"
                    SUBMITTED=$((SUBMITTED + 1))
                fi
            done
        done
    done
done

echo ""
echo "Total jobs: $TOTAL"
if $DRY_RUN; then
    echo "(Dry run — no jobs submitted)"
else
    echo "Submitted: $SUBMITTED"
fi
