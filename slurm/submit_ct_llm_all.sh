#!/bin/bash
# Submit all CT LLM benchmark jobs on Cayuga.
#
# Grid: 5 models × 4 tasks × (1 zero-shot + 3 three-shot seeds) = 80 jobs
#
# Provider routing (CT-specific SLURM templates):
#   llama70b, qwen32b  → run_ct_llm_local.slurm  (GPU, vLLM)
#   gpt-4o-mini        → run_ct_llm_openai.slurm  (CPU, API)
#   gemini-2.5-flash   → run_ct_llm_gemini.slurm  (CPU, API)
#   claude-haiku-4-5   → run_ct_llm_anthropic.slurm (CPU, API)
#
# Usage:
#   bash slurm/submit_ct_llm_all.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

CT_TASKS=("ct-l1" "ct-l2" "ct-l3" "ct-l4")
CONFIGS=("zero-shot" "3-shot")
FEWSHOT_SETS=(0 1 2)

# Model definitions: short_name → model_arg, slurm_template, extra_flags
# For local models: short name matches SLURM case statement
# For API models: model name is passed directly to --model
declare -A MODEL_ARG
declare -A MODEL_SLURM
declare -A MODEL_EXTRA

MODEL_ARG["llama70b"]="llama70b"
MODEL_SLURM["llama70b"]="run_ct_llm_local.slurm"
MODEL_EXTRA["llama70b"]=""

MODEL_ARG["qwen32b"]="qwen32b"
MODEL_SLURM["qwen32b"]="run_ct_llm_local.slurm"
MODEL_EXTRA["qwen32b"]=""

MODEL_ARG["gpt-4o-mini"]="gpt-4o-mini"
MODEL_SLURM["gpt-4o-mini"]="run_ct_llm_openai.slurm"
MODEL_EXTRA["gpt-4o-mini"]=""

MODEL_ARG["gemini-2.5-flash"]="gemini-2.5-flash"
MODEL_SLURM["gemini-2.5-flash"]="run_ct_llm_gemini.slurm"
MODEL_EXTRA["gemini-2.5-flash"]=""

MODEL_ARG["claude-haiku-4-5"]="claude-haiku-4-5"
MODEL_SLURM["claude-haiku-4-5"]="run_ct_llm_anthropic.slurm"
MODEL_EXTRA["claude-haiku-4-5"]=""

TOTAL=0
SUBMITTED=0

for MODEL in "${!MODEL_ARG[@]}"; do
    MODEL_NAME="${MODEL_ARG[$MODEL]}"
    SLURM_TMPL="${MODEL_SLURM[$MODEL]}"
    EXTRA="${MODEL_EXTRA[$MODEL]}"

    for TASK in "${CT_TASKS[@]}"; do
        for CONFIG in "${CONFIGS[@]}"; do
            if [[ "$CONFIG" == "zero-shot" ]]; then
                FS_SETS=(0)
            else
                FS_SETS=("${FEWSHOT_SETS[@]}")
            fi

            for FS in "${FS_SETS[@]}"; do
                TOTAL=$((TOTAL + 1))
                JOB_NAME="ct_${TASK}_${MODEL}_${CONFIG}_fs${FS}"

                if $DRY_RUN; then
                    echo "  [DRY] $JOB_NAME → $SLURM_TMPL"
                else
                    echo "  Submitting: $JOB_NAME"
                    $SBATCH \
                        --job-name="$JOB_NAME" \
                        --export="ALL,TASK=$TASK,MODEL=$MODEL_NAME,CONFIG=$CONFIG,FS=$FS,EXTRA=$EXTRA" \
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
