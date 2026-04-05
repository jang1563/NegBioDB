#!/bin/bash
# Submit all 20 VP LLM zero-shot jobs (5 models x 4 tasks)
# Run after LLM dataset build completes.

set -euo pipefail

SCRATCH="/athena/masonlab/scratch/users/jak4013"
PROJECT_DIR="${SCRATCH}/negbiodb"
SLURM_DIR="${PROJECT_DIR}/slurm"
SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"
TASKS=(vp-l1 vp-l2 vp-l3 vp-l4)
CONFIG="zero-shot"
FS=0

n=0

# Haiku (Anthropic)
for TASK in "${TASKS[@]}"; do
    ${SBATCH} --export=ALL,TASK=${TASK},CONFIG=${CONFIG},FEWSHOT_SET=${FS} \
        "${SLURM_DIR}/run_vp_llm_anthropic.slurm"
    n=$((n+1))
done

# Gemini
for TASK in "${TASKS[@]}"; do
    ${SBATCH} --export=ALL,TASK=${TASK},CONFIG=${CONFIG},FEWSHOT_SET=${FS} \
        "${SLURM_DIR}/run_vp_llm_gemini.slurm"
    n=$((n+1))
done

# GPT-4o-mini (OpenAI)
for TASK in "${TASKS[@]}"; do
    ${SBATCH} --export=ALL,TASK=${TASK},CONFIG=${CONFIG},FEWSHOT_SET=${FS} \
        "${SLURM_DIR}/run_vp_llm_openai.slurm"
    n=$((n+1))
done

# Qwen2.5-7B (vLLM)
for TASK in "${TASKS[@]}"; do
    ${SBATCH} \
        --export=ALL,TASK=${TASK},MODEL=Qwen/Qwen2.5-7B-Instruct,CONFIG=${CONFIG},FEWSHOT_SET=${FS} \
        "${SLURM_DIR}/run_vp_llm_vllm.slurm"
    n=$((n+1))
done

# Llama-3.1-8B (vLLM)
for TASK in "${TASKS[@]}"; do
    ${SBATCH} \
        --export=ALL,TASK=${TASK},MODEL=meta-llama/Llama-3.1-8B-Instruct,CONFIG=${CONFIG},FEWSHOT_SET=${FS} \
        "${SLURM_DIR}/run_vp_llm_vllm.slurm"
    n=$((n+1))
done

echo "Submitted ${n} VP LLM zero-shot jobs"
