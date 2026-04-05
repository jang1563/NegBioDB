#!/bin/bash
#SBATCH --job-name=rl_baseline
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rl_baseline_%j.log
#SBATCH --error=logs/rl_baseline_%j.err

# Run Qwen3-8B baseline (no fine-tuning) on all domains for before/after comparison.

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
EVAL_DOMAINS="${EVAL_DOMAINS:-dti ct ppi ge vp}"

SCRATCH="/athena/masonlab/scratch/users/jak4013"
CONDA_ENV="${SCRATCH}/conda_env/negbiorl-train"
OUTPUT_DIR="results/negbiorl/phase4_eval/baseline"

set -euo pipefail
mkdir -p logs "${OUTPUT_DIR}"

echo "=== NegBioRL Baseline Evaluation ==="
echo "Model: ${BASE_MODEL}"
echo "Domains: ${EVAL_DOMAINS}"

source /home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PYTHONPATH=src:${PYTHONPATH:-}

# Use vLLM for fast inference (no adapter)
python -c "
import json
from pathlib import Path
from vllm import LLM, SamplingParams
from negbiorl.data_registry import get_export_path, get_gold_answer_field, load_jsonl
from negbiorl.sft_data import _format_prompt

llm = LLM(model='${BASE_MODEL}', trust_remote_code=True)
params = SamplingParams(temperature=0.0, max_tokens=512)
output_dir = Path('${OUTPUT_DIR}')

for domain in '${EVAL_DOMAINS}'.split():
    for task in ['l1', 'l4']:
        try:
            export_path = get_export_path(domain, task)
        except (ValueError, FileNotFoundError):
            continue
        if not export_path.exists():
            continue
        records = [r for r in load_jsonl(export_path) if r.get('split') == 'test']
        if not records:
            continue
        gold_field = get_gold_answer_field(domain)
        prompts, golds, qids = [], [], []
        for rec in records:
            s, u = _format_prompt(domain, task, rec)
            prompts.append(f'{s}\n\nUser: {u}\n\nAssistant:')
            golds.append(rec.get(gold_field, ''))
            qids.append(rec.get('question_id', ''))
        print(f'  {domain}/{task}: {len(prompts)} prompts')
        outputs = llm.generate(prompts, params)
        out_path = output_dir / f'{domain}_{task}_predictions.jsonl'
        with open(out_path, 'w') as f:
            for qid, gold, out in zip(qids, golds, outputs):
                f.write(json.dumps({'question_id': qid, 'prediction': out.outputs[0].text.strip(), 'gold_answer': gold}) + '\n')
print('Done.')
"

echo "=== Baseline saved to ${OUTPUT_DIR} ==="
