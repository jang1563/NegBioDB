#!/bin/bash
#SBATCH --job-name=md_l3_judge
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_l3_judge_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_l3_judge_%j.err

# MD L3 judge: score free-text reasoning responses via LLM-as-judge
# Usage: sbatch --export=ALL,RESPONSES_FILE=md_l3_responses_gemini_0shot.jsonl,JUDGE=gemini-2.5-flash slurm/run_md_l3_judge.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

RESPONSES_FILE="${RESPONSES_FILE:-md_l3_responses_claude-sonnet-4-6_0shot.jsonl}"
JUDGE="${JUDGE:-gemini-2.5-flash}"
RESULTS_DIR="${PROJECT_DIR}/results/md_llm"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

if [[ -f "${HOME}/.api_keys" ]]; then
    source "${HOME}/.api_keys"
fi

echo "=== MD L3 Judge ==="
echo "Responses: ${RESPONSES_FILE}, Judge: ${JUDGE}"
echo "Start: $(date)"

python - <<PYEOF
import json, sys, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
project_dir = Path("${PROJECT_DIR}")
sys.path.insert(0, str(project_dir / "src"))

from negbiodb_md.llm_eval import L3_AXES, save_results
from negbiodb_md.llm_dataset import load_md_candidate_pool
from negbiodb.llm_client import call_llm

responses_path = Path("${RESULTS_DIR}") / "${RESPONSES_FILE}"
if not responses_path.exists():
    print(f"Responses file not found: {responses_path}")
    sys.exit(1)

dataset_path = project_dir / "exports" / "md_llm" / "md_l3.jsonl"
records = [json.loads(l) for l in open(dataset_path) if l.strip()]
resp_by_id = {}
for line in open(responses_path):
    d = json.loads(line.strip())
    resp_by_id[d["record_id"]] = d["response"]

JUDGE_PROMPT = '''You are a scientific evaluator assessing the quality of reasoning about null metabolomics findings.

Score the following response on 4 axes (0-5 each):
- metabolite_biology: Accuracy of biochemical role and metabolic pathway description
- disease_mechanism: Accuracy of disease pathophysiology explanation
- study_context: Reference to study limitations (platform, biofluid, sample size)
- alternative_hypothesis: Quality of testable alternative explanations

Context: {context}
Question: {question}
Gold reasoning: {gold_reasoning}
Model response: {response}

Return JSON: {{"metabolite_biology": N, "disease_mechanism": N, "study_context": N, "alternative_hypothesis": N}}'''

judge_scores = []
for rec in records:
    rid = rec["record_id"]
    resp = resp_by_id.get(rid, "")
    prompt = JUDGE_PROMPT.format(
        context=rec.get("context", ""),
        question=rec.get("question", ""),
        gold_reasoning=rec.get("gold_reasoning", ""),
        response=resp,
    )
    raw = call_llm(model="${JUDGE}", system="You are a scientific evaluator.", user=prompt)
    try:
        import re
        m = re.search(r'\{.*\}', raw or "", re.DOTALL)
        scores = json.loads(m.group()) if m else {}
    except Exception:
        scores = {}
    judge_scores.append(scores)

# Compute metrics
from negbiodb_md.llm_eval import eval_l3_with_judge
metrics = eval_l3_with_judge(records, list(resp_by_id.values()), judge_scores)
print(f"MD-L3 judge results: {json.dumps(metrics, indent=2)}")

model_name = "${RESPONSES_FILE}".replace("md_l3_responses_", "").replace(".jsonl", "")
out_path = Path("${RESULTS_DIR}") / f"md_l3_{model_name}_judged.json"
save_results({"judge": "${JUDGE}", "model": model_name, "metrics": metrics}, out_path)
PYEOF

echo "End: $(date)"
