"""Domain → path mapping, field name handling, and eval module resolution.

Central registry that abstracts away cross-domain differences so GRPO reward
functions, SFT data builders, and evaluation pipelines can operate generically.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable

# Project root — one level up from src/negbiorl/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Domain Registry
# ---------------------------------------------------------------------------

DOMAIN_REGISTRY: dict[str, dict[str, Any]] = {
    "dti": {
        "label": "Drug-Target Interaction",
        "eval_module": "negbiodb.llm_eval",
        "prompt_module": "negbiodb.llm_prompts",
        "task_prefix": "",                        # DTI uses bare "l1", "l4"
        "exports_dir": "exports/llm_benchmarks",
        "results_dir": "results/llm",
        "gold_answer_field": "correct_answer",   # DTI uses 'correct_answer'
        "gold_class_field": "class",              # DTI uses 'class'
        "l1_choices": 4,                          # A/B/C/D
        "l1_file": "l1_mcq.jsonl",
        "l2_file": "l2_candidates.jsonl",
        "l3_file": "l3_reasoning_pilot.jsonl",
        "l4_file": "l4_tested_untested.jsonl",
        "format_prompt": "format_prompt",
        "parse_l1": "parse_l1_answer",
        "parse_l4": "parse_l4_answer",
        "l4_returns_tuple": True,                 # (answer, evidence)
        "temporal_groups": ["pre_2023", "post_2024"],
        "evidence_keywords": [
            "ChEMBL", "PubChem", "BindingDB", "IC50", "Ki", "Kd",
            "pChEMBL", "bioassay", "PMID", "DOI",
        ],
        "l3_judge_dims": ["accuracy", "reasoning", "completeness", "specificity"],
    },
    "ct": {
        "label": "Clinical Trial Failure",
        "eval_module": "negbiodb_ct.llm_eval",
        "prompt_module": "negbiodb_ct.llm_prompts",
        "task_prefix": "ct-",                    # CT uses "ct-l1", "ct-l4"
        "exports_dir": "exports/ct_llm",
        "results_dir": "results/ct_llm",
        "gold_answer_field": "gold_answer",
        "gold_class_field": "gold_category",
        "l1_choices": 5,                          # A/B/C/D/E
        "l1_file": "ct_l1_dataset.jsonl",
        "l2_file": "ct_l2_dataset.jsonl",
        "l3_file": "ct_l3_dataset.jsonl",
        "l4_file": "ct_l4_dataset.jsonl",
        "format_prompt": "format_ct_prompt",
        "parse_l1": "parse_ct_l1_answer",
        "parse_l4": "parse_ct_l4_answer",
        "l4_returns_tuple": True,
        "temporal_groups": ["pre_2020", "post_2023"],
        "evidence_keywords": [
            "NCT", "clinicaltrials.gov", "Phase", "trial",
            "FDA", "EMA", "DSMB", "interim analysis",
        ],
        "l3_judge_dims": ["accuracy", "reasoning", "completeness", "specificity"],
    },
    "ppi": {
        "label": "Protein-Protein Interaction",
        "eval_module": "negbiodb_ppi.llm_eval",
        "prompt_module": "negbiodb_ppi.llm_prompts",
        "task_prefix": "ppi-",                   # PPI uses "ppi-l1", "ppi-l4"
        "exports_dir": "exports/ppi_llm",
        "results_dir": "results/ppi_llm",
        "gold_answer_field": "gold_answer",
        "gold_class_field": "gold_category",
        "l1_choices": 4,
        "l1_file": "ppi_l1_dataset.jsonl",
        "l2_file": "ppi_l2_dataset.jsonl",
        "l3_file": "ppi_l3_dataset.jsonl",
        "l4_file": "ppi_l4_dataset.jsonl",
        "format_prompt": "format_ppi_prompt",
        "parse_l1": "parse_ppi_l1_answer",
        "parse_l4": "parse_ppi_l4_answer",
        "l4_returns_tuple": True,
        "temporal_groups": ["pre_2015", "post_2020"],
        "evidence_keywords": [
            "UniProt", "HuRI", "BioGRID", "IntAct", "STRING",
            "co-IP", "yeast two-hybrid", "co-fractionation", "PMID",
        ],
        "l3_judge_dims": [
            "biological_plausibility", "structural_reasoning",
            "mechanistic_completeness", "specificity",
        ],
    },
    "ge": {
        "label": "Gene Essentiality",
        "eval_module": "negbiodb_depmap.llm_eval",
        "prompt_module": "negbiodb_depmap.llm_prompts",
        "task_prefix": "ge-",                    # GE uses "ge-l1", "ge-l4"
        "exports_dir": "exports/ge_llm",
        "results_dir": "results/ge_llm",
        "gold_answer_field": "gold_answer",
        "gold_class_field": "gold_category",
        "l1_choices": 4,
        "l1_file": "ge_l1_dataset.jsonl",
        "l2_file": "ge_l2_dataset.jsonl",
        "l3_file": "ge_l3_dataset.jsonl",
        "l4_file": "ge_l4_dataset.jsonl",
        "format_prompt": "format_ge_prompt",
        "parse_l1": "parse_ge_l1_answer",
        "parse_l4": "parse_ge_l4_answer",
        "l4_returns_tuple": False,                # returns str | None only
        "temporal_groups": ["old_release", "new_release"],
        "evidence_keywords": [
            "DepMap", "CRISPR", "RNAi", "Chronos", "DEMETER",
            "gene effect", "dependency score", "cell line",
        ],
        "l3_judge_dims": [
            "biological_plausibility", "pathway_reasoning",
            "context_specificity", "mechanistic_depth",
        ],
    },
    "dc": {
        "label": "Drug Combination Synergy",
        "eval_module": "negbiodb_dc.llm_eval",
        "prompt_module": "negbiodb_dc.llm_prompts",
        "task_prefix": "dc-",
        "exports_dir": "exports/dc_llm",
        "results_dir": "results/dc_llm",
        "gold_answer_field": "gold_answer",
        "gold_class_field": "gold_category",
        "l1_choices": 4,
        "l1_file": "dc_l1_dataset.jsonl",
        "l2_file": "dc_l2_dataset.jsonl",
        "l3_file": "dc_l3_dataset.jsonl",
        "l4_file": "dc_l4_dataset.jsonl",
        "format_prompt": "format_dc_prompt",
        "parse_l1": "parse_dc_l1_answer",
        "parse_l4": "parse_dc_l4_answer",
        "l4_returns_tuple": True,
        "temporal_groups": [],
        "evidence_keywords": [
            "DrugComb", "SynergyFinder", "Bliss", "Loewe", "HSA", "ZIP",
            "synergy score", "antagonism", "combination index", "PMID",
        ],
        "l3_judge_dims": [
            "mechanistic_reasoning", "pathway_analysis",
            "pharmacological_context", "therapeutic_relevance",
        ],
    },
    "vp": {
        "label": "Variant Pathogenicity",
        "eval_module": "negbiodb_vp.llm_eval",
        "prompt_module": "negbiodb_vp.llm_prompts",
        "task_prefix": "vp-",                    # VP uses "vp-l1", "vp-l4"
        "exports_dir": "exports/vp_llm",
        "results_dir": "results/vp_llm",
        "gold_answer_field": "gold_answer",
        "gold_class_field": "gold_category",
        "l1_choices": 4,
        "l1_file": "vp_l1_dataset.jsonl",
        "l2_file": "vp_l2_dataset.jsonl",
        "l3_file": "vp_l3_dataset.jsonl",
        "l4_file": "vp_l4_dataset.jsonl",
        "format_prompt": "format_vp_prompt",
        "parse_l1": "parse_vp_l1_answer",
        "parse_l4": "parse_vp_l4_answer",
        "l4_returns_tuple": True,
        "temporal_groups": ["pre_2020", "post_2023"],
        "evidence_keywords": [
            "ClinVar", "gnomAD", "CADD", "REVEL", "AlphaMissense",
            "ACMG", "PVS1", "PM2", "PP3", "BA1", "ClinGen",
        ],
        "l3_judge_dims": [
            "population_reasoning", "computational_evidence",
            "functional_reasoning", "gene_disease_specificity",
        ],
    },
}

# Domains used for training (VP reserved for zero-shot transfer test)
TRAIN_DOMAINS = ["dti", "ct", "ppi", "ge", "dc"]
TRANSFER_TEST_DOMAIN = "vp"
ALL_DOMAINS = list(DOMAIN_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Accessor helpers
# ---------------------------------------------------------------------------

def get_domain(domain: str) -> dict[str, Any]:
    """Return registry entry for a domain, raising KeyError if unknown."""
    if domain not in DOMAIN_REGISTRY:
        raise KeyError(
            f"Unknown domain '{domain}'. Choose from: {ALL_DOMAINS}"
        )
    return DOMAIN_REGISTRY[domain]


def get_export_path(domain: str, task: str) -> Path:
    """Return absolute path to a domain's JSONL export for a given task level."""
    reg = get_domain(domain)
    filename = reg.get(f"{task}_file")
    if filename is None:
        raise ValueError(f"No export file registered for domain={domain}, task={task}")
    return PROJECT_ROOT / reg["exports_dir"] / filename


def get_results_dir(domain: str) -> Path:
    """Return absolute path to a domain's results directory."""
    return PROJECT_ROOT / get_domain(domain)["results_dir"]


def get_gold_answer_field(domain: str) -> str:
    """Return the field name for gold answers (handles DTI 'correct_answer' quirk)."""
    return get_domain(domain)["gold_answer_field"]


def get_gold_class_field(domain: str) -> str:
    """Return the field name for gold class/category."""
    return get_domain(domain)["gold_class_field"]


def get_prefixed_task(domain: str, task: str) -> str:
    """Return the domain-prefixed task name for prompt formatters.

    DTI uses bare "l1", others use "ct-l1", "ppi-l1", etc.
    """
    prefix = get_domain(domain).get("task_prefix", "")
    return f"{prefix}{task}"


# ---------------------------------------------------------------------------
# Dynamic parser loading
# ---------------------------------------------------------------------------

def _import_function(module_path: str, func_name: str) -> Callable:
    """Import a function from a module path string."""
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def get_l1_parser(domain: str) -> Callable[[str], str | None]:
    """Return the L1 answer parser for a domain."""
    reg = get_domain(domain)
    return _import_function(reg["eval_module"], reg["parse_l1"])


def get_l4_parser(domain: str) -> Callable:
    """Return the L4 answer parser for a domain.

    Note: DTI/CT/PPI/VP return tuple[str|None, str|None],
    but GE returns str|None. Check reg['l4_returns_tuple'].
    """
    reg = get_domain(domain)
    return _import_function(reg["eval_module"], reg["parse_l4"])


def parse_l4_unified(
    response: str, domain: str
) -> tuple[str | None, str | None]:
    """Parse L4 response, normalizing GE's non-tuple return to tuple format."""
    reg = get_domain(domain)
    parser = get_l4_parser(domain)
    result = parser(response)
    if reg["l4_returns_tuple"]:
        return result  # already (answer, evidence)
    # GE returns str|None → wrap as (answer, None)
    return (result, None)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_export(domain: str, task: str) -> list[dict]:
    """Load a domain's LLM benchmark export for a given task level."""
    path = get_export_path(domain, task)
    if not path.exists():
        raise FileNotFoundError(
            f"Export not found: {path}. Run the domain's LLM dataset builder first."
        )
    return load_jsonl(path)


def load_predictions(
    domain: str, model: str, task: str, fewshot: str = "fs0",
    shot_config: str = "zero-shot",
) -> list[dict]:
    """Load model predictions for a domain/task/fewshot configuration.

    Actual file naming conventions:
    - DTI: results/llm/{task}_{full_model}_{shot_config}_{seed}/predictions.jsonl
    - CT:  results/ct_llm/ct-{task}_{full_model}_{shot_config}_{seed}/predictions.jsonl
    - PPI: results/ppi_llm/ppi-{task}_{full_model}_{shot_config}_{seed}/predictions.jsonl
    - GE:  results/ge_llm/ge-{task}_{full_model}_{shot_config}_{seed}/predictions.jsonl

    Args:
        domain: Domain key
        model: Short model name (haiku, gemini, gpt, qwen, llama)
        task: Task level (l1, l2, l3, l4)
        fewshot: Seed identifier (fs0, fs1, fs2)
        shot_config: Shot configuration (zero-shot, 3-shot)
    """
    results_dir = get_results_dir(domain)

    # Domain prefix in directory names
    prefix_map = {"dti": "", "ct": "ct-", "ppi": "ppi-", "ge": "ge-", "dc": "dc-", "vp": "vp-"}
    prefix = prefix_map.get(domain, "")

    # Try exact pattern first: {prefix}{task}_{model-substring}_{shot}_{seed}/predictions.jsonl
    pattern = f"{prefix}{task}_*{model}*_{shot_config}_{fewshot}"
    matches = sorted(results_dir.glob(f"{pattern}/predictions.jsonl"))
    if matches:
        # If multiple matches (e.g. gemini-2-5-flash vs gemini-2-5-flash-lite),
        # prefer the shorter directory name (more specific model)
        if len(matches) > 1:
            matches.sort(key=lambda p: len(p.parent.name))
        return load_jsonl(matches[0])

    # Broader glob fallback (task comes before model in directory names)
    matches = sorted(results_dir.glob(f"*{task}*{model}*{fewshot}*/predictions.jsonl"))
    if matches:
        return load_jsonl(matches[0])

    raise FileNotFoundError(
        f"No predictions found for domain={domain}, model={model}, "
        f"task={task}, shot={shot_config}, seed={fewshot} in {results_dir}"
    )


# ---------------------------------------------------------------------------
# Constants for models
# ---------------------------------------------------------------------------

BENCHMARK_MODELS = [
    "haiku",    # claude-haiku-4-5
    "gemini",   # gemini-2.0-flash
    "gpt",      # gpt-4o-mini
    "qwen",     # Qwen2.5-7B-Instruct
    "llama",    # Llama-3.1-8B-Instruct
]

TRAINING_MODELS = {
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "role": "primary",
        "generation": "current",
    },
    "qwen25-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "role": "historical_baseline",
        "generation": "2024",
    },
    "llama31-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "role": "historical_baseline",
        "generation": "2024",
    },
}
