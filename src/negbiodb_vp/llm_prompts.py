"""Prompt templates for VP LLM benchmark tasks VP-L1 through VP-L4.

Each task has zero-shot and 3-shot templates.
3-shot: 3 independent example sets (fewshot_set=0,1,2) for variance reporting.

Mirrors src/negbiodb_ppi/llm_prompts.py structure from PPI domain.
"""

from __future__ import annotations

import json

VP_SYSTEM_PROMPT = (
    "You are a clinical geneticist with expertise in variant pathogenicity "
    "classification, ACMG/AMP guidelines, population genetics, and computational "
    "variant effect prediction. Provide precise, evidence-based answers."
)

# ── VP-L1: 4-way variant classification MCQ ──────────────────────────────

VP_L1_CATEGORIES = {
    "A": "Pathogenic - variant causes disease through established mechanism",
    "B": "Likely benign - evidence suggests variant does not cause disease",
    "C": "Uncertain significance (VUS) - insufficient evidence to classify",
    "D": "Benign - strong evidence variant does not cause disease",
}

# Few-shot set seeds (3 independent sets for variance)
FEWSHOT_SEEDS = [42, 43, 44]


VP_L1_QUESTION = (
    "Based on the variant information below, classify its pathogenicity.\n\n"
    "{context_text}\n\n"
    "Categories:\n"
    "A) Pathogenic - variant causes disease through established mechanism\n"
    "B) Likely benign - evidence suggests variant does not cause disease\n"
    "C) Uncertain significance (VUS) - insufficient evidence to classify\n"
    "D) Benign - strong evidence variant does not cause disease\n"
)

VP_L1_ANSWER_FORMAT = "Respond with ONLY a single letter (A, B, C, or D)."

VP_L1_FEW_SHOT = """Here are examples of variant pathogenicity classification:

{examples}

Now classify the following:

{context_text}

Categories:
A) Pathogenic - variant causes disease through established mechanism
B) Likely benign - evidence suggests variant does not cause disease
C) Uncertain significance (VUS) - insufficient evidence to classify
D) Benign - strong evidence variant does not cause disease
"""


# ── VP-L2: Structured variant interpretation extraction ─────────────────

VP_L2_QUESTION = (
    "Extract all variant pathogenicity assessments from the following clinical "
    "genetics report. Return a JSON object with the fields specified below.\n\n"
    "{context_text}\n\n"
    "Required JSON fields:\n"
    "- variants: list of objects, each with:\n"
    "    - gene: gene symbol\n"
    "    - hgvs: HGVS notation (coding or protein)\n"
    "    - classification: one of [pathogenic, likely_pathogenic, uncertain_significance, "
    "likely_benign, benign]\n"
    "    - acmg_criteria_met: list of ACMG criteria codes (e.g., [\"BA1\", \"BS1\", \"BP4\"])\n"
    "    - population_frequency: gnomAD global allele frequency (number or null)\n"
    "    - condition: associated disease/condition name\n"
    "- total_variants_discussed: total number of variants discussed in the report\n"
    "- classification_method: classification framework used (e.g., \"ACMG/AMP\")\n\n"
    "Return ONLY valid JSON, no additional text."
)

VP_L2_FEW_SHOT = """Extract variant pathogenicity assessments from clinical genetics reports.

{examples}

Now extract from this report:

{context_text}

Return ONLY valid JSON, no additional text."""


# ── VP-L3: Benign variant reasoning ────────────────────────────────────

VP_L3_QUESTION = (
    "The following DNA variant has been classified as benign or likely benign by "
    "clinical laboratories. Based on the variant and gene information below, "
    "provide a scientific explanation for why this variant is unlikely to cause "
    "disease.\n\n"
    "{context_text}\n\n"
    "Your explanation should address:\n"
    "1. Population reasoning - Interpret allele frequency evidence (BA1/BS1 "
    "thresholds, population-specific patterns)\n"
    "2. Computational evidence - Assess in silico predictor results (CADD, REVEL, "
    "AlphaMissense, conservation scores)\n"
    "3. Functional reasoning - Explain protein function impact, domain context, "
    "biochemical consequence of amino acid change\n"
    "4. Gene-disease specificity - Discuss gene-disease relationship, inheritance "
    "pattern, and why variant doesn't disrupt disease mechanism\n\n"
    "Provide a thorough explanation in 3-5 paragraphs."
)

VP_L3_FEW_SHOT = """Here are examples of scientific reasoning about benign variant classification:

{examples}

Now explain the following:

{context_text}"""


# ── VP-L4: Tested vs Untested Discrimination ───────────────────────────

VP_L4_QUESTION = (
    "Based on your knowledge of variant pathogenicity databases (ClinVar, gnomAD) "
    "and clinical genetics studies, determine whether the following variant-disease "
    "pair has ever been assessed for pathogenicity.\n\n"
    "{context_text}\n\n"
    "On the first line, respond with ONLY 'tested' or 'untested'.\n"
    "On the second line, provide brief evidence for your answer (e.g., ClinVar "
    "submissions, clinical laboratory reports, population data, or reasoning for "
    "why it was/wasn't assessed)."
)

VP_L4_ANSWER_FORMAT = (
    "On the first line, respond with ONLY 'tested' or 'untested'. "
    "On the second line, provide brief evidence."
)

VP_L4_FEW_SHOT = """Here are examples of tested/untested variant-disease pair determination:

{examples}

Now determine:

{context_text}"""


# ── Helper functions ────────────────────────────────────────────────────


def format_vp_l1_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format VP-L1 MCQ prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = VP_L1_QUESTION.format(context_text=context) + "\n" + VP_L1_ANSWER_FORMAT
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\nAnswer: {ex['gold_answer']}"
            for ex in fewshot_examples
        )
        user = (
            VP_L1_FEW_SHOT.format(examples=examples_text, context_text=context)
            + "\n"
            + VP_L1_ANSWER_FORMAT
        )

    return VP_SYSTEM_PROMPT, user


def format_vp_l2_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format VP-L2 extraction prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = VP_L2_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\n\nExtracted:\n{json.dumps(ex.get('gold_extraction', {}), indent=2)}"
            for ex in fewshot_examples
        )
        user = VP_L2_FEW_SHOT.format(examples=examples_text, context_text=context)

    return VP_SYSTEM_PROMPT, user


_L3_MAX_EXAMPLE_CHARS = 1200  # ~300 tokens per example context
_L3_MAX_REASONING_CHARS = 600  # ~150 tokens per example reasoning


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text at word boundary, appending '[...]' if truncated."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + " [...]"


def format_vp_l3_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format VP-L3 reasoning prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = VP_L3_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{_truncate_text(ex['context_text'], _L3_MAX_EXAMPLE_CHARS)}\n\n"
            f"Explanation:\n{_truncate_text(ex.get('gold_reasoning', 'N/A'), _L3_MAX_REASONING_CHARS)}"
            for ex in fewshot_examples
        )
        user = VP_L3_FEW_SHOT.format(examples=examples_text, context_text=context)

    return VP_SYSTEM_PROMPT, user


def format_vp_l4_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format VP-L4 tested/untested prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = VP_L4_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\nAnswer: {ex['gold_answer']}"
            for ex in fewshot_examples
        )
        user = (
            VP_L4_FEW_SHOT.format(examples=examples_text, context_text=context)
            + "\n"
            + VP_L4_ANSWER_FORMAT
        )

    return VP_SYSTEM_PROMPT, user


VP_TASK_FORMATTERS = {
    "vp-l1": format_vp_l1_prompt,
    "vp-l2": format_vp_l2_prompt,
    "vp-l3": format_vp_l3_prompt,
    "vp-l4": format_vp_l4_prompt,
}


def format_vp_prompt(
    task: str,
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Dispatch to task-specific formatter.

    Args:
        task: 'vp-l1', 'vp-l2', 'vp-l3', or 'vp-l4'
        record: dict with at least 'context_text' key
        config: 'zero-shot' or '3-shot'
        fewshot_examples: list of example dicts for 3-shot

    Returns:
        (system_prompt, user_prompt) tuple
    """
    formatter = VP_TASK_FORMATTERS.get(task)
    if formatter is None:
        raise ValueError(f"Unknown task: {task}. Choose from {list(VP_TASK_FORMATTERS)}")
    return formatter(record, config, fewshot_examples)
