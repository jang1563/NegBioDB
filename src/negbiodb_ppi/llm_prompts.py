"""Prompt templates for PPI LLM benchmark tasks PPI-L1 through PPI-L4.

Each task has zero-shot and 3-shot templates.
3-shot: 3 independent example sets (fewshot_set=0,1,2) for variance reporting.

Mirrors src/negbiodb_ct/llm_prompts.py structure from CT domain.
"""

from __future__ import annotations

import json

PPI_SYSTEM_PROMPT = (
    "You are a protein biochemist with expertise in protein-protein interactions, "
    "structural biology, and proteomics experimental methods. "
    "Provide precise, evidence-based answers."
)

# ── PPI-L1: 4-way evidence quality classification ────────────────────────

PPI_L1_CATEGORIES = {
    "A": "Direct experimental — A specific binding assay (co-IP, pulldown, SPR, etc.) found no physical interaction",
    "B": "Systematic screen — A high-throughput binary screen (Y2H, LUMIER, etc.) found no interaction",
    "C": "Computational inference — ML analysis of co-fractionation or complex data predicts no interaction",
    "D": "Database score absence — Zero or negligible combined interaction score across multiple evidence channels",
}

# Few-shot set seeds (3 independent sets for variance)
FEWSHOT_SEEDS = [42, 43, 44]


PPI_L1_QUESTION = (
    "Based on the evidence description below, classify the type and quality of "
    "evidence supporting this protein non-interaction.\n\n"
    "{context_text}\n\n"
    "Categories:\n"
    "A) Direct experimental — A specific binding assay (co-IP, pulldown, SPR, etc.) found no physical interaction\n"
    "B) Systematic screen — A high-throughput binary screen (Y2H, LUMIER, etc.) found no interaction\n"
    "C) Computational inference — ML analysis of co-fractionation or complex data predicts no interaction\n"
    "D) Database score absence — Zero or negligible combined interaction score across multiple evidence channels\n"
)

PPI_L1_ANSWER_FORMAT = "Respond with ONLY a single letter (A, B, C, or D)."

PPI_L1_FEW_SHOT = """Here are examples of protein non-interaction evidence classification:

{examples}

Now classify the following:

{context_text}

Categories:
A) Direct experimental — A specific binding assay (co-IP, pulldown, SPR, etc.) found no physical interaction
B) Systematic screen — A high-throughput binary screen (Y2H, LUMIER, etc.) found no interaction
C) Computational inference — ML analysis of co-fractionation or complex data predicts no interaction
D) Database score absence — Zero or negligible combined interaction score across multiple evidence channels
"""


# ── PPI-L2: Non-interaction evidence extraction ─────────────────────────

PPI_L2_QUESTION = (
    "Extract all protein pairs reported as non-interacting from the following "
    "evidence summary. Return a JSON object with the fields specified below.\n\n"
    "{context_text}\n\n"
    "Required JSON fields:\n"
    "- non_interacting_pairs: list of objects, each with:\n"
    "    - protein_1: gene symbol or UniProt accession\n"
    "    - protein_2: gene symbol or UniProt accession\n"
    "    - method: experimental method used (e.g., 'co-immunoprecipitation', 'yeast two-hybrid')\n"
    "    - evidence_strength: one of [strong, moderate, weak]\n"
    "- total_negative_count: total number of non-interacting pairs mentioned\n"
    "- positive_interactions_mentioned: true if any positive interactions are also mentioned\n\n"
    "Return ONLY valid JSON, no additional text."
)

PPI_L2_FEW_SHOT = """Extract non-interacting protein pairs from evidence summaries.

{examples}

Now extract from this evidence summary:

{context_text}

Return ONLY valid JSON, no additional text."""


# ── PPI-L3: Non-interaction reasoning ───────────────────────────────────

PPI_L3_QUESTION = (
    "The following two proteins have been experimentally tested and confirmed to "
    "NOT physically interact. Based on the protein information below, provide a "
    "scientific explanation for why they are unlikely to form a physical interaction.\n\n"
    "{context_text}\n\n"
    "Your explanation should address:\n"
    "1. Biological plausibility — Are there biological reasons (function, pathway, "
    "localization) that make interaction unlikely?\n"
    "2. Structural reasoning — Do domain architectures, binding interfaces, or "
    "steric factors argue against interaction?\n"
    "3. Mechanistic completeness — Are multiple relevant factors considered "
    "(expression timing, tissue specificity, post-translational regulation)?\n"
    "4. Specificity — Are claims specific to these proteins or generic statements?\n\n"
    "Provide a thorough explanation in 3-5 paragraphs."
)

PPI_L3_FEW_SHOT = """Here are examples of scientific reasoning about protein non-interactions:

{examples}

Now explain the following:

{context_text}"""


# ── PPI-L4: Tested vs Untested Discrimination ───────────────────────────

PPI_L4_QUESTION = (
    "Based on your knowledge of protein-protein interaction databases and "
    "experimental studies, determine whether the following protein pair has ever "
    "been experimentally tested for physical interaction.\n\n"
    "{context_text}\n\n"
    "On the first line, respond with ONLY 'tested' or 'untested'.\n"
    "On the second line, provide brief evidence for your answer (e.g., database "
    "sources, experimental methods, or reasoning for why it was/wasn't tested)."
)

PPI_L4_ANSWER_FORMAT = (
    "On the first line, respond with ONLY 'tested' or 'untested'. "
    "On the second line, provide brief evidence."
)

PPI_L4_FEW_SHOT = """Here are examples of tested/untested protein pair determination:

{examples}

Now determine:

{context_text}"""


# ── Helper functions ────────────────────────────────────────────────────


def format_ppi_l1_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format PPI-L1 MCQ prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = PPI_L1_QUESTION.format(context_text=context) + "\n" + PPI_L1_ANSWER_FORMAT
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\nAnswer: {ex['gold_answer']}"
            for ex in fewshot_examples
        )
        user = (
            PPI_L1_FEW_SHOT.format(examples=examples_text, context_text=context)
            + "\n"
            + PPI_L1_ANSWER_FORMAT
        )

    return PPI_SYSTEM_PROMPT, user


def format_ppi_l2_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format PPI-L2 extraction prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = PPI_L2_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\n\nExtracted:\n{json.dumps(ex.get('gold_extraction', {}), indent=2)}"
            for ex in fewshot_examples
        )
        user = PPI_L2_FEW_SHOT.format(examples=examples_text, context_text=context)

    return PPI_SYSTEM_PROMPT, user


def format_ppi_l3_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format PPI-L3 reasoning prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = PPI_L3_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\n\nExplanation:\n{ex.get('gold_reasoning', 'N/A')}"
            for ex in fewshot_examples
        )
        user = PPI_L3_FEW_SHOT.format(examples=examples_text, context_text=context)

    return PPI_SYSTEM_PROMPT, user


def format_ppi_l4_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format PPI-L4 tested/untested prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = PPI_L4_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\nAnswer: {ex['gold_answer']}"
            for ex in fewshot_examples
        )
        user = (
            PPI_L4_FEW_SHOT.format(examples=examples_text, context_text=context)
            + "\n"
            + PPI_L4_ANSWER_FORMAT
        )

    return PPI_SYSTEM_PROMPT, user


PPI_TASK_FORMATTERS = {
    "ppi-l1": format_ppi_l1_prompt,
    "ppi-l2": format_ppi_l2_prompt,
    "ppi-l3": format_ppi_l3_prompt,
    "ppi-l4": format_ppi_l4_prompt,
}


def format_ppi_prompt(
    task: str,
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Dispatch to task-specific formatter.

    Args:
        task: 'ppi-l1', 'ppi-l2', 'ppi-l3', or 'ppi-l4'
        record: dict with at least 'context_text' key
        config: 'zero-shot' or '3-shot'
        fewshot_examples: list of example dicts for 3-shot

    Returns:
        (system_prompt, user_prompt) tuple
    """
    formatter = PPI_TASK_FORMATTERS.get(task)
    if formatter is None:
        raise ValueError(f"Unknown task: {task}. Choose from {list(PPI_TASK_FORMATTERS)}")
    return formatter(record, config, fewshot_examples)
