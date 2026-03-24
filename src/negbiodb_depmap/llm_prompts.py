"""Prompt templates for GE LLM benchmark tasks GE-L1 through GE-L4.

Each task has zero-shot and 3-shot templates.
3-shot: 3 independent example sets (fewshot_set=0,1,2) for variance reporting.

Mirrors src/negbiodb_ppi/llm_prompts.py structure.
"""

from __future__ import annotations

import json

GE_SYSTEM_PROMPT = (
    "You are a cancer biologist with expertise in genetic dependencies, "
    "CRISPR screening, and functional genomics. "
    "Provide precise, evidence-based answers."
)

# ── GE-L1: 4-way essentiality classification ─────────────────────────────

GE_L1_CATEGORIES = {
    "A": "Common essential — Required for viability in nearly all cell types",
    "B": "Selective essential — Required specifically in this lineage/context",
    "C": "Non-essential — Knockout has no significant effect on viability",
    "D": "Unknown/Untested — Insufficient data to determine essentiality",
}

FEWSHOT_SEEDS = [42, 43, 44]

GE_L1_QUESTION = (
    "Classify the essentiality status of the gene described below in the "
    "given cell line context.\n\n"
    "{context_text}\n\n"
    "Categories:\n"
    "A) Common essential — Required for viability in nearly all cell types\n"
    "B) Selective essential — Required specifically in this lineage/context\n"
    "C) Non-essential — Knockout has no significant effect on viability\n"
    "D) Unknown/Untested — Insufficient data to determine essentiality\n"
)

GE_L1_ANSWER_FORMAT = "Respond with ONLY a single letter (A, B, C, or D)."

GE_L1_FEW_SHOT = """Here are examples of gene essentiality classification:

{examples}

Now classify the following:

{context_text}

Categories:
A) Common essential — Required for viability in nearly all cell types
B) Selective essential — Required specifically in this lineage/context
C) Non-essential — Knockout has no significant effect on viability
D) Unknown/Untested — Insufficient data to determine essentiality
"""


# ── GE-L2: Essentiality information extraction ───────────────────────────

GE_L2_QUESTION = (
    "Extract gene essentiality information from the following text. "
    "Return a JSON object with the fields specified below.\n\n"
    "{context_text}\n\n"
    "Required JSON fields:\n"
    "- genes: list of objects, each with:\n"
    "    - gene_name: gene symbol\n"
    "    - cell_line_or_tissue: cell line name or tissue type\n"
    "    - screen_method: experimental method (e.g., 'CRISPR-Cas9', 'RNAi')\n"
    "    - essentiality_status: one of [essential, non-essential, context-dependent]\n"
    "    - dependency_score: numeric score if mentioned, null otherwise\n"
    "- total_genes_mentioned: total number of genes discussed\n"
    "- screen_type: primary screening technology used\n\n"
    "Return ONLY valid JSON, no additional text."
)

GE_L2_FEW_SHOT = """Extract gene essentiality information from evidence descriptions.

{examples}

Now extract from this text:

{context_text}

Return ONLY valid JSON, no additional text."""


# ── GE-L3: Non-essentiality reasoning ────────────────────────────────────

GE_L3_QUESTION = (
    "Gene {gene_symbol} has been confirmed as NON-ESSENTIAL in {cell_line} "
    "({lineage}). CRISPR knockout of this gene has no significant effect on "
    "cell viability in this context.\n\n"
    "{context_text}\n\n"
    "Explain why this gene's knockout has no significant effect on viability "
    "in this context. Your explanation should address:\n"
    "1. Biological plausibility — Are there biological reasons (function, "
    "pathway, tissue specificity) that make essentiality unlikely?\n"
    "2. Pathway reasoning — Are redundant pathways or paralog compensation "
    "mechanisms available?\n"
    "3. Context specificity — Why might this gene be non-essential specifically "
    "in this lineage/cell type?\n"
    "4. Mechanistic depth — What molecular mechanisms explain the lack of "
    "dependency?\n\n"
    "Provide a thorough explanation in 3-5 paragraphs."
)

GE_L3_FEW_SHOT = """Here are examples of scientific reasoning about gene non-essentiality:

{examples}

Now explain the following:

{context_text}"""


# ── GE-L4: Tested vs Untested Discrimination ─────────────────────────────

GE_L4_QUESTION = (
    "Based on your knowledge of cancer dependency screening projects "
    "(DepMap, Project SCORE), determine whether the following gene has been "
    "experimentally tested for essentiality via CRISPR knockout in the "
    "specified cell line.\n\n"
    "{context_text}\n\n"
    "On the first line, respond with ONLY 'tested' or 'untested'.\n"
    "On the second line, provide brief evidence for your answer."
)

GE_L4_ANSWER_FORMAT = (
    "On the first line, respond with ONLY 'tested' or 'untested'. "
    "On the second line, provide brief evidence."
)

GE_L4_FEW_SHOT = """Here are examples of tested/untested gene-cell line determination:

{examples}

Now determine:

{context_text}"""


# ── Helper functions ──────────────────────────────────────────────────────

_L3_MAX_EXAMPLE_CHARS = 1200
_L3_MAX_REASONING_CHARS = 600


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text at word boundary, appending '[...]' if truncated."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + " [...]"


def format_ge_l1_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format GE-L1 MCQ prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = GE_L1_QUESTION.format(context_text=context) + "\n" + GE_L1_ANSWER_FORMAT
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\nAnswer: {ex['gold_answer']}"
            for ex in fewshot_examples
        )
        user = (
            GE_L1_FEW_SHOT.format(examples=examples_text, context_text=context)
            + "\n"
            + GE_L1_ANSWER_FORMAT
        )

    return GE_SYSTEM_PROMPT, user


def format_ge_l2_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format GE-L2 extraction prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = GE_L2_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\n\nExtracted:\n{json.dumps(ex.get('gold_extraction', {}), indent=2)}"
            for ex in fewshot_examples
        )
        user = GE_L2_FEW_SHOT.format(examples=examples_text, context_text=context)

    return GE_SYSTEM_PROMPT, user


def format_ge_l3_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format GE-L3 reasoning prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]
    gene_symbol = record.get("gene_symbol", "UNKNOWN")
    cell_line = record.get("cell_line", "UNKNOWN")
    lineage = record.get("lineage", "UNKNOWN")

    if config == "zero-shot" or not fewshot_examples:
        user = GE_L3_QUESTION.format(
            gene_symbol=gene_symbol,
            cell_line=cell_line,
            lineage=lineage,
            context_text=context,
        )
    else:
        examples_text = "\n\n---\n\n".join(
            f"{_truncate_text(ex['context_text'], _L3_MAX_EXAMPLE_CHARS)}\n\n"
            f"Explanation:\n{_truncate_text(ex.get('gold_reasoning', 'N/A'), _L3_MAX_REASONING_CHARS)}"
            for ex in fewshot_examples
        )
        user = GE_L3_FEW_SHOT.format(examples=examples_text, context_text=context)

    return GE_SYSTEM_PROMPT, user


def format_ge_l4_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format GE-L4 tested/untested prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = GE_L4_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\nAnswer: {ex['gold_answer']}"
            for ex in fewshot_examples
        )
        user = (
            GE_L4_FEW_SHOT.format(examples=examples_text, context_text=context)
            + "\n"
            + GE_L4_ANSWER_FORMAT
        )

    return GE_SYSTEM_PROMPT, user


GE_TASK_FORMATTERS = {
    "ge-l1": format_ge_l1_prompt,
    "ge-l2": format_ge_l2_prompt,
    "ge-l3": format_ge_l3_prompt,
    "ge-l4": format_ge_l4_prompt,
}


def format_ge_prompt(
    task: str,
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Dispatch to task-specific formatter.

    Args:
        task: 'ge-l1', 'ge-l2', 'ge-l3', or 'ge-l4'
        record: dict with at least 'context_text' key
        config: 'zero-shot' or '3-shot'
        fewshot_examples: list of example dicts for 3-shot

    Returns:
        (system_prompt, user_prompt) tuple
    """
    formatter = GE_TASK_FORMATTERS.get(task)
    if formatter is None:
        raise ValueError(f"Unknown task: {task}. Choose from {list(GE_TASK_FORMATTERS)}")
    return formatter(record, config, fewshot_examples)
