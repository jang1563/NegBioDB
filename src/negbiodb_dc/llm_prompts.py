"""Prompt templates for DC LLM benchmark tasks DC-L1 through DC-L4.

Each task has zero-shot and 3-shot templates.
3-shot: 3 independent example sets (fewshot_set=0,1,2) for variance reporting.

Mirrors src/negbiodb_vp/llm_prompts.py structure.
"""

from __future__ import annotations

import json

from negbiodb_dc.llm_dataset import FEWSHOT_SEEDS  # noqa: F401 — re-exported for convenience

DC_SYSTEM_PROMPT = (
    "You are a pharmacologist with expertise in drug combination therapy, "
    "synergy/antagonism mechanisms, and cancer biology. "
    "Provide precise, evidence-based answers."
)

# ── DC-L1: 4-way drug combination classification MCQ ──────────────────────

DC_L1_CATEGORIES = {
    "A": "Strongly synergistic — Drugs amplify each other's effects (ZIP > 10)",
    "B": "Weakly synergistic/Additive — Combined effect near sum of individual effects",
    "C": "Antagonistic — Drugs interfere with each other (ZIP < -5)",
    "D": "Strongly antagonistic — Drugs strongly counteract each other (ZIP < -10)",
}

DC_L1_QUESTION = (
    "Classify the likely pharmacological outcome when these two drugs are "
    "combined.\n\n"
    "{context_text}\n\n"
    "Categories:\n"
    "A) Strongly synergistic — Drugs amplify each other's effects (ZIP > 10)\n"
    "B) Weakly synergistic/Additive — Combined effect near sum of individual effects\n"
    "C) Antagonistic — Drugs interfere with each other (ZIP < -5)\n"
    "D) Strongly antagonistic — Drugs strongly counteract each other (ZIP < -10)\n"
)

DC_L1_ANSWER_FORMAT = "Respond with ONLY a single letter (A, B, C, or D)."

DC_L1_FEW_SHOT = """Here are examples of drug combination outcome classification:

{examples}

Now classify the following:

{context_text}

Categories:
A) Strongly synergistic — Drugs amplify each other's effects (ZIP > 10)
B) Weakly synergistic/Additive — Combined effect near sum of individual effects
C) Antagonistic — Drugs interfere with each other (ZIP < -5)
D) Strongly antagonistic — Drugs strongly counteract each other (ZIP < -10)
"""


# ── DC-L2: Mechanism extraction ───────────────────────────────────────────

DC_L2_QUESTION = (
    "Extract drug combination interaction details from the following "
    "pharmacology report. Return a JSON object with the fields specified "
    "below.\n\n"
    "{context_text}\n\n"
    "Required JSON fields:\n"
    '- drug_a: {{"name": "...", "mechanism": "...", "primary_targets": ["..."]}}\n'
    '- drug_b: {{"name": "...", "mechanism": "...", "primary_targets": ["..."]}}\n'
    "- shared_targets: list of shared gene targets (empty list if none)\n"
    "- interaction_type: one of [antagonistic, synergistic, additive]\n"
    "- mechanism_of_interaction: one of [competitive_binding, pathway_crosstalk, "
    "metabolic_interference, pharmacokinetic_interaction, target_redundancy, "
    "feedback_activation, opposing_pathways, unknown]\n"
    "- affected_pathways: list of affected signaling pathways\n\n"
    "Return ONLY valid JSON, no additional text."
)

DC_L2_FEW_SHOT = """Extract drug combination interaction details from pharmacology reports.

{examples}

Now extract from this report:

{context_text}

Return ONLY valid JSON, no additional text."""


# ── DC-L3: Antagonism reasoning ───────────────────────────────────────────

DC_L3_QUESTION = (
    "The following drug combination has been experimentally confirmed as "
    "antagonistic — the drugs interfere with each other's therapeutic "
    "effects when combined.\n\n"
    "{context_text}\n\n"
    "Explain why this drug combination is antagonistic (fails to show synergy). "
    "Your explanation should address:\n"
    "1. Mechanistic reasoning — How do the drugs' mechanisms of action conflict "
    "or interfere with each other?\n"
    "2. Pathway analysis — What pathway-level interactions (competition, feedback "
    "loops, crosstalk) explain the antagonism?\n"
    "3. Pharmacological context — Are there PK/PD interactions, metabolic "
    "interference, or dosing considerations that contribute?\n"
    "4. Therapeutic relevance — What are the clinical implications? What "
    "alternative combinations might work better?\n\n"
    "Provide a thorough explanation in 3-5 paragraphs."
)

DC_L3_FEW_SHOT = """Here are examples of scientific reasoning about drug combination antagonism:

{examples}

Now explain the following:

{context_text}"""


# ── DC-L4: Tested vs Untested Discrimination ─────────────────────────────

DC_L4_QUESTION = (
    "Based on your knowledge of drug combination screening databases "
    "(DrugComb, NCI-ALMANAC, AstraZeneca-Sanger DREAM), determine whether "
    "the following drug pair has ever been experimentally tested for "
    "synergy/antagonism.\n\n"
    "{context_text}\n\n"
    "On the first line, respond with ONLY 'tested' or 'untested'.\n"
    "On the second line, provide brief evidence for your answer."
)

DC_L4_ANSWER_FORMAT = (
    "On the first line, respond with ONLY 'tested' or 'untested'. "
    "On the second line, provide brief evidence."
)

DC_L4_FEW_SHOT = """Here are examples of tested/untested drug combination determination:

{examples}

Now determine:

{context_text}"""


# ── Helper functions ──────────────────────────────────────────────────────

_L3_MAX_EXAMPLE_CHARS = 1200
_L3_MAX_REASONING_CHARS = 600


def _truncate_text(text: str | None, max_chars: int) -> str:
    """Truncate text at word boundary, appending '[...]' if truncated."""
    if not text:
        return "N/A"
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + " [...]"


def format_dc_l1_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format DC-L1 MCQ prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = DC_L1_QUESTION.format(context_text=context) + "\n" + DC_L1_ANSWER_FORMAT
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\nAnswer: {ex['gold_answer']}"
            for ex in fewshot_examples
        )
        user = (
            DC_L1_FEW_SHOT.format(examples=examples_text, context_text=context)
            + "\n"
            + DC_L1_ANSWER_FORMAT
        )

    return DC_SYSTEM_PROMPT, user


def format_dc_l2_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format DC-L2 extraction prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = DC_L2_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\n\nExtracted:\n{json.dumps(ex.get('gold_extraction', {}), indent=2)}"
            for ex in fewshot_examples
        )
        user = DC_L2_FEW_SHOT.format(examples=examples_text, context_text=context)

    return DC_SYSTEM_PROMPT, user


def format_dc_l3_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format DC-L3 reasoning prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = DC_L3_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{_truncate_text(ex['context_text'], _L3_MAX_EXAMPLE_CHARS)}\n\n"
            f"Explanation:\n{_truncate_text(ex.get('gold_reasoning', 'N/A'), _L3_MAX_REASONING_CHARS)}"
            for ex in fewshot_examples
        )
        user = DC_L3_FEW_SHOT.format(examples=examples_text, context_text=context)

    return DC_SYSTEM_PROMPT, user


def format_dc_l4_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format DC-L4 tested/untested prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = DC_L4_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\nAnswer: {ex['gold_answer']}"
            for ex in fewshot_examples
        )
        user = (
            DC_L4_FEW_SHOT.format(examples=examples_text, context_text=context)
            + "\n"
            + DC_L4_ANSWER_FORMAT
        )

    return DC_SYSTEM_PROMPT, user


DC_TASK_FORMATTERS = {
    "dc-l1": format_dc_l1_prompt,
    "dc-l2": format_dc_l2_prompt,
    "dc-l3": format_dc_l3_prompt,
    "dc-l4": format_dc_l4_prompt,
}


def format_dc_prompt(
    task: str,
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Dispatch to task-specific formatter.

    Args:
        task: 'dc-l1', 'dc-l2', 'dc-l3', or 'dc-l4'
        record: dict with at least 'context_text' key
        config: 'zero-shot' or '3-shot'
        fewshot_examples: list of example dicts for 3-shot

    Returns:
        (system_prompt, user_prompt) tuple
    """
    formatter = DC_TASK_FORMATTERS.get(task)
    if formatter is None:
        raise ValueError(f"Unknown task: {task}. Choose from {list(DC_TASK_FORMATTERS)}")
    return formatter(record, config, fewshot_examples)
