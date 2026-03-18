"""Prompt templates for LLM benchmark tasks L1–L4.

Each task has zero-shot and 3-shot templates.
3-shot: 3 independent example sets (fewshot_set=0,1,2) for variance reporting.
"""

import json

SYSTEM_PROMPT = (
    "You are a pharmaceutical scientist with expertise in drug-target interactions, "
    "assay development, and medicinal chemistry. Provide precise, evidence-based answers."
)

# ── L1: Multiple Choice Classification ────────────────────────────────────────

L1_ZERO_SHOT = """{context}"""

L1_FEW_SHOT = """Here are some examples of drug-target interaction classification:

{examples}

Now classify the following:

{context}"""

L1_ANSWER_FORMAT = (
    "Respond with ONLY the letter of the correct answer: A, B, C, or D."
)

# ── L2: Structured Extraction from Abstract ──────────────────────────────────

L2_ZERO_SHOT = """Extract all negative drug-target interaction results from the following abstract.

Abstract:
{abstract_text}

For each negative result found, extract:
- compound: compound/drug name
- target: target protein/gene name
- target_uniprot: UniProt accession (if determinable)
- activity_type: type of measurement (IC50, Ki, Kd, EC50, etc.)
- activity_value: reported value with units
- activity_relation: relation (=, >, <, ~)
- assay_format: biochemical, cell-based, or in vivo
- outcome: inactive, weak, or inconclusive

Also report:
- total_inactive_count: total number of inactive results mentioned
- positive_results_mentioned: true/false

Respond in JSON format."""

L2_FEW_SHOT = """Extract negative drug-target interaction results from abstracts.

{examples}

Now extract from this abstract:

Abstract:
{abstract_text}

Respond in JSON format."""

# ── L3: Reasoning (Pilot) ────────────────────────────────────────────────────

L3_ZERO_SHOT = """{context}

Provide a detailed scientific explanation (3-5 paragraphs) covering:
1. Structural compatibility between compound and target binding site
2. Known selectivity profile and mechanism of action
3. Relevant SAR (structure-activity relationship) data
4. Pharmacological context and therapeutic implications"""

L3_FEW_SHOT = """Here are examples of scientific reasoning about inactive drug-target interactions:

{examples}

Now explain the following:

{context}"""

# ── L4: Tested vs Untested Discrimination ────────────────────────────────────

L4_ZERO_SHOT = """{context}"""

L4_FEW_SHOT = """Here are examples of tested/untested compound-target pair determination:

{examples}

Now determine:

{context}"""

L4_ANSWER_FORMAT = (
    "Respond with 'tested' or 'untested' on the first line. "
    "If tested, provide the evidence source (database, assay ID, or DOI) on the next line."
)


# ── Helper functions ──────────────────────────────────────────────────────────


def format_l1_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format L1 MCQ prompt.

    Returns (system_prompt, user_prompt).
    """
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = L1_ZERO_SHOT.format(context=context) + "\n\n" + L1_ANSWER_FORMAT
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\nAnswer: {ex['correct_answer']}"
            for ex in fewshot_examples
        )
        user = (
            L1_FEW_SHOT.format(examples=examples_text, context=context)
            + "\n\n"
            + L1_ANSWER_FORMAT
        )

    return SYSTEM_PROMPT, user


def format_l2_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format L2 extraction prompt."""
    abstract = record["abstract_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = L2_ZERO_SHOT.format(abstract_text=abstract)
    else:
        examples_text = "\n\n---\n\n".join(
            f"Abstract:\n{ex['abstract_text']}\n\nExtracted:\n{json.dumps(ex.get('gold_extraction', {}), indent=2)}"
            for ex in fewshot_examples
        )
        user = L2_FEW_SHOT.format(examples=examples_text, abstract_text=abstract)

    return SYSTEM_PROMPT, user


def format_l3_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format L3 reasoning prompt."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = L3_ZERO_SHOT.format(context=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\n\nExplanation:\n{ex.get('gold_reasoning', 'N/A')}"
            for ex in fewshot_examples
        )
        user = L3_FEW_SHOT.format(examples=examples_text, context=context)

    return SYSTEM_PROMPT, user


def format_l4_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format L4 tested/untested prompt."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = L4_ZERO_SHOT.format(context=context) + "\n\n" + L4_ANSWER_FORMAT
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\nAnswer: {ex['correct_answer']}"
            for ex in fewshot_examples
        )
        user = (
            L4_FEW_SHOT.format(examples=examples_text, context=context)
            + "\n\n"
            + L4_ANSWER_FORMAT
        )

    return SYSTEM_PROMPT, user


TASK_FORMATTERS = {
    "l1": format_l1_prompt,
    "l2": format_l2_prompt,
    "l3": format_l3_prompt,
    "l4": format_l4_prompt,
}


def format_prompt(
    task: str,
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Dispatch to task-specific formatter.

    Returns (system_prompt, user_prompt).
    """
    formatter = TASK_FORMATTERS.get(task)
    if formatter is None:
        raise ValueError(f"Unknown task: {task}. Choose from {list(TASK_FORMATTERS)}")
    return formatter(record, config, fewshot_examples)

