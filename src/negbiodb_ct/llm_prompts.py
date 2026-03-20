"""Prompt templates for CT LLM benchmark tasks CT-L1 through CT-L4.

Each task has zero-shot and 3-shot templates.
3-shot: 3 independent example sets (fewshot_set=0,1,2) for variance reporting.

Mirrors src/negbiodb/llm_prompts.py structure from DTI domain.
"""

from __future__ import annotations

CT_SYSTEM_PROMPT = (
    "You are a clinical trial expert with deep knowledge of drug development, "
    "regulatory science, and clinical pharmacology."
)

# ── CT-L1: 5-way categories ────────────────────────────────────────────────

CT_L1_CATEGORIES = {
    "A": "Safety — Trial failed due to drug toxicity, adverse events, or safety signals",
    "B": "Efficacy — Trial failed to demonstrate therapeutic benefit vs control",
    "C": "Enrollment — Trial failed to recruit sufficient participants",
    "D": "Strategic — Trial was discontinued for business, strategic, or portfolio reasons",
    "E": "Other — Trial failed due to study design flaws, regulatory issues, or other reasons",
}

# 8-way DB categories → 5-way MCQ mapping
CATEGORY_TO_MCQ: dict[str, str] = {
    "safety": "A",
    "efficacy": "B",
    "enrollment": "C",
    "strategic": "D",
    "design": "E",
    "regulatory": "E",
    "pharmacokinetic": "E",
    "other": "E",
}

# Few-shot set seeds (3 independent sets for variance)
FEWSHOT_SEEDS = [42, 43, 44]


# ── CT-L1: MCQ Classification ──────────────────────────────────────────────

CT_L1_QUESTION = (
    "Based on the clinical trial information below, classify the primary "
    "reason for this trial's failure.\n\n"
    "{context_text}\n\n"
    "Categories:\n"
    "A) Safety — Trial failed due to drug toxicity, adverse events, or safety signals\n"
    "B) Efficacy — Trial failed to demonstrate therapeutic benefit vs control\n"
    "C) Enrollment — Trial failed to recruit sufficient participants\n"
    "D) Strategic — Trial was discontinued for business, strategic, or portfolio reasons\n"
    "E) Other — Trial failed due to study design flaws, regulatory issues, or other reasons\n"
)

CT_L1_ANSWER_FORMAT = "Respond with ONLY a single letter (A, B, C, D, or E)."

CT_L1_FEW_SHOT = """Here are examples of clinical trial failure classification:

{examples}

Now classify the following:

{context_text}

Categories:
A) Safety — Trial failed due to drug toxicity, adverse events, or safety signals
B) Efficacy — Trial failed to demonstrate therapeutic benefit vs control
C) Enrollment — Trial failed to recruit sufficient participants
D) Strategic — Trial was discontinued for business, strategic, or portfolio reasons
E) Other — Trial failed due to study design flaws, regulatory issues, or other reasons
"""


# ── CT-L2: Structured Extraction ───────────────────────────────────────────

CT_L2_QUESTION = (
    "Extract structured failure information from the following clinical trial "
    "termination report. Return a JSON object with the fields specified below.\n\n"
    "{context_text}\n\n"
    "Required JSON fields:\n"
    "- failure_category: one of [efficacy, safety, pharmacokinetic, enrollment, strategic, design, regulatory, other]\n"
    "- failure_subcategory: specific reason (e.g., 'futility', 'hepatotoxicity', 'slow accrual')\n"
    "- affected_system: organ system affected (null if not applicable)\n"
    "- severity_indicator: one of [mild, moderate, severe, fatal, null]\n"
    "- quantitative_evidence: true if text mentions specific numbers or statistics\n"
    "- decision_maker: who terminated [sponsor, dsmb, regulatory, investigator, null]\n"
    "- patient_impact: brief description of patient safety impact (null if not mentioned)\n\n"
    "Return ONLY valid JSON, no additional text."
)

CT_L2_FEW_SHOT = """Extract structured failure information from clinical trial termination reports.

{examples}

Now extract from this report:

{context_text}

Return ONLY valid JSON, no additional text."""


# ── CT-L3: Reasoning ───────────────────────────────────────────────────────

CT_L3_QUESTION = (
    "The following clinical trial was confirmed as a FAILURE. Based on the trial "
    "data below, provide a scientific explanation for why this drug failed in this "
    "clinical trial.\n\n"
    "{context_text}\n\n"
    "Your explanation should address:\n"
    "1. Mechanism — What is the drug's mechanism of action and why might it be "
    "insufficient for this condition?\n"
    "2. Evidence interpretation — What do the statistical results tell us about "
    "the magnitude and confidence of the failure?\n"
    "3. Clinical factors — What trial design, patient population, or disease "
    "biology factors may have contributed?\n"
    "4. Broader context — How does this failure relate to the known challenges "
    "of treating this condition or drug class?\n\n"
    "Provide a thorough explanation in 3-5 paragraphs."
)

CT_L3_FEW_SHOT = """Here are examples of scientific reasoning about clinical trial failures:

{examples}

Now explain the following:

{context_text}"""


# ── CT-L4: Tested vs Untested Discrimination ──────────────────────────────

CT_L4_QUESTION = (
    "Based on your knowledge of clinical trials and drug development, determine "
    "whether the following drug-condition combination has ever been tested in a "
    "registered clinical trial (e.g., on ClinicalTrials.gov).\n\n"
    "{context_text}\n\n"
    "On the first line, respond with ONLY 'tested' or 'untested'.\n"
    "On the second line, provide brief evidence for your answer (e.g., trial "
    "identifiers, known results, or reasoning for why it was/wasn't tested)."
)

CT_L4_ANSWER_FORMAT = (
    "On the first line, respond with ONLY 'tested' or 'untested'. "
    "On the second line, provide brief evidence."
)

CT_L4_FEW_SHOT = """Here are examples of tested/untested drug-condition pair determination:

{examples}

Now determine:

{context_text}"""


# ── Helper functions ────────────────────────────────────────────────────────


def format_ct_l1_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format CT-L1 MCQ prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = CT_L1_QUESTION.format(context_text=context) + "\n" + CT_L1_ANSWER_FORMAT
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\nAnswer: {ex['gold_answer']}"
            for ex in fewshot_examples
        )
        user = (
            CT_L1_FEW_SHOT.format(examples=examples_text, context_text=context)
            + "\n"
            + CT_L1_ANSWER_FORMAT
        )

    return CT_SYSTEM_PROMPT, user


def format_ct_l2_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format CT-L2 extraction prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = CT_L2_QUESTION.format(context_text=context)
    else:
        import json

        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\n\nExtracted:\n{json.dumps(ex.get('gold_extraction', {}), indent=2)}"
            for ex in fewshot_examples
        )
        user = CT_L2_FEW_SHOT.format(examples=examples_text, context_text=context)

    return CT_SYSTEM_PROMPT, user


def format_ct_l3_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format CT-L3 reasoning prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = CT_L3_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\n\nExplanation:\n{ex.get('gold_reasoning', 'N/A')}"
            for ex in fewshot_examples
        )
        user = CT_L3_FEW_SHOT.format(examples=examples_text, context_text=context)

    return CT_SYSTEM_PROMPT, user


def format_ct_l4_prompt(
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Format CT-L4 tested/untested prompt. Returns (system_prompt, user_prompt)."""
    context = record["context_text"]

    if config == "zero-shot" or not fewshot_examples:
        user = CT_L4_QUESTION.format(context_text=context)
    else:
        examples_text = "\n\n---\n\n".join(
            f"{ex['context_text']}\nAnswer: {ex['gold_answer']}"
            for ex in fewshot_examples
        )
        user = (
            CT_L4_FEW_SHOT.format(examples=examples_text, context_text=context)
            + "\n"
            + CT_L4_ANSWER_FORMAT
        )

    return CT_SYSTEM_PROMPT, user


CT_TASK_FORMATTERS = {
    "ct-l1": format_ct_l1_prompt,
    "ct-l2": format_ct_l2_prompt,
    "ct-l3": format_ct_l3_prompt,
    "ct-l4": format_ct_l4_prompt,
}


def format_ct_prompt(
    task: str,
    record: dict,
    config: str = "zero-shot",
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Dispatch to task-specific formatter.

    Args:
        task: 'ct-l1', 'ct-l2', 'ct-l3', or 'ct-l4'
        record: dict with at least 'context_text' key
        config: 'zero-shot' or '3-shot'
        fewshot_examples: list of example dicts for 3-shot

    Returns:
        (system_prompt, user_prompt) tuple
    """
    formatter = CT_TASK_FORMATTERS.get(task)
    if formatter is None:
        raise ValueError(f"Unknown task: {task}. Choose from {list(CT_TASK_FORMATTERS)}")
    return formatter(record, config, fewshot_examples)
