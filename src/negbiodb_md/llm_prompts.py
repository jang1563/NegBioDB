"""Prompt templates for MD LLM benchmark tasks MD-L1 through MD-L4.

Each task has zero-shot and 3-shot templates.
3-shot: 3 independent example sets (fewshot_set=0,1,2) for variance reporting.

Mirrors src/negbiodb_dc/llm_prompts.py structure.
"""

from __future__ import annotations

import json

from negbiodb_md.llm_dataset import FEWSHOT_SEEDS  # noqa: F401

MD_SYSTEM_PROMPT = (
    "You are a metabolomics expert with deep knowledge of human metabolism, "
    "biomarker discovery, and clinical metabolomics studies. "
    "Provide precise, evidence-based answers grounded in metabolomics science."
)

# ── MD-L1: 4-way biomarker MCQ ────────────────────────────────────────────────

MD_L1_QUESTION = (
    "Given the following study context, identify which metabolite is NOT a "
    "significant biomarker for the specified disease.\n\n"
    "{context_text}\n\n"
    "Which of the following metabolites is NOT a significant biomarker for "
    "{disease_name} in this study?\n\n"
    "A) {choice_a}\n"
    "B) {choice_b}\n"
    "C) {choice_c}\n"
    "D) {choice_d}\n"
)

MD_L1_ANSWER_FORMAT = "Respond with ONLY a single letter (A, B, C, or D)."

MD_L1_FEW_SHOT = """Here are examples of identifying non-biomarkers:

{examples}

Now answer the following:

{context_text}

Which of the following metabolites is NOT a significant biomarker for {disease_name} in this study?

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}
"""


def format_l1_prompt(record: dict, few_shot_examples: list[dict] | None = None) -> str:
    """Format a prompt for MD-L1.

    Args:
        record:            L1 record dict from md_l1.jsonl
        few_shot_examples: list of example dicts (zero-shot if None)

    Returns:
        Formatted prompt string
    """
    choices = record.get("choices", {})
    context = record.get("context", "")
    disease = record.get("metadata", {}).get("disease_name", "the disease")
    # Reconstruct disease name from question if metadata missing
    question = record.get("question", "")
    if "NOT a significant biomarker for" in question:
        disease = question.split("NOT a significant biomarker for")[1].split(" in this study")[0].strip()

    if few_shot_examples:
        examples_text = "\n\n".join(_format_l1_example(ex) for ex in few_shot_examples)
        return MD_L1_FEW_SHOT.format(
            examples=examples_text,
            context_text=context,
            disease_name=disease,
            choice_a=choices.get("A", ""),
            choice_b=choices.get("B", ""),
            choice_c=choices.get("C", ""),
            choice_d=choices.get("D", ""),
        )

    return MD_L1_QUESTION.format(
        context_text=context,
        disease_name=disease,
        choice_a=choices.get("A", ""),
        choice_b=choices.get("B", ""),
        choice_c=choices.get("C", ""),
        choice_d=choices.get("D", ""),
    ) + "\n" + MD_L1_ANSWER_FORMAT


def _format_l1_example(rec: dict) -> str:
    choices = rec.get("choices", {})
    disease = rec.get("question", "").split("NOT a significant biomarker for")[-1].split(" in this study")[0].strip()
    return (
        f"Context:\n{rec.get('context', '')}\n"
        f"Question: Which metabolite is NOT a biomarker for {disease}?\n"
        f"A) {choices.get('A', '')}  B) {choices.get('B', '')}  "
        f"C) {choices.get('C', '')}  D) {choices.get('D', '')}\n"
        f"Answer: {rec.get('gold_answer', '')}"
    )


# ── MD-L2: Structured field extraction ───────────────────────────────────────

MD_L2_QUESTION = (
    "Extract metabolomics study result details from the following report. "
    "Return a JSON object with the fields specified below.\n\n"
    "{context_text}\n\n"
    "Required JSON fields:\n"
    '- metabolite: name of the metabolite measured\n'
    '- disease: name of the disease studied\n'
    '- fold_change: numeric fold change (null if not reported)\n'
    '- platform: analytical platform (nmr, lc_ms, gc_ms, or other)\n'
    '- biofluid: sample type (blood, urine, csf, tissue, or other)\n'
    '- outcome: one of [significant, not_significant]\n\n'
    "Return ONLY valid JSON, no additional text."
)

MD_L2_FEW_SHOT = """Extract metabolomics study result details from reports.

{examples}

Now extract from the following:

{context_text}

Required JSON fields:
- metabolite: name of the metabolite measured
- disease: name of the disease studied
- fold_change: numeric fold change (null if not reported)
- platform: analytical platform (nmr, lc_ms, gc_ms, or other)
- biofluid: sample type (blood, urine, csf, tissue, or other)
- outcome: one of [significant, not_significant]

Return ONLY valid JSON, no additional text."""


def format_l2_prompt(record: dict, few_shot_examples: list[dict] | None = None) -> str:
    context = record.get("context", "")
    if few_shot_examples:
        examples_text = "\n\n---\n\n".join(_format_l2_example(ex) for ex in few_shot_examples)
        return MD_L2_FEW_SHOT.format(examples=examples_text, context_text=context)
    return MD_L2_QUESTION.format(context_text=context)


def _format_l2_example(rec: dict) -> str:
    gold = rec.get("gold_fields", {})
    return (
        f"Report:\n{rec.get('context', '')}\n\n"
        f"JSON output:\n{json.dumps(gold, indent=2)}"
    )


# ── MD-L3: Free-text reasoning ────────────────────────────────────────────────

MD_L3_QUESTION = (
    "Analyze the following metabolomics result and explain why the metabolite "
    "was NOT found to be a significant biomarker for the disease.\n\n"
    "{context_text}\n\n"
    "{question_text}\n\n"
    "Your explanation should address:\n"
    "1. Metabolite biology: the biochemical role and metabolic pathway of this compound\n"
    "2. Disease mechanism: why this disease may or may not affect this metabolite\n"
    "3. Study context: limitations of this specific study (platform, biofluid, sample size)\n"
    "4. Alternative hypotheses: testable explanations for the null finding\n"
)

MD_L3_FEW_SHOT = """Analyze metabolomics null findings and explain the biology.

{examples}

Now analyze the following:

{context_text}

{question_text}

Your explanation should address:
1. Metabolite biology: the biochemical role and metabolic pathway of this compound
2. Disease mechanism: why this disease may or may not affect this metabolite
3. Study context: limitations of this specific study (platform, biofluid, sample size)
4. Alternative hypotheses: testable explanations for the null finding
"""


def format_l3_prompt(record: dict, few_shot_examples: list[dict] | None = None) -> str:
    context = record.get("context", "")
    question = record.get("question", "")
    if few_shot_examples:
        examples_text = "\n\n---\n\n".join(_format_l3_example(ex) for ex in few_shot_examples)
        return MD_L3_FEW_SHOT.format(
            examples=examples_text,
            context_text=context,
            question_text=question,
        )
    return MD_L3_QUESTION.format(context_text=context, question_text=question)


def _format_l3_example(rec: dict) -> str:
    return (
        f"Context:\n{rec.get('context', '')}\n\n"
        f"Question: {rec.get('question', '')}\n\n"
        f"Reasoning:\n{rec.get('gold_reasoning', '')}"
    )


# ── MD-L4: Real vs synthetic discrimination ───────────────────────────────────

MD_L4_QUESTION = (
    "You will be shown a metabolite-disease non-association record. "
    "Determine whether this represents a REAL finding from an actual "
    "metabolomics study, or a SYNTHETIC (randomly generated) pair that "
    "was never experimentally tested.\n\n"
    "{context_text}\n\n"
    "Is this metabolite-disease non-association REAL (from an actual study) "
    "or SYNTHETIC (randomly paired, never tested)?\n\n"
    "A) Real — from an actual metabolomics study where this metabolite was "
    "measured and found non-significant\n"
    "B) Synthetic — randomly generated pair, never experimentally tested\n"
)

MD_L4_ANSWER_FORMAT = "Respond with ONLY a single letter (A or B)."

MD_L4_FEW_SHOT = """Distinguish real metabolomics null findings from synthetic (untested) pairs.

{examples}

Now classify:

{context_text}

Is this metabolite-disease non-association REAL or SYNTHETIC?

A) Real — from an actual metabolomics study
B) Synthetic — randomly generated pair, never tested
"""


def format_l4_prompt(record: dict, few_shot_examples: list[dict] | None = None) -> str:
    context = record.get("context", "")
    if few_shot_examples:
        examples_text = "\n\n---\n\n".join(_format_l4_example(ex) for ex in few_shot_examples)
        return MD_L4_FEW_SHOT.format(examples=examples_text, context_text=context)
    return MD_L4_QUESTION.format(context_text=context) + "\n" + MD_L4_ANSWER_FORMAT


def _format_l4_example(rec: dict) -> str:
    label_letter = "A" if rec.get("label") == 1 else "B"
    return (
        f"Record:\n{rec.get('context', '')}\n\n"
        f"Answer: {label_letter} ({'Real' if rec.get('label') == 1 else 'Synthetic'})"
    )
